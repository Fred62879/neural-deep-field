
import torch
import torch.nn as nn

from wisp.utils import PerfTimer
from wisp.utils.common import get_input_latents_dim

from wisp.models.encoders import Encoder
from wisp.models.decoders import BasicDecoder, Siren
from wisp.models.activations import get_activation_class
from wisp.models.layers import get_layer_class, Normalization
from wisp.models.hypers.hps_converter import HyperSpectralConverter
from wisp.models.hypers.hps_integrator import HyperSpectralIntegrator


class HyperSpectralDecoder(nn.Module):

    def __init__(self, integrate=True, scale=True, **kwargs):

        super(HyperSpectralDecoder, self).__init__()

        self.kwargs = kwargs
        self.scale = scale

        self.init_wave_encoder()
        self.convert = HyperSpectralConverter(self.wave_encoder, **kwargs)
        self.decode = self.init_decoder()
        self.norm = Normalization(kwargs["mlp_output_norm_method"])
        self.inte = HyperSpectralIntegrator(integrate=integrate, **kwargs)

    def init_wave_encoder(self):
        wave_embedder_args = (
            1, self.kwargs["wave_embed_dim"], self.kwargs["wave_embed_omega"],
            self.kwargs["wave_embed_sigma"], self.kwargs["wave_embed_bias"],
            self.kwargs["wave_embed_seed"])

        self.wave_encoder = Encoder(
            encode_method=self.kwargs["wave_encode_method"],
            embedder_args=wave_embedder_args, **self.kwargs)

    def init_decoder(self):
        # encode ra/dec coords first and then combine with wave
        if self.kwargs["quantize_latent"]:
            latents_dim = self.kwargs["qtz_latent_dim"]
        else: latents_dim = get_input_latents_dim(**self.kwargs)

        if self.kwargs["wave_encode_method"] == "positional":
            if self.kwargs["hps_combine_method"] == "add":
                assert(self.kwargs["wave_encode_dim"] == latents_dim)
                input_dim = self.kwargs["wave_embed_dim"]
            elif self.kwargs["hps_combine_method"] == "concat":
                input_dim = self.kwargs["wave_embed_dim"] + latents_dim

        else: # coords and wave are not encoded
            input_dim = 3

        if self.kwargs["hps_decod_activation_type"] == "relu":
            decoder = BasicDecoder(
                input_dim, 1, get_activation_class(self.kwargs["hps_decod_activation_type"]),
                True, layer=get_layer_class(self.kwargs["hps_decod_layer_type"]),
                num_layers=self.kwargs["hps_decod_num_layers"]+1,
                hidden_dim=self.kwargs["hps_decod_hidden_dim"], skip=[])

        elif self.kwargs["hps_decod_activation_type"] == "sin":
            decoder = Siren(
                input_dim, 1, self.kwargs["hps_decod_num_layers"], self.kwargs["hps_decod_hidden_dim"],
                self.kwargs["hps_siren_first_w0"], self.kwargs["hps_siren_hidden_w0"],
                self.kwargs["hps_siren_seed"], self.kwargs["hps_siren_coords_scaler"],
                self.kwargs["hps_siren_last_linear"])

        else: raise ValueError("Unrecognized hyperspectral decoder activation type.")
        return decoder

    def reconstruct_spectra(self, wave, latents, scaler, redshift, scale=True):
        hps_latents = self.convert(wave, latents, redshift)
        if self.kwargs["print_shape"]: print('hps_decoder', hps_latents.shape)

        spectra = self.decode(hps_latents)[...,0]
        if self.scale and scaler is not None: spectra *= scaler
        spectra = self.norm(spectra)
        return spectra

    def train_with_full_wave(self, latents, full_wave, num_spectra_coords, ret):
        """ During training, some latents will be decoded, combining with full wave.
            Currently only supports spectra coords (incl. gt, dummy that requires
              spectrum plotting during training time).
            Latents that require full wave are at the end of the tensor.
        """
        latents = latents[-num_spectra_coords:]
        scaler = None if ret["scaler"] is None else ret["scaler"][-num_spectra_coords:]
        redshift = None if ret["redshift"] is None else ret["redshift"][-num_spectra_coords:]
        full_wave = full_wave[None,:,None].tile(num_spectra_coords,1,1)

        if self.kwargs["print_shape"]: print('hps_decoder', spectra_latents.shape)
        if self.kwargs["print_shape"]: print('hps_decoder, full wave', full_wave.shape)

        ret["spectra"] = self.reconstruct_spectra(full_wave, latents, scaler, redshift)

    def forward(self, latents, wave, trans, nsmpl, ret, full_wave=None, num_spectra_coords=-1):
        """ @Param
              latents:   (encoded or original) coords. [bsz,num_samples,coords_encode_dim or 2 or 3]
              wave:      lambda values, used to convert ra/dec to hyperspectral latents. [bsz,num_samples]
              trans:     corresponding transmission values of lambda. [bsz,num_samples]
              nsmpl:     average number of lambda samples falling within each band. [num_bands]
              full_wave: not None if do spectra supervision. [num_spectra_coords,full_num_samples]
              num_spectra_coords: > 0 if spectra supervision.

              ret (output from nerf and/or quantization): {
                "scaler":        unique scaler value for each coord. [bsz,1]
                "redshift":      unique redshift value for each coord. [bsz,1]
                "embed_ids":     ids of embedding each pixel's latent is quantized to.
                "codebook_loss": loss for codebook optimization.
              }

            @Return (add new fields to input data)
              intensity: reconstructed pixel values
              spectra:   reconstructed spectra
        """
        if self.kwargs["print_shape"]: print('hps_decoder',latents.shape)

        if num_spectra_coords > 0:
            self.train_with_full_wave(latents, full_wave, num_spectra_coords, ret)
            latents = latents[:-num_spectra_coords]
            if ret["scaler"] is not None: ret["scaler"] = ret["scaler"][:-num_spectra_coords]
            if ret["redshift"] is not None: ret["redshift"] = ret["redshift"][:-num_spectra_coords]
            if self.kwargs["print_shape"]: print('hps_decoder', latents.shape)

        if latents.shape[0] > 0:
            spectra = self.reconstruct_spectra(wave, latents, ret["scaler"], ret["redshift"])
            if "spectra" not in ret: ret["spectra"] = spectra
            if self.kwargs["print_shape"]: print('hps_decoder', spectra.shape)

            intensity = self.inte(spectra, trans, nsmpl)
            if self.kwargs["print_shape"]: print('hps_decoder', intensity.shape)
            ret["intensity"] = intensity
