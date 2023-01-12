
import torch
import torch.nn as nn

from wisp.utils import PerfTimer
from wisp.models.embedders import Encoder
from wisp.models.decoders import Decoder
from wisp.models.layers import Normalization
from wisp.models.hypers.hps_converter import HyperSpectralConverter
from wisp.models.hypers.hps_integrator import HyperSpectralIntegrator


class HyperSpectralDecoder(nn.Module):

    def __init__(self, integrate=True, scale=True, **kwargs):

        super(HyperSpectralDecoder, self).__init__()

        self.kwargs = kwargs
        self.scale = scale

        self.init_encoder()
        self.convert = HyperSpectralConverter(self.wave_encoder, **kwargs)
        self.spectra_decoder = Decoder(**kwargs)
        self.norm = Normalization(kwargs["mlp_output_norm_method"])
        self.inte = HyperSpectralIntegrator(integrate=integrate, **kwargs)

    def init_encoder(self):
        embedder_args = (
            1,
            self.kwargs["wave_embed_dim"],
            self.kwargs["wave_embed_omega"],
            self.kwargs["wave_embed_sigma"],
            self.kwargs["wave_embed_bias"],
            self.kwargs["wave_embed_seed"]
        )
        self.wave_encoder = Encoder(
            input_dim=1,
            encode_method=self.kwargs["wave_encode_method"],
            embedder_args=embedder_args,
            **self.kwargs
        )

    def reconstruct_spectra(self, wave, latents, scaler, redshift, scale=True):
        latents = self.convert(wave, latents, redshift)
        if self.kwargs["print_shape"]: print('hps_decoder', latents.shape)

        spectra = self.spectra_decoder(latents)[...,0]
        if self.scale and scaler is not None:
            spectra = torch.exp((scaler * spectra.T).T)
            #spectra = (scaler * spectra.T).T
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
            #assert(latents[2080] == latents[4096])
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
