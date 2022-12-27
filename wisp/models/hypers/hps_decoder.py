
import torch
import torch.nn as nn

from wisp.utils import PerfTimer
from wisp.models.embedders.pe import RandGausLinr
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
        self.spectra_supervision = "spectra_supervision" in kwargs["tasks"]
        if self.spectra_supervision:
            self.num_spectra_coords = kwargs["num_supervision_spectra"]

        wave_embedder = self.init_wave_embedder()
        self.convert = HyperSpectralConverter(wave_embedder, **kwargs)
        self.decod = self.init_decoder()
        self.norm = Normalization(kwargs["mlp_output_norm_method"])
        self.inte = HyperSpectralIntegrator(integrate=integrate, **kwargs)

    def init_wave_embedder(self):
        wave_embed_method = self.kwargs["wave_embed_method"]
        if wave_embed_method == "positional":
            pe_dim = self.kwargs["wave_embed_dim"]
            sigma = 1
            omega = 1
            pe_bias = True
            embedder = RandGausLinr((1, pe_dim, sigma, omega, pe_bias, False))
        else:
            raise ValueError("Unrecognized wave embedding method.")
        return embedder

    def init_decoder(self):
        # encode ra/dec coords first and then combine with wave
        if self.kwargs["hps_combine_method"] == "add":
            assert(self.kwargs["wave_embed_dim"] == self.kwargs["coords_embed_dim"])
            input_dim = self.kwargs["wave_embed_dim"]

        elif self.kwargs["hps_combine_method"] == "concat":
            if self.kwargs["coords_embed_method"] == "positional":
                assert(self.kwargs["hps_decod_activation_type"] == "relu")
                coords_embed_dim = self.kwargs["coords_embed_dim"]

            elif self.kwargs["coords_embed_method"] == "grid":
                assert(self.kwargs["hps_decod_activation_type"] == "relu")
                coords_embed_dim = self.kwargs["feature_dim"]
                if self.kwargs["multiscale_type"] == 'cat':
                    coords_embed_dim *= self.kwargs["num_lods"]

            input_dim = self.kwargs["wave_embed_dim"] + coords_embed_dim
        else:
            # coords and wave are not embedded
            input_dim = 3

        '''
        # encode ra/dec coords with wave together
        if self.kwargs["quantize_latent"]:
            input_dim = self.kwargs["qtz_latent_dim"]

        elif self.kwargs["coords_embed_method"] == "grid":
            assert(self.kwargs["hps_decod_activation_type"] == "relu")
            input_dim = self.kwargs["feature_dim"]
            if self.kwargs["multiscale_type"] == 'cat':
                input_dim *= self.kwargs["num_lods"]

        elif self.kwargs["coords_embed_method"] == "positional":
            assert(self.kwargs["hps_decod_activation_type"] == "relu")
            input_dim = self.kwargs["coords_embed_dim"]

        else:
            assert(self.kwargs["hps_decod_activation_type"] == "sin")
            input_dim = 3
        '''

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

    def reconstruct_spectra(self, coords, scaler):
        spectra = self.decod(coords)
        if scaler is not None: spectra *= scaler
        spectra = self.norm(spectra)
        return spectra

    def reconstruct_supervision_spectra(self, latents, wave, scaler, redshift):
        """ Reconstruct gt spectra.
            @Param
              latetns: (original or embedded) ra/dec coords.
              wave: all lambda values. [num_spectra_coords,full_num_samples,1]
              redshift: corresponding redshift values of each wave.
        """
        # slice spectra data
        spectra_latents = latents[-self.num_spectra_coords:]
        spectra_scaler = None if scaler is None else \
            scaler[-self.num_spectra_coords:]
        spectra_redshift = None if redshift is None else \
            redshift[-self.num_spectra_coords:]

        # generate hyperspectral latents
        spectra_hps_latents = self.convert(wave, spectra_latents, redshift=spectra_redshift)

        # generate spectra
        recon_spectra = self.reconstruct_spectra(spectra_hps_latents, spectra_scaler)
        return recon_spectra

    def forward(self, data, **kwargs):
        """ @Param
              data: output from nerf, including:
                    latents:  (embedded or original) coords. [bsz,num_samples,coords_embed_dim or 2 or 3]
                    scaler:   (if perform quantization) unique scaler value for each coord. [bsz,1]
                    redshift: (if perform quantization) unique redshift value for each coord. [bsz,1]

              kwargs (includes other data):
                    wave: lambda values, used to convert ra/dec to hyperspectral latents. [bsz,num_samples]
                    trans: corresponding transmission values of lambda. [bsz,num_samples]
                    full_wave: not None if do spectra supervision. [num_spectra_coords,full_num_samples]

            @Return (add new fields to input data)
              intensity: reconstructed pixel values
              spectra:   reconstructed spectra
        """
        latents = data["latents"]
        scaler = None if "scaler" not in data or not self.scale else data["scaler"]
        redshift = None if "redshift" not in data or not self.scale else data["redshift"]

        if self.spectra_supervision:
            full_wave = kwargs["full_wave"][:,None,:,None].tile(1,self.num_spectra_coords,1,1) # ****** replace
            data["spectra"] = self.reconstruct_supervision_spectra(
                latents, full_wave, scaler, redshift)

            # slice out spectra supervision data
            latents = latents[:-self.num_spectra_coords]
            scaler = None if scaler is None else scaler[:-self.num_spectra_coords]
            redshift = None if redshift is None else redshift[:-self.num_spectra_coords]

        hps_latents = self.convert(kwargs["wave"], latents, redshift=redshift)
        spectra = self.reconstruct_spectra(hps_latents, scaler)
        intensity = self.inte(spectra[...,0], **kwargs)
        data["intensity"] = intensity
        return data
