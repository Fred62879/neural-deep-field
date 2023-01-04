
import torch
import torch.nn as nn

from wisp.utils import PerfTimer
from wisp.models.embedders.pe import RandGaus
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
            seed = 0
            embedder = RandGaus((1, pe_dim, omega, sigma, pe_bias, seed, False))
        else:
            embedder = None
        return embedder

    def get_input_coords_dim(self):
        """ Infer the dimension of the input RA/DEC coordinate.
        """
        if self.kwargs["quantize_latent"]:
            coords_dim = self.kwargs["qtz_latent_dim"]
        elif self.kwargs["coords_embed_method"] == "positional":
            coords_dim = self.kwargs["coords_embed_dim"]
        elif self.kwargs["coords_embed_method"] == "grid":
            coords_dim = self.kwargs["feature_dim"]
            if self.kwargs["multiscale_type"] == 'cat':
                coords_dim *= self.kwargs["num_lods"]
        else:
            coords_dim = 2
        return coords_dim

    def init_decoder(self):
        # encode ra/dec coords first and then combine with wave
        coords_dim = self.get_input_coords_dim()

        if self.kwargs["wave_embed_method"] == "positional":
            if self.kwargs["hps_combine_method"] == "add":
                assert(self.kwargs["wave_embed_dim"] == coords_dim)
                input_dim = self.kwargs["wave_embed_dim"]
            elif self.kwargs["hps_combine_method"] == "concat":
                input_dim = self.kwargs["wave_embed_dim"] + coords_dim

        else: # coords and wave are not embedded
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
        if self.kwargs["print_shape"]: print('hps_decoder', coords.shape)
        spectra = self.decod(coords)
        if scaler is not None: spectra *= scaler
        spectra = self.norm(spectra)
        return spectra

    def train_with_full_wave(self, dataholder, data, **net_args):
        """ During training, some latents will be decoded, combining with full wave.
            Currently only supports spectra coords (incl. gt, dummy that requires
              spectrum plotting during training time).
            Latents that require full wave are in the front of the tensor.
            **TODO** pass only supervision gt spectra here (get rid of non-supervision
              gt, and dummy coords).
        """
        num_spectra_coords = net_args["num_spectra_coords"]
        has_scaler = "scaler" in data and data["scaler"] is not None
        has_redshift = "redshift" in data and data["redshift"] is not None

        # get data that requires full wave
        spectra_latents = data["latents"][-num_spectra_coords:]
        spectra_scaler = None if not has_scaler else data["scaler"][-num_spectra_coords:]
        spectra_redshift = None if not has_redshift else data["redshift"][-num_spectra_coords:]

        if self.kwargs["print_shape"]: print('hps_decoder', spectra_latents.shape)
        if self.kwargs["print_shape"] and spectra_scaler is not None:
            print('hps_decoder', spectra_scaler.shape)
        if self.kwargs["print_shape"] and spectra_redshift is not None:
            print('hps_decoder', spectra_redshift.shape)

        # generate hyperspectral latents
        if self.kwargs["print_shape"]: print(net_args["full_wave"].shape)
        full_wave = net_args["full_wave"][None,:,None].tile(num_spectra_coords,1,1)
        if self.kwargs["print_shape"]: print('hps_decoder, full wave', full_wave.shape)
        spectra_hps_latents = self.convert(full_wave, spectra_latents, redshift=spectra_redshift)

        # generate spectra
        dataholder["spectra"] = self.reconstruct_spectra(spectra_hps_latents, spectra_scaler)

        # leave only latents that require sampled wave
        data["latents"] = data["latents"][:-num_spectra_coords]
        if has_scaler:   data["scaler"]   = data["scaler"][:-num_spectra_coords]
        if has_redshift: data["redshift"] = data["redshift"][:-num_spectra_coords]

    def forward(self, dataholder, data, **net_args):
        """ @Param
              dataholder: temporarily holds return data.
              data: output from nerf and/or quantization, including:
                    latents:  (embedded or original) coords. [bsz,num_samples,coords_embed_dim or 2 or 3]
                    scaler:   (if perform quantization) unique scaler value for each coord. [bsz,1]
                    redshift: (if perform quantization) unique redshift value for each coord. [bsz,1]
                    codebook_loss
                    latents
                    embed_ids
              net_args (other data):
                    wave: lambda values, used to convert ra/dec to hyperspectral latents. [bsz,num_samples]
                    trans: corresponding transmission values of lambda. [bsz,num_samples]
                    full_wave: not None if do spectra supervision. [num_spectra_coords,full_num_samples]
                    spectra_supervision_train: bool, indicates whether currently doing training with supervision
                    num_spectra_coords:

            @Return (add new fields to input data)
              intensity: reconstructed pixel values
              spectra:   reconstructed spectra
        """

        if self.kwargs["print_shape"]: print('hps_decoder',data["latents"].shape)

        if net_args["spectra_supervision_train"]:
            self.train_with_full_wave(dataholder, data, **net_args)
            if self.kwargs["print_shape"]: print('hps_decoder', data["latents"].shape)

        # still have latents to decode besides full wave training latents (e.g. spectra supervision)
        if data["latents"].shape[0] > 0:
            scaler = None if "scaler" not in data else data["scaler"]
            redshift = None if "redshift" not in data else data["redshift"]

            hps_latents = self.convert(net_args["wave"], data["latents"], redshift=redshift)
            if self.kwargs["print_shape"]: print('hps_decoder', hps_latents.shape)

            spectra = self.reconstruct_spectra(hps_latents, scaler)
            if self.kwargs["print_shape"]: print('hps_decoder', spectra.shape)

            intensity = self.inte(spectra[...,0], **net_args)
            if self.kwargs["print_shape"]: print('hps_decoder', intensity.shape)

            if "spectra" not in dataholder and "spectra" in net_args["other_channels"]:
                dataholder["spectra"] = spectra
            dataholder["intensity"] = intensity

        return dataholder
