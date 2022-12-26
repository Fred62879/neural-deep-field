
import torch
import torch.nn as nn

from wisp.utils import PerfTimer
from wisp.models.decoders import BasicDecoder, Siren
from wisp.models.activations import get_activation_class
from wisp.models.layers import get_layer_class, Normalization
from wisp.models.hypers.hps_converter import HyperSpectralConverter
from wisp.models.hypers.hps_integrator import HyperSpectralIntegrator


class HyperSpectralDecoder(nn.Module):

    def __init__(self, **kwargs):

        super(HyperSpectralDecoder, self).__init__()

        self.convert = HyperSpectralConverter(**kwargs)

        # encode ra/dec coords first and then combine with wave
        if kwargs["hps_convert_method"] == "add":
            assert(kwargs["wave_embed_dim"] == kwargs["coords_embed_dim"])
            input_dim = kwargs["wave_embed_dim"]

        elif kwargs["hps_convert_method"] == "concat":
            if kwargs["coords_embed_method"] == "positional":
                assert(kwargs["hps_decod_activation_type"] == "relu")
                coords_embed_dim = kwargs["coords_embed_dim"]

            elif kwargs["coords_embed_method"] == "grid":
                assert(kwargs["hps_decod_activation_type"] == "relu")
                coords_embed_dim = kwargs["feature_dim"]
                if kwargs["multiscale_type"] == 'cat':
                    coords_embed_dim *= kwargs["num_lods"]

            input_dim = kwargs["wave_embed_dim"] + coords_embed_dim

        else:
            # coords and wave are not embedded
            input_dim = 3

        '''
        # encode ra/dec coords with wave together
        if kwargs["quantize_latent"]:
            input_dim = kwargs["qtz_latent_dim"]

        elif kwargs["coords_embed_method"] == "grid":
            assert(kwargs["hps_decod_activation_type"] == "relu")
            input_dim = kwargs["feature_dim"]
            if kwargs["multiscale_type"] == 'cat':
                input_dim *= kwargs["num_lods"]

        elif kwargs["coords_embed_method"] == "positional":
            assert(kwargs["hps_decod_activation_type"] == "relu")
            input_dim = kwargs["coords_embed_dim"]

        else:
            assert(kwargs["hps_decod_activation_type"] == "sin")
            input_dim = 3
        '''

        if kwargs["hps_decod_activation_type"] == "relu":
            self.decode = BasicDecoder(
                input_dim, 1, get_activation_class(kwargs["hps_decod_activation_type"]),
                True, layer=get_layer_class(kwargs["hps_decod_layer_type"]),
                num_layers=kwargs["hps_decod_num_layers"]+1,
                hidden_dim=kwargs["hps_decod_hidden_dim"], skip=[])

        elif kwargs["hps_decod_activation_type"] == "sin":
            self.decode = Siren(
                input_dim, 1, kwargs["hps_decod_num_layers"], kwargs["hps_decod_hidden_dim"],
                kwargs["hps_siren_first_w0"], kwargs["hps_siren_hidden_w0"],
                kwargs["hps_siren_seed"], kwargs["hps_siren_coords_scaler"],
                kwargs["hps_siren_last_linear"])

        else: raise ValueError("Unrecognized hyperspectral decoder activation type.")

        self.norm = Normalization(kwargs["mlp_output_norm_method"])
        self.integrate = HyperSpectralIntegrator(**kwargs)

    def forward(self, data, **kwargs):
        """ @Param
              data: output from nerf, including:
                   latent: (embedded or original) coords [bsz,num_samples,coords_embed_dim or 2 or 3]
                   scaler: (if perform quantization) unique scaler value for each coord [bsz,1]
                   redshift: (if perform quantization) unique redshift value for each coord [bsz,1]
              wave: lambda values, used to convert ra/dec to hyperspectral coords [bsz,num_samples]
              trans: corresponding transmission values of lambda [bsz,num_samples]
        """
        latents = data["latents"]
        scaler = None if "scaler" not in data else data["scaler"]
        redshift = None if "redshift" not in data else data["redshift"]

        hyperspectral_coords = self.convert(kwargs["wave"], latents, redshift=redshift)
        #hyperspectral_coords = latents

        spectra = self.decode(hyperspectral_coords)
        spectra = self.norm(spectra)
        if scaler is not None: spectra *= scaler
        data["spectra"] = spectra

        intensity = self.integrate(spectra[...,0], **kwargs)
        data["intensity"] = intensity
        return data
