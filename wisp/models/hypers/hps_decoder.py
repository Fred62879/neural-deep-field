
import torch
import torch.nn as nn

from wisp.utils import PerfTimer
from wisp.models.decoders import BasicDecoder
from wisp.models.layers import get_layer_class, Fn
from wisp.models.activations import get_activation_class
from wisp.models.hypers.hps_converter import HyperSpectralConverter
from wisp.models.hypers.hps_integrator import HyperSpectralIntegrator


class HyperSpectralDecoder(nn.Module):

    def __init__(self, **kwargs):

        super(HyperSpectralDecoder, self).__init__()

        #self.convert = HyperSpectralConverter(**kwargs)

        '''
        # encode ra/dec coords with wave separately
        if kwargs["hps_convert_method"] == "add":
            assert(kwargs["wave_embed_dim"] == kwargs["coords_embed_dim"])
            input_dim = kwargs["wave_embed_dim"]
        elif kwargs["hps_convert_method"] == "concat":
            input_dim = kwargs["wave_embed_dim"] + kwargs["coords_embed_dim"]
        else:
            # coords and wave are not embedded
            input_dim = 3
        '''

        # encode ra/dec coords with wave together
        if kwargs["coords_embed_method"] == "grid":
            if kwargs["multiscale_type"] == 'cat':
                input_dim = kwargs["feature_dim"] * kwargs["num_lods"]
            else: input_dim = kwargs["feature_dim"]
        elif kwargs["coords_embed_method"] == "positional":
            input_dim = kwargs["coords_embedder_dim"]

        self.decode = BasicDecoder(
            input_dim, 1, get_activation_class(kwargs["hps_decod_activation_type"]),
            True, layer=get_layer_class(kwargs["hps_decod_layer_type"]),
            num_layers=kwargs["hps_decod_num_layers"]+1,
            hidden_dim=kwargs["hps_decod_hidden_dim"], skip=[])

        self.sinh = Fn(torch.sinh)

        self.integrate = HyperSpectralIntegrator(**kwargs)

    def forward(self, data, **kwargs):
        """
            @Param
              data: output from nerf, including:
                   latent: (embedded or original) ra/dec coords [bsz,coords_embed_dim or 2]
                   scaler: (if perform quantization) unique scaler value for each coord [bsz,1]
                   redshift: (if perform quantization) unique redshift value for each coord [bsz,1]
              wave: lambda values, used to convert ra/dec to hyperspectral coords [bsz,num_samples]
              trans: corresponding transmission values of lambda [bsz,num_samples]
        """
        latents = data["latents"]
        #print(latents.shape, latents[0])
        scaler = None if "scaler" not in data else data["scaler"]
        redshift = None if "redshift" not in data else data["redshift"]

        #hyperspectral_coords = self.convert(kwargs["wave"], latents, redshift=redshift)
        hyperspectral_coords = latents

        spectra = self.decode(hyperspectral_coords)
        spectra = self.sinh(spectra)
        if scaler is not None: spectra *= scaler

        intensity = self.integrate(spectra[...,0], **kwargs)
        data["intensity"] = intensity
        return data
