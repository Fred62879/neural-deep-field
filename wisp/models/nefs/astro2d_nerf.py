
import torch

from wisp.models.grids import *
from wisp.utils import PerfTimer
from wisp.models.encoders import Encoder
from wisp.models.nefs import BaseNeuralField
from wisp.models.activations import get_activation_class
from wisp.models.layers import get_layer_class, Normalization
from wisp.models.decoders import BasicDecoder, MLP_Relu, Siren


class AstroNerf(BaseNeuralField):
    """ Model for encoding RA/DEC coordinates.
        In hyperspectral setup, output is embedded latent variables.
        Otherwise, output is decoded pixel intensities.

        Different from how the base_nef works,
          here we use either the positional encoding or the grid for embedding.

        Usage:
          2D: embedding (positional/grid) -- relu -- sinh
              no embedding -- siren
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.space_dim = kwargs["space_dim"]
        self.num_bands = kwargs["num_bands"]

        super(AstroNerf, self).__init__(**kwargs)

        self.init_encoder()
        self.init_decoder()
        self.norm = Normalization(self.kwargs["mlp_output_norm_method"])

        torch.cuda.empty_cache()
        self._forward_functions = {}
        self.register_forward_functions()
        self.supported_channels = set([
            channel for channels in self._forward_functions.values()
            for channel in channels])

    def init_encoder(self):
        """ Initialize the encoder (positional encoding or grid interpolaton)
              for both ra/dec coordinates and wave (lambda values).
        """
        coords_embedder_args = (
            2, self.kwargs["coords_embed_dim"], self.kwargs["coords_embed_omega"],
            self.kwargs["coords_embed_sigma"], self.kwargs["coords_embed_bias"],
            self.kwargs["coords_embed_seed"])

        self.coords_encoder = Encoder(
            encode_method=self.kwargs["coords_encode_method"],
            embedder_args=coords_embedder_args, **self.kwargs)

    def init_decoder(self):
        """ Initializes the decoder object.
        """
        if self.kwargs["coords_encode_method"] == "positional":
            assert(self.activation_type == "relu")
            input_dim = self.kwargs["coords_embed_dim"]
        elif self.kwargs["coords_encode_method"] == "grid":
            assert(self.activation_type == "relu")
            input_dim = self.effective_feature_dim
        else:
            assert(self.activation_type == "sin")
            input_dim = self.space_dim

        if self.activation_type == "relu":
            self.decoder = BasicDecoder(
                input_dim, self.output_dim,
                get_activation_class(self.activation_type),
                True, layer=get_layer_class(self.layer_type),
                num_layers=self.num_layers+1,
                hidden_dim=self.hidden_dim, skip=[])

        elif self.activation_type == "sin":
            self.decoder = Siren(
                input_dim, self.num_bands, self.num_layers, self.hidden_dim,
                self.kwargs["siren_first_w0"], self.kwargs["siren_hidden_w0"],
                self.kwargs["siren_seed"], self.kwargs["siren_coords_scaler"],
                self.kwargs["siren_last_linear"])

        else: raise ValueError("Unrecognized decoder activation type.")

    def get_nef_type(self):
        return 'astro2d'

    def register_forward_functions(self):
        """ Register forward functions with the channels that they output.
        """
        channels = ["intensity"]
        self._register_forward_function( self.coords_to_pixel, channels )

    def coords_to_pixel(self, coords, pidx=None, lod_idx=None):
        """ Compute hyperspectral intensity for the provided coordinates.
            @Params:
              coords (torch.FloatTensor): tensor of shape [batch, num_samples, 2]
              pidx (torch.LongTensor): SPC point_hierarchy indices of shape [batch].
                                       Unused in the current implementation.
              lod_idx (int): index into active_lods. If None, will use the maximum LOD.
                             Currently interpolation doesn't use this.
            @Return
              {"indensity": torch.FloatTensor }:
                - Output intensity tensor of shape [batch, num_samples, num_bands]
        """
        timer = PerfTimer(activate=False, show_memory=True)

        batch, num_samples, _ = coords.shape

        coords = self.coords_encoder(coords)

        intensity = self.decoder(coords).view(-1, self.num_bands)
        intensity = self.norm(intensity)
        timer.check("rf_hyperspectral_decode")

        ret = dict(intensity=intensity)
        return ret
