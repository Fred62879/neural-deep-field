
import torch
import logging as log

from wisp.models.grids import *
from wisp.utils import PerfTimer
from wisp.models.nefs import BaseNeuralField
from wisp.models.decoders import BasicDecoder
from wisp.models.layers import get_layer_class, Fn
from wisp.models.activations import get_activation_class
from wisp.models.embedders import get_positional_embedder, RandGausLinr


class NeuralHyperSpectral(BaseNeuralField):
    """ Model for encoding RA/DEC coordinates.
        In hyperspectral setup, output is embedded latent variables.
        Otherwise, output is decoded pixel intensities.

        Discrete from how the base_nef works,
          here we use either the positional encoding or the grid for embedding.
    """

    def init_embedder(self):
        if self.kwargs["coords_embed_method"] != "positional": return

        pe_dim = self.kwargs["coords_embed_dim"]
        sigma = 1
        omega = 1
        pe_bias = True
        #float_tensor = torch.cuda.FloatTensor
        verbose = False
        self.embedder = RandGausLinr((3, pe_dim, sigma, omega, pe_bias, verbose))

    def init_grid(self):
        """ Initialize the grid object. """
        if self.kwargs["coords_embed_method"] != "grid": return

        if self.grid_type == "OctreeGrid":
            grid_class = OctreeGrid
        elif self.grid_type == "CodebookOctreeGrid":
            grid_class = CodebookOctreeGrid
        elif self.grid_type == "TriplanarGrid":
            grid_class = TriplanarGrid
        elif self.grid_type == "HashGrid":
            grid_class = HashGrid
        else:
            raise NotImplementedError

        self.grid = grid_class \
            (self.feature_dim, space_dim=self.space_dim, base_lod=self.base_lod, num_lods=self.num_lods,
             interpolation_type=self.interpolation_type,
             multiscale_type=self.multiscale_type, **self.kwargs)

        self.grid.init_from_geometric(
            self.kwargs["min_grid_res"], self.kwargs["max_grid_res"], self.num_lods)

        if self.multiscale_type == 'cat':
            self.effective_feature_dim = self.grid.feature_dim * self.num_lods
        else:
            self.effective_feature_dim = self.grid.feature_dim

    def init_decoder(self):
        """ Initializes the decoder object.
        """
        # hyperspectral setup doesn't need decoder
        if self.space_dim == 3: return

        if self.kwargs["coords_embed_method"] == "positional":
            input_dim = self.kwargs["coords_embed_dim"]
        elif self.kwargs["coords_embed_method"] == "grid":
            input_dim = self.effective_feature_dim
        else:
            input_dim = 3 #2 ^^

        self.decoder_intensity = BasicDecoder \
            (input_dim, self.output_dim, get_activation_class(self.activation_type),
             True, layer=get_layer_class(self.layer_type), num_layers=self.num_layers+1,
             hidden_dim=self.hidden_dim, skip=[])

        self.sinh_scaling = Fn(torch.sinh)

    def get_nef_type(self):
        return 'hyperspectral'

    def register_forward_functions(self):
        """ Register forward functions with the channels that they output.
            For hyperspectral setup, we need latent embedding as output.
            Otherwise, we need pixel intensity values as output.
        """
        channels = ["intensity"] if self.space_dim == 2 else ["latents"]
        self._register_forward_function( self.hyperspectral, channels )

    def hyperspectral(self, coords, pidx=None, lod_idx=None):
        """ Compute hyperspectral intensity for the provided coordinates.
            @Params:
              coords (torch.FloatTensor): tensor of shape [1, batch, num_samples, 2/3]
              pidx (torch.LongTensor): SPC point_hierarchy indices of shape [batch].
                                       Unused in the current implementation.
              lod_idx (int): index into active_lods. If None, will use the maximum LOD.
                             Currently interpolation doesn't use this.
            @Return
              {"indensity": torch.FloatTensor }:
                - Output intensity tensor of shape [batch, num_samples, 3]
        """
        timer = PerfTimer(activate=False, show_memory=True)

        coords = coords[0] # ****** replace
        if coords.ndim == 2:
            batch, _ = coords.shape
        elif coords.ndim == 3:
            batch, num_samples, _ = coords.shape
        else:
            raise Exception("Wrong coordinate dimension.")

        timer.check("rf_hyperspectral_preprocess")

        # embed 2D coords into high-dimensional vectors with PE or the grid
        if self.kwargs["coords_embed_method"] == "positional":
            feats = self.embedder(coords) # [bsz,num_samples,coords_embed_dim]
            timer.check("rf_hyperspectral_pe")

        elif self.kwargs["coords_embed_method"] == "grid":
            if lod_idx is None:
                lod_idx = len(self.grid.active_lods) - 1

            feats = self.grid.interpolate(coords, lod_idx)
            feats = feats.reshape(-1, self.effective_feature_dim)
            timer.check("rf_hyperspectra_interpolate")
        else:
            feats = coords
            log.info("no embedding performed on the coordinates.")

        timer.check("rf_hyperspectral_embedding")

        if self.space_dim == 3:
            feats = feats.reshape(batch, num_samples, -1)
            return dict(latents=feats)

        intensity = self.decoder_intensity(feats)
        intensity = self.sinh_scaling(intensity)
        timer.check("rf_hyperspectral_decode")
        return dict(intensity=intensity)
