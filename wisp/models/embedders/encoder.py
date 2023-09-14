
import torch.nn as nn

from wisp.utils import PerfTimer
from wisp.models.grids import *
from wisp.utils.common import get_input_latents_dim

import sys
sys.path.insert(0, './wisp/models/embedders')
from wisp.models.embedders.pe import RandGaus


class Encoder(nn.Module):
    """ Wrapper class for different encoding method:
         - Positional encoding
         - Grid
    """
    def __init__(self, input_dim, encode_method, embedder_args, **kwargs):
        super(Encoder, self).__init__()

        self.kwargs = kwargs
        self.input_dim = input_dim
        self.encode_method = encode_method

        if encode_method == "positional_encoding":
            self.embedder = RandGaus(embedder_args)
        elif encode_method == "grid":
            self.init_grid()
        else: raise NotImplementedError

    def init_grid(self):
        grid_type = self.kwargs["grid_type"]
        if grid_type == "OctreeGrid":
            grid_class = OctreeGrid
        elif grid_type == "CodebookOctreeGrid":
            grid_class = CodebookOctreeGrid
        elif grid_type == "TriplanarGrid":
            grid_class = TriplanarGrid
        elif grid_type == "HashGrid":
            grid_class = HashGrid
        elif grid_type == "DenseGrid":
            grid_class = DenseGrid
        else: raise NotImplementedError

        self.grid = grid_class(
            self.kwargs["grid_feature_dim"],
            base_lod=self.kwargs["grid_base_lod"],
            num_lods=self.kwargs["grid_num_lods"],
            interpolation_type=self.kwargs["grid_interpolation_type"],
            multiscale_type=self.kwargs["grid_multiscale_type"],
            feature_std=self.kwargs["grid_feature_std"],
            **self.kwargs)

        self.grid.init_from_geometric(
            self.kwargs["min_grid_res"],
            self.kwargs["max_grid_res"],
            self.kwargs["grid_num_lods"]
        )
        self.effective_feature_dim = get_input_latents_dim(**self.kwargs)

    def forward(self, coords, lod_idx=None):
        """ Encode given coords
            @Param
              coords: [...,batch_size,num_samples,coord_dim]
            @Return
              latents: [batch_size,num_samples,latent_dim]
        """
        timer = PerfTimer(activate=self.kwargs["activate_model_timer"], show_memory=False)

        (batch, num_samples) = coords.shape[-3:-1]

        if self.encode_method == "positional_encoding":
            latents = self.embedder(coords) # [bsz,num_samples,coords_embed_dim]
        elif self.encode_method == "grid":
            if lod_idx is None:
                lod_idx = len(self.grid.active_lods) - 1
            latents = self.grid.interpolate(coords, lod_idx)
            latents = latents.reshape(batch, num_samples, self.effective_feature_dim)
        else:
            latents = coords
        return latents
