
import torch.nn as nn

from wisp.utils import PerfTimer
from wisp.models.grids import *
from wisp.models.embedders import RandGaus


class Encoder(nn.Module):
    """ Encoder class for coordinates.
    """
    def __init__(self, encode_method, embedder_args, *grid_args, **kwargs):
        super(Encoder, self).__init__()

        self.kwargs = kwargs
        self.encode_method = encode_method

        if encode_method == "positional":
            self.embedder = RandGaus(embedder_args)
        elif encode_method == "grid":
            self.init_grid(*grid_args)

    def init_grid(self, *grid_args):
        (grid_type, grid_feature_dim, grid_dim, grid_base_lod, grid_num_lods,
         grid_interpolation_type, grid_multiscale_type, min_grid_res, max_grid_res) = grid_args

        if grid_type == "OctreeGrid":
            grid_class = OctreeGrid
        elif grid_type == "CodebookOctreeGrid":
            grid_class = CodebookOctreeGrid
        elif grid_type == "TriplanarGrid":
            grid_class = TriplanarGrid
        elif grid_type == "HashGrid":
            grid_class = HashGrid
        else:
            raise NotImplementedError

        self.grid = grid_class(
            grid_feature_dim,
            base_lod=grid_base_lod,
            num_lods=grid_num_lods,
            interpolation_type=grid_interpolation_type,
            multiscale_type=grid_multiscale_type, **self.kwargs)

        self.grid.init_from_geometric(min_grid_res, max_grid_res, grid_num_lods)

        if grid_multiscale_type == 'cat':
            self.effective_feature_dim = self.grid.feature_dim * grid_num_lods
        else: grid_effective_feature_dim = self.grid.feature_dim

    def forward(self, coords, lod_idx=None):
        """ Encode given coords
            @Param
              coords: [batch_size,num_samples,coord_dim]
            @Return
              latents: [batch_size,num_samples,latent_dim]
        """
        timer = PerfTimer(activate=False, show_memory=True)

        batch, num_samples, _ = coords.shape

        if self.encode_method == "positional":
            latents = self.embedder(coords) # [bsz,num_samples,coords_embed_dim]
            timer.check("rf_hyperspectral_pe")

        elif self.encode_method == "grid":
            if lod_idx is None:
                lod_idx = len(self.grid.active_lods) - 1
            latents = self.grid.interpolate(coords, lod_idx)
            latents = latents.reshape(batch, num_samples, self.effective_feature_dim)
            timer.check("rf_hyperspectra_interpolate")
        else:
            latents = coords

        return latents
