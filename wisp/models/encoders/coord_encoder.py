
import torch.nn as nn

from wisp.utils import PerfTimer
from wisp.models.grids import *
from wisp.models.embedders import RandGaus


class Encoder(nn.Module):
    """ Encoder class for coordinates.
    """
    def __init__(self, encode_method, embedder_args, **kwargs):
        super(Encoder, self).__init__()

        self.encode_method = encode_method

        if encode_method == "positional":
            self.embedder = RandGaus(embedder_args)
        elif encode_method == "grid":
            self.init_grid()

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
        else:
            raise NotImplementedError

        self.grid = grid_class(
            self.kwargs["grid_feature_dim"],
            grid_dim=self.kwargs["grid_dim"],
            base_lod=self.kwargs["grid_base_lod"],
            num_lods=self.kwargs["grid_num_lods"],
            interpolation_type=self.kwargs["grid_interpolation_type"],
            multiscale_type=self.kwargs["grid_multiscale_type"],
            **self.kwargs)

        self.grid.init_from_geometric(
            self.kwargs["min_grid_res"], self.kwargs["max_grid_res"], self.kwargs["grid_num_lods"])

        if self.multiscale_type == 'cat':
            self.effective_feature_dim = self.grid.feature_dim * self.kwargs["grid_num_lods"]
        else: self.effective_feature_dim = self.grid.feature_dim

    def forward(self, coords):
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
