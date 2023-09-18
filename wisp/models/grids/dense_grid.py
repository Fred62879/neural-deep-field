
import math
import torch
import numpy as np
import torch.nn as nn
import logging as log
import wisp.ops.grid as grid_ops

from wisp.models.grids import BLASGrid
from wisp.utils import PsDebugger, PerfTimer


class DenseGrid(nn.Module):
    """ This is a feature grid where the features are defined in a dense grid to interpolate.
    """
    def __init__(self,
        feature_dim        : int,
        grid_dim           : int   = 2,
        feature_std        : float = 0.0,
        feature_bias       : float = 0.0,
        multiscale_type    : str   = 'cat',
        interpolation_type : str   = 'linear',
        **kwargs
    ):
        """ Initialize the dense grid class.
            @Param
              dim (int): The dimension of the output data
              feature_dim (int): The dimension of the features stored on the grid.
              grid_dim (int): The dimension of the grid (2 for image, 3 for shapes).
              interpolation_type (str): The type of interpolation function.
              multiscale_type (str): The type of multiscale aggregation. Usually 'sum' or 'cat'.
                                     Note that 'cat' will change the decoder input dimension.
              feature_std (float): The features are initialized with a Gaussian
                                   distribution with the given standard deviation.
              feature_bias (float): The mean of the Gaussian distribution.
              codebook_bitwidth (int): The bitwidth of the codebook.
              blas_level (int): The level of the octree to be used as the BLAS.
        """

        super(DenseGrid, self).__init__()

        self.kwargs = kwargs
        self.grid_dim = grid_dim
        self.feature_dim = feature_dim
        self.feature_std = feature_std
        self.feature_bias = feature_bias
        self.multiscale_type = multiscale_type
        self.interpolation_type = interpolation_type
        self.align_corners = kwargs["interpolation_align_corners"]
        assert self.grid_dim == 2

    def init_from_geometric(self, min_width, max_width, num_lods):
        """ Build the multiscale grid with a geometric sequence.
            This is an implementation of the geometric multiscale grid from
              instant-ngp (https://nvlabs.github.io/instant-ngp/).
            See Section 3 Equations 2 and 3 for more details.
        """
        b = np.exp((np.log(max_width) - np.log(min_width)) / num_lods)
        log.info(f"value of b is {b}")
        resolutions = [int(np.floor(min_width*(b**l))) for l in range(num_lods)]
        self.init_from_resolutions(resolutions)

    def init_from_resolutions(self, resolutions):
        """ Build a multiscale grid from a list of resolutions.
        """
        log.info(f"Active Resolutions: {resolutions}")

        self.resolutions = resolutions
        self.num_lods = len(resolutions)
        self.active_lods = [x for x in range(self.num_lods)]
        self.max_lod = self.num_lods - 1

        self.codebook = nn.ParameterList([])
        for res in resolutions:
            fts = torch.zeros((1, self.feature_dim, res, res))
            fts += torch.randn_like(fts) * self.feature_std
            self.codebook.append(nn.Parameter(fts))

    def interpolate(self, coords, lod_idx, pidx=None):
        """  Query multiscale features.
             @Param
               coords (torch.FloatTensor): coords of shape [batch, num_samples, 3]
               lod_idx  (int): int specifying the index to ``active_lods``
               pidx (torch.LongTensor): Primitive indices of shape [batch]. Unused here.
             @Return
               interpolated features of shape [batch, num_samples, feature_dim]
        """
        timer = PerfTimer(activate=self.kwargs["activate_model_timer"],
                          show_memory=self.kwargs["show_memory"])
        batch, num_samples, _ = coords.shape
        feats = grid_ops.dense_grid(
            coords, self.resolutions, lod_idx, self.codebook, self.align_corners
        )

        if self.multiscale_type == 'cat':
            ret = feats
        elif self.multiscale_type == 'sum':
            ret = feats.reshape(batch, num_samples, self.num_lods,
                                feats.shape[-1] // self.num_lods).sum(-2)
        else:
            raise NotImplementedError
        return ret
