
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import logging as log
import time
import math

from wisp.utils import PsDebugger, PerfTimer
from wisp.ops.spc import sample_spc

import wisp.ops.spc as wisp_spc_ops
import wisp.ops.grid as grid_ops

from wisp.models.decoders import BasicDecoder

import kaolin.ops.spc as spc_ops

from wisp.accelstructs import OctreeAS

class DenseGrid():
    """ This is a feature grid where the features are defined in a dense grid to interpolate.
    """
    def __init__(self,
        feature_dim        : int,
        grid_dim           : int   = 3,
        interpolation_type : str   = 'linear',
        multiscale_type    : str   = 'cat',
        feature_std        : float = 0.0,
        feature_bias       : float = 0.0,
        codebook_bitwidth  : int   = 16,
        blas_level         : int   = 7,
        **kwargs
    ):
        """Initialize the hash grid class.

        Args:
            dim (int): The dimension of the output data
            feature_dim (int): The dimension of the features stored on the grid.
            grid_dim (int): The dimension of the grid (2 for image, 3 for shapes).
            interpolation_type (str): The type of interpolation function.
            multiscale_type (str): The type of multiscale aggregation. Usually 'sum' or 'cat'.
                                   Note that 'cat' will change the decoder input dimension.
            feature_std (float): The features are initialized with a Gaussian distribution with the given
                                 standard deviation.
            feature_bias (float): The mean of the Gaussian distribution.
            codebook_bitwidth (int): The bitwidth of the codebook.
            blas_level (int): The level of the octree to be used as the BLAS.

        Returns:
            (void): Initializes the class.
        """
        super().__init__()
        self.grid_dim = grid_dim
        self.feature_dim = feature_dim
        self.interpolation_type = interpolation_type
        self.multiscale_type = multiscale_type

        self.feature_std = feature_std
        self.feature_bias = feature_bias
        self.codebook_bitwidth = codebook_bitwidth
        self.blas_level = blas_level

        self.kwargs = kwargs

        self.blas = OctreeAS()
        self.blas.init_dense(self.blas_level)
        self.dense_points = spc_ops.unbatched_get_level_points(self.blas.points, self.blas.pyramid, self.blas_level).clone()
        self.num_cells = self.dense_points.shape[0]
        self.occupancy = torch.zeros(self.num_cells)


    def interpolate(self, coords, lod_idx, pidx=None):
        """ Query multiscale features.

            coords (torch.FloatTensor): coords of shape [batch, num_samples, 3]
            lod_idx  (int): int specifying the index to ``active_lods``
            pidx (torch.LongTensor): Primitive indices of shape [batch]. Unused here.

        Returns:
            (torch.FloatTensor): interpolated features of shape [batch, num_samples, feature_dim]
        """
        timer = PerfTimer(activate=False, show_memory=False)

        batch, num_samples, _ = coords.shape

        feats = grid_ops.hashgrid(coords, self.resolutions, self.codebook_bitwidth,
                                  self.grid_dim, lod_idx, self.codebook)

        if self.multiscale_type == 'cat':
            return feats
        elif self.multiscale_type == 'sum':
            return feats.reshape(batch, num_samples, len(self.resolutions), feats.shape[-1] // len(self.resolutions)).sum(-2)
        else:
            raise NotImplementedError
