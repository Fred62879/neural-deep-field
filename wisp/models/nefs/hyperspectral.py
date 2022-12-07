
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import logging as log
import time
import math
import inspect
from abc import abstractmethod

from wisp.ops.spc import sample_spc
from wisp.utils import PsDebugger, PerfTimer
from wisp.ops.geometric import sample_unif_sphere

from wisp.models.nefs import BaseNeuralField
from wisp.models.embedders import get_positional_embedder
from wisp.accelstructs import OctreeAS
from wisp.models.layers import get_layer_class
from wisp.models.activations import get_activation_class
from wisp.models.decoders import BasicDecoder
from wisp.models.grids import *

import kaolin.ops.spc as spc_ops


class NeuralHyperSpectral(BaseNeuralField):
    """Model for encoding hyperspectral cube
    """
    def init_embedder(self):
        """Initialize positional embedding objects.
        """
        return

    def init_decoder(self):
        """Initializes the decoder object.
        """
        if self.multiscale_type == 'cat':
            self.effective_feature_dim = self.grid.feature_dim * self.num_lods
        else:
            self.effective_feature_dim = self.grid.feature_dim

        self.input_dim = self.effective_feature_dim

        if self.position_input:
            self.input_dim += self.pos_embed_dim

        self.decoder_intensity = BasicDecoder \
            (self.input_dim, self.output_dim, get_activation_class(self.activation_type),
             True, layer=get_layer_class(self.layer_type), num_layers=self.num_layers+1,
             hidden_dim=self.hidden_dim, skip=[])

        '''
        self.decoder_hyperspectral = BasicDecoder \
            (self.input_dim + self.wave_pe_dim, self.spectra_dim,
             get_activation_class(self.activation_type),
             True, layer=get_layer_class(self.layer_type),
             num_layers=self.num_layers+1,
             hidden_dim=self.hidden_dim, skip=[])
        '''

    def init_grid(self):
        """Initialize the grid object.
        """
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
            (self.feature_dim, base_lod=self.base_lod, num_lods=self.num_lods,
             interpolation_type=self.interpolation_type,
             multiscale_type=self.multiscale_type, **self.kwargs)

    def get_nef_type(self):
        """Returns a text keyword of the neural field type.
        Returns:
            (str): The key type
        """
        return 'hyperspectral'

    def register_forward_functions(self):
        """Register forward functions with the channels that they output.

        This function should be overrided and call `self._register_forward_function` to
        tell the class which functions output what output channels. The function can be called
        multiple times to register multiple functions.
        """
        self._register_forward_function(self.hyperspectral, ["density"])

    def hyperspectral(self, coords, pidx=None, lod_idx=None):
        """Compute hyperspectral intensity for the provided coordinates.

        Args:
            coords (torch.FloatTensor): tensor of shape [batch, num_samples, 3]
            pidx (torch.LongTensor): SPC point_hierarchy indices of shape [batch].
                                     Unused in the current implementation.
            lod_idx (int): index into active_lods. If None, will use the maximum LOD.

        Returns:
            {"indensity": torch.FloatTensor }:
                - Output intensity tensor of shape [batch, num_samples, 3]
        """
        timer = PerfTimer(activate=False, show_memory=True)
        if lod_idx is None:
            lod_idx = len(self.grid.active_lods) - 1
        batch, num_samples, _ = coords.shape
        timer.check("rf_hyperspectral_preprocess")

        # Embed coordinates into high-dimensional vectors with the grid.
        feats = self.grid.interpolate(coords, lod_idx).reshape(-1, self.effective_feature_dim)
        timer.check("rf_hyperspectra_interpolate")

        if self.position_input:
            raise NotImplementedError

        # Decode high-dimensional vectors to output intensity.
        density = self.decoder_intensity(feats)
        timer.check("rf_hyperspectral_decode")

        return dict(density=density)
