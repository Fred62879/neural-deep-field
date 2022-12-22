
import torch

from wisp.models.grids import *
from wisp.utils import PerfTimer
from wisp.models.nefs import BaseNeuralField
from wisp.models.decoders import BasicDecoder
from wisp.models.hypers.pe import Rand_Gaus_Linr
from wisp.models.layers import get_layer_class, Fn
from wisp.models.activations import get_activation_class
from wisp.models.embedders import get_positional_embedder


class NeuralHyperSpectral(BaseNeuralField):
    """Model for encoding hyperspectral cube
    """
    def init_embedder(self):
        """Initialize positional embedding objects.
        """
        return

    def init_decoder(self, pe=False):
        """Initializes the decoder object.
        """
        if self.multiscale_type == 'cat':
            self.effective_feature_dim = self.grid.feature_dim * self.num_lods
        else:
            self.effective_feature_dim = self.grid.feature_dim

        self.pe = pe
        if self.pe:
            self.input_dim = 2
            pe_dim = 2000
            sigma = 1
            omega = 1
            pe_bias = True
            float_tensor = torch.cuda.FloatTensor
            verbose = False
            self.pe = Rand_Gaus_Linr((self.input_dim, pe_dim, sigma, omega, pe_bias, float_tensor, verbose))
            self.input_dim = pe_dim
        else:
            self.input_dim = self.effective_feature_dim

        if self.position_input:
            self.input_dim += self.pos_embed_dim

        self.decoder_intensity = BasicDecoder \
            (self.input_dim, self.output_dim, get_activation_class(self.activation_type),
             True, layer=get_layer_class(self.layer_type), num_layers=self.num_layers+1,
             hidden_dim=self.hidden_dim, skip=[])

        self.sinh_scaling = Fn(torch.sinh)

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
            (self.feature_dim, space_dim=self.space_dim, base_lod=self.base_lod, num_lods=self.num_lods,
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
        self._register_forward_function(
            self.hyperspectral,
            ["density", "recon_spectra", "cdbk_loss", "embd_ids", "latents"]
        )

    def hyperspectral(self, coords, pidx=None, lod_idx=None):
        """Compute hyperspectral intensity for the provided coordinates.
        Args:
            coords (torch.FloatTensor): tensor of shape [batch, num_samples, 3]
            pidx (torch.LongTensor): SPC point_hierarchy indices of shape [batch].
                                     Unused in the current implementation.
            lod_idx (int): index into active_lods. If None, will use the maximum LOD.
                           Currently interpolation doesn't use this.
        Returns:
            {"indensity": torch.FloatTensor }:
                - Output intensity tensor of shape [batch, num_samples, 3]
        """
        timer = PerfTimer(activate=False, show_memory=True)
        if lod_idx is None:
            lod_idx = len(self.grid.active_lods) - 1

        coords = coords[0]
        #print('forward', coords.shape)
        if coords.ndim == 2:
            batch, _ = coords.shape
        elif coords.ndim == 3:
            batch, num_samples, _ = coords.shape
        else:
            raise Exception("Wrong coordinate dimension.")
        timer.check("rf_hyperspectral_preprocess")

        if self.pe:
            feats = self.pe([coords,None])[:,0]
        else:
            # Embed coordinates into high-dimensional vectors with the grid.
            feats = self.grid.interpolate(coords, lod_idx).reshape(-1, self.effective_feature_dim)
            timer.check("rf_hyperspectra_interpolate")

            if self.position_input:
                raise NotImplementedError

        #print(feats.shape)
        # Decode high-dimensional vectors to output intensity.
        density = self.decoder_intensity(feats)
        #print(density.shape)
        density = self.sinh_scaling(density)
        timer.check("rf_hyperspectral_decode")
        return dict(density=density,recon_spectra=None,cdbk_loss=None,embd_ids=None,latents=None)
