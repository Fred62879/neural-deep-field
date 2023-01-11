
import torch

from wisp.utils import PerfTimer
from wisp.models.embedders import Encoder
from wisp.models.decoders import Decoder
from wisp.models.nefs import BaseNeuralField
from wisp.models.layers import Normalization


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
        super().__init__()

        self.kwargs = kwargs
        self.num_bands = kwargs["num_bands"]

        self.encoder = Encoder(kwargs["coords_encode_method"], **kwargs)
        self.decoder = Decoder(**kwargs)
        self.norm = Normalization(kwargs["mlp_output_norm_method"])

        torch.cuda.empty_cache()

    def get_nef_type(self):
        return 'astro2d'

    def register_forward_functions(self):
        """ Register forward functions with the channels that they output.
        """
        channels = ["intensity"]
        self._register_forward_function( self.coords_to_intensity, channels )

    def coords_to_intensity(self, coords, pidx=None, lod_idx=None):
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

        coords = self.encoder(coords)
        intensity = self.decoder(coords).view(-1, self.num_bands)
        intensity = self.norm(intensity)
        timer.check("rf_hyperspectral_decode")

        ret = dict(intensity=intensity)
        return ret
