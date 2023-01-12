import numpy as np
import torch

from collections import defaultdict

from wisp.utils import PerfTimer
from wisp.models.embedders import Encoder
from wisp.models.nefs import BaseNeuralField
from wisp.models.decoders import QuantizedDecoder
from wisp.models.hypers import HyperSpectralDecoder


class AstroHyperSpectralNerf(BaseNeuralField):
    """ Model for encoding RA/DEC coordinates with lambda
          values from Monte Carlo sampling.

        Different from how the base_nef works,
          here we use either the positional encoding or the grid for encoding.
    """
    def __init__(self, integrate=True, scale=True, qtz_calculate_loss=True, **kwargs):
        self.kwargs = kwargs

        super(AstroHyperSpectralNerf, self).__init__()

        self.space_dim = kwargs["space_dim"]

        self.init_encoder()
        if kwargs["quantize_latent"]:
            self.qtz_decoder = QuantizedDecoder(qtz_calculate_loss, **kwargs)
        self.hps_decoder = HyperSpectralDecoder(
            integrate=integrate, scale=scale, **kwargs)

        torch.cuda.empty_cache()

    def init_encoder(self):
        embedder_args = (
            2,
            self.kwargs["coords_embed_dim"],
            self.kwargs["coords_embed_omega"],
            self.kwargs["coords_embed_sigma"],
            self.kwargs["coords_embed_bias"],
            self.kwargs["coords_embed_seed"]
        )
        self.coord_encoder = Encoder(
            input_dim=2,
            encode_method=self.kwargs["coords_encode_method"],
            embedder_args=embedder_args,
            **self.kwargs
        )

    def get_nef_type(self):
        return 'hyperspectral'

    def register_forward_functions(self):
        """ Register forward functions with the channels that they output.
        """
        channels = ["intensity","latents","spectra"]
        if self.kwargs["quantize_latent"]:
            channels.extend(["scaler","redshift","codebook_loss","min_embed_ids"])
        self._register_forward_function( self.hyperspectral, channels )

    def hyperspectral(self, coords, wave=None, trans=None, nsmpl=None, full_wave=None, num_spectra_coords=-1, pidx=None, lod_idx=None):
        """ Compute hyperspectral intensity for the provided coordinates.
            @Params:
              coords (torch.FloatTensor): tensor of shape [batch, num_samples, 2/3]
              pidx (torch.LongTensor): SPC point_hierarchy indices of shape [batch].
                                       Unused in the current implementation.
              lod_idx (int): index into active_lods. If None, will use the maximum LOD.
                             Currently interpolation doesn't use this.
            @Return
              {"indensity": Output intensity tensor of shape [batch, num_samples, 3]
               "spectra":
              }
        """
        timer = PerfTimer(activate=False, show_memory=True)

        ret = defaultdict(lambda: None)

        latents = self.coord_encoder(coords, lod_idx=lod_idx)
        if self.kwargs["quantize_latent"]:
            latents = self.qtz_decoder(latents, ret)

        np.save('../../data/tmp.npy', latents.detach().cpu().numpy())
        latents = torch.FloatTensor(np.load('../../data/tmp.npy')).to('cuda:0')
        #latents = coords
        #print(latents)
        self.hps_decoder(latents, wave, trans, nsmpl, ret, full_wave, num_spectra_coords)
        return ret
