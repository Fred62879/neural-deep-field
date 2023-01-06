
import torch

from collections import defaultdict

from wisp.utils import PerfTimer
from wisp.models.encoders import Encoder
from wisp.models.nefs import BaseNeuralField
from wisp.models.decoders import QuantizedDecoder
from wisp.models.hypers import HyperSpectralDecoder


class AstroHyperSpectral(BaseNeuralField):
    """ Model for encoding RA/DEC coordinates with lambda
          values from Monte Carlo sampling.

        Different from how the base_nef works,
          here we use either the positional encoding or the grid for encoding.
    """
    def __init__(self, integrate=True, scale=True, qtz_calculate_loss=True, **kwargs):
        super(AstroHyperSpectral, self).__init__(**kwargs)

        self.kwargs = kwargs
        self.space_dim = kwargs["space_dim"]

        self.init_encoder()
        if kwargs["quantize_latent"]:
            self.quantized_decoder = QuantizedDecoder(
                qtz_calculate_loss, **kwargs)
        self.hps_decoder = HyperSpectralDecoder(
            integrate=integrate, scale=scale, **kwargs)

        torch.cuda.empty_cache()
        self._forward_functions = {}
        self.register_forward_functions()
        self.supported_channels = set([
            channel for channels in self._forward_functions.values()
            for channel in channels])

    def init_encoder(self):
        """ Initialize the encoder (positional encoding or grid interpolaton)
              for both ra/dec coordinates and wave (lambda values).
        """
        coords_embedder_args = (
            2, self.kwargs["coords_embed_dim"], self.kwargs["coords_embed_omega"],
            self.kwargs["coords_embed_sigma"], self.kwargs["coords_embed_bias"],
            self.kwargs["coords_embed_seed"])

        self.coords_encoder = Encoder(
            encode_method=self.kwargs["coords_encode_method"],
            embedder_args=coords_embedder_args, **self.kwargs)

    def get_nef_type(self):
        return 'hyperspectral'

    def register_forward_functions(self):
        """ Register forward functions with the channels that they output.
        """
        channels = ["intensity","latents","spectra"]
        if self.kwargs["quantize_latent"]:
            channels.extend(["codebook_loss","min_embed_ids"])
        self._register_forward_function( self.hyperspectral, channels )

    def hyperspectral(self, coords, wave, trans, nsmpl, full_wave=None, num_spectra_coords=-1, rpidx=None, lod_idx=None):
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

        latents = self.coords_encoder(coords)
        if self.kwargs["quantize_latent"]:
            latents = self.quantized_decoder(latents, ret)
        self.hps_decoder(latents, wave, trans, nsmpl, ret, full_wave, num_spectra_coords)
        #self.hps_decoder(coords, wave, trans, nsmpl, ret, full_wave, num_spectra_coords)

        return ret
