
import time
import torch
import numpy as np

from collections import defaultdict

from wisp.utils import PerfTimer
from wisp.models.embedders import Encoder
from wisp.models.layers import init_codebook
from wisp.models.nefs import BaseNeuralField
from wisp.models.decoders import SpatialDecoder
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

        self.init_codebook()
        self.init_encoder()
        self.init_decoder(integrate, scale, qtz_calculate_loss)
        torch.cuda.empty_cache()

    def init_codebook(self):
        if self.kwargs["quantize_latent"] or self.kwargs["quantize_spectra"]:
            self.codebook = init_codebook(
                self.kwargs["qtz_seed"],
                self.kwargs["qtz_num_embed"],
                self.kwargs["qtz_latent_dim"]
            )
        else: self.codebook = None

    def init_encoder(self):
        if not self.kwargs["encode_coords"]: return

        embedder_args = (
            2,
            self.kwargs["coords_embed_dim"],
            self.kwargs["coords_embed_omega"],
            self.kwargs["coords_embed_sigma"],
            self.kwargs["coords_embed_bias"],
            self.kwargs["coords_embed_seed"]

        )
        self.spatial_encoder = Encoder(
            input_dim=2,
            encode_method=self.kwargs["coords_encode_method"],
            embedder_args=embedder_args,
            **self.kwargs
        )

    def init_decoder(self, integrate, scale, calculate_loss):
        self.spatial_decoder = SpatialDecoder(
            output_scaler=self.kwargs["generate_scaler"],
            output_redshift=self.kwargs["redshift_supervision"],
            apply_redshift=self.kwargs["apply_gt_redshift"],
            qtz_calculate_loss=calculate_loss, **self.kwargs)

        self.hps_decoder = HyperSpectralDecoder(
            integrate=integrate, scale=scale, **self.kwargs)

    def get_nef_type(self):
        return 'hyperspectral'

    def register_forward_functions(self):
        """ Register forward functions with the channels that they output.
        """
        channels = ["intensity","latents","spectra"]
        if self.kwargs["quantize_latent"] or self.kwargs["quantize_spectra"]:
            channels.extend(["scaler","redshift","codebook_loss",
                             "min_embed_ids","codebook","soft_qtz_weights"])

        self._register_forward_function(self.hyperspectral, channels)

    def hyperspectral(self, coords, wave, full_wave_bound,
                      trans=None, nsmpl=None,
                      full_wave=None, num_spectra_coords=-1,
                      qtz_args=None, pidx=None, lod_idx=None):
        """ Compute hyperspectral intensity for the provided coordinates.
            @Params:
              coords (torch.FloatTensor): tensor of shape [batch, num_samples, 2/3]

              - qtz_args:
                temperature: temperature for soft quantization, if performed
                find_embed_id: whether find embed id or not (used for soft quantization)
                save_codebook: save codebook weights value to local
                save_soft_qtz_weights: save weights for each code (when doing soft qtz)

              - grid_args:
                pidx (torch.LongTensor): SPC point_hierarchy indices of shape [batch].
                                         Unused in the current implementation.
                lod_idx (int): index into active_lods. If None, will use the maximum LOD.
                               Currently interpolation doesn't use this.
            @Return
              {
                "indensity": Output intensity tensor of shape [batch, num_samples, 3]
                "spectra":
              }
        """
        ret = defaultdict(lambda: None)
        timer = PerfTimer(activate=self.kwargs["activate_timer"], show_memory=False)
        # timer.check("hyper nef encode coord")

        if self.kwargs["encode_coords"]:
            latents = self.spatial_encoder(coords, lod_idx=lod_idx)
        else: latents = coords
        latents = self.spatial_decoder(latents, self.codebook, qtz_args, ret)

        self.hps_decoder(latents, wave, trans, nsmpl, full_wave_bound,
                         full_wave, num_spectra_coords,
                         self.codebook, qtz_args, self.kwargs["quantize_spectra"], ret)

        if self.codebook is not None:
            ret["codebook"] = self.codebook.weight
        return ret
