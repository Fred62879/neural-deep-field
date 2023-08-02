
import time
import torch
import numpy as np

from collections import defaultdict

from wisp.utils import PerfTimer
from wisp.utils.common import print_shape
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
    def __init__(self, integrate=True, scale=True, add_bias=True,
                 qtz_calculate_loss=True, _model_redshfit=True, **kwargs):
        self.kwargs = kwargs

        super(AstroHyperSpectralNerf, self).__init__()

        self.init_codebook()
        self.init_encoder()
        self.init_decoder(
            integrate, scale, qtz_calculate_loss, _model_redshfit)
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

    def init_decoder(self, integrate, scale, add_bias, calculate_loss, _model_redshift):
        self.spatial_decoder = SpatialDecoder(
            output_bias=add_bias,
            output_scaler=scale,
            qtz_calculate_loss=calculate_loss, **self.kwargs
        )
        self.hps_decoder = HyperSpectralDecoder(
            scale=scale,
            add_bias=add_bias,
            integrate=integrate,
            _model_redshift=_model_redshift, **self.kwargs
        )

    def get_nef_type(self):
        return 'hyperspectral'

    def register_forward_functions(self):
        """ Register forward functions with the channels that they output.
        """
        channels = ["intensity","latents","spectra"]
        if self.kwargs["quantize_latent"] or self.kwargs["quantize_spectra"]:
            channels.extend(["scaler","redshift","codebook_loss",
                             "min_embed_ids","codebook","qtz_weights","codebook_spectra"])

        self._register_forward_function(self.hyperspectral, channels)

    def hyperspectral(self, coords, wave, wave_range,
                      trans=None, nsmpl=None,
                      specz=None, sup_id=None,
                      full_wave=None, num_spectra_coords=-1,
                      qtz_args=None, pidx=None, lod_idx=None):
        """ Compute hyperspectral intensity for the provided coordinates.
            @Params:
              coords (torch.FloatTensor): tensor of shape [batch, num_samples, 2/3]

              - qtz_args:
                temperature: temperature for soft quantization, if performed
                find_embed_id: whether find embed id or not (used for soft quantization)
                save_codebook: save codebook weights value to local
                save_qtz_weights: save weights for each code (when doing soft qtz)

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
        timer = PerfTimer(activate=self.kwargs["activate_model_timer"], show_memory=False)
        timer.reset()

        if self.kwargs["encode_coords"]:
            latents = self.spatial_encoder(coords, lod_idx=lod_idx)
        else: latents = coords
        timer.check("nef::spatial encoding done")

        latents = self.spatial_decoder(
            latents, self.codebook, qtz_args, ret, specz=specz, sup_id=sup_id)
        timer.check("nef::spatial decoding done")

        self.hps_decoder(latents, wave, trans, nsmpl, wave_range,
                         qtz_args=qtz_args, ret=ret, full_wave=full_wave,
                         codebook=self.codebook, num_spectra_coords=num_spectra_coords)
        timer.check("nef::hyperspectral decoding done")

        if self.codebook is not None:
            ret["codebook"] = self.codebook.weight
        return ret
