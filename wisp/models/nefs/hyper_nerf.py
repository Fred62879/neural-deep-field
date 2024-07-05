
import time
import torch
import numpy as np

from collections import defaultdict

from wisp.utils import PerfTimer
from wisp.utils.common import get_bool_encode_coords

from wisp.models.embedders import Encoder
from wisp.models.layers import init_codebook
from wisp.models.nefs import BaseNeuralField
from wisp.models.decoders import SpatialDecoder
from wisp.models.hypers import HyperSpectralDecoder, HyperSpectralDecoderB


class AstroHyperSpectralNerf(BaseNeuralField):
    """ Model for encoding RA/DEC coordinates with lambda
          values from Monte Carlo sampling.

        Different from how the base_nef works,
          here we use either the positional encoding or the grid for encoding.
    """
    def __init__(self, integrate=True, **kwargs):
        self.kwargs = kwargs

        super(AstroHyperSpectralNerf, self).__init__()

        self.init_codebook()
        self.init_encoder()
        self.init_decoder(integrate)
        torch.cuda.empty_cache()

    def init_codebook(self):
        if self.kwargs["quantize_latent"] or self.kwargs["quantize_spectra"]:
            self.codebook = init_codebook(
                self.kwargs["qtz_seed"], self.kwargs["qtz_num_embed"],
                self.kwargs["qtz_latent_dim"])
        else: self.codebook = None

    def init_encoder(self):
        self.encode_coords = get_bool_encode_coords(**self.kwargs)
        if not self.encode_coords: return

        encode_method=self.kwargs["coords_encode_method"]
        if encode_method == "positional_encoding":
            embedder_args = (
                2, self.kwargs["coords_embed_dim"],
                self.kwargs["coords_embed_omega"],
                self.kwargs["coords_embed_sigma"],
                self.kwargs["coords_embed_bias"],
                self.kwargs["coords_embed_seed"])
        else: embedder_args = None
        self.spatial_encoder = Encoder(
            input_dim=2, encode_method=encode_method,
            embedder_args=embedder_args, **self.kwargs)

    def init_decoder(self, integrate):
        self.spatial_decoder = SpatialDecoder(
            output_bias=self.kwargs["decode_bias"],
            output_scaler=self.kwargs["decode_scaler"],
            _apply_gt_redshift=False,
            qtz_calculate_loss=self.kwargs["quantization_calculate_loss"],
            **self.kwargs
        )

        if self.kwargs["use_batched_hps_model"]:
            hps_decoder_cls = HyperSpectralDecoderB
        else: hps_decoder_cls = HyperSpectralDecoder
        self.hps_decoder = hps_decoder_cls(
            scale=self.kwargs["decode_scaler"],
            add_bias=self.kwargs["decode_bias"],
            integrate=integrate,
            intensify=self.kwargs["intensify_intensity"],
            qtz_spectra=self.kwargs["quantize_spectra"],
            _model_redshift=self.kwargs["model_redshift"],
            **self.kwargs
        )

    def get_nef_type(self):
        return 'hyperspectral'

    def register_forward_functions(self):
        """ Register forward functions with the channels that they output.
        """
        channels = ["intensity","latents","spectra"]
        if self.kwargs["spectra_supervision"]:
            channels.append("sup_spectra")
        if self.kwargs["quantize_latent"] or self.kwargs["quantize_spectra"]:
            channels.extend([
                "scaler","codebook_loss","min_embed_ids",
                "codebook","qtz_weights","codebook_spectra"
            ])
        if self.kwargs["model_redshift"]:
            channels.append("redshift")
            if self.kwargs["redshift_model_method"] == "classification":
                channels.append("redshift_logits")

        self._register_forward_function(self.hyperspectral, channels)

    def hyperspectral(self, coords, wave, wave_range,
                      trans=None, nsmpl=None,
                      specz=None, sup_id=None,
                      sup_spectra_wave=None, num_sup_spectra=0,
                      qtz_args=None, pidx=None, lod_idx=None):
        """ Compute hyperspectral intensity for the provided coordinates.
            @Params:
              coords: coords or spatial embedding [bsz,nsmpls,2/3]

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
        timer = PerfTimer(activate=self.kwargs["activate_model_timer"],
                          show_memory=self.kwargs["show_memory"])
        timer.reset()

        if self.encode_coords:
            latents = self.spatial_encoder(coords, lod_idx=lod_idx)
        else: latents = coords
        timer.check("nef::spatial encoding done")

        latents = self.spatial_decoder(
            latents, self.codebook, qtz_args, ret, specz=specz, sup_id=sup_id)
        timer.check("nef::spatial decoding done")

        self.hps_decoder(
            latents, wave, trans, nsmpl, wave_range,
            codebook=self.codebook,
            qtz_args=qtz_args, ret=ret,
            # num_sup_spectra=num_sup_spectra,
            # sup_spectra_wave=sup_spectra_wave
        )
        timer.check("nef::hyperspectral decoding done")

        if self.codebook is not None:
            ret["codebook"] = self.codebook.weight
        return ret
