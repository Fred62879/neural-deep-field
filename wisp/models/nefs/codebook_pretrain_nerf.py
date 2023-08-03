
import torch
import torch.nn as nn

from wisp.utils import PerfTimer
from collections import defaultdict
from wisp.models.nefs import BaseNeuralField
from wisp.models.embedders.encoder import Encoder
from wisp.models.hypers import HyperSpectralDecoder
from wisp.models.decoders import BasicDecoder, SpatialDecoder
from wisp.models.layers import get_layer_class, init_codebook, Quantization


class CodebookPretrainNerf(BaseNeuralField):
    def __init__(self, pretrain_pixel_supervision, _model_redshift=True, **kwargs):
        super(CodebookPretrainNerf, self).__init__()

        self.kwargs = kwargs
        self.model_redshift = _model_redshift
        self.pixel_supervision = pretrain_pixel_supervision
        self.init_model()

    def get_nef_type(self):
        return "codebook_pretrain"

    def register_forward_functions(self):
        """ Register forward functions with the channels that they output.
        """
        channels = ["intensity","spectra","redshift","qtz_weights","codebook_spectra"]
        self._register_forward_function(self.pretrain, channels)

    def init_model(self):
        if self.kwargs["quantize_latent"] or self.kwargs["quantize_spectra"]:
            self.codebook = init_codebook(
                self.kwargs["qtz_seed"],
                self.kwargs["qtz_num_embed"],
                self.kwargs["qtz_latent_dim"])
        else: self.codebook = None

        assert self.kwargs["model_redshift"] and \
            self.kwargs["apply_gt_redshift"] and \
            not self.kwargs["redshift_unsupervision"] and \
            not self.kwargs["redshift_semi_supervision"]

        self.spatial_decoder = SpatialDecoder(
            output_bias=False,
            output_scaler=False,
            qtz_calculate_loss=False,
            **self.kwargs)

        self.hps_decoder = HyperSpectralDecoder(
            scale=False,
            add_bias=False,
            integrate=self.pixel_supervision,
            intensify=False,
            qtz_spectra=self.kwargs["quantize_spectra"],
            _model_redshift=self.model_redshift,
            **self.kwargs)

    def pretrain(self, coords, wave, wave_range,
                 trans=None, trans_mask=None, nsmpl=None, qtz_args=None, specz=None
    ):
        """ Pretrain codebook.
            @Param
              coords: trainable latent variable [num_supervision_spectra,1,latent_dim]
              wave:   lambda values [bsz,nsmpl,1]
              wave_range: range of lambda used for linear norm [2] (min/max)
              trans:  transmission values (padded with -1 at two ends)
              trans_mask: mask for trans (0 for padded region, 1 for actual trans region)
        """
        timer = PerfTimer(activate=self.kwargs["activate_model_timer"], show_memory=False)
        timer.check("forward starts")

        ret = defaultdict(lambda: None)
        bsz = coords.shape[0]
        coords = coords[:,None]
        # print(coords)

        # `latents` is either logits or qtz latents or latents dep on qtz method
        latents = self.spatial_decoder(coords, self.codebook, qtz_args, ret, specz=specz)
        timer.check("spatial decoding done")

        self.hps_decoder(
            latents, wave, trans, nsmpl, wave_range,
            trans_mask=trans_mask,
            codebook=self.codebook,
            qtz_args=qtz_args, ret=ret
        )
        timer.check("hps decoding done")
        return ret
