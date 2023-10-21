
import torch
import torch.nn as nn

from collections import defaultdict

from wisp.utils import PerfTimer
from wisp.utils.common import get_bool_classify_redshift

from wisp.models.nefs import BaseNeuralField
from wisp.models.embedders.encoder import Encoder
from wisp.models.decoders import BasicDecoder, SpatialDecoder
from wisp.models.hypers import HyperSpectralDecoder, HyperSpectralDecoderB
from wisp.models.layers import get_layer_class, init_codebook, Quantization


class CodebookPretrainNerf(BaseNeuralField):
    def __init__(self, decode_redshift=False,
                 codebook_pretrain_pixel_supervision=False,
                 **kwargs
    ):
        self.kwargs = kwargs

        super(CodebookPretrainNerf, self).__init__()

        self.decode_redshift = decode_redshift
        self.pixel_supervision = codebook_pretrain_pixel_supervision
        self.use_latents_as_coords = kwargs["use_latents_as_coords"]
        self.init_model()

    def get_nef_type(self):
        return "codebook_pretrain"

    def set_latents(self, latents):
        self.latents = latents

    def set_redshift_latents(self, redshift_latents):
        self.redshift_latents = redshift_latents

    def set_batch_reduction_order(self, order="qtz_first"):
        self.hps_decoder.set_batch_reduction_order(order=order)

    def set_bayesian_redshift_logits_calculation(self, loss, mask, gt_spectra):
        self.hps_decoder.set_bayesian_redshift_logits_calculation(loss, mask, gt_spectra)

    def register_forward_functions(self):
        """ Register forward functions with the channels that they output.
        """
        channels = ["intensity","spectra","spectra_all_bins","qtz_weights",
                    "codebook_spectra","full_range_codebook_spectra","min_embed_ids","latents"]

        if self.kwargs["model_redshift"]:
            channels.append("redshift")
            if get_bool_classify_redshift(**self.kwargs):
                channels.append("redshift_logits")

        self._register_forward_function(self.pretrain, channels)

    def init_model(self):
        assert self.kwargs["model_redshift"] and \
            (self.kwargs["apply_gt_redshift"] or self.decode_redshift)

        if self.kwargs["quantize_latent"] or self.kwargs["quantize_spectra"]:
            self.codebook = init_codebook(
                self.kwargs["qtz_seed"],
                self.kwargs["qtz_num_embed"],
                self.kwargs["qtz_latent_dim"])
        else: self.codebook = None

        self.spatial_decoder = SpatialDecoder(
            output_bias=False,
            output_scaler=False,
            decode_redshift=self.decode_redshift,
            qtz_calculate_loss=False,
            **self.kwargs)

        if self.kwargs["use_batched_hps_model"]:
            hps_decoder_cls = HyperSpectralDecoderB
        else: hps_decoder_cls = HyperSpectralDecoder
        self.hps_decoder = hps_decoder_cls(
            scale=False,
            add_bias=False,
            integrate=self.pixel_supervision,
            intensify=False,
            qtz_spectra=self.kwargs["quantize_spectra"],
            _model_redshift=self.kwargs["model_redshift"],
            **self.kwargs)

        # init latent variables **** DELETE ****
        if self.kwargs["debug_lbfgs"]:
            self.num_spectra = self.kwargs["redshift_pretrain_num_spectra"]
            if self.use_latents_as_coords:
                self.latents = nn.Embedding(
                    self.num_spectra, self.kwargs["qtz_num_embed"])
            if not self.kwargs["apply_gt_redshift"] and self.kwargs["split_latent"]:
                self.redshift_latents = nn.Embedding(
                    self.num_spectra, self.kwargs["redshift_logit_latent_dim"])

    def index_latents(self, data, selected_ids, idx):
        ret = data
        if selected_ids is not None:
            ret = ret[selected_ids]
        if idx is not None:
            ret = ret[idx]
        return ret

    def pretrain(self, coords, wave, wave_range,
                 trans=None, trans_mask=None, nsmpl=None,
                 full_emitted_wave=None,
                 qtz_args=None, specz=None,
                 scaler_latents=None,    # DELETE
                 redshift_latents=None,  # DELETE
                 idx=None, selected_ids=None,
                 init_redshift_prob=None # debug
    ):
        """ Pretrain codebook.
            @Param
              coords: trainable latent variable [num_supervision_spectra,1,latent_dim]
              wave:   lambda values [bsz,nsmpl,1]
              wave_range: range of lambda used for linear norm [2] (min/max)

              trans:  transmission values (padded with -1 at two ends)
              trans_mask: mask for trans (0 for padded region, 1 for actual trans region)
              nsmpl:  number of lambda sample within each band

              full_emitted_wave:

              qtz_args: quantization arguments

              specz: gt redshift
        """
        timer = PerfTimer(activate=self.kwargs["activate_model_timer"],
                          show_memory=self.kwargs["show_memory"])
        timer.check("forward starts")

        ret = defaultdict(lambda: None)

        if self.use_latents_as_coords:
            assert coords is None
            coords = self.latents
            coords = self.index_latents(coords, selected_ids, idx)

        coords = coords[:,None]
        # print(coords.device)

        if not self.kwargs["apply_gt_redshift"] and self.kwargs["split_latent"]:
            redshift_latents = self.redshift_latents
            redshift_latents = self.index_latents(redshift_latents, selected_ids, idx)
        else: redshift_latents = None
        # print(redshift_latents.device)

        # `latents` is either logits or qtz latents or latents dep on qtz method
        latents = self.spatial_decoder(
            coords, self.codebook, qtz_args, ret,
            specz=specz,
            scaler_latents=scaler_latents,
            redshift_latents=redshift_latents,
            init_redshift_prob=init_redshift_prob
        )
        timer.check("spatial decoding done")

        self.hps_decoder(
            latents, wave, trans, nsmpl, wave_range,
            trans_mask=trans_mask,
            codebook=self.codebook,
            qtz_args=qtz_args, ret=ret,
            full_emitted_wave=full_emitted_wave
        )
        timer.check("hps decoding done")
        return ret
