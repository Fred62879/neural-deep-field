
import torch
import torch.nn as nn

from collections import defaultdict

from wisp.utils import PerfTimer
from wisp.utils.common import get_bool_classify_redshift, \
    get_bool_has_redshift_latents

from wisp.models.nefs import BaseNeuralField
from wisp.models.embedders.encoder import Encoder
from wisp.models.decoders import BasicDecoder, SpatialDecoder
from wisp.models.hypers import HyperSpectralDecoder, HyperSpectralDecoderB
from wisp.models.layers import get_layer_class, init_codebook, Quantization


class CodebookPretrainNerf(BaseNeuralField):
    def __init__(self, codebook_pretrain_pixel_supervision=False, **kwargs):
        self.kwargs = kwargs

        super(CodebookPretrainNerf, self).__init__()

        assert kwargs["model_redshift"], "we must model redshift during pretrain"

        self.split_latents = kwargs["split_latent"]
        self.use_latents_as_coords = kwargs["use_latents_as_coords"]
        self.pixel_supervision = codebook_pretrain_pixel_supervision
        self.has_redshift_latents = get_bool_has_redshift_latents(**kwargs)

        self.init_model()

    def get_nef_type(self):
        return "codebook_pretrain"

    def get_addup_latents(self):
        return self.addup_latents

    def set_latents(self, latents):
        self.latents = latents

    def add_latents(self):
        """
        Add `addup_latents` to `base_latents`
        @Param
          base_latents: [bsz,dim]
          addup_latents: [bsz,nbins,dim]
        """
        raise NotImplementedError()
        # nbins = self.addup_latents.shape[1]
        # self.latents = self.base_latents[:,None].tile(1,nbins,1) + self.addup_latents
        # self.latents = self.addup_latents + torch.zeros(self.addup_latents.shape).to("cuda:0")

    def combine_latents_all_bins(self, gt_bin_ids, wrong_bin_ids, gt_bin_masks):
        """
        @Params
           gt_bin_ids: [2,bsz]
           wrong_bin_ids: [2,bsz,nbins-1]
           gt_bin_masks: [bsz,nbins]

           gt_bin_latents: [bsz,1,dim]
           wrong_bin_latents: [bsz,nbins-1,dim]
        """
        raise NotImplementedError()
        bsz, nbins = gt_bin_masks.shape
        gt_bin_ids = gt_bin_ids[...,None]
        print(torch.argmax(gt_bin_masks.to(torch.long), dim=-1))

        self.latents = torch.zeros((
            bsz,nbins,self.kwargs["spectra_latent_dim"])).to(self.gt_bin_latents.device)

        print(gt_bin_ids.shape)
        self.latents[gt_bin_ids[0],gt_bin_ids[1],:] = self.gt_bin_latents
        print(self.latents.shape, wrong_bin_ids.shape, self.wrong_bin_latents.shape)
        self.latents[wrong_bin_ids[0],wrong_bin_ids[1],:] = self.wrong_bin_latents
        print(torch.argmax( (self.latents[...,0] != 1).to(torch.long), dim=-1 ))
        assert 0

    def set_base_latents(self, latents):
        self.base_latents = latents

    def set_addup_latents(self, latents):
        self.addup_latents = latents

    def set_gt_bin_latents(self, latents):
        self.gt_bin_latents = latents

    def set_wrong_bin_latents(self, latents):
        self.wrong_bin_latents = latents

    def set_redshift_latents(self, redshift_latents):
        self.redshift_latents = redshift_latents

    def set_batch_reduction_order(self, order="qtz_first"):
        self.hps_decoder.set_batch_reduction_order(order=order)

    def set_bayesian_redshift_logits_calculation(self, loss, mask, gt_spectra):
        self.hps_decoder.set_bayesian_redshift_logits_calculation(loss, mask, gt_spectra)

    def register_forward_functions(self):
        """ Register forward functions with the channels that they output.
        """
        channels = ["coords","intensity","spectra","spectra_all_bins","qtz_weights",
                    "codebook_spectra","codebook_logits","spectra_latents",
                    "full_range_codebook_spectra","min_embed_ids","latents"]

        if self.kwargs["model_redshift"]:
            channels.append("redshift")
            if get_bool_classify_redshift(**self.kwargs):
                channels.append("redshift_logits")
                if self.kwargs["calculate_binwise_spectra_loss"]:
                    channels.extend([
                        "spectra_binwise_loss","spectra_binwise_loss_l2",
                        "spectra_all_bins","optimal_bin_ids",\
                        "gt_redshift_bin_ids","gt_redshift_bin_masks"
                    ])
                    if self.kwargs["plot_spectrum_under_gt_bin"]:
                        channels.append("gt_bin_spectra")

            if self.kwargs["plot_spectrum_color_based_on_loss"] or \
               self.kwargs["plot_spectrum_with_loss"]:
                channels.append("spectra_lambdawise_loss")

        self._register_forward_function(self.pretrain, channels)

    def init_model(self):
        if self.kwargs["quantize_latent"] or self.kwargs["quantize_spectra"]:
            self.codebook = init_codebook(
                self.kwargs["qtz_seed"],
                self.kwargs["qtz_num_embed"],
                self.kwargs["qtz_latent_dim"])
        else: self.codebook = None

        self.spatial_decoder = SpatialDecoder(
            output_bias=False,
            output_scaler=False,
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
                 idx=None, selected_ids=None,
                 init_redshift_prob=None, # debug
                 spectra_masks=None,
                 spectra_loss_func=None,
                 spectra_l2_loss_func=None,
                 spectra_source_data=None,
                 gt_redshift_bin_ids=None,
                 gt_redshift_bin_masks=None
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

              gt_redshift_bin_ids: [bsz,nbins]
              gt_redshift_bin_masks: [bsz,nbins]
        """
        timer = PerfTimer(activate=self.kwargs["activate_model_timer"],
                          show_memory=self.kwargs["show_memory"])
        timer.check("forward starts")

        ret = defaultdict(lambda: None)

        if self.use_latents_as_coords:
            assert coords is None
            if self.kwargs["regularize_binwise_spectra_latents"]:
                nbins = self.addup_latents.shape[1]
                addup_latents = self.index_latents(self.addup_latents, selected_ids, idx)
                base_latents = self.index_latents(
                    self.base_latents, selected_ids, idx)[:,None].tile(1,nbins,1)
                coords = base_latents + addup_latents
            else:
                coords = self.latents
                coords = self.index_latents(coords, selected_ids, idx)

            if gt_redshift_bin_masks is not None:
                assert 0
                # check gt bin id, useful only when we load pretrained latents to gt bin only
                if self.kwargs["dont_optimize_gt_bin"]:
                    gt_redshift_bin_masks = ~gt_redshift_bin_masks
                #_gt_bin_ids = torch.argmin(
                #    torch.tensor(gt_redshift_bin_masks).to(torch.long), dim=-1)
                gt_redshift_bin_masks = gt_redshift_bin_masks[...,None]
                print(coords[0,74:77])
                coords = coords * gt_redshift_bin_masks \
                    + (coords * (~gt_redshift_bin_masks)).detach()

        coords = coords[:,None]
        ret["coords"] = coords

        if self.has_redshift_latents:
            redshift_latents = self.redshift_latents
            redshift_latents = self.index_latents(redshift_latents, selected_ids, idx)
        else: redshift_latents = None

        # `latents` is either coefficients or qtz latents or latents
        #   dep on whether we qtz and qtz method
        latents = self.spatial_decoder(
            coords, self.codebook, qtz_args, ret,
            specz=specz,
            redshift_latents=redshift_latents,
            init_redshift_prob=init_redshift_prob
        )
        timer.check("spatial decoding done")

        self.hps_decoder(
            latents, wave, trans, nsmpl, wave_range,
            trans_mask=trans_mask,
            codebook=self.codebook,
            qtz_args=qtz_args, ret=ret,
            full_emitted_wave=full_emitted_wave,
            spectra_masks=spectra_masks,
            spectra_loss_func=spectra_loss_func,
            spectra_l2_loss_func=spectra_l2_loss_func,
            spectra_source_data=spectra_source_data,
            gt_redshift_bin_ids=gt_redshift_bin_ids
        )

        timer.check("hps decoding done")
        return ret
