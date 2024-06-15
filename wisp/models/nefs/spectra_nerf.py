
import torch
import torch.nn as nn

from collections import defaultdict

from wisp.utils import PerfTimer
from wisp.utils.common import get_bool_classify_redshift, \
    get_bool_has_redshift_latents, get_bool_weight_spectra_loss_with_global_restframe_loss, \
    get_bool_save_redshift_classification_data, get_bool_sanity_check_sample_bins, \
    get_bool_sanity_check_sample_bins_per_step

from wisp.models.nefs import BaseNeuralField
from wisp.models.embedders.encoder import Encoder
from wisp.models.decoders import BasicDecoder, SpatialDecoder
from wisp.models.hypers import HyperSpectralDecoder, HyperSpectralDecoderB
from wisp.models.layers import get_layer_class, init_codebook, Quantization


class SpectraNerf(BaseNeuralField):
    def __init__(self, **kwargs):
        assert kwargs["model_redshift"], "we must model redshift during pretrain"

        self.kwargs = kwargs
        self.split_latents = kwargs["split_latent"]
        self.use_latents_as_coords = kwargs["use_latents_as_coords"]
        self.pixel_supervision = kwargs["pretrain_pixel_supervision"]
        self.optimize_bins_separately = kwargs["optimize_gt_bin_only"] or \
            kwargs["dont_optimize_gt_bin"]

        self.has_redshift_latents = \
            get_bool_has_redshift_latents(**kwargs)
        self.sanity_check_sample_bins = \
            get_bool_sanity_check_sample_bins(**kwargs)
        self.sanity_check_sample_bins_per_step = \
            get_bool_sanity_check_sample_bins_per_step(**kwargs)
        self.save_lambdawise_spectra_loss = \
            get_bool_save_redshift_classification_data(**kwargs)

        super(SpectraNerf, self).__init__()

        self.init_model()

    def get_nef_type(self):
        return "spectra_pretrain"

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
        # nbins = self.addup_latents.shape[1]
        # self.latents = self.base_latents[:,None].tile(1,nbins,1) + self.addup_latents
        # self.latents = self.addup_latents + torch.zeros(self.addup_latents.shape).to("cuda:0")
        raise NotImplementedError()

    def combine_latents_all_bins(self, gt_bin_ids, wrong_bin_ids, redshift_bins_mask):
        """
        @Params
           gt_bin_ids: [2,bsz]
           wrong_bin_ids: [2,bsz,nbins-1]
           redshift_bins_mask: [bsz,nbins]

           gt_bin_latents: [bsz,1,dim]
           wrong_bin_latents: [bsz,nbins-1,dim]
        """
        # bsz, nbins = redshift_bins_mask.shape
        # gt_bin_ids = gt_bin_ids[...,None]
        # self.latents = torch.zeros((
        #     bsz,nbins,self.kwargs["spectra_latent_dim"])).to(self.gt_bin_latents.device)

        # self.latents[gt_bin_ids[0],gt_bin_ids[1],:] = self.gt_bin_latents
        # self.latents[wrong_bin_ids[0],wrong_bin_ids[1],:] = self.wrong_bin_latents
        raise NotImplementedError()

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

    def toggle_sample_bins(self, sample: bool):
        self.hps_decoder.toggle_sample_bins(sample)

    def register_forward_functions(self):
        """ Register forward functions with the channels that they output.
        """
        channels = ["coords","intensity","spectra","spectra_all_bins","qtz_weights",
                    "codebook_spectra","codebook_logits","spectra_latents",
                    "full_range_codebook_spectra","min_embed_ids","latents"]

        if self.kwargs["model_redshift"]:
            channels.append("redshift")
            if self.kwargs["apply_gt_redshift"]:
                channels.extend(["spectrawise_loss","spectrawise_loss_l2"])
            elif get_bool_classify_redshift(**self.kwargs):
                channels.extend(["redshift_logits","redshift_logits_l2"])
                if self.kwargs["brute_force_redshift"]:
                    channels.extend([
                        "spectra_binwise_loss","spectra_binwise_loss_l2",
                        "spectra_all_bins","redshift_bins_mask"
                    ])
                    if self.kwargs["plot_spectrum_under_gt_bin"]:
                        channels.append("gt_bin_spectra")
                    if self.sanity_check_sample_bins or \
                       self.sanity_check_sample_bins_per_step:
                        channels.append("selected_bins_mask")

            if self.save_lambdawise_spectra_loss or \
               self.kwargs["plot_spectrum_with_loss"] or \
               self.kwargs["plot_spectrum_color_based_on_loss"] or \
               "plot_global_lambdawise_spectra_loss" in self.kwargs["tasks"]:
                channels.extend(["spectra_lambdawise_loss","spectra_lambdawise_loss_l2"])

        if get_bool_weight_spectra_loss_with_global_restframe_loss(**self.kwargs):
            channels.extend(["lambdawise_weights","optimal_bin_ids"])
        elif self.kwargs["use_global_spectra_loss_as_lambdawise_weights"]:
            channels.append("global_restframe_spectra_loss")

        self._register_forward_function(self.run, channels)

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
            scale=False, add_bias=False, integrate=self.pixel_supervision,
            intensify=False, qtz_spectra=self.kwargs["quantize_spectra"],
            _model_redshift=self.kwargs["model_redshift"],
            **self.kwargs)

    ##########
    # forward
    ##########

    def run(
            self, wave_range, full_emitted_wave=None,
            coords=None, trans=None, trans_mask=None, nsmpl=None,
            qtz_args=None, specz=None, idx=None, selected_ids=None, optm_bin_ids=None,
            spectra_loss_func=None, spectra_l2_loss_func=None,
            spectra_source_data=None, global_restframe_spectra_loss=None,
            spectra_mask=None, redshift_bins_mask=None, selected_bins_mask=None,
    ):
        """
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

          optm_bin_ids: [bsz] ids of bin with best spectra quality at pervious step
          selected_bins_mask: [bsz,nbins]
          redshift_bins_mask: [bsz,nbins]
        """
        timer = PerfTimer(
            activate=self.kwargs["activate_model_timer"], show_memory=self.kwargs["show_memory"])
        timer.check("forward starts")

        ret = defaultdict(lambda: None)
        ret["coords"] = self.set_coords(
            coords, selected_ids, idx, redshift_bins_mask, selected_bins_mask)

        if self.has_redshift_latents:
            redshift_latents = self.redshift_latents
            redshift_latents = self.index_latents(redshift_latents, selected_ids, idx)
        else: redshift_latents = None

        # `latents` is either coefficients or qtz latents or latents
        #   dep on whether we qtz and qtz method
        latents = self.spatial_decoder(
            ret["coords"], self.codebook, qtz_args, ret,
            specz=specz, redshift_latents=redshift_latents
        )
        timer.check("spatial decoding done")

        wave = spectra_source_data[:,0][...,None] # [bsz,nsmpl,1]
        self.hps_decoder(
            latents, wave, trans, nsmpl, wave_range,
            trans_mask=trans_mask,
            codebook=self.codebook,
            qtz_args=qtz_args, ret=ret,
            full_emitted_wave=full_emitted_wave,
            spectra_mask=spectra_mask,
            selected_bins_mask=selected_bins_mask,
            spectra_loss_func=spectra_loss_func,
            spectra_l2_loss_func=spectra_l2_loss_func,
            spectra_source_data=spectra_source_data,
            optm_bin_ids=optm_bin_ids,
            redshift_bins_mask=redshift_bins_mask,
            global_restframe_spectra_loss=global_restframe_spectra_loss,
        )
        timer.check("hps decoding done")
        return ret

    def set_coords(self, coords, selected_ids, idx, redshift_bins_mask, selected_bins_mask):
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

            # if self.sanity_check_sample_bins:
            #     assert selected_bins_mask is not None
            #     bsz, _, nsmpl = coords.shape
            #     coords = coords[selected_bins_mask[...,None].tile(1,1,nsmpl)]
            #     coords = coords.view(bsz,-1,nsmpl)

            if self.optimize_bins_separately:
                # assert redshift_bins_mask is not None:
                # check gt bin id, useful only when we load pretrained latents to gt bin only
                # if self.kwargs["dont_optimize_gt_bin"]:
                #     redshift_bins_mask = ~redshift_bins_mask
                # #_gt_bin_ids = torch.argmin(
                # #    torch.tensor(redshift_bins_mask).to(torch.long), dim=-1)
                # redshift_bins_mask = redshift_bins_mask[...,None]
                # coords = coords * redshift_bins_mask \
                #     + (coords * (~redshift_bins_mask)).detach()
                raise NotImplementedError()

        if self.sanity_check_sample_bins_per_step:
            # assert selected_bins_mask is not None
            raise NotImplementedError()

        coords = coords[:,None]
        return coords

    def index_latents(self, data, selected_ids, idx):
        ret = data
        # index with `selected_ids` first
        if selected_ids is not None: ret = ret[selected_ids]
        if idx is not None:          ret = ret[idx]
        return ret
