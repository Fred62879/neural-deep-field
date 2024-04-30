
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

from functools import partial
from wisp.utils import PerfTimer
from wisp.utils.common import get_input_latent_dim, get_bool_classify_redshift, \
    create_batch_ids

from wisp.models.decoders import BasicDecoder, Siren, Garf
from wisp.models.hypers.hps_integrator import HyperSpectralIntegrator
from wisp.models.hypers.hps_converter_batched import HyperSpectralConverter
from wisp.models.layers import Intensifier, Quantization, ArgMax, \
    calculate_spectra_loss, calculate_redshift_logits

class HyperSpectralDecoderB(nn.Module):

    def __init__(self, scale=True,
                 add_bias=True,
                 integrate=True,
                 intensify=True,
                 qtz_spectra=True,
                 _model_redshift=True,
                 **kwargs
    ):
        super(HyperSpectralDecoderB, self).__init__()

        self.kwargs = kwargs
        self.scale = scale
        self.add_bias = add_bias
        self.intensify = intensify
        self.qtz_spectra = qtz_spectra
        self.reduction_order = "qtz_first"

        self.classify_redshift = _model_redshift and kwargs["apply_gt_redshift"]
        self.classify_redshift = _model_redshift and get_bool_classify_redshift(**kwargs)
        self.classify_redshift_based_on_l2 = self.classify_redshift and \
            kwargs["classify_redshift_based_on_l2"]

        self.regress_lambdawise_weights = kwargs["regress_lambdawise_weights"]
        self.regress_lambdawise_weights_share_latents = self.regress_lambdawise_weights and \
            kwargs["regress_lambdawise_weights_share_latents"]

        # self.recon_codebook_spectra = kwargs["regu_codebook_spectra"] and \
        #     self.kwargs["space_dim"] == 3 and self.kwargs["quantize_spectra"]

        self.convert = HyperSpectralConverter(
            _qtz_spectra=qtz_spectra,
            _model_redshift=_model_redshift, **kwargs
        )

        self.calculate_lambdawise_spectra_loss = \
            (kwargs["plot_spectrum_color_based_on_loss"] or \
             kwargs["plot_spectrum_with_loss"]) and \
             ("codebook_pretrain_infer" in kwargs["tasks"] or \
              "sanity_check_infer" in kwargs["tasks"] or \
              "generalization_infer" in kwargs["tasks"])

        self.init_decoder()
        if self.qtz_spectra:
            self.qtz = Quantization(False, **kwargs)
        if self.intensify:
            self.intensifier = Intensifier(kwargs["intensification_method"])
        self.inte = HyperSpectralIntegrator(integrate=integrate, **kwargs)

    ##################
    # Initialization
    ##################

    def get_input_dim(self):
        if self.kwargs["space_dim"] == 2:

            if self.kwargs["coords_encode_method"] == "positional_encoding":
                assert(self.kwargs["decoder_activation_type"] == "relu")
                input_dim = self.kwargs["coords_embed_dim"]
            elif self.kwargs["coords_encode_method"] == "grid":
                assert(self.kwargs["decoder_activation_type"] == "relu")
                input_dim = get_input_latent_dim(**self.kwargs)
            else:
                assert(self.kwargs["decoder_activation_type"] == "sin")
                input_dim = self.kwargs["space_dim"]
        else:
            assert self.kwargs["space_dim"] == 3
            if self.kwargs["quantize_latent"] or self.kwargs["quantize_spectra"]:
                latents_dim = self.kwargs["qtz_latent_dim"]
            elif self.kwargs["decode_spatial_embedding"]:
                latents_dim = self.kwargs["spatial_decod_output_dim"]
            else:
                latents_dim = get_input_latent_dim(**self.kwargs)

            if self.kwargs["encode_wave"]:
                if self.kwargs["hps_combine_method"] == "add":
                    assert(self.kwargs["wave_embed_dim"] == latents_dim)
                    input_dim = self.kwargs["wave_embed_dim"]
                elif self.kwargs["hps_combine_method"] == "concat":
                    input_dim = self.kwargs["wave_embed_dim"] + latents_dim
                else:
                    raise ValueError()
            elif self.kwargs["decoder_activation_type"] == "sin" or \
                 self.kwargs["decoder_activation_type"] == "gaussian":
                input_dim = self.kwargs["spectra_latent_dim"] + 1
            else:
                # coords and wave are not encoded
                input_dim = 3

        return input_dim

    def get_output_dim(self):
        if self.kwargs["space_dim"] == 2:
            return self.kwargs["num_bands"]
        return 1

    def init_decoder(self):
        input_dim = self.get_input_dim()
        output_dim = self.get_output_dim()

        if self.kwargs["decoder_activation_type"] == "sin":
            self.spectra_decoder = Siren(
                input_dim, output_dim,
                self.kwargs["decoder_num_hidden_layers"],
                self.kwargs["decoder_hidden_dim"],
                self.kwargs["siren_first_w0"],
                self.kwargs["siren_hidden_w0"],
                self.kwargs["siren_seed"],
                1, #self.kwargs["siren_coords_scaler"],
                self.kwargs["siren_last_linear"])

        elif self.kwargs["decoder_activation_type"] == "gaussian":
            self.spectra_decoder = Garf(
                input_dim, output_dim,
                self.kwargs["decoder_num_hidden_layers"],
                self.kwargs["decoder_hidden_dim"],
                self.kwargs["decoder_gaussian_sigma"],
                self.kwargs["decoder_skip_layers"],
                self.kwargs["decoder_latents_skip_all_layers"])
        else:
            self.spectra_decoder = BasicDecoder(
                input_dim, output_dim, True,
                num_layers=self.kwargs["decoder_num_hidden_layers"] + 1,
                hidden_dim=self.kwargs["decoder_hidden_dim"],
                batch_norm=self.kwargs["decoder_batch_norm"],
                layer_type=self.kwargs["decoder_layer_type"],
                activation_type=self.kwargs["decoder_activation_type"],
                skip=self.kwargs["decoder_skip_layers"],
                skip_method=self.kwargs["decoder_latents_skip_method"],
                skip_all_layers=self.kwargs["decoder_latents_skip_all_layers"],
                activate_before_skip=self.kwargs["decoder_activate_before_latents_skip"],
                skip_add_conversion_method=self.kwargs["decoder_latents_skip_add_conversion_method"])
        if self.regress_lambdawise_weights:
            self.lambdawise_weights_decoder = BasicDecoder(
                input_dim, 1, True,
                num_layers=self.kwargs["lambdawise_weights_decoder_num_layers"],
                hidden_dim=self.kwargs["lambdawise_weights_decoder_hidden_dim"])

    ##################
    # Setters
    ##################

    def set_batch_reduction_order(self, order="qtz_first"):
        """ When we do spectra quantization over multiple redshift bins,
            our forwared spectra has two batch dim.
            If we want to save spectra under each bin, we need to perform qtz first.
            If we want to plot codebook spectra, we need to average over all redshift bins first.
            @Param
              spectra: [num_redshift_bins,num_embed,bsz,num_smpl]
              order: `qtz_first` or `bin_avg_first`
        """
        self.reduction_order = order

    def set_bayesian_redshift_logits_calculation(self, loss, mask, gt_spectra):
        """ Set function to calculate bayesian logits for redshift classification.
        """
        self.calculate_bayesian_redshift_logits = partial(
            calculate_bayesian_redshift_logits, loss, mask, gt_spectra)

    ##################
    # Quantization helpers
    ##################

    def quantize_spectra(self, logits, codebook_spectra, ret, qtz_args):
        _, spectra = self.qtz(logits, codebook_spectra, ret, qtz_args)
        spectra = torch.squeeze(spectra, dim=-2) # [...,bsz,nsmpl]
        return spectra

    def _qtz_first(
            self, input, spectra, ret, qtz_args, spectra_masks,
            spectra_loss_func, gt_spectra, gt_redshift_bin_ids
    ):
        spectra = self.quantize_spectra(input, spectra, ret, qtz_args)
        if self.classify_redshift:
            ret["spectra_all_bins"] = spectra

            if self.kwargs["calculate_binwise_spectra_loss"]:
                calculate_spectra_loss(
                    spectra_loss_func, spectra_masks, gt_spectra,
                    spectra, ret, self.calculate_lambdawise_spectra_loss, **self.kwargs
                )
                calculate_redshift_logits(self.kwargs["binwise_loss_beta"], ret)
            spectra = self.classify_redshift3D(spectra, gt_redshift_bin_ids, ret)
        return spectra

    def _bin_avg_first(self, input, spectra, ret, qtz_args):
        if self.classify_redshift:
            spectra = self.classify_redshift4D(spectra, ret)
        ret["codebook_spectra"] = spectra.permute(1,0,2)
        spectra = self.quantize_spectra(input, spectra, ret, qtz_args)
        return spectra

    ##################
    # Classification helpers
    ##################

    def classify_redshift4D(self, spectra, ret):
        """
        @Param
          spectra [num_redshift_bins,num_embed,bsz,nsmpl]
          ret["redshift_logits"] [bsz,num_redshift_bins]
        @Return
          spectra [num_embed,bsz,nsmpl]
        """
        num_embed = spectra.shape[1]
        spectra = torch.einsum("ij,jkim->kim", ret["redshift_logits"], spectra)
        return spectra

    def classify_redshift3D(self, spectra, gt_redshift_bin_ids, ret):
        """
        @Param
              spectra [num_redshift_bins,bsz,nsmpl]
          ret["redshift_logits"] [bsz,num_redshift_bins]
        @Return
          spectra [bsz,nsmpl]
        """
        if self.kwargs["calculate_binwise_spectra_loss"]:
            # index with argmax, this spectra is for visualization only
            #  optimization relies on spectra loss calculated for each bin
            ret["redshift_logits"] = ret["redshift_logits"].detach()
            ids = torch.argmax(ret["redshift_logits"], dim=-1)
            ret["optimal_bin_ids"] = ids
            ids = torch.tensor(
                create_batch_ids(ids.detach().cpu().numpy()), dtype=ids.dtype
            ).to(ids.device)
            if gt_redshift_bin_ids is not None:
                ret["gt_bin_spectra"] = spectra[
                    gt_redshift_bin_ids[1], gt_redshift_bin_ids[0]]
            spectra = spectra[ids[1], ids[0]]
        else:
            spectra = torch.matmul(
                ret["redshift_logits"][:,None], spectra.permute(1,0,2))[:,0]
        # elif self.kwargs["redshift_classification_method"] == "argmax":
        #     # spectra = ArgMax.apply(ret["redshift_logits"], spectra)
        #     ids = ArgMax.apply(ret["redshift_logits"])
        #     spectra = torch.matmul(ids[:,None], spectra.permute(1,0,2))[:,0]
        # elif self.kwargs["redshift_classification_method"] == "bayesian_weighted_avg":
        #     logits = self.calculate_bayesian_redshift_logits(
        #         spectra, ret["redshift_logits"], **self.kwargs) # [bsz,num_bins]
        #     spectra = torch.matmul(logits, spectra.permute(1,0,2))[:,0]
        # else: raise ValueError()
        return spectra

    def _classify_redshift(
            self, spectra, ret, spectra_masks, spectra_loss_func, spectra_l2_loss_func,
            gt_spectra, gt_redshift_bin_ids
    ):
        assert self.kwargs["calculate_binwise_spectra_loss"] or print(
            "only support brute force in case of redshift classification")

        assert spectra.ndim == 3
        ret["spectra_all_bins"] = spectra
        calculate_spectra_loss(
            spectra_loss_func, spectra_masks, gt_spectra,
            spectra, ret, self.calculate_lambdawise_spectra_loss, **self.kwargs)

        if spectra_l2_loss_func is not None:
            calculate_spectra_loss(
                spectra_l2_loss_func, spectra_masks, gt_spectra,
                spectra, ret, self.calculate_lambdawise_spectra_loss,
                loss_name_suffix="_l2", **self.kwargs)

        calculate_redshift_logits(
            self.kwargs["binwise_loss_beta"], ret,
            suffix="_l2" if self.classify_redshift_based_on_l2 else ""
        )
        spectra = self.classify_redshift3D(spectra, gt_redshift_bin_ids, ret)
        return spectra

    ##################
    # Model forward
    ##################

    def spectra_dim_reduction(
            self, input, spectra, ret, qtz_args, spectra_masks, spectra_loss_func,
            spectra_l2_loss_func, gt_spectra, gt_redshift_bin_ids
    ):
        if self.qtz_spectra:
            if self.reduction_order == "qtz_first":
                spectra = self._qtz_first(
                    input, spectra, ret, qtz_args, spectra_masks,
                    spectra_loss_func, gt_spectra, gt_redshift_bin_ids
                )
            elif self.reduction_order == "bin_avg_first":
                spectra = self._bin_avg_first(input, spectra, ret, qtz_args)
            else:
                raise ValueError()
        else:
            if self.classify_redshift:
                spectra = self._classify_redshift(
                    spectra, ret, spectra_masks, spectra_loss_func, spectra_l2_loss_func,
                    gt_spectra, gt_redshift_bin_ids
                )
            elif self.apply_gt_redshift and self.calculate_lambdawise_spectra_loss:
                assert spectra.ndim == 2
                calculate_spectra_loss(
                    spectra_loss_func, spectra_masks, gt_spectra,
                    spectra, ret, True, **self.kwargs)

                if spectra_l2_loss_func is not None:
                    calculate_spectra_loss(
                        spectra_loss_func, spectra_masks, gt_spectra,
                        spectra, ret, self.calculate_lambdawise_spectra_loss,
                        loss_name_suffix="_l2", **self.kwargs)
        return spectra

    def reconstruct_spectra(
            self, input, wave, scaler, bias, redshift, wave_bound, ret, codebook,
            qtz_args, spectra_masks, spectra_loss_func, spectra_l2_loss_func, gt_spectra,
            gt_redshift_bin_ids, optm_bin_ids
    ):
        """
        Reconstruct emitted spectra (under possibly multiple redshift values)
          using given input and wave.
        Scale spectra intensity using scaler, bias, and sinh func, if needed.
        @Param:
           input: 2D coords or embedded latents or logits [bsz,1,2/embed_dim]
           wave:  [bsz,num_samples,1]
           redshift: [num_bins] if model redshift using classification model
                     [bsz] o.w.
           wave_bound: lo and hi lambda values used for linear normalization [2]
           codebook: nn.Parameter([num_embed,embed_dim])
           optm_bin_ids: id of bin with best spectra quality from previous optim round
        @Return
           spectra: reconstructed emitted spectra [bsz,num_nsmpl]
        """
        bsz = wave.shape[0]

        if self.qtz_spectra:
            codes = codebook.weight[:,None,None].tile(1,bsz,1,1)
            # [(num_bins,)num_embed,bsz,nsmpl,dim]
            latents = self.convert(wave, codes, redshift, wave_bound)

            # each spectra has different lambda thus needs its own codebook spectra
            # each redshift bin shift lambda differently thus needs its own codebook spectra
            # codebook spectra [(num_bins,)num_embed,bsz,nsmpl]
            spectra = self.spectra_decoder(latents)[...,0]
        else:
            latents = self.convert(wave, input, redshift, wave_bound) # [...,bsz,nsmpl,dim]
            spectra = self.spectra_decoder(latents)[...,0] # [...,bsz,nsmpl]

        if self.regress_lambdawise_weights:
            # latents [...,bsz,nsmpl,dim], weights [...,bsz,nsmpl,1]
            if self.regress_lambdawise_weights_share_latents:
                optm_bin_ids = create_batch_ids(optm_bin_ids)
                # print('*', optm_bin_ids.shape, optm_bin_ids, latents.shape)
                weight_latents = latents[optm_bin_ids[1],optm_bin_ids[0]]
            else: weight_latents = latents

            weights = self.lambdawise_weights_decoder(weight_latents)[...,0]
            weights = F.softmax(weights, dim=-1)
            if self.regress_lambdawise_weights_share_latents and latents.ndim == 4:
                weights = weights[None,...].tile(latents.shape[0],1,1)
            ret["lambdawise_weights"] = weights

        spectra = self.spectra_dim_reduction(
            input, spectra, ret, qtz_args,
            spectra_masks, spectra_loss_func,
            spectra_l2_loss_func, gt_spectra,
            gt_redshift_bin_ids) # [bsz,nsmpl]

        if self.scale:
            assert scaler is not None
            spectra = (scaler * spectra.T).T
        if self.add_bias:
            assert bias is not None
            spectra = spectra + bias[:,None]
        if self.intensify:
            spectra = self.intensifier(spectra)

        return latents, spectra

    def forward_codebook_spectra(self, codebook, full_emitted_wave, full_wave_bound, ret):
        """ @Params
              codebook: [num_embed,dim]
              full_wave: [nsmpl]
              full_wave_masks: [nsmpl]
            @Return
              ret["codebook_spectra"]: [num_embed,nsmpl]
        """
        n = codebook.weight.shape[0]
        latents = self.convert(
            full_emitted_wave[None,:,None].tile(n,1,1),
            codebook.weight[:,None], None, full_wave_bound
        ) # [num_embed,nsmpl,dim]
        ret["full_range_codebook_spectra"] = self.spectra_decoder(latents)[...,0]

    def forward(self, latents,
                wave, trans, nsmpl, full_wave_bound,
                trans_mask=None,
                full_emitted_wave=None,
                codebook=None, qtz_args=None, ret=None,
                spectra_masks=None,
                spectra_loss_func=None,
                spectra_l2_loss_func=None,
                spectra_source_data=None,
                optm_bin_ids=None,
                gt_redshift_bin_ids=None
    ):
        """
        @Param
        latents:   (encoded or original) coords or logits for quantization.
                     [bsz,1,...,space_dim or coords_encode_dim]

        - hyperspectral
          wave:      lambda values, used to convert ra/dec to hyperspectral latents.
                     [bsz,num_samples,1]
          trans:     corresponding transmission values of lambda. [(bsz,)nbands,num_samples]
          nsmpl:     average number of lambda samples falling within each band. [num_bands]
          full_wave_bound: min and max value of lambda
                           used for linear normalization of lambda
        - spectra qtz
          codebook    nn.Parameter [num_embed,embed_dim]
          qtz_args

        - loss
          spectra_loss_func: spectra loss of choice for training
          spectra_l2_loss_func: when train with ssim, we may calculate l2 loss as well

        ret (output from nerf and/or quantization): {
            "scaler":        unique scaler value for each coord. [bsz,1]
            "redshift":      unique redshift value for each coord. [bsz,1]
            "embed_ids":     ids of embedding each pixel's latent is quantized to.
            "codebook_loss": loss for codebook optimization.
          }

        - full_wave
          full_wave_massk

        @Return (added to `ret`)
          spectra:   reconstructed spectra
          intensity: reconstructed pixel values
          sup_spectra: reconstructed spectra used for spectra supervision
        """
        bsz = latents.shape[0]
        timer = PerfTimer(activate=self.kwargs["activate_model_timer"],
                          show_memory=self.kwargs["show_memory"])
        timer.reset()

        if full_emitted_wave is not None:
            self.forward_codebook_spectra(codebook, full_emitted_wave, full_wave_bound, ret)
            timer.check("hps_decoder::codebook spectra reconstruced")

        redshift = None if ret["redshift"] is None else \
            ret["redshift"] if self.classify_redshift else ret["redshift"][:bsz]
        timer.check("hps_decoder::got redshift")

        if self.classify_redshift_based_on_l2:
            assert self.kwargs["spectra_loss_cho"] == "l2" or spectra_l2_loss_func is not None

        latents, ret["spectra"] = self.reconstruct_spectra(
            latents, wave,
            None if ret["scaler"] is None else ret["scaler"][:bsz],
            None if ret["bias"] is None else ret["bias"][:bsz],
            redshift,
            full_wave_bound, ret, codebook, qtz_args,
            spectra_masks,
            spectra_loss_func,
            spectra_l2_loss_func,
            spectra_source_data,
            gt_redshift_bin_ids,
            optm_bin_ids
        )
        timer.check("hps_decoder::spectra reconstruced")

        intensity = self.inte(ret["spectra"], trans, trans_mask, nsmpl)
        ret["intensity"] = intensity
        timer.check("hps_decoder::integration done")
