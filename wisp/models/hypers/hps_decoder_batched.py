
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

from functools import partial
from wisp.utils import PerfTimer
from wisp.utils.common import create_batch_ids, get_input_latent_dim, \
    get_bool_classify_redshift, get_bool_plot_lambdawise_spectra_loss, \
    get_bool_save_redshift_classification_data , \
    get_bool_classify_redshift_based_on_l2, \
    get_bool_weight_spectra_loss_with_global_restframe_loss

from wisp.models.decoders import BasicDecoder, Siren, Garf
from wisp.models.hypers.hps_integrator import HyperSpectralIntegrator
from wisp.models.hypers.hps_converter_batched import HyperSpectralConverter
from wisp.models.layers import Intensifier, Quantization, ArgMax, \
    calculate_spectra_loss, calculate_redshift_logits, normalize_Linear

class HyperSpectralDecoderB(nn.Module):

    def __init__(
        self, scale=True, add_bias=True, integrate=True, intensify=True,
        qtz_spectra=True, _model_redshift=True, **kwargs
    ):
        super(HyperSpectralDecoderB, self).__init__()

        self.kwargs = kwargs
        self.scale = scale
        self.add_bias = add_bias
        self.integrate = integrate
        self.intensify = intensify
        self.qtz_spectra = qtz_spectra
        self.reduction_order = "qtz_first"
        self.model_redshift = _model_redshift

        self.apply_gt_redshift = _model_redshift and kwargs["apply_gt_redshift"]
        self.classify_redshift = _model_redshift and get_bool_classify_redshift(**kwargs)
        self.classify_redshift_based_on_l2 = _model_redshift and \
            get_bool_classify_redshift_based_on_l2(**kwargs)

        self.plot_lambdawise_spectra_loss = \
            get_bool_plot_lambdawise_spectra_loss(**kwargs)
        self.save_lambdawise_spectra_loss = \
            get_bool_save_redshift_classification_data(**kwargs)
        #     get_bool_save_lambdawise_spectra_loss(**kwargs)
        self.use_lambdawise_weights = \
            get_bool_weight_spectra_loss_with_global_restframe_loss(**kwargs)
        self.save_lambdawise_loss = self.plot_lambdawise_spectra_loss or \
            self.save_lambdawise_spectra_loss

        self.init_net()

    ##################
    # Initialization
    ##################

    def init_net(self):
        self.convert = HyperSpectralConverter(
            _qtz_spectra=self.qtz_spectra,
            _model_redshift=self.model_redshift, **self.kwargs)

        self.init_decoder()
        if self.qtz_spectra:
            self.qtz = Quantization(False, **self.kwargs)
        if self.intensify:
            self.intensifier = Intensifier(self.kwargs["intensification_method"])
        self.inte = HyperSpectralIntegrator(integrate=self.integrate, **self.kwargs)

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
                skip_add_conversion_method=\
                    self.kwargs["decoder_latents_skip_add_conversion_method"])

        if self.use_lambdawise_weights and self.kwargs["regress_lambdawise_weights"]:
            self.lambdawise_weights_decoder = BasicDecoder(
                input_dim, 1, True,
                num_layers=self.kwargs["lambdawise_weights_decoder_num_layers"],
                hidden_dim=self.kwargs["lambdawise_weights_decoder_hidden_dim"])

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

    #########
    # Setters
    #########

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

    def toggle_sample_bins(self, sample: bool):
        self.convert.toggle_sample_bins(sample)

    #########
    # Helpers
    #########

    def quantize_spectra(self, logits, codebook_spectra, ret, qtz_args):
        _, spectra = self.qtz(logits, codebook_spectra, ret, qtz_args)
        spectra = torch.squeeze(spectra, dim=-2) # [...,bsz,nsmpl]
        return spectra

    def classify_redshift4D(self, spectra, ret):
        """
        Get the final reconstructed spectra based on spectra under all bins
          (i.e. weighted average of spectra under all bins) (for codebook quantization usage).
        @Param
          spectra [nbins,num_embed,bsz,nsmpls]
          ret["redshift_logits"] [bsz,nbins]
        @Return
          spectra [num_embed,bsz,nsmpl]
        """
        num_embed = spectra.shape[1]
        spectra = torch.einsum("ij,jkim->kim", ret["redshift_logits"], spectra)
        return spectra

    def classify_redshift3D(self, spectra, redshift_bins_mask, ret):
        """
        Get the final reconstructed spectra based on spectra under all bins
          (i.e. weighted average or argmax of spectra under all bins) (no qtz usage).
        @Param
          spectra [nbins,bsz,nsmpl]
          redshift_bins_mask [bsz,nbins]
          ret["redshift_logits"] [bsz,nbins]
        @Return
          spectra [bsz,nsmpl]
        """
        if self.kwargs["brute_force_redshift"]:
            """
            Get argmax spectra which is for visualization only.
            Optimization requires loss of spectra under each bin.
            """
            if redshift_bins_mask is not None:
                nbins, bsz, nsmpl = spectra.shape
                # print(spectra.shape, redshift_bins_mask.shape)
                # print(redshift_bins_mask)
                # mask = (redshift_bins_mask.T)[...,None].tile(1,1,nsmpl)
                # print(mask.shape)
                # print(spectra[mask].shape)
                # ret["gt_bin_spectra"] = spectra[mask].view(bsz,nsmpl)

                ret["gt_bin_spectra"] = (spectra.permute(1,0,2)[redshift_bins_mask])

                # ids = torch.tensor([[0,1],[55,35]], dtype=torch.long).to("cuda:0")
                # print(spectra.shape, ids[0], ids[1])
                # ret["gt_bin_spectra"] = spectra[ids[1],ids[0]]

            suffix = "_l2" if self.classify_redshift_based_on_l2 else ""
            redshift_logits = ret[f"redshift_logits{suffix}"].detach()
            ids = torch.argmax(redshift_logits, dim=-1)
            ret["optimal_bin_ids"] = ids
            ids = torch.tensor(
                create_batch_ids(ids.cpu().numpy()), dtype=ids.dtype
            ).to(ids.device)
            spectra = spectra[ids[1], ids[0]]
        else:
            spectra = torch.matmul(
                ret["redshift_logits"][:,None], spectra.permute(1,0,2))[:,0]
        return spectra

    ###################
    # Spectra reduction
    ###################

    def reduce_codebook_spectra(
            self, input, spectra, ret, qtz_args, spectra_mask,
            spectra_loss_func, spectra_l2_loss_func, gt_spectra, redshift_bins_mask
    ):
        """
        Reduce `codebook` (and `bin`) dim of the given spectra.
        @Param
          input:   2D coords or embedded latents or logits [bsz,1,2/embed_dim]
          spectra: codebook spectra reconstructed (under all redshift bins)
        """
        if self.reduction_order == "qtz_first":
            spectra = self.quantize_spectra(input, spectra, ret, qtz_args)
            if self.classify_redshift:
                ret["spectra_all_bins"] = spectra

                if self.kwargs["brute_force_redshift"]:
                    if spectra_loss_func is not None:
                        calculate_spectra_loss(
                            spectra_loss_func, spectra_mask, gt_spectra,
                            spectra, ret, self.save_lambdawise_loss,
                            self.use_lambdawise_weights, **self.kwargs)
                        calculate_redshift_logits(self.kwargs["binwise_loss_beta"], ret)

                    if spectra_l2_loss_func is not None:
                        calculate_spectra_loss(
                            spectra_l2_loss_func, spectra_mask, gt_spectra,
                            spectra, ret, self.save_lambdawise_loss,
                            self.use_lambdawise_weights, suffix="_l2", **self.kwargs)
                        calculate_redshift_logits(
                            self.kwargs["binwise_loss_beta"], ret, suffix="_l2")

                spectra = self.classify_redshift3D(spectra, redshift_bins_mask, ret)

        elif self.reduction_order == "bin_avg_first":
            if self.classify_redshift:
                spectra = self.classify_redshift4D(spectra, ret)
            ret["codebook_spectra"] = spectra.permute(1,0,2)
            spectra = self.quantize_spectra(input, spectra, ret, qtz_args)
        else:
            raise ValueError()

    def reduce_spectra(
            self, input, spectra, ret, qtz_args, spectra_mask, spectra_loss_func,
            spectra_l2_loss_func, gt_spectra, redshift_bins_mask
    ):
        """
        Reduce `bin` dim of the given spectra.
        @Param
          input:   2D coords or embedded latents or logits [bsz,1,2/embed_dim]
          spectra: spectra reconstructed under all redshift bins
        """
        if self.classify_redshift:
            assert spectra.ndim == 3
        elif self.apply_gt_redshift:
            assert spectra.ndim == 2
        else: raise ValueError()

        if self.classify_redshift:
            ret["spectra_all_bins"] = spectra

        if spectra_loss_func is not None:
            calculate_spectra_loss(
                spectra_loss_func, spectra_mask, gt_spectra,
                spectra, ret, self.save_lambdawise_loss,
                self.use_lambdawise_weights, **self.kwargs)

        if spectra_l2_loss_func is not None:
            calculate_spectra_loss(
                spectra_l2_loss_func, spectra_mask, gt_spectra,
                spectra, ret, self.save_lambdawise_loss,
                suffix="_l2", **self.kwargs)

        if self.classify_redshift:
            if spectra_loss_func is not None:
                calculate_redshift_logits(self.kwargs["binwise_loss_beta"], ret)
            if spectra_l2_loss_func is not None:
                calculate_redshift_logits(self.kwargs["binwise_loss_beta"], ret, suffix="_l2")

            spectra = self.classify_redshift3D(spectra, redshift_bins_mask, ret)

        return spectra

    ###############
    # Model forward
    ###############

    def generate_lambdawise_weights(
        self, latents, optm_bin_ids, global_restframe_spectra_loss, ret
    ):
        """
        @Param
          latents [...,bsz,nsmpl,dim]
          weights [...,bsz,nsmpl,1]
        """
        if self.kwargs["use_global_spectra_loss_as_lambdawise_weights"] or \
           self.kwargs["infer_use_global_loss_as_lambdawise_weights"]:
            assert global_restframe_spectra_loss is not None
            emitted_wave = ret["emitted_wave"] # [nbins,bsz,nsmpl]
            sp = emitted_wave.shape
            emitted_wave = emitted_wave.flatten()
            global_loss = global_restframe_spectra_loss(emitted_wave.detach().cpu().numpy())

            # `global_loss` may contain `nan` which are result of out of range interpolation
            #   we replace them with largest loss
            invalid = np.isnan(global_loss)
            global_loss[invalid] = np.max(global_loss[~invalid])

            # `global_loss` may also contain negative values, replace with 0
            nonpos = global_loss <= 0
            global_loss[nonpos] = np.min(global_loss[~nonpos])

            global_loss = torch.tensor(
                global_loss, dtype=emitted_wave.dtype).to(emitted_wave.device)
            weights = torch.exp(-global_loss).view(sp)

        elif self.kwargs["regress_lambdawise_weights"]:
            if self.kwargs["regress_lambdawise_weights_share_latents"]:
                optm_bin_ids = create_batch_ids(optm_bin_ids)
                weight_latents = latents[optm_bin_ids[1],optm_bin_ids[0]]
            else: weight_latents = latents

            weights = self.lambdawise_weights_decoder(weight_latents)[...,0] # [...,bsz,nsmpl]
            weights = normalize_Linear(weights)
            if self.kwargs["regress_lambdawise_weights_share_latents"] and latents.ndim == 4:
                # add extra dim for `num_of_bins`
                weights = weights[None,...].tile(latents.shape[0],1,1) # [nbins,bsz,nsmpl]

        ret["lambdawise_weights"] = weights

    def forward_codebook_spectra(self, codebook, full_emitted_wave, full_wave_bound, ret):
        """ @Params
              codebook: [num_embed,dim]
              full_wave: [nsmpl]
              full_wave_mask: [nsmpl]
            @Return
              ret["codebook_spectra"]: [num_embed,nsmpl]
        """
        n = codebook.weight.shape[0]
        latents = self.convert(
            full_emitted_wave[None,:,None].tile(n,1,1),
            codebook.weight[:,None], None, full_wave_bound, ret
        ) # [num_embed,nsmpl,dim]
        ret["full_range_codebook_spectra"] = self.spectra_decoder(latents)[...,0]

    def reconstruct_spectra(
            self, input, wave, scaler, bias, redshift, wave_bound,
            codebook, qtz_args, spectra_loss_func, spectra_l2_loss_func,
            spectra_mask, redshift_bins_mask, selected_bins_mask,
            gt_spectra, optm_bin_ids, global_restframe_spectra_loss, ret
    ):
        """
        Reconstruct emitted (codebook) spectra (under possibly multiple redshift values)
          using given input and wave.
        Scale spectra intensity using scaler, bias, and sinh func, if needed.
        @Param:
           input: 2D coords or embedded latents or logits [bsz,1,2/embed_dim]
           wave:  [bsz,num_samples,1]
           redshift: [num_bins] if model redshift using classification model
                     [bsz] o.w.
           wave_bound: lo and hi lambda values used for linear normalization [2]
           codebook: nn.Parameter([num_embed,embed_dim])
        @Return
           spectra: reconstructed emitted spectra [bsz,num_nsmpl]
        """
        bsz = wave.shape[0]

        if self.qtz_spectra:
            """
            each spectra has different lambda thus needs its own codebook spectra
            each redshift bin shift lambda differently thus needs its own codebook spectra
            latents [(num_bins,)num_embed,bsz,nsmpl,dim]
            spectra (codebook spectra) [(num_bins,)num_embed,bsz,nsmpl]
            """
            codes = codebook.weight[:,None,None].tile(1,bsz,1,1)
            latents = self.convert(wave, codes, redshift, wave_bound, selected_bins_mask, ret)
            codebook_spectra = self.spectra_decoder(latents)[...,0] # [...,nbins,bsz,nsmpl]
            if self.use_lambdawise_weights:
                self.generate_lambdawise_weights(
                    latents, optm_bin_ids, global_restframe_spectra_loss, ret)
            spectra = self.reduce_codebook_spectra(
                input, codebook_spectra, ret, qtz_args, spectra_mask,
                spectra_loss_func, gt_spectra, redshift_bins_mask) # [bsz,nsmpl]
        else:
            latents = self.convert(
                wave, input, redshift, wave_bound, selected_bins_mask, ret) # [...,bsz,nsmpl,dim]
            spectra = self.spectra_decoder(latents)[...,0] # [...,bsz,nsmpl]
            if self.use_lambdawise_weights:
                self.generate_lambdawise_weights(
                    latents, optm_bin_ids, global_restframe_spectra_loss, ret)
            spectra = self.reduce_spectra(
                input, spectra, ret, qtz_args, spectra_mask, spectra_loss_func,
                spectra_l2_loss_func, gt_spectra, redshift_bins_mask) # [bsz,nsmpl]

        if self.scale:
            assert scaler is not None
            spectra = (scaler * spectra.T).T
        if self.add_bias:
            assert bias is not None
            spectra = spectra + bias[:,None]
        if self.intensify:
            spectra = self.intensifier(spectra)

        return latents, spectra

    def forward(
            self, latents, wave, trans, nsmpl, full_wave_bound,
            spectra_source_data=None, codebook=None, qtz_args=None,
            optm_bin_ids=None, global_restframe_spectra_loss=None, ret=None,
            spectra_loss_func=None, spectra_l2_loss_func=None, full_emitted_wave=None,
            trans_mask=None, spectra_mask=None, selected_bins_mask=None, redshift_bins_mask=None
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

        - mask
          redshift_bins_mask: 1 for gt bin, 0 for all other bins
          selected_bins_mask: 1 for bins we sample at current step, 0 for others
                              (gt bin always selected)

        ret (output from nerf and/or quantization): {
            "scaler":        unique scaler value for each coord. [bsz,1]
            "redshift":      unique redshift value for each coord. [bsz,1]
            "embed_ids":     ids of embedding each pixel's latent is quantized to.
            "codebook_loss": loss for codebook optimization.
          }

        - full_wave
          full_wave_massk

        optm_bin_ids: id of bin with best spectra quality from previous optim round

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

        latents, spectra = self.reconstruct_spectra(
            latents, wave,
            None if ret["scaler"] is None else ret["scaler"][:bsz],
            None if ret["bias"] is None else ret["bias"][:bsz],
            redshift, full_wave_bound, codebook, qtz_args,
            spectra_loss_func, spectra_l2_loss_func,
            spectra_mask, redshift_bins_mask, selected_bins_mask,
            spectra_source_data, optm_bin_ids, global_restframe_spectra_loss, ret
        )
        timer.check("hps_decoder::spectra reconstruced")

        intensity = self.inte(spectra, trans, trans_mask, nsmpl)

        ret["spectra"] = spectra
        ret["intensity"] = intensity
        timer.check("hps_decoder::integration done")
