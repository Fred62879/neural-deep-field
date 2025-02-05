import os
import math
import time
import torch
import random
import warnings
import nvidia_smi
import numpy as np
import torch.nn as nn
import logging as log
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tqdm import tqdm
from pathlib import Path
from functools import partial
from os.path import exists, join
from torch.autograd import Variable
from scipy.interpolate import interp1d
from torch.utils.data import BatchSampler, \
    SequentialSampler, RandomSampler, DataLoader

from wisp.utils.common import *
from wisp.utils import PerfTimer
from wisp.trainers import BaseTrainer
from wisp.optimizers import multi_optimizer
from wisp.utils.plot import plot_grad_flow, plot_multiple, \
    plot_redshift_estimation_stats_together, \
    plot_redshift_estimation_stats_individually
from wisp.loss import get_loss, get_reduce, spectra_supervision_loss, pretrain_pixel_loss


torch.backends.cudnn.benchmark = True

class SpectraTrainer(BaseTrainer):
    def __init__(self, pipeline, dataset, optim_cls, optim_params, device, mode, **extra_args):
        super().__init__(pipeline, dataset, optim_cls, optim_params, device, mode, **extra_args)

        assert(extra_args["space_dim"] == 3)

        self.summarize_training_tasks()
        self.set_path()

        self.init_data()
        self.init_net()
        self.collect_model_params()
        # self.init_data() # moved above
        self.init_loss()
        self.init_optimizer()

    def train(self):
        torch.autograd.set_detect_anomaly(True)
        self.begin_train()

        n = self.num_epochs if self.train_based_on_epochs else self.num_steps
        iter = tqdm(range(n + 1)) if self.use_tqdm else range(n + 1)

        for self.cur_iter in iter:
            self.begin_iter()

            if self.train_based_on_epochs:
                for self.cur_batch in range(self.num_batches_cur_epoch):
                    data = self.next_batch()
                    ret = self.step(data)
                    self.post_step(data, ret)
            else:
                assert 0
                # need to figure out how to sample
                idx = self.sample_data()
                data = self.dataset.__getitem__(idx)
                ret = self.step(data)
                self.post_step(data, ret)

            self.end_iter()
        self.end_train()

    #############
    # Initializations
    #############

    def summarize_training_tasks(self):
        assert has_common([self.mode], [
            "redshift_pretrain","gasnet_pretrain","spectra_pretrain",
            "sanity_check","generalization","redshift_classification_train"
        ])
        self.spectra_baseline = self.mode == "redshift_pretrain"
        self.gasnet_baseline = self.mode == "gasnet_pretrain"
        self.spectra_pretrain = self.mode == "spectra_pretrain"
        self.sanity_check = self.mode == "sanity_check"
        self.generalization = self.mode == "generalization"
        self.freeze_spectra = self.mode == "redshift_classification_train"
        self.pre_classification_mode = self.sanity_check or self.generalization

        tasks = set(self.extra_args["tasks"])
        self.save_redshift = "save_redshift" in tasks
        self.save_qtz_weights = "save_qtz_weights" in tasks
        self.recon_spectra = "recon_spectra" in tasks
        self.save_pixel_values = "save_pixel_values" in tasks
        self.plot_codebook_logits = "plot_codebook_logits" in tasks
        self.recon_codebook_spectra_individ = "recon_codebook_spectra_individ" in tasks
        self.plot_gt_bin_spectra = self.extra_args["plot_spectrum_under_gt_bin"]

        self.generalize_train_first_layer = self.generalization and \
            self.extra_args["generalize_train_first_layer"]

        if self.spectra_pretrain or self.spectra_baseline or self.gasnet_baseline:
            self.num_spectra = self.dataset.get_num_supervision_spectra()
        elif self.sanity_check or self.freeze_spectra:
            self.num_spectra = self.dataset.get_num_validation_spectra()
        elif self.generalization:
            self.num_spectra = self.dataset.get_num_test_spectra()
        else: raise ValueError("Invalid mode!")

        # quantization setups
        self.qtz_latent = self.space_dim == 3 and self.extra_args["quantize_latent"]
        self.qtz_spectra = self.space_dim == 3 and self.extra_args["quantize_spectra"]
        self.qtz = self.qtz_latent or self.qtz_spectra
        self.qtz_n_embd = self.extra_args["qtz_num_embed"]
        self.qtz_strategy = self.extra_args["quantization_strategy"]

        assert not self.extra_args["temped_qtz"]
        assert not (self.qtz_latent and self.qtz_spectra)
        # qtz and latents decoding are mutually exclusive
        assert sum([self.qtz, self.extra_args["decode_spatial_embedding"]]) <= 1

        # we do epoch based training only in autodecoder setting
        assert self.train_based_on_epochs or self.qtz_spectra
        # in codebook qtz setting, we only do step based training
        assert not self.train_based_on_epochs or not self.qtz_spectra

        """
        model_redshift
          |_apply_gt_redshift
          |_regress_redshift
          |_classify_redshift
                |_brute_force
                |   |_neg_sup_wrong_redshift                            |-pretrain
                |   |_regularize_binwise_spectra_latents               -
                |   |_optimize_latents_for_each_redshift_bin            |- sanity check OR
                |   |_optimize_one_latent_for_all_redshift_bins         |  generalization
                |        |_calculate_spectra_loss_based_on_optimal_bin  |
                |        |_calculate_spectra_loss_based_on_top_n_bins  -
                |
                |_freeze_spectra
        """
        self.model_redshift = self.extra_args["model_redshift"]
        self.regress_redshift = get_bool_regress_redshift(**self.extra_args)
        self.classify_redshift = get_bool_classify_redshift(**self.extra_args)
        self.apply_gt_redshift = self.model_redshift and self.extra_args["apply_gt_redshift"]
        assert not self.model_redshift or \
            sum([self.regress_redshift, self.classify_redshift, self.apply_gt_redshift]) == 1

        self.has_redshift_latents = get_bool_has_redshift_latents(**self.extra_args)
        self.brute_force = self.model_redshift and self.classify_redshift and \
            self.extra_args["brute_force_redshift"]
        self.freeze_spectra = self.model_redshift and \
            self.classify_redshift and self.freeze_spectra
        assert not (self.brute_force and self.freeze_spectra)

        # redshift regression only needs train and test phase
        assert not self.regress_redshift or (
            self.spectra_baseline or self.gasnet_baseline or self.generalization)
        # if we want to calculate binwise spectra loss when we do quantization
        #  we need to perform qtz first in hyperspectral_decoder::dim_reduction
        assert not self.brute_force or \
            (not self.qtz or self.extra_args["spectra_batch_reduction_order"] == "qtz_first")

        self.classify_redshift_based_on_l2 = self.classify_redshift and \
            self.extra_args["classify_redshift_based_on_l2"] and \
            self.extra_args["spectra_loss_cho"] != "l2"

        self.neg_sup_wrong_redshift = \
            self.spectra_pretrain and self.brute_force and \
            self.extra_args["negative_supervise_wrong_redshift"]
        self.neg_sup_optimize_alternately = \
            self.neg_sup_wrong_redshift and \
            self.extra_args["neg_sup_optimize_alternately"]

        # pretrain mandates either apply gt redshift directly or brute force
        assert not self.spectra_pretrain or (
            self.apply_gt_redshift or self.brute_force)
        # brute force during pretrain mandates negative supervision
        assert not(self.spectra_pretrain and self.brute_force) or \
            self.neg_sup_wrong_redshift
        # sanity check & generalization mandates brute force
        assert not (self.sanity_check or self.generalization) or \
                    self.brute_force

        # three different brute force strategies during sc & generalization
        self.regularize_binwise_spectra_latents = self.brute_force and \
            self.brute_force and \
            self.extra_args["regularize_binwise_spectra_latents"]
        self.optimize_latents_for_each_redshift_bin = self.brute_force and \
            self.brute_force and \
            self.extra_args["optimize_latents_for_each_redshift_bin"]
        self.optimize_one_latent_for_all_redshift_bins = self.brute_force and \
            self.brute_force and \
            not self.regularize_binwise_spectra_latents and \
            not self.extra_args["optimize_latents_for_each_redshift_bin"]
        self.calculate_spectra_loss_based_on_optimal_bin = \
            self.optimize_one_latent_for_all_redshift_bins and \
            self.extra_args["calculate_spectra_loss_based_on_optimal_bin"]
        self.calculate_spectra_loss_based_on_top_n_bins = \
            self.optimize_one_latent_for_all_redshift_bins and \
            self.extra_args["calculate_spectra_loss_based_on_top_n_bins"]
        assert not (self.calculate_spectra_loss_based_on_optimal_bin and \
                    self.calculate_spectra_loss_based_on_top_n_bins)

        if self.classify_redshift:
            self.num_redshift_bins = self.dataset.get_num_redshift_bins()

        # * weighted spectra training
        self.regress_lambdawise_weights = self.brute_force and \
            self.extra_args["regress_lambdawise_weights"]
        self.use_global_spectra_loss_as_lambdawise_weights = self.brute_force and \
            self.extra_args["use_global_spectra_loss_as_lambdawise_weights"]
        assert not (self.regress_lambdawise_weights and \
                    self.use_global_spectra_loss_as_lambdawise_weights)
        if self.use_global_spectra_loss_as_lambdawise_weights:
            log.info("train with restframe loss as weights!")
        """
        When we perform sanity check with lambdawise weights, we fine-tune the model
          such that gt bin gets better quality spectra than that of all other wrong bins.
        """
        self.sanity_check_with_weights = self.sanity_check and \
            (self.regress_lambdawise_weights or \
             self.use_global_spectra_loss_as_lambdawise_weights)
        self.sanity_check_finetune = self.extra_args["sanity_check_finetune"] and \
            self.sanity_check
        assert not self.sanity_check_with_weights or self.sanity_check_finetune

        self.sanity_check_sample_bins = \
            get_bool_sanity_check_sample_bins(**self.extra_args)
        self.sanity_check_sample_bins_per_step = \
            get_bool_sanity_check_sample_bins_per_step(**self.extra_args)
        assert not (self.sanity_check_sample_bins and self.sanity_check_sample_bins_per_step)

        # ** use global restframe loss
        if self.use_global_spectra_loss_as_lambdawise_weights:
            self.global_restframe_spectra_loss_fname = join(
                self.log_dir, "..", self.extra_args["global_restframe_spectra_loss_fname"])

        self.limit_redshift_to_pretrain_range = \
            self.use_global_spectra_loss_as_lambdawise_weights and \
            self.extra_args["limit_redshift_to_pretrain_range"]

        # before the 1st iter, we dum init optimal bin id
        self.dum_init_optm_bin_ids = self.regress_lambdawise_weights
        self.regress_lambdawise_weights_share_latents = \
            self.regress_lambdawise_weights and \
            self.extra_args["regress_lambdawise_weights_share_latents"]
        self.regress_lambdawise_weights_use_gt_bin_latent = \
            self.regress_lambdawise_weights_share_latents and \
            self.extra_args["regress_lambdawise_weights_use_gt_bin_latent"]
        self.plot_spectra_with_weights = \
            self.recon_spectra and \
            self.regress_lambdawise_weights and \
            self.extra_args["plot_spectrum_with_weights"]

        # * negative supervision
        if self.neg_sup_optimize_alternately:
            # alternate between `latent` and `network (w or w.o codebook)`
            self.neg_sup_alternation_steps = self.extra_args["neg_sup_alternation_steps"]
            self.neg_sup_alternation_starts_with = self.extra_args["neg_sup_alternation_starts_with"]
            assert self.neg_sup_alternation_starts_with == "latents"

        # wave sampling setup
        self.sample_wave = not self.extra_args["pretrain_use_all_wave"]
        self.pixel_supervision = self.extra_args["pretrain_pixel_supervision"]
        # self.train_within_wave_range = not self.pixel_supervision and \
        #     self.extra_args["learn_spectra_within_wave_range"]
        self.trans_sample_method = self.extra_args["trans_sample_method"]
        if self.spectra_baseline or self.gasnet_baseline:
            assert not self.sample_wave

        # all others
        self.plot_loss = self.extra_args["plot_loss"]
        self.plot_l2_loss = self.extra_args["plot_l2_loss"] and \
            self.extra_args["spectra_loss_cho"] == "ssim1d"
        self.plot_gt_bin_loss = self.extra_args["plot_gt_bin_loss"] and \
            self.brute_force and self.brute_force
        self.plot_individ_spectra_loss = self.apply_gt_redshift and \
            self.extra_args["plot_individ_spectra_loss"]

        self.index_latent = True # index latents as coords in model
        self.split_latent = self.extra_args["split_latent"] and self.brute_force

        self.regularize_redshift_logits = self.extra_args["regularize_redshift_logits"]
        self.redshift_logits_regu_method = self.extra_args["redshift_logits_regu_method"]
        self.regularize_codebook_logits = self.qtz_spectra and \
            self.extra_args["regularize_codebook_logits"]
        self.regularize_spectra_latents = self.extra_args["regularize_spectra_latents"] and \
            not self.freeze_spectra
        self.regularize_within_codebook_spectra = self.qtz_spectra and \
            self.extra_args["regularize_within_codebook_spectra"]
        self.regularize_across_codebook_spectra = self.qtz_spectra and \
            self.extra_args["regularize_across_codebook_spectra"]
        self.regularize_codebook_spectra = self.spectra_pretrain and \
            (self.regularize_within_codebook_spectra or self.regularize_across_codebook_spectra)
        assert not self.regularize_redshift_logits or \
            (self.classify_redshift and \
             self.extra_args["redshift_classification_method"] == "weighted_avg")
        assert self.regularize_within_codebook_spectra + \
            self.regularize_across_codebook_spectra <= 1

        self.optimize_spectra_latents = self.extra_args["optimize_spectra_latents"]
        self.optimize_redshift_latents = self.has_redshift_latents and \
            self.extra_args["optimize_redshift_latents"] and self.brute_force

        # latents are optimized as codebook coefficients
        self.optimize_spectra_latents_as_logits = \
            self.extra_args["optimize_spectra_latents_as_logits"]
        self.optimize_redshift_latents_as_logits = self.has_redshift_latents and \
            self.optimize_redshift_latents and \
            self.extra_args["optimize_redshift_latents_as_logits"] and self.brute_force

        self.load_pretrained_spectra_latents = \
            self.extra_args["load_pretrained_spectra_latents"]
        self.load_pretrained_spectra_latents_to_gt_bin_only = \
            self.extra_args["load_pretrained_spectra_latents_to_gt_bin_only"]
        self.optimize_gt_bin_only = self.extra_args["optimize_gt_bin_only"]
        self.dont_optimize_gt_bin = self.extra_args["dont_optimize_gt_bin"]
        self.optimize_bins_separately = self.optimize_gt_bin_only or self.dont_optimize_gt_bin
        assert not self.load_pretrained_spectra_latents or self.brute_force
        assert not self.load_pretrained_spectra_latents_to_gt_bin_only or \
            self.load_pretrained_spectra_latents
        assert not self.optimize_bins_separately or \
            self.load_pretrained_spectra_latents
        assert sum([self.optimize_gt_bin_only, self.dont_optimize_gt_bin]) <= 1

        assert not self.optimize_spectra_latents_as_logits or self.qtz_spectra
        # no pretrained latents to load

        self.em_alternation_steps = self.extra_args["em_alternation_steps"]
        self.em_alternation_starts_with = self.extra_args["em_alternation_starts_with"]
        self.use_lbfgs = self.extra_args["optimize_latents_use_lbfgs"]

        # alternate (em) optimization for spectra & redshift latents
        self.optimize_latents_alternately = \
            self.extra_args["pretrain_optimize_latents_alternately"]

        assert not self.use_lbfgs or \
            (self.optimize_spectra_latents_as_logits and \
             self.optimize_redshift_latents_as_logits), \
             "Doesn't support optimizing alternately when using lbfgs!"
        assert not self.optimize_latents_alternately or \
            (self.optimize_spectra_latents_as_logits and \
             self.optimize_redshift_latents_as_logits), \
             "Doesn't support optimizing alternately when using autodecoder arch!"

    def set_path(self):
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        log.info(f"logging to {self.log_dir}")

        for cur_path, cur_pname, in zip(
                ["loss_dir","model_dir","spectra_dir","redshift_dir",
                 "codebook_spectra_dir","codebook_logits_dir","qtz_weight_dir"],
                ["losses","models","train_spectra","train_redshift",
                 "train_codebook_spectra","train_codebook_logits","qtz_weights"]
        ):
            path = join(self.log_dir, cur_pname)
            setattr(self, cur_path, path)
            Path(path).mkdir(parents=True, exist_ok=True)

        self.grad_fname = join(self.log_dir, "grad.png")

        if self.plot_loss:
            self.loss_fname = join(self.loss_dir, "loss")
            if self.plot_gt_bin_loss:
                self.gt_bin_loss_fname = join(self.loss_dir, "gt_bin_loss")
            if self.neg_sup_wrong_redshift:
                self.gt_bin_loss_fname = join(self.loss_dir, "gt_bin_loss")
                self.wrong_bin_regu_fname = join(self.loss_dir, "wrong_bin_regu")
                self.wrong_bin_loss_fname = join(self.loss_dir, "wrong_bin_loss")

        if self.plot_l2_loss:
            self.l2_loss_fname = join(self.loss_dir, "l2_loss")
            if self.plot_gt_bin_loss:
                self.gt_bin_l2_loss_fname = join(self.loss_dir, "gt_bin_l2_loss")

        if self.sanity_check or self.generalization:
            _, self.pretrained_model_fname = get_pretrained_model_fname(
                self.log_dir,
                self.extra_args["pretrain_log_dir"],
                self.extra_args["pretrained_model_name"])

        if self.extra_args["resume_train"]:
            pretrained_dir, self.resume_train_model_fname = get_pretrained_model_fname(
                self.log_dir,
                self.extra_args["resume_log_dir"],
                self.extra_args["resume_model_fname"])

            loss_dir = join(pretrained_dir, "losses")
            self.resume_loss_fname = join(loss_dir, "loss.npy")
            self.resume_gt_bin_loss_fname = join(loss_dir, "gt_bin_loss.npy")
            self.resume_wrong_bin_regu_fname = join(loss_dir, "wrong_bin_regu.npy")
            self.resume_wrong_bin_loss_fname = join(loss_dir, "wrong_bin_loss.npy")

            if self.plot_l2_loss:
                self.resume_l2_loss_fname = join(loss_dir, "l2_loss.npy")
                self.resume_gt_bin_l2_loss_fname = join(loss_dir, "gt_bin_l2_loss.npy")

        if self.extra_args["plot_logits_for_gt_bin"]:
            self.gt_bin_logits_fname = join(self.log_dir, "gt_bin_logits")

    def init_net(self):
        if self.qtz:
            self.pipeline.set_batch_reduction_order(
                self.extra_args["spectra_batch_reduction_order"])

        if not self.freeze_spectra and \
           not self.spectra_baseline and not self.gasnet_baseline:
            # latents here are used to
            #  EITHER generate codebook coefficients (wo softmax) in codebook qtz setting
            #  OR concat with lambda to be directly decoded as spectra in autodecoder setting
            latents, redshift_latents = self.init_latents()

            if self.optimize_bins_separately:
                # gt_redshift_bin_ids = get_redshift_bin_ids()
                # redshift_bins_mask = create_redshift_bins_mask(
                #     self.num_redshift_bins)
                # wrong_redshift_bin_ids = \
                #     self.dataset.get_wrong_redshift_bin_ids(redshift_bins_mask)
                # gt_bin_latents, wrong_bin_latents = latents
                # self.pipeline.set_gt_bin_latents(gt_bin_latents)
                # self.pipeline.set_wrong_bin_latents(wrong_bin_latents)
                # self.pipeline.combine_latents_all_bins(
                #     gt_redshift_bin_ids, wrong_redshift_bin_ids, redshift_bins_mask)
                raise NotImplementedError()
            elif self.regularize_binwise_spectra_latents:
                base_latents, addup_latents = latents
                self.pipeline.set_base_latents(base_latents)
                self.pipeline.set_addup_latents(addup_latents)
                # self.pipeline.add_latents()
            else:
                self.pipeline.set_latents(latents)

            if redshift_latents is not None:
                self.pipeline.set_redshift_latents(redshift_latents)
            self.freeze_and_load()

        log.info(self.pipeline)
        log.info("Total number of parameters: {}".format(
            sum(p.numel() for p in self.pipeline.parameters())))

    def collect_model_params(self):
        # collect all parameters from network and trainable latents
        self.params_dict = {}
        for name, param in self.pipeline.named_parameters():
            self.params_dict[name] = param

    def init_data(self):
        self.save_data = False
        self.configure_dataset()
        self.batch_size = min(self.extra_args["pretrain_batch_size"], len(self.dataset))

        if self.train_based_on_epochs:
            self.shuffle_dataloader = True
            self.dataloader_drop_last = False
            self.init_dataloader()

    def init_loss(self):
        if self.spectra_baseline or self.gasnet_baseline:
            if self.regress_redshift:
                self.redshift_loss = nn.MSELoss(reduction="mean")
            elif self.classify_redshift:
                if self.extra_args["redshift_classification_strategy"] == "binary":
                    self.redshift_loss = nn.BCEWithLogitsLoss(reduction="none")
                elif self.extra_args["redshift_classification_strategy"] == "multi_class":
                    self.redshift_loss = nn.CrossEntropyLoss()
                else: raise ValueError()
            else: raise ValueError()
        elif self.freeze_spectra:
            # self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
            self.bce_loss = nn.BCELoss(reduction="none")
        else:
            if self.plot_individ_spectra_loss:
                # if we need loss for each spectra in a batch
                #   we don't reduce when calculating spectra loss
                assert self.extra_args["spectra_loss_cho"][-4:] == "none"
                raise NotImplementedError()

            self.spectra_reduce_func = get_reduce(self.extra_args["spectra_loss_reduction"])

            loss_func = get_loss(
                self.extra_args["spectra_loss_cho"], "none", self.cuda,
                filter_size=self.extra_args["spectra_ssim_loss_filter_size"],
                filter_sigma=self.extra_args["spectra_ssim_loss_filter_sigma"]
            )
            self.spectra_loss_func = spectra_supervision_loss(
                loss_func, self.extra_args["weight_by_wave_coverage"])

            if self.plot_l2_loss:
                l2_loss_func = get_loss("l2", "none", self.cuda)
                self.spectra_l2_loss_func = spectra_supervision_loss(
                    l2_loss_func, self.extra_args["weight_by_wave_coverage"])

            if self.pixel_supervision:
                loss = get_loss(self.extra_args["pixel_loss_cho"], self.cuda)
                self.pixel_loss = partial(pretrain_pixel_loss, loss)

    def init_dataloader(self):
        """ (Re-)Initialize dataloader.
        """
        if self.shuffle_dataloader: sampler_cls = RandomSampler
        else: sampler_cls = SequentialSampler
        # sampler_cls = SequentialSampler
        # sampler_cls = RandomSampler

        sampler = BatchSampler(
            sampler_cls(self.dataset),
            batch_size=self.batch_size,
            drop_last=self.dataloader_drop_last
        )

        self.train_data_loader = DataLoader(
            self.dataset,
            batch_size=None,
            sampler=sampler,
            pin_memory=True,
            num_workers=self.extra_args["dataset_num_workers"]
        )

    def init_optimizer(self):
        if self.spectra_pretrain:
            latents, net_params = self._assign_pretrain_optimization_params()
            if self.neg_sup_optimize_alternately:
                assert not self.use_lbfgs
                latents_optm = self.optim_cls(latents, **self.optim_params)
                net_optm = self.optim_cls(net_params, **self.optim_params)
                optms = { "latents": latents_optm, "net": net_optm }
            else:
                if self.use_lbfgs:
                    assert len(net_params) == 0
                    params = {
                        "lr": self.extra_args["redshift_latents_lr"],
                        "max_iter": 4,
                        "history_size": 10
                    }
                    optms = { "latents_optimizer": torch.optim.LBFGS(latents, **params) }
                else:
                    net_params.extend(latents)
                    optms = { "all_params": self.optim_cls(net_params, **self.optim_params) }

        elif self.pre_classification_mode:
            if self.optimize_latents_alternately:
                spectra_latents, redshift_latents, net_params = \
                    self._assign_brute_force_optimization_params()
                if self.use_lbfgs:
                    raise NotImplementedError()
                else:
                    redshift_latents_optm = self.optim_cls(
                        redshift_latents, **self.optim_params)
                    spectra_latents_optm = self.optim_cls(
                        spectra_latents, **self.optim_params)
                    optms = {
                        "redshift_latents": redshift_latents_optm,
                        "spectra_latents": spectra_latents_optm }
            else:
                latents, net_params = self._assign_brute_force_optimization_params()
                if self.use_lbfgs:
                    assert len(net_params) == 0
                    params = {
                        "lr": self.extra_args["redshift_latents_lr"],
                        "max_iter": 4,
                        "history_size": 10
                    }
                    optms = { "latents_optimizer": torch.optim.LBFGS(latents, **params) }
                else:
                    net_params.extend(latents)
                    optms = { "all_params": self.optim_cls(net_params, **self.optim_params) }

        elif self.freeze_spectra or self.spectra_baseline or self.gasnet_baseline:
            if self.freeze_spectra: lr = self.extra_args["classifier_lr"]
            else: lr = self.extra_args["baseline_lr"]
            net_params = self._assign_model_optimization_params(lr)
            # net_params = self._assign_model_optimization_params()
            optms = { "all_params": self.optim_cls(net_params, **self.optim_params) }
        else:
            raise ValueError()

        self.optimizer = multi_optimizer(**optms)
        if self.verbose: log.info(self.optimizer)

    def configure_dataset(self):
        """ Configure dataset with selected fields and set length accordingly.
        """
        self.dataset.set_mode(self.mode)
        fields = ["spectra_sup_bounds"]

        if self.freeze_spectra:
            self.set_redshift_classification_data_fields()
            fields = self.set_redshift_classification_data(fields)
            self.dataset.toggle_classifier_train_sample_bins(
                self.extra_args["classifier_train_sample_bins"])

        elif self.spectra_baseline or self.gasnet_baseline:
            fields.extend([
                "spectra_mask","spectra_redshift","spectra_source_data"])
            if self.classify_redshift:
                fields.extend(["gt_redshift_bin_ids","redshift_bins_mask"])
        else:
            fields.extend(["wave_data","spectra_mask","gt_redshift_bin_ids",
                           "spectra_redshift","spectra_source_data"])
            if self.sanity_check_sample_bins:
                fields.extend(["redshift_bins_mask","selected_bins_mask"])
                self.dataset.toggle_sanity_check_sample_bins(True)
            elif self.sanity_check_sample_bins_per_step:
                fields.append("idx")
                self.dataset.toggle_sanity_check_sample_bins_per_step(True)

        # todo, codebook pretrain "coords" not handled
        if self.pixel_supervision:
            fields.append("spectra_pixels")
        if self.plot_gt_bin_loss or self.neg_sup_wrong_redshift:
            fields.extend(["gt_redshift_bin_ids","redshift_bins_mask"])

        # use original spectra wave
        self.dataset.set_wave_source("spectra")

        # set spectra data source
        # tmp: in sanity check mode, we manually change sup_id and val_id in spectra_data
        if self.generalization:
            self.dataset.set_spectra_source("test")
        elif self.sanity_check or self.freeze_spectra:
            self.dataset.set_spectra_source("val")
        else: self.dataset.set_spectra_source("sup")

        # set input latents for codebook net
        fields.append("idx")
        self.dataset.toggle_integration(self.pixel_supervision)
        self.dataset.toggle_wave_sampling(self.sample_wave)
        if self.sample_wave:
            self.dataset.set_num_wave_samples(self.extra_args["pretrain_num_wave_samples"])
            self.dataset.set_wave_sample_method(self.extra_args["pretrain_wave_sample_method"])

        # if self.freeze_spectra:
        #     length = self.num_spectra * self.num_redshift_bins
        # else: length = self.num_spectra
        length = self.num_spectra
        self.dataset.set_length(length)
        if self.extra_args["infer_selected"]:
            self.selected_ids = select_inferrence_ids(
                length, self.extra_args["pretrain_num_infer_upper_bound"])
        else: self.selected_ids = np.arange(length)

        # weighted spectra training
        if self.use_global_spectra_loss_as_lambdawise_weights:
            emitted_wave, loss = np.load(self.global_restframe_spectra_loss_fname)
            global_restframe_spectra_loss = interp1d(
                emitted_wave, loss, bounds_error=False, fill_value=math.nan)
            self.dataset.set_hardcode_data(
                "global_restframe_spectra_loss", global_restframe_spectra_loss)
            fields.append("global_restframe_spectra_loss")

        elif self.regress_lambdawise_weights:
            if self.regress_lambdawise_weights_share_latents:
                if self.regress_lambdawise_weights_use_gt_bin_latent:
                    raise NotImplementedError()
                else:
                    if self.dum_init_optm_bin_ids:
                        # we perform dum initialization only before the first iteration
                        self.optm_bin_ids = torch.zeros(self.num_spectra).to(torch.long)
                        self.dataset.set_hardcode_data("optm_bin_ids", self.optm_bin_ids)
                    fields.append("optm_bin_ids")

        self.dataset.set_fields(fields)

    def set_redshift_classification_data(self, fields):
        assert self.freeze_spectra

        log_dir = get_redshift_classification_data_dir(self.mode, **self.extra_args)
        log_dir = join(self.log_dir, "..", log_dir)
        prefix = self.extra_args["redshift_classification_data_fname_prefix"]

        for field in self.redshift_classification_batched_fields:
            if field == "baseline_logits":
                base_log_dir = get_log_dir(**self.extra_args)
                suffix = self.extra_args["baseline_logits_fname_suffix"]
                fname = join(base_log_dir, self.extra_args["baseline_logits_path"],
                             "pretrain_redshift", f"{suffix}_redshift_logits.npy")
            else: fname = join(log_dir, f"{prefix}_{field}.npy")
            field = get_redshift_classification_data_field_name(field)
            fields.append(field)
            self.dataset.set_hardcode_data(field, np.load(fname))

        for field in self.redshift_classification_unbatched_fields:
            # todo, single file loading
            assert 0

        return fields

    def set_redshift_classification_data_fields(self):
        assert self.freeze_spectra

        self.redshift_classification_need_loss = \
            self.extra_args["classify_based_on_loss"] or \
            self.extra_args["classify_based_on_concat_wave_loss"]
        self.redshift_classification_need_spectra = \
            self.extra_args["classify_based_on_concat_spectra"] or \
            self.extra_args["classify_based_on_concat_wave_spectra"]
        self.redshift_classification_need_emit_wave = \
            self.extra_args["classify_based_on_concat_wave_loss"] or \
            self.extra_args["classify_based_on_concat_wave_spectra"]

        data_fields = [ "spectra_mask","redshift_bins_mask" ]
        if self.extra_args["classifier_train_add_baseline_logits"]:
            data_fields.append("baseline_logits")
        if self.extra_args["classifier_train_use_bin_sampled_data"]:
            data_fields.append("selected_bins_mask")
        if self.redshift_classification_need_loss:
            data_fields.append("spectra_lambdawise_losses")
        if self.redshift_classification_need_spectra:
            data_fields.extend(["gt_spectra","recon_spectra"])
        if self.redshift_classification_need_emit_wave:
            data_fields.extend([
                "wave_range","spectra_wave","spectra_redshift"])
        self.classification_forward_data_fields = list(
            set(data_fields) -
            set(["redshift_bins_mask","selected_bins_mask","baseline_logits"]))

        if self.extra_args["load_classification_data_individually"]:
            self.redshift_classification_batched_fields = \
                self.extra_args["redshift_classification_batched_data_fields"]
            self.redshift_classification_unbatched_fields = list(
                set(data_fields) - set(batched_fields))
            self.redshift_classification_num_files_saved_offset = 0
        else:
            self.redshift_classification_batched_fields = data_fields
            self.redshift_classification_unbatched_fields = []

    #############
    # Training logic
    #############

    def begin_train(self):
        self.total_steps = 0 # used for resume training purpose

        # total steps of pretrain used only when we do temperaturized qtz
        # this value should not change as we freeze codebook & qtz operation together
        self.spectra_pretrain_total_steps = 0

        if self.plot_loss:
            self.loss = []
            if self.plot_gt_bin_loss:
                self.gt_bin_loss = []
            if self.neg_sup_wrong_redshift:
                self.gt_bin_loss = []
                self.wrong_bin_regus = []
                self.wrong_bin_loss = []
        if self.plot_l2_loss:
            self.l2_loss = []
            if self.plot_gt_bin_loss:
                self.gt_bin_l2_loss = []
        if self.extra_args["plot_logits_for_gt_bin"]:
            self.gt_bin_logits = []
        if self.plot_individ_spectra_loss:
            self.spectra_individ_loss = []

        # for n, p in self.pipeline.named_parameters():
            # print(n, p.requires_grad)

        if self.extra_args["resume_train"]:
            self.resume_train()

        # check model state (frozen or not)
        # for n, p in self.pipeline.named_parameters():
            # print(n, p.requires_grad)

    def end_train(self):
        self.writer.close()

        if self.extra_args["plot_logits_for_gt_bin"]:
            self._plot_logits_for_gt_bin()

        if self.plot_grad_every != -1:
            plt.savefig(self.grad_fname)
            plt.close()

        if self.plot_loss:
            self._plot_loss(self.loss, self.loss_fname)
            if self.plot_gt_bin_loss:
                self._plot_loss(self.gt_bin_loss, self.gt_bin_loss_fname)
            if self.neg_sup_wrong_redshift:
                self._plot_loss(self.gt_bin_loss, self.gt_bin_loss_fname)
                self._plot_loss(self.wrong_bin_regus, self.wrong_bin_regu_fname)
                self._plot_loss(self.wrong_bin_loss, self.wrong_bin_loss_fname)
            if self.plot_individ_spectra_loss:
                self._plot_individ_spectra_loss()
        if self.plot_l2_loss:
            self._plot_loss(self.l2_loss, self.l2_loss_fname)
            if self.plot_gt_bin_loss:
                self._plot_loss(self.gt_bin_l2_loss, self.gt_bin_l2_loss_fname)

        if self.extra_args["log_gpu_every"] != -1:
            nvidia_smi.nvmlShutdown()

    #############
    # begin/end each iter
    #############

    def begin_iter(self):
        if self.train_based_on_epochs:
            self.cur_batch = 0
            self.set_num_batches()
            self.reset_data_iterator()

        self.init_log_dict()
        self.configure_alternate_optimization()
        self.pipeline.train()

        if self.extra_args["plot_logits_for_gt_bin"]:
            self.cur_gt_bin_logits = []
        if self.plot_individ_spectra_loss:
            self.cur_spectra_individ_loss = []
        if self.save_model_every > -1 and self.cur_iter % self.save_model_every == 0:
            self.save_model()
        if self.save_data_every > -1 and self.cur_iter % self.save_data_every == 0:
            self.pre_save_data()

        if self.regress_lambdawise_weights and \
           self.regress_lambdawise_weights_share_latents and \
           not self.regress_lambdawise_weights_use_gt_bin_latent:
            self.optm_bin_ids = []
            self.dum_init_optm_bin_ids = False

        self.timer.check("begun iteration")

    def end_iter(self):
        self.pipeline.eval()

        if self.plot_loss: self.add_loss()
        if self.log_tb_every > -1 and self.cur_iter % self.log_tb_every == 0:
            self.log_tb()
        if self.log_cli_every > -1 and self.cur_iter % self.log_cli_every == 0:
            self.log_cli()
        if self.render_tb_every > -1 and self.cur_iter % self.render_tb_every == 0:
            self.render_tb()
        if self.save_data_every > -1 and self.cur_iter % self.save_data_every == 0:
            self.post_save_data()

        if self.regress_lambdawise_weights_share_latents and \
           not self.regress_lambdawise_weights_use_gt_bin_latent:
            self.optm_bin_ids = torch.stack(self.optm_bin_ids).detach().cpu()
            self.dataset.set_hardcode_data("optm_bin_ids", self.optm_bin_ids)

        self._toggle_grad(on_off="on")
        freeze_layers_incl( # always freeze pe
            self.pipeline, incls=["wave_encoder"])

        self.timer.check("iter ended")

    def reset_data_iterator(self):
        """ Rewind the iterator for the new epoch.
        """
        self.train_data_loader_iter = iter(self.train_data_loader)

    def init_log_dict(self):
        """ Custom log dict.
        """
        super().init_log_dict()
        self.log_dict["pixel_loss"] = 0.0
        self.log_dict["gt_bin_loss"] = 0.0
        self.log_dict["spectra_loss"] = 0.0
        self.log_dict["total_l2_loss"] = 0.0
        self.log_dict["gt_bin_l2_loss"] = 0.0
        self.log_dict["spectra_l2_loss"] = 0.0
        self.log_dict["wrong_bin_regus"] = 0.0
        self.log_dict["wrong_bin_loss"] = 0.0
        self.log_dict["redshift_logits_regu"] = 0.0
        self.log_dict["spectra_latents_regu"] = 0.0
        self.log_dict["codebook_spectra_regu"] = 0.0
        self.log_dict["spectra_wrong_bin_regu"] = 0.0
        self.log_dict["binwise_spectra_latents_regu"] = 0.0

    #############
    # One step
    #############

    def pre_step(self):
        # since we are optimizing latents which are inputs for the pipeline
        # we should also update the input each time
        # however since dataset receives a reference to the latents,
        # we don't need to update dataset with updated latents manually
        # self.dataset.set_hardcode_data("spectra_latents", self.latents.weight)
        pass

    def step(self, data):
        """ Advance the training by one step using the batched data supplied.
            @Param:
              data (dict): Dictionary of the input batch from the DataLoader.
        """
        def closure(keep_ret=False):
            self.optimizer.zero_grad()
            self.timer.check("zero grad")
            loss, ret = self.calculate_loss(data)
            loss.backward()
            self.timer.check("backward done")
            if keep_ret: return loss, ret
            return loss

        if self.use_lbfgs:
            # todo, use lbfgs in em (uncomment line below and debug)
            # self.optimizer.step(target=self.cur_optm_target, closure=closure)
            self.optimizer.step(closure=closure)
            self.timer.check("stepped")

        loss, ret = closure(True) # forward backward without step
        if self._plot_grad_now():
            plot_grad_flow(self.params_dict.items(), self.grad_fname)

        if not self.use_lbfgs:
            self.optimizer.step(target=self._get_optm_target())
            self.timer.check("stepped")

        return ret

    def post_step(self, data, ret):
        if self.extra_args["plot_logits_for_gt_bin"]:
            gt_bin_ids = data["gt_redshift_bin_ids"][:,None]
            batch_ids = np.arange(len(gt_bin_ids))[:,None]
            ids = np.concatenate((batch_ids, gt_bin_ids), axis=-1)
            self.cur_gt_bin_logits.extend(
                ret["redshift_logits"][ids[:,0],ids[:,1]].detach().cpu().numpy())

        if self.save_data:
            if self.save_redshift:
                self.gt_redshift.extend(data["spectra_redshift"])
                if not self.apply_gt_redshift:
                    if self.classify_redshift:
                        ids = torch.argmax(ret["redshift_logits"], dim=-1)
                        argmax_redshift = ret["redshift"][ids]
                        self.est_redshift.extend(argmax_redshift)
                        self.redshift_logits.extend(ret["redshift_logits"])
                        self.binwise_loss.extend(ret["spectra_binwise_loss"])
                    else:
                        self.est_redshift.extend(ret["redshift"])
                else:
                    self.est_redshift.extend(data["spectra_redshift"])

            if self.save_qtz_weights:
                self.qtz_weights.extend(ret["qtz_weights"])
            if self.save_pixel_values:
                self.recon_pixel_vals.extend(ret["intensity"])
                self.gt_pixel_vals.extend(data["spectra_pixels"])
            if self.plot_codebook_logits:
                self.codebook_logits.extend(ret["codebook_logits"])
            if self.recon_spectra:
                self.recon_fluxes.extend(ret["spectra"])
                self.gt_fluxes.extend(data["spectra_source_data"][:,1])
                self.spectra_wave.extend(data["spectra_source_data"][:,0])
                self.spectra_mask.extend(data["spectra_mask"])
                if self.plot_gt_bin_spectra:
                    self.gt_bin_fluxes.extend(ret["gt_bin_spectra"])
            if self.plot_spectra_with_weights:
                assert ret["lambdawise_weights"].ndim == 3 # [nbins,bsz,nsmpl]
                if self.regress_lambdawise_weights_share_latents:
                    self.lambdawise_weights.extend(ret["lambdawise_weights"][0])
                else:
                    self.lambdawise_weights.extend(
                        ret["lambdawise_weights"].permute(1,0,2))
            if self.recon_codebook_spectra_individ:
                self.codebook_spectra.extend(ret["codebook_spectra"])
                self.spectra_wave_c.extend(data["spectra_source_data"][:,0])
                self.spectra_mask_c.extend(data["spectra_mask"])

        self.total_steps += 1
        self.timer.check("batch ended")

    #############
    # Model Helpers
    #############

    def init_latents(self):
        if self.spectra_pretrain:
            assert not self.split_latent
            latents = self.init_pretrain_spectra_latents()
            redshift_latents = None
        elif self.brute_force:
            latents = self.init_brute_force_spectra_latents()
            if not self.has_redshift_latents:
                redshift_latents = None
            else: redshift_latents = self.init_brute_force_redshift_latents()
        else:
            raise ValueError("Invalid spectra trainer mode!")
        return latents, redshift_latents

    def freeze_and_load(self):
        """ For redshift pretrain (sanity check), we load part of the pretrained
            model (from codebook pretrain) and freeze.
            Note: if we also resume, then `resume_train` should overwrite model states
                  i.e. `resume_train` called after this func
        """
        if self.spectra_pretrain or self.freeze_spectra:
            pass
        elif self.sanity_check or self.generalization:
            load_excls, freeze_excls = [], []

            # we load latents in `init_latents` only
            load_excls.append("model.latents")

            if self.optimize_spectra_latents:
                if self.regularize_binwise_spectra_latents:
                    freeze_excls.extend(["model.base_latents","model.addup_latents"])
                else:
                    freeze_excls.append("model.latents")

            if self.optimize_spectra_latents_as_logits:
                pass # directly optimize latents as coefficients
            else:
                # optimize a mlp to decode latents to coefficients
                if self.extra_args["optimize_codebook_logits_mlp"]:
                    freeze_excls.append("spatial_decoder.decode")
                if not self.extra_args["load_pretrained_codebook_logits_mlp"]:
                    load_excls.append("spatial_decoder.decode")

            # redshift bin logits
            if self.classify_redshift:
                if self.optimize_redshift_latents_as_logits:
                    # directly optimize logits
                    freeze_excls.append("model.redshift_latents")
                else:
                    # optimize an autodecoder
                    freeze_excls.append("redshift_decoder")
                    freeze_excls.append("model.redshift_latents")

            if self.generalize_train_first_layer:
                assert(exists(self.pretrained_model_fname))
                checkpoint = torch.load(self.pretrained_model_fname)
                for n in checkpoint["model_state_dict"].keys():
                    if "wave_encoder" in n or "spectra_decoder.layers.0" in n:
                        freeze_excls.append(n)
                        load_excls.append(n)

            if self.regress_lambdawise_weights and self.sanity_check:
                # in case of weighted sanity check, we dont freeze any layers
                assert(exists(self.pretrained_model_fname))
                checkpoint = torch.load(self.pretrained_model_fname)
                for n in checkpoint["model_state_dict"].keys():
                    freeze_excls.append(n)

            # we dont freeze layers of the current model that are not in the pretrained model
            assert(exists(self.pretrained_model_fname))
            checkpoint = torch.load(self.pretrained_model_fname)
            nms = set(checkpoint["model_state_dict"].keys())
            for n, p in self.pipeline.named_parameters():
                if n not in nms:
                    freeze_excls.append(n)

            # sc sample debug #REMOVE
            # checkpoint = torch.load(self.pretrained_model_fname)
            # for n in checkpoint["model_state_dict"].keys():
            #     freeze_excls.append(n)
            # done

            if self.generalization or not self.sanity_check_finetune:
                freeze_layers_excl(self.pipeline, excls=freeze_excls)
                log.info(f"only {freeze_excls} not frozen!")
            else: log.info("no layers frozen!")
            self.load_model(self.pretrained_model_fname, excls=load_excls)
        else:
            raise ValueError()

    def init_pretrain_spectra_latents(self):
        if self.optimize_spectra_latents_as_logits:
            dim = self.extra_args["qtz_num_embed"]
        else: dim = self.extra_args["spectra_latent_dim"]
        if self.optimize_latents_for_each_redshift_bin:
            sp = (self.num_spectra, self.num_redshift_bins, dim)
        else: sp = (self.num_spectra, dim)
        latents = self.create_latents(
            sp, seed=self.extra_args["seed"] + 1,
            zero_init=self.extra_args["zero_init_spectra_latents"],
            freeze=not self.optimize_spectra_latents
        )
        return latents

    def init_brute_force_redshift_latents(self):
        if not self.apply_gt_redshift and self.split_latent:
            sp = (self.num_spectra, self.extra_args["redshift_logit_latent_dim"])
            latents = self.create_latents(
                sp, seed=self.extra_args["seed"] + 2,
                zero_init=self.extra_args["zero_init_redshift_latents"],
                freeze=not self.optimize_redshift_latents)
        else: latents = None
        return latents

    def init_brute_force_spectra_latents(self):
        """ Initialize latents for spectra generation during sanity.
        """
        latents = None
        if self.optimize_spectra_latents_as_logits:
            sp_z_dim = self.extra_args["qtz_num_embed"]
        else: sp_z_dim = self.extra_args["spectra_latent_dim"]

        if self.sanity_check_sample_bins:
            n_bins = self.extra_args["sanity_check_num_bins_to_sample"]
        else: n_bins = self.num_redshift_bins

        if self.regularize_binwise_spectra_latents:
            sp = [(self.num_spectra, sp_z_dim),
                  (self.num_spectra, n_bins, sp_z_dim)]
        elif self.optimize_latents_for_each_redshift_bin:
            sp = (self.num_spectra, n_bins, sp_z_dim)
        else: sp = (self.num_spectra, sp_z_dim)

        if not self.load_pretrained_spectra_latents:
            pretrained = None
        else:
            assert self.sanity_check
            # checkpoint comes from codebook pretrain (use sup spectra)
            checkpoint = torch.load(self.pretrained_model_fname)
            assert checkpoint["model_state_dict"]["model.latents"].shape[-1] == sp_z_dim

            # sanity check use permuted sup spectra
            permute_ids = self.dataset.get_sanity_check_spectra_ids()
            pretrained = checkpoint["model_state_dict"]["model.latents"][permute_ids].detach()

            if self.optimize_latents_for_each_redshift_bin:
                if self.spectra_pretrain:
                    # when we pretrain with brute force
                    assert pretrained.ndim == 3 and self.brute_force
                    raise NotImplementedError()
                else:
                    pretrained = self.load_pretrained_latents_all_bins(pretrained)

        assert pretrained == None or type(pretrained) == list or pretrained.shape == sp

        latents = self.create_latents(
            sp, seed=self.extra_args["seed"], pretrained=pretrained,
            zero_init=self.extra_args["zero_init_spectra_latents"],
            freeze=not self.extra_args["optimize_spectra_latents"])

        return latents

    def load_pretrained_latents_all_bins(self, pretrained):
        """ This is only called when we pretrain via applying GT redshift directly;
              and sanity check with brute force method.
        """
        n_bins = self.extra_args["sanity_check_num_bins_to_sample"] \
            if self.sanity_check_sample_bins else self.num_redshift_bins

        if self.load_pretrained_spectra_latents_to_gt_bin_only:
            # only load pretrained latents to gt bin, all other bins are 0
            assert self.dataset.get_spectra_source() == "val"

            if self.optimize_bins_separately:
                sp = (len(pretrained), n_bins-1, pretrained.shape[-1])
                wrong_bin_latents = torch.ones(sp, dtype=pretrained.dtype)
                pretrained = [pretrained[:,None], wrong_bin_latents]
            else:
                redshift = self.dataset.get_spectra_redshift() # validation spectra
                gt_bin_ids = get_gt_redshift_bin_ids(redshift, **self.extra_args)
                bsz, dim = pretrained.shape
                sp = (bsz, self.num_redshift_bins, dim)
                to_load = torch.ones(sp, dtype=pretrained.dtype).to(pretrained.device)
                to_load[gt_bin_ids[0],gt_bin_ids[1]] = pretrained

                if self.sanity_check_sample_bins:
                    selected_bins_mask = torch.tensor(self.dataset.get_selected_bins_mask())
                    to_load = to_load[
                        selected_bins_mask[...,None].tile(1,1,dim)].view(bsz,n_bins,dim)

                pretrained = to_load
        else:
            if self.optimize_bins_separately:
                pretrained = [pretrained[:,None],pretrained[:,None].tile(1, n_bins-1, 1)]
            else: pretrained = pretrained[:,None].detach().tile(1, n_bins, 1)

        return pretrained

    def create_latents(self, sp, pretrained=None, zero_init=False, freeze=False, seed=0):
        if pretrained is not None:
            if type(pretrained) == list:
                latents = [cur_latents.to(self.device) for cur_latents in pretrained]
            else:
                latents = pretrained.to(self.device)
        elif zero_init:
            if type(sp) == list:
                latents = [torch.zeros(cur_sp).to(self.device) for cur_sp in sp]
            else: latents = torch.zeros(sp).to(self.device)
            # latents = 0.01 * torch.ones(n,m)
        else:
            torch.manual_seed(seed)
            if type(sp) == list:
                latents = [torch.rand(cur_sp).to(self.device) for cur_sp in sp]
            else: latents = torch.rand(sp).to(self.device)

        if type(latents) == list:
            latents = [nn.Parameter(cur_latents, requires_grad=not freeze)
                       for cur_latents in latents]
        else: latents = nn.Parameter(latents, requires_grad=not freeze)
        return latents

    def load_model(self, model_fname, excls=[]):
        log.info(f"saved model found, loading {model_fname}")
        checkpoint = torch.load(model_fname)
        load_pretrained_model_weights(
            self.pipeline, checkpoint["model_state_dict"], excls=excls)

        self.spectra_pretrain_total_steps = 0
        if self.sanity_check or self.generalization:
            """
            `pretrain_total_steps` is needed only when we do temperaturized quantization.
            After codebook pretrain, we only have `total_steps` in the saved model and we
              use it for the current (very first) sanity check run.
            After the first sanity check, we save `total_steps` of codebook pretrain as
              `spectra_pretrain_total_steps` in the saved model. And `total_steps` in
              saved model now is the total iterations of the sanity check run.
            """
            if "spectra_pretrain_total_steps" not in checkpoint:
                # first sanity check
                self.spectra_pretrain_total_steps = checkpoint["total_steps"]
            else:
                # resume from previous sanity check
                self.spectra_pretrain_total_steps = \
                    checkpoint["spectra_pretrain_total_steps"]

        self.pipeline.train()
        return checkpoint

    def save_model(self):
        if self.train_based_on_epochs:
            fname = f"model-ep{self.cur_iter}-bch{self.cur_batch}.pth"
        else: fname = f"model-step{self.cur_iter}.pth"
        model_fname = os.path.join(self.model_dir, fname)
        if self.verbose: log.info(f"Saving model checkpoint to: {model_fname}")

        checkpoint = {
            "total_steps": self.total_steps,
            "model_state_dict": self.pipeline.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }
        if self.brute_force:
            checkpoint["spectra_pretrain_total_steps"] = self.spectra_pretrain_total_steps
        if self.regress_lambdawise_weights and \
           self.regress_lambdawise_weights_share_latents and \
           not self.regress_lambdawise_weights_use_gt_bin_latent:
            checkpoint["optimal_bin_ids"] = self.optm_bin_ids
        torch.save(checkpoint, model_fname)
        return checkpoint

    def resume_train(self):
        try:
            checkpoint = self.load_model(self.resume_train_model_fname)

            # re-init
            self.collect_model_params()
            self.init_data() # |_ these two can be
            self.init_loss() # |  ommitted
            self.init_optimizer()

            self.total_steps = checkpoint["total_steps"]
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if self.plot_loss:
                if exists(self.resume_loss_fname):
                    self.loss = list(np.load(self.resume_loss_fname))
                if self.plot_l2_loss and exists(self.resume_l2_loss_fname):
                    self.l2_loss = list(np.load(self.resume_l2_loss_fname))
                if self.plot_gt_bin_loss:
                    if exists(self.resume_gt_bin_loss_fname):
                        self.gt_bin_loss = list(np.load(self.resume_gt_bin_loss_fname))
                    if exists(self.resume_gt_bin_l2_loss_fname):
                        self.gt_bin_l2_loss = list(np.load(self.resume_gt_bin_l2_loss_fname))
                if self.neg_sup_wrong_redshift:
                    if exists(self.resume_gt_bin_loss_fname):
                        self.gt_bin_loss = list(np.load(self.resume_gt_bin_loss_fname))
                    if exists(self.resume_wrong_bin_regu_fname):
                        self.wrong_bin_regus = list(np.load(self.resume_wrong_bin_regu_fname))
                    if exists(self.resume_wrong_bin_loss_fname):
                        self.wrong_bin_loss = list(np.load(self.resume_wrong_bin_loss_fname))
                if self.plot_individ_spectra_loss:
                    fname = self.resume_loss_fname[:-4] + "_individ.npy"
                    if exists(fname):
                        self.spectra_individ_loss = list(np.load(fname).T)

            log.info("resumed training")

        except Exception as e:
            log.info(e)
            log.info("start training from begining")

    #############
    # Data Helpers
    #############

    def sample_data(self):
        n = len(self.dataset)
        ids = np.arange(n)
        random.shuffle(ids)
        if self.extra_args["step_based_sample_w_replace"]:
            assert 0
        elif self.extra_args["step_based_sample_wo_replace"]:
            ids = ids[:self.batch_size]
        else: assert 0
        return ids

    def set_num_batches(self):
        """ Set number of batches/iterations and batch size for each epoch.
            At certain epochs, we may not need all data and can break before
              iterating thru all data.
        """
        length = len(self.dataset)
        if self.dataloader_drop_last:
            self.num_batches_cur_epoch = int(length // self.batch_size)
        else:
            self.num_batches_cur_epoch = int(np.ceil(length / self.batch_size))
        # log.info(f"num of batches: {self.num_batches_cur_epoch}")

    def add_loss(self):
        m = len(self.train_data_loader) if self.train_based_on_epochs else 1

        # we need to store below data on cpu
        # i.e. self.log_dict["xxx"] should be on cpu rather than gpu
        self.loss.append(self.log_dict["total_loss"] / m)
        if self.plot_gt_bin_loss:
            self.gt_bin_loss.append(self.log_dict["gt_bin_loss"] / m)
        if self.plot_l2_loss:
            self.l2_loss.append(self.log_dict["total_l2_loss"] / m)
            if self.plot_gt_bin_loss:
                self.gt_bin_l2_loss.append(self.log_dict["gt_bin_l2_loss"] / m)
        if self.neg_sup_optimize_alternately:
            self.gt_bin_loss.append(self.log_dict["gt_bin_loss"] / m)
            self.wrong_bin_loss.append(self.log_dict["wrong_bin_loss"] / m)
            if self.cur_neg_sup_target == "net":
                self.wrong_bin_regus.append(self.log_dict["wrong_bin_regus"] / m)
        if self.extra_args["plot_logits_for_gt_bin"]:
            self.gt_bin_logits.append(self.cur_gt_bin_logits)
        if self.plot_individ_spectra_loss:
            self.spectra_individ_loss.append(self.cur_spectra_individ_loss)

    def pre_save_data(self):
        self.save_data = True

        if self.save_redshift:
            self.gt_redshift = []
            self.est_redshift = []
            if self.classify_redshift:
                self.redshift_logits = []
                if self.brute_force:
                    self.binwise_loss = []
        if self.save_qtz_weights:
            self.qtz_weights = []
        if self.save_pixel_values:
            self.gt_pixel_vals = []
            self.recon_pixel_vals = []
        if self.recon_spectra:
            self.gt_fluxes = []
            self.recon_fluxes = []
            self.spectra_wave = []
            self.spectra_mask = []
            if self.plot_gt_bin_spectra:
                self.gt_bin_fluxes = []
        if self.plot_codebook_logits:
            self.codebook_logits = []
        if self.plot_spectra_with_weights:
            self.lambdawise_weights = []
        if self.recon_codebook_spectra_individ:
            self.spectra_wave_c = []
            self.spectra_mask_c = []
            self.codebook_spectra = []

        # re-init dataloader to make sure pixels are in order
        self.use_all_pixels = True
        self.shuffle_dataloader = False
        self.sample_wave = not self.extra_args["pretrain_use_all_wave"]
        self.dataset.toggle_wave_sampling(self.sample_wave)
        self.set_num_batches()
        self.init_dataloader()
        self.reset_data_iterator()
        warnings.warn("dataloader state is modified in spectra trainer, \
        ensure this is for validation purpose only!")

    def post_save_data(self):
        """ Save data locally and restore trainer state.
        """
        self.save_local()
        self.use_all_pixels = False
        self.shuffle_dataloader = True
        self.save_data = False
        self.sample_wave = not self.extra_args["pretrain_use_all_wave"]
        self.dataset.toggle_wave_sampling(self.sample_wave)
        self.configure_dataset()
        self.set_num_batches()
        self.init_dataloader()
        self.reset_data_iterator()
        warnings.warn("dataloader state is modified in spectra trainer, \
        ensure this is for validation purpose only!")

    def _plot_grad_now(self):
        return self.plot_grad_every != -1 and \
            (self.cur_iter == 0 or self.cur_iter % self.plot_grad_every == 0)

    def save_local(self):
        if self.recon_spectra:
            self._recon_spectra()
        if self.recon_codebook_spectra_individ:
            self._recon_codebook_spectra_individ()
        if self.save_qtz_weights:
            self._save_qtz_weights()
        if self.save_pixel_values:
            self._save_pixel_values()
        if self.plot_codebook_logits:
            self._plot_codebook_logits()
        if self.save_redshift:
            self._save_redshift()
            if self.classify_redshift:
                self._plot_redshift_logits()
                self._log_redshift_residual_outlier()
                if self.brute_force:
                    self._plot_binwise_spectra_loss()

    def _save_redshift(self):
        self.gt_redshift = torch.stack(
            self.gt_redshift)[self.selected_ids].detach().cpu().numpy()
        self.est_redshift = torch.stack(
            self.est_redshift)[self.selected_ids].detach().cpu().numpy()
        np.set_printoptions(suppress=True)
        np.set_printoptions(precision=3)
        log.info(f"gt redshift values: {self.gt_redshift}")
        log.info(f"est redshift values: {self.est_redshift}")

    def _save_pixel_values(self):
        gt_vals = torch.stack(self.gt_pixel_vals).detach().cpu().numpy()[self.selected_ids,0]
        recon_vals = torch.stack(self.recon_pixel_vals).detach().cpu().numpy()[self.selected_ids]

        np.set_printoptions(suppress=True)
        np.set_printoptions(precision=3)
        if self.extra_args["log_pixel_ratio"]:
            ratio = recon_vals / gt_vals
            log.info(f"Recon./GT. ratio: {ratio}")
        else:
            log.info(f"GT. vals {gt_vals}")
            log.info(f"Recon. vals {recon_vals}")

    def _save_qtz_weights(self):
        weights = torch.stack(self.qtz_weights).detach().cpu().numpy()
        fname = join(self.qtz_weight_dir, f"model-ep{self.cur_iter}-bch{self.cur_batch}.pth")
        np.save(fname, weights)
        np.set_printoptions(suppress=True)
        np.set_printoptions(precision=3)
        w = weights[self.selected_ids,0]
        log.info(f"Qtz weights {w}")

    def _recon_spectra(self):
        log.info("reconstructing gt spectrum")

        self.gt_fluxes = torch.stack(self.gt_fluxes).view(
            self.num_spectra, -1).detach().cpu().numpy()[self.selected_ids]
        # [n_spectra,nsmpl]

        self.recon_fluxes = torch.stack(self.recon_fluxes).view(
            self.num_spectra, self.extra_args["spectra_neighbour_size"]**2, -1
        ).detach().cpu().numpy()[self.selected_ids]

        if self.plot_gt_bin_spectra:
            gt_bin_fluxes = torch.stack(self.gt_bin_fluxes).view(
                self.num_spectra, -1).detach().cpu().numpy()[self.selected_ids]
        else: gt_bin_fluxes = None

        if self.plot_spectra_with_weights:
            lambdawise_weights = torch.stack(
                self.lambdawise_weights).detach().cpu().numpy()[self.selected_ids]
            if not self.regress_lambdawise_weights_share_latents:
                lambdawise_weights = lambdawise_weights.permute(1,0,2)
        else: lambdawise_weights = None

        self.spectra_wave = torch.stack(self.spectra_wave).view(
            self.num_spectra, -1).detach().cpu().numpy()[self.selected_ids]
        self.spectra_mask = torch.stack(self.spectra_mask).bool().view(
            self.num_spectra, -1).detach().cpu().numpy()[self.selected_ids]

        # plot spectrum in multiple figures, each figure contains several spectrum
        n_spectra = len(self.selected_ids)
        titles = np.char.mod("%d", np.arange(n_spectra))
        n_spectrum_per_fig = self.extra_args["num_spectrum_per_fig"]
        n_figs = int(np.ceil( n_spectra / n_spectrum_per_fig ))

        for i in range(n_figs):
            fname = f"ep{self.cur_iter}-bch{self.cur_batch}-plot{i}"
            lo = i * n_spectrum_per_fig
            hi = min(lo + n_spectrum_per_fig, n_spectra)

            if self.plot_gt_bin_spectra:
                cur_gt_bin_fluxes = gt_bin_fluxes[lo:hi]
            else: cur_gt_bin_fluxes = None
            if self.plot_spectra_with_weights:
                cur_lambdawise_weights = lambdawise_weights[lo:hi],
            else: cur_lambdawise_weights = None

            cur_metrics = self.dataset.plot_spectrum(
                self.spectra_dir, fname,
                self.extra_args["flux_norm_cho"], None,
                self.spectra_wave[lo:hi], None, self.gt_fluxes[lo:hi],
                self.spectra_wave[lo:hi], self.recon_fluxes[lo:hi],
                recon_fluxes2=cur_gt_bin_fluxes,
                lambdawise_losses=None,
                lambdawise_weights=cur_lambdawise_weights,
                clip=self.extra_args["plot_clipped_spectrum"],
                gt_mask=self.spectra_mask[lo:hi],
                recon_mask=self.spectra_mask[lo:hi],
                calculate_metrics=False,
                titles=titles[lo:hi])

    def _recon_codebook_spectra_individ(self):
        """ Reconstruct codebook spectra for each spectra individually.
        """
        log.info("reconstructing codebook spectrum")

        self.codebook_spectra = torch.stack(self.codebook_spectra).view(
            self.num_spectra, self.qtz_n_embd, -1
        ).detach().cpu().numpy()[self.selected_ids]
        self.spectra_wave_c = torch.stack(self.spectra_wave_c).view(
            self.num_spectra, -1).detach().cpu().numpy()[self.selected_ids]
        self.spectra_mask_c = torch.stack(self.spectra_mask_c).bool().view(
            self.num_spectra, -1).detach().cpu().numpy()[self.selected_ids]

        if self.extra_args["infer_selected"]:
            input("plot codebook spectra for all spectra, press Enter to confirm...")

        prefix = "individ-"
        for i, (wave, mask, codebook_spectra) in enumerate(
                zip(self.spectra_wave_c, self.spectra_mask_c, self.codebook_spectra)
        ):
            cur_dir = join(self.codebook_spectra_dir, f"spectra-{i}")
            Path(cur_dir).mkdir(parents=True, exist_ok=True)

            fname = f"{prefix}ep{self.cur_iter}-bch{self.cur_batch}"
            wave = np.tile(wave, self.qtz_n_embd).reshape(self.qtz_n_embd, -1)
            mask = np.tile(mask, self.qtz_n_embd).reshape(self.qtz_n_embd, -1)

            self.dataset.plot_spectrum(
                cur_dir, fname, self.extra_args["flux_norm_cho"],
                None, None, wave, codebook_spectra,
                is_codebook=True,
                save_spectra_together=True,
                clip=self.extra_args["plot_clipped_spectrum"],
                recon_mask=mask)

    def _plot_logits_for_gt_bin(self):
        logits = np.array(self.gt_bin_logits) # [nepochs,nspectra]
        plot_multiple(
            self.extra_args["num_spectrum_per_fig"],
            self.extra_args["num_spectrum_per_row"],
            logits.T, self.gt_bin_logits_fname)

    def _plot_codebook_logits(self):
        codebook_logits = torch.stack(self.codebook_logits).detach().cpu().numpy()
        fname = join(self.codebook_logits_dir, f"ep{self.cur_iter}-bch{self.cur_batch}_logits")
        np.save(fname, codebook_logits)

        plot_multiple(
            self.extra_args["num_spectrum_per_fig"],
            self.extra_args["num_spectrum_per_row"],
            codebook_logits, fname, hist=True)

    def _plot_individ_spectra_loss(self):
        losses = np.array(self.spectra_individ_loss).T
        np.save(self.loss_fname + "_individ.npy", losses)
        plot_multiple(
            self.extra_args["num_spectrum_per_fig"],
            self.extra_args["num_spectrum_per_row"],
            losses, self.loss_fname + "_individ")

        plot_multiple(
            self.extra_args["num_spectrum_per_fig"],
            self.extra_args["num_spectrum_per_row"],
            np.log10(losses), self.loss_fname + "_individ_log10")

    def _log_redshift_residual_outlier(self):
        self.redshift_residual = self.est_redshift - self.gt_redshift
        fname = join(self.redshift_dir,
                     f"ep{self.cur_iter}-bch{self.cur_batch}_redshift_residual.txt")
        log_data(self, "redshift_residual", fname=fname, log_to_console=False)

        ids = np.arange(len(self.redshift_residual))
        outlier = ids[np.abs(self.redshift_residual) > self.extra_args["redshift_bin_width"]]
        outlier_gt = self.gt_redshift[outlier]
        outlier_est = self.est_redshift[outlier]
        to_save = np.array(list(outlier) + list(outlier_gt) + list(outlier_est)).reshape(3,-1)
        log.info(f"outlier spectra: {outlier}")
        log.info(f"gt_redshift: {outlier_gt}")
        log.info(f"argmax_redshift: {outlier_est}")
        fname = join(self.redshift_dir,
                     f"ep{self.cur_iter}-bch{self.cur_batch}_redshift_outlier.txt")
        with open(fname, "w") as f: f.write(f"{to_save}")

    def _plot_binwise_spectra_loss(self):
        losses = torch.stack(self.binwise_loss).detach().cpu().numpy()
        bin_centers = init_redshift_bins(**self.extra_args)
        fname = join(self.redshift_dir, f"ep{self.cur_iter}-bch{self.cur_batch}_loss")
        np.save(fname, np.concatenate((bin_centers[None,:], losses), axis=0))
        plot_multiple(
            self.extra_args["num_spectrum_per_fig"],
            self.extra_args["num_spectrum_per_row"],
            losses, fname, x=bin_centers,vertical_xs=self.gt_redshift)

    def _plot_redshift_logits(self):
        redshift_logits = torch.stack(self.redshift_logits).detach().cpu().numpy()
        bin_centers = init_redshift_bins(**self.extra_args)
        fname = join(self.redshift_dir, f"ep{self.cur_iter}-bch{self.cur_batch}_logits")
        np.save(fname, np.concatenate((bin_centers[None,:], redshift_logits), axis=0))
        plot_multiple(
            self.extra_args["num_spectrum_per_fig"],
            self.extra_args["num_spectrum_per_row"],
            redshift_logits, fname, x=bin_centers,vertical_xs=self.gt_redshift
        )

    def _plot_redshift_est_stats(self):
        raise NotImplementedError()
        if self.extra_args["plot_redshift_est_stats_individually"]:
            plot_redshift_estimation_stats_individually(
                self.gt_bin_ids, self.gt_redshift, self.extra_args["redshift_lo"],
                self.extra_args["redshift_hi"], self.extra_args["redshift_bin_width"],
                self.extra_args["num_spectrum_per_row"], f"{fname}_precision_recall.png",
                self.extra_args["num_redshift_est_stats_residual_levels"],
                cho=self.extra_args["redshift_est_stats_cho"])
        else:
            plot_redshift_estimation_stats_together(
                self.gt_bin_ids, self.gt_redshift, self.extra_args["redshift_lo"],
                self.extra_args["redshift_hi"], self.extra_args["redshift_bin_width"],
                f"{fname}_precision_recall",
                self.extra_args["num_redshift_est_stats_residual_levels"],
                cho=self.extra_args["redshift_est_stats_cho"])

        log.info("redshift logits plotting done")

    def _plot_loss(self, losses, fname):
        x = np.arange(len(losses))
        plt.plot(x, losses); plt.title("Loss")
        plt.savefig(fname + ".png")
        plt.close()
        np.save(fname + ".npy", np.array(losses))

        plt.plot(x, np.log10(np.array(losses)))
        plt.title("Log10 loss")
        plt.savefig(fname + "_log10.png")
        plt.close()

    def _get_all_bin_losses(self, ret):
        if self.classify_redshift_based_on_l2:
            all_bin_losses = ret["spectra_binwise_loss_l2"] # [bsz,nbins]
        else: all_bin_losses = ret["spectra_binwise_loss"] # [bsz,nbins]
        return all_bin_losses

    def _get_optimal_wrong_bin_data(self, ret, data):
        all_bin_losses = self._get_all_bin_losses(ret)
        ids, optimal_wrong_bin_losses = get_optimal_wrong_bin_ids(
            all_bin_losses, data["redshift_bins_mask"]
        )
        ids = create_batch_ids(ids.detach().cpu().numpy())
        return ids, optimal_wrong_bin_losses

    #############
    # Loss Helpers
    #############

    def calculate_loss(self, data):
        total_loss = 0
        add_to_device(data, self.gpu_fields, self.device)
        self.timer.check("added to gpu")

        spectra_loss_func, spectra_l2_loss_func = None, None
        if not (self.spectra_baseline or self.gasnet_baseline or self.freeze_spectra):
            spectra_loss_func = self.spectra_loss_func
            if self.plot_l2_loss:
                spectra_l2_loss_func = self.spectra_l2_loss_func

        if self.brute_force:
            steps = self.spectra_pretrain_total_steps
        else: steps = self.total_steps

        if self.freeze_spectra:
            clsfy_forward_data_fields = self.classification_forward_data_fields
        else: clsfy_forward_data_fields = None

        if self.sanity_check_sample_bins or \
           self.sanity_check_sample_bins_per_step:
            self.pipeline.toggle_sample_bins(True)

        ret = spectra_redshift_forward(
            data,
            self.pipeline,
            qtz=self.qtz,
            qtz_strategy=self.qtz_strategy,
            spectra_loss_func=spectra_loss_func,
            spectra_l2_loss_func=spectra_l2_loss_func,
            index_latent=self.index_latent,
            plot_l2_loss=self.plot_l2_loss,
            baseline=self.spectra_baseline or \
                     self.gasnet_baseline,
            apply_gt_redshift=self.apply_gt_redshift,
            regress_redshift=self.regress_redshift,
            classify_redshift=self.classify_redshift,
            brute_force=self.brute_force,
            freeze_spectra=self.freeze_spectra,
            freeze_spectra_data_fields=clsfy_forward_data_fields,
            sanity_check_sample_bins=self.sanity_check_sample_bins,
            sanity_check_sample_bins_per_step=self.sanity_check_sample_bins_per_step,
            calculate_binwise_spectra_loss= self.brute_force,
            save_coords=self.regularize_spectra_latents,
            save_spectra=self.recon_spectra,
            save_redshift=self.save_data and self.save_redshift,
            save_gt_bin_spectra=self.save_data and self.plot_gt_bin_spectra,
            save_redshift_logits=self.regularize_redshift_logits or \
                                 (self.save_data and self.classify_redshift) or \
                                 self.freeze_spectra
        )
        self.timer.check("forwarded")

        spectra_loss, spectra_l2_loss, spectra_wrong_bin_regu = \
            self._calculate_spectra_loss(ret, data)

        recon_loss = 0
        if self.pixel_supervision:
            gt_pixels = data["spectra_pixels"]
            recon_pixels = ret["intensity"]
            recon_loss = self.pixel_loss(gt_pixels, recon_pixels)
            recon_loss *= extra_args["pretrain_pixel_beta"]
            self.log_dict["pixel_loss"] += recon_loss.item()

        redshift_logits_regu = 0
        if self.regularize_redshift_logits:
            logits = ret["redshift_logits"]
            if self.redshift_logits_regu_method == "l1":
                redshift_logits_regu = torch.mean(torch.sum(logits, dim=-1))
            elif self.redshift_logits_regu_method == "l1_excl_largest":
                largest, _ = torch.max(logits, dim=-1)
                redshift_logits_regu = torch.mean(torch.sum(logits, dim=-1) - largest)
            elif self.redshift_logits_regu_method == "laplace":
                redshift_logits_regu = torch.mean(
                    -torch.log( torch.exp(-logits) + torch.exp(-(1-logits)) ) + \
                    torch.log( torch.FloatTensor(1 + 1/torch.exp(torch.tensor(1))
                    ).to(self.device)))
            else:
                raise ValueError("Invalid redshift logit regularization method!")

            redshift_logits_regu *= self.extra_args["redshift_logits_regu_beta"]
            self.log_dict["redshift_logits_regu"] += redshift_logits_regu.item()

        codebook_logits_regu= 0
        if self.regularize_codebook_logits:
            codebook_logits_regu = torch.abs(
                torch.mean(torch.sum(ret["codebook_logits"],dim=-1)))
            codebook_logits_regu *= self.extra_args["codebook_logits_regu_beta"]
            self.log_dict["codebook_logits_regu"] += codebook_logits_regu.item()

        spectra_latents_regu = 0
        if self.regularize_spectra_latents:
            spectra_latents_regu = torch.mean(ret["coords"]**2)
            spectra_latents_regu *= self.extra_args["spectra_latents_regu_beta"]
            self.log_dict["spectra_latents_regu"] += spectra_latents_regu.item()

        binwise_spectra_latents_regu = 0
        if self.regularize_binwise_spectra_latents:
            addup_latents = self.pipeline.get_addup_latents() # [bsz,nbins,dim]
            binwise_spectra_latents_regu = torch.sum(torch.abs(addup_latents))
            binwise_spectra_latents_regu *= self.extra_args["spectra_latents_regu_beta"]
            self.log_dict["binwise_spectra_latents_regu"] += \
                binwise_spectra_latents_regu.item()

        codebook_spectra_regu = 0
        if self.regularize_codebook_spectra:
            sp = ret["full_range_codebook_spectra"].shape # [num_embed,nsmpl]
            dtp = ret["full_range_codebook_spectra"].device
            if self.regularize_within_codebook_spectra:
                codebook_spectra_regu = F.l1_loss(
                    ret["full_range_codebook_spectra"],
                    torch.zeros(sp).to(dtp))

            elif self.regularize_across_codebook_spectra:
                codebook_spectra_regu = F.l1_loss(
                    ret["full_range_codebook_spectra"],
                    torch.zeros(sp).to(dtp),
                    reduction='none')
                codebook_spectra_regu = torch.mean(torch.sum(codebook_spectra_regu, dim=0))

            codebook_spectra_regu *= self.extra_args["codebook_spectra_regu_beta"]
            self.log_dict["codebook_spectra_regu"] += codebook_spectra_regu

        total_loss = spectra_loss + recon_loss + spectra_latents_regu + \
            spectra_wrong_bin_regu + redshift_logits_regu + \
            codebook_spectra_regu + binwise_spectra_latents_regu
        self.log_dict["total_loss"] += total_loss.item()

        if self.plot_l2_loss:
            total_l2_loss = spectra_l2_loss + recon_loss + spectra_latents_regu + \
                redshift_logits_regu + codebook_spectra_regu + binwise_spectra_latents_regu
            self.log_dict["total_l2_loss"] += total_l2_loss.item()

        self.timer.check("loss calculated")
        return total_loss, ret

    def _calculate_spectra_loss(self, ret, data):
        spectra_wrong_bin_regu, spectra_l2_loss = 0, None

        if self.spectra_baseline or self.gasnet_baseline:
            if self.regress_redshift:
                spectra_loss = self.redshift_loss(
                    ret["redshift"], data["spectra_redshift"])
            elif self.classify_redshift:
                if self.extra_args["redshift_classification_strategy"] == "binary":
                    gt = data["redshift_bins_mask"].flatten().to(torch.float32)
                    logits = ret["redshift_logits"].flatten()
                    spectra_loss = self.redshift_loss(logits, gt)
                    weight = torch.ones_like(spectra_loss)
                    weight[gt==1.] = self.num_redshift_bins
                    spectra_loss = (spectra_loss * weight).mean()
                elif self.extra_args["redshift_classification_strategy"] == "multi_class":
                    spectra_loss = self.redshift_loss(
                        ret["redshift_logits"], data["gt_redshift_bin_ids"][1])
                else: raise ValueError()
            else: raise ValueError()

        elif self.freeze_spectra:
            est = ret["redshift_logits"].flatten()
            gt_field = get_redshift_classification_data_field_name(
                "redshift_bins_mask")
            gt = data[gt_field].flatten().to(torch.float32)
            if self.extra_args["classifier_train_add_baseline_logits"]:
                baseline_logits = data[get_redshift_classification_data_field_name("baseline_logits")]
                if self.extra_args["classifier_train_use_bin_sampled_data"]:
                    selected_bins_mask = data[get_redshift_classification_data_field_name("selected_bins_mask")]
                    baseline_logits = baseline_logits[selected_bins_mask]

                # print(torch.max(est), torch.min(est))
                # print(torch.max(baseline_logits), torch.min(baseline_logits))
                est = (est + baseline_logits) / 2

            spectra_loss = self.bce_loss(est, gt)
            weight = torch.ones_like(spectra_loss)
            weight[gt==1.] = self.extra_args["classifier_train_num_bins_to_sample"] - 1
            spectra_loss = (spectra_loss * weight).mean()

        elif self.brute_force:
            if self.neg_sup_wrong_redshift:
                spectra_loss = self._calculate_neg_sup_loss(ret, data)
            else:
                spectra_loss, spectra_wrong_bin_regu = \
                    self._calculate_all_bin_losses(ret, data)
                if self.plot_l2_loss:
                    spectra_l2_loss, _ = self._calculate_all_bin_losses(
                        ret, data, loss_name_suffix="_l2")
        else: # `apply_gt_redshift`
            spectra_loss = self._calculate_gt_bin_loss(ret, data)
            if self.plot_l2_loss:
                spectra_l2_loss = self._calculate_gt_bin_loss(
                    ret, data, loss_name_suffix="_l2")

        self.log_dict["spectra_loss"] += spectra_loss.item()
        if self.plot_l2_loss:
            self.log_dict["spectra_l2_loss"] += spectra_l2_loss.item()
        return spectra_loss, spectra_l2_loss, spectra_wrong_bin_regu

    def _calculate_gt_bin_loss(self, ret, data, loss_name_suffix=""):
        loss_name = "spectrawise_loss" + loss_name_suffix
        spectrawise_loss = ret[loss_name] # [bsz]
        if self.plot_individ_spectra_loss:
            self.cur_spectra_individ_loss.extend(
                spectrawise_loss.detach().cpu().numpy())
        spectra_loss = self.spectra_reduce_func(spectrawise_loss)
        return spectra_loss

    def _calculate_all_bin_losses(self, ret, data, loss_name_suffix=""):
        spectra_regu = 0
        loss_name = "spectra_binwise_loss" + loss_name_suffix
        all_bin_loss = ret[loss_name] # [bsz,n_bins] no `nan` or any invalid values

        if self.optimize_gt_bin_only:
            mask = data["redshift_bins_mask"].to(all_bin_loss.device)
            spectra_loss = self.spectra_reduce_func(all_bin_loss * mask)
        elif self.dont_optimize_gt_bin:
            # spectra_loss = self.spectra_reduce_func(
            #     all_bin_loss[~data["redshift_bins_mask"]])
            inv_mask = ~data["redshift_bins_mask"].to(all_bin_loss.device)
            spectra_loss = self.spectra_reduce_func(all_bin_loss * inv_mask)
        else: # optimize all bins together
            if self.calculate_spectra_loss_based_on_optimal_bin:
                spectra_loss, _ = torch.min(all_bin_loss, dim=-1)
                spectra_loss = self.spectra_reduce_func(spectra_loss)
            elif self.calculate_spectra_loss_based_on_top_n_bins:
                bsz = len(all_bin_loss)
                ids = torch.argsort(all_bin_loss, dim=-1)
                ids = ids[:,:self.extra_args["num_bins_to_calculate_spectra_loss"]]
                ids = create_batch_ids(ids).view(2,-1)
                spectra_loss = (all_bin_loss[ids[0],ids[1]]).view(bsz,-1)
                spectra_loss = self.spectra_reduce_func(spectra_loss)
            else:
                if self.sanity_check_with_weights:
                    gt_bin_loss = all_bin_loss[data["redshift_bins_mask"]] # [bsz]
                    _, optimal_wrong_bin_loss = \
                        self._get_optimal_wrong_bin_data(ret, data) # [bsz]
                    spectra_regu = gt_bin_loss - optimal_wrong_bin_loss
                    spectra_regu[spectra_regu < 0] = 0
                    spectra_regu = self.spectra_reduce_func(spectra_regu)
                    spectra_regu *= self.extra_args["spectra_wrong_bin_regu_beta"]
                    self.log_dict["spectra_wrong_bin_regu"] += spectra_regu

                if self.regress_lambdawise_weights_share_latents:
                    self.optm_bin_ids.extend(torch.argmin(all_bin_loss, dim=-1))

                spectra_loss = self.spectra_reduce_func(all_bin_loss)

        if self.plot_gt_bin_loss:
            mask = data["redshift_bins_mask"]
            if self.sanity_check_sample_bins_per_step:
                mask = mask[data["selected_bins_mask"]].view(mask.shape[0], -1)
            all_spectra_gt_bin_loss = all_bin_loss[mask]
            gt_bin_loss = self.spectra_reduce_func(all_spectra_gt_bin_loss)
            loss_name = f"gt_bin{loss_name_suffix}_loss"
            self.log_dict[loss_name] += gt_bin_loss.item()

        return spectra_loss, spectra_regu

    def _calculate_neg_sup_loss(self, ret, data):
        """ Calculate spectra loss under negative supervision settings.
        """
        all_bin_loss = ret["spectra_binwise_loss"] # [bsz,n_bins]
        gt_bin_loss = torch.mean(all_bin_loss[data["redshift_bins_mask"]])
        self.log_dict["gt_bin_loss"] += gt_bin_loss.item()

        if self.cur_neg_sup_target == "latents":
            spectra_loss = torch.mean(ret["spectra_binwise_loss"])
            if self.extra_args["neg_sup_with_optimal_wrong_bin"]:
                _, optimal_wrong_bin_loss = self._get_optimal_wrong_bin_data(ret, data)
                self.log_dict["wrong_bin_loss"] += torch.mean(optimal_wrong_bin_loss).item()
            else:
                wrong_bin_loss = all_bin_loss[~data["redshift_bins_mask"]] # [n,]
                self.log_dict["wrong_bin_loss"] += torch.mean(wrong_bin_loss).item()

        elif self.cur_neg_sup_target == "net":
            if self.extra_args["neg_sup_with_optimal_wrong_bin"]:
                _, optimal_wrong_bin_loss = self._get_optimal_wrong_bin_data(ret, data)
                self.log_dict["wrong_bin_loss"] += \
                    torch.mean(optimal_wrong_bin_loss).item()
                wrong_bin_loss = self.extra_args["neg_sup_constant"] - \
                    optimal_wrong_bin_loss
                wrong_bin_loss[wrong_bin_loss < 0] = 0
            else:
                wrong_bin_loss = all_bin_loss[~data["redshift_bins_mask"]] # [n,nbins-1]
                self.log_dict["wrong_bin_loss"] += torch.mean(wrong_bin_loss).item()
                # wrong_bin_loss = self.extra_args["neg_sup_constant"] - wrong_bin_loss
                wrong_bin_loss = wrong_bin_loss - self.extra_args["neg_sup_constant"]
                wrong_bin_loss[wrong_bin_loss < 0] = 0
                wrong_bin_loss = torch.sum(wrong_bin_loss, dim=-1) / \
                    (self.num_redshift_bins-1)

            wrong_bin_loss = torch.mean(wrong_bin_loss)
            wrong_bin_regus = self.extra_args["neg_sup_beta"] * wrong_bin_loss
            self.log_dict["wrong_bin_regus"] += wrong_bin_regus.item()
            spectra_loss = gt_bin_loss + wrong_bin_regus
        else:
            raise ValueError()

        return spectra_loss

    def log_cli(self):
        """ Controls CLI logging.
            By default, this function only runs every epoch.
        """
        # Average over iterations
        m = len(self.train_data_loader) if self.train_based_on_epochs else 1
        n = self.num_epochs if self.train_based_on_epochs else self.num_steps
        name = "EPOCH" if self.train_based_on_epochs else "STEP"

        log_text = f"{name} {self.cur_iter}/{n}"
        log_text += " | total loss: {:>.3E}".format(self.log_dict["total_loss"] / m)
        log_text += " | spectra loss: {:>.3E}".format(self.log_dict["spectra_loss"] / m)

        if self.plot_l2_loss:
            log_text += " | total l2 loss: {:>.3E}".format(
                self.log_dict["total_l2_loss"] / m)
            log_text += " | spectra l2 loss: {:>.3E}".format(
                self.log_dict["spectra_l2_loss"] / m)

        if self.pixel_supervision:
            log_text += " | pixel loss: {:>.3E}".format(self.log_dict["pixel_loss"] / m)
        if self.neg_sup_wrong_redshift:
            log_text += " | gt bin loss: {:>.3E}".format(
                self.log_dict["gt_bin_loss"] / m)
            log_text += " | wrong bin regu: {:>.3E}".format(
                self.log_dict["wrong_bin_regus"] / m)
            log_text += " | wrong bin loss: {:>.3E}".format(
                self.log_dict["wrong_bin_loss"] / m)

        if self.regularize_redshift_logits:
            log_text += " | redshift logits regu: {:>.3E}".format(
                self.log_dict["redshift_logits_regu"] / m)
        if self.regularize_codebook_logits:
            log_text += " | codebook logits regu: {:>.3E}".format(
                self.log_dict["codebook_logits_regu"] / m)
        if self.regularize_spectra_latents:
            log_text += " | spectra latents regu: {:>.3E}".format(
                self.log_dict["spectra_latents_regu"] / m)
        if self.regularize_binwise_spectra_latents:
            log_text += " | binwise spectra latents regu: {:>.3E}".format(
                self.log_dict["binwise_spectra_latents_regu"] / m)
        if self.regularize_codebook_spectra:
            log_text += " | codebook spectra regu: {:>.3E}".format(
                self.log_dict["codebook_spectra_regu"] / m)
        if self.regress_lambdawise_weights:
            log_text += " | spectra wrong bin regu: {:>.3E}".format(
                self.log_dict["spectra_wrong_bin_regu"] / m)

        log.info(log_text)

    ###################
    # Optimizer Helpers
    ###################

    def configure_alternate_optimization(self):
        self._get_current_em_latents_target()
        self._get_current_neg_sup_target()
        self._toggle_grad(on_off="off")

    def _toggle_grad(self, on_off="on"):
        if self.optimize_latents_alternately:
            raise NotImplementedError()
        elif self.neg_sup_optimize_alternately:
            if self.cur_neg_sup_target == "latents":
                self._toggle_net_grad(on_off)
            elif self.cur_neg_sup_target == "net":
                self._toggle_spectra_latents_grad(on_off)
            else: raise ValueError()

    def _toggle_net_grad(self, on_off):
        for n,p in self.pipeline.named_parameters():
            if n != "model.latents": p.requires_grad = on_off == "on"

    def _toggle_spectra_latents_grad(self, on_off):
        for n,p in self.pipeline.named_parameters():
            if n == "model.latents": p.requires_grad = on_off == "on"

    def _get_optm_target(self):
        """ Get the current optimization target when doing alternate optimization.
        """
        if self.neg_sup_optimize_alternately:
            target = self.cur_neg_sup_target
        elif self.optimize_latents_alternately:
            target = self.cur_optm_target
        else: target = None
        return target

    def _get_current_em_latents_target(self):
        if not self.optimize_latents_alternately: return
        residu = self.cur_iter % (sum(self.em_alternation_steps))
        if self.em_alternation_starts_with == "spectra_latents":
           self.cur_optm_target = "spectra_latents" \
               if residu < self.em_alternation_steps[0] else "redshift_latents"
        elif self.em_alternation_starts_with == "redshift_latents":
            self.cur_optm_target = "redshift_latents" \
                if residu < self.em_alternation_steps[0] else "spectra_latents"
        else: raise ValueError()

    def _get_current_neg_sup_target(self):
        if not self.neg_sup_optimize_alternately: return
        residu = self.cur_iter % (sum(self.neg_sup_alternation_steps))
        if self.neg_sup_alternation_starts_with == "net":
            self.cur_neg_sup_target = "net" \
                if residu < self.neg_sup_alternation_steps[0] else "latents"
        elif self.neg_sup_alternation_starts_with == "latents":
            self.cur_neg_sup_target = "latents" \
                if residu < self.neg_sup_alternation_steps[0] else "net"
        else: raise ValueError()

    def _assign_pretrain_optimization_params(self):
        """ Configure optimization parameters for codebook pretrain.
        """
        net_params, spectra_latents = [], None
        for name in self.params_dict:
            if name == "model.latents":
                spectra_latents = self.params_dict[name]
            else: net_params.append(self.params_dict[name])

        latents_group, net_params_group = [], []
        self._add_spectra_latents(spectra_latents, latents_group)
        net_params_group.append({"params": net_params,
                                 "lr": self.extra_args["spectra_pretrain_lr"]})
        return latents_group, net_params_group

    def _assign_model_optimization_params(self, lr):
        model_params = []
        for name in self.params_dict:
            model_params.append(self.params_dict[name])
        # return model_params
        model_params_group = []
        model_params_group.append({"params": model_params, "lr": lr})
        return model_params_group

    def _assign_brute_force_optimization_params(self):
        spectra_latents, redshift_latents = None, None
        base_spectra_latents, addup_spectra_latents = None, None
        gt_bin_spectra_latents, wrong_bin_spectra_latents  = None, None
        codebook_logit_params, redshift_logit_params = [], []

        for name in self.params_dict:
            if name == "model.latents":
                spectra_latents = self.params_dict[name]
            elif name == "model.base_latents":
                base_spectra_latents = self.params_dict[name]
            elif name == "model.addup_latents":
                addup_spectra_latents = self.params_dict[name]
            elif name == "model.gt_bin_latents":
                gt_bin_spectra_latents = self.params_dict[name]
            elif name == "model.wrong_bin_latents":
                wrong_bin_spectra_latents = self.params_dict[name]
            elif name == "model.redshift_latents":
                redshift_latents = self.params_dict[name]
            elif "redshift_decoder" in name:
                redshift_logit_params.append(self.params_dict[name])
            elif "spatial_decoder.decode" in name:
                codebook_logit_params.append(self.params_dict[name])

        if self.optimize_latents_alternately:
            spectra_latents_group, redshift_latents_group = [], []
        else: latents_group = []
        net_params_group = []

        # redshift latents & parameters
        if self.has_redshift_latents:
            if self.optimize_latents_alternately:
                self._add_redshift_latents(
                    redshift_latents, redshift_logit_params,
                    redshift_latents_group, net_params_group)
            else:
                self._add_redshift_latents(
                    redshift_latents, redshift_logit_params,
                    latents_group, net_params_group)

        # spectra latents
        if self.optimize_spectra_latents:
            # optimize only part of the spectra latents
            # if self.optimize_gt_bin_only:
            #     spectra_latents = spectra_latents[redshift_bins_mask] # gt bin latents
            # elif self.dont_optimize_gt_bin:
            #     spectra_latents = spectra_latents[~redshift_bins_mask] # wrong bin latents
            if self.optimize_latents_alternately: # em
                self._add_spectra_latents(spectra_latents, spectra_latents_group)
            elif self.regularize_binwise_spectra_latents:
                self._add_spectra_latents(base_spectra_latents, latents_group)
                self._add_spectra_latents(addup_spectra_latents, latents_group)
            elif self.optimize_bins_separately:
                self._add_spectra_latents(gt_bin_spectra_latents, latents_group)
                self._add_spectra_latents(wrong_bin_spectra_latents, latents_group)
            else:
                # if self.sanity_check_sample_bins:
                #     selected_bins_mask = self.dataset.get_selected_bins_mask()
                #     bsz,nbins,dim = spectra_latents.shape
                #     mask = torch.tensor(selected_bins_mask[...,None]).tile(1,1,dim)
                #     spectra_latents = spectra_latents[mask].view(bsz,-1,dim)
                self._add_spectra_latents(spectra_latents, latents_group)

        # codebook coefficients parameters
        if self.optimize_spectra_latents_as_logits:
            pass
        else:
            if self.extra_args["optimize_codebook_logits_mlp"]:
                net_params_group.append({"params": codebook_logit_params,
                                         "lr": self.extra_args["spectra_pretrain_lr"]})

        if self.optimize_latents_alternately:
            return spectra_latents_group, redshift_latents_group, net_params_group
        return latents_group, net_params_group

    def _add_spectra_latents(self, spectra_latents, latents_group):
        if self.use_lbfgs:
            assert self.optimize_spectra_latents_as_logits
            latents_group.append(spectra_latents)
        else:
            latents_group.append({"params": spectra_latents,
                                  "lr": self.extra_args["spectra_latents_lr"]})

    def _add_redshift_latents(self, redshift_latents, redshift_logit_params,
                              latents_group, net_params_group
    ):
        if self.optimize_redshift_latents_as_logits:
            assert self.optimize_redshift_latents
            if self.use_lbfgs:
                latents_group.append(redshift_latents)
            else:
                latents_group.append({"params": redshift_latents,
                                      "lr": self.extra_args["redshift_latents_lr"]})
        else: # autodecoder arch
            if self.optimize_redshift_latents:
                latents_group.append({"params": redshift_latents,
                                      "lr": self.extra_args["redshift_latents_lr"]})
            if self.optimize_redshift_logits_mlp:
                net_params_group.append({"params": redshift_logit_params,
                                         "lr": self.extra_args["spectra_pretrain_lr"]})

    def validate(self):
        pass
