import os
import time
import torch
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
from torch.utils.data import BatchSampler, \
    SequentialSampler, RandomSampler, DataLoader

from wisp.utils import PerfTimer
from wisp.trainers import BaseTrainer
from wisp.optimizers import multi_optimizer
from wisp.utils.plot import plot_grad_flow, plot_multiple, \
    plot_precision_recall_individually, plot_precision_recall_together
from wisp.loss import spectra_supervision_loss, \
    spectra_supervision_emd_loss, pretrain_pixel_loss
from wisp.utils.common import get_gpu_info, add_to_device, sort_alphanumeric, \
    select_inferrence_ids, load_embed, load_model_weights, forward, \
    load_pretrained_model_weights, get_pretrained_model_fname, init_redshift_bins, \
    get_bool_classify_redshift, get_bool_has_redshift_latents, get_optimal_wrong_bin_ids, \
    freeze_layers_incl, freeze_layers_excl, get_loss, create_latent_mask, set_seed, log_data


class CodebookTrainer(BaseTrainer):
    """ Trainer class for codebook pretraining.
    """
    def __init__(self, pipeline, dataset, optim_cls, optim_params, device, **extra_args):
        super().__init__(pipeline, dataset, optim_cls, optim_params, device, **extra_args)

        assert(extra_args["space_dim"] == 3 and extra_args["pretrain_codebook"])

        self.summarize_training_tasks()
        self.set_path()

        self.init_net()
        self.collect_model_params()

        self.init_data()
        self.init_loss()
        self.init_optimizer()

    def train(self):
        torch.autograd.set_detect_anomaly(True)
        self.begin_train()

        if self.use_tqdm: iter = tqdm(range(self.num_epochs + 1))
        else: iter = range(self.num_epochs + 1)

        for self.epoch in iter:
            self.begin_epoch()

            for batch in range(self.num_iterations_cur_epoch):
                data = self.pre_step()
                ret = self.step(data)
                self.post_step(data, ret)

            self.end_epoch()
        self.end_train()

    #############
    # Initializations
    #############

    def summarize_training_tasks(self):
        tasks = set(self.extra_args["tasks"])

        self.save_redshift = "save_redshift" in tasks
        self.save_qtz_weights = "save_qtz_weights" in tasks
        self.recon_gt_spectra = "recon_gt_spectra" in tasks
        self.save_pixel_values = "save_pixel_values" in tasks
        self.plot_codebook_logits = "plot_codebook_logits" in tasks
        self.recon_codebook_spectra_individ = "recon_codebook_spectra_individ" in tasks

        if "codebook_pretrain" in tasks:
            self.mode = "codebook_pretrain"
        elif "redshift_pretrain" in tasks:
            self.mode = "redshift_pretrain"
        else: raise ValueError()

        if self.mode == "redshift_pretrain":
            if self.extra_args["sample_from_codebook_pretrain_spectra"]:
                self.num_spectra = self.extra_args["redshift_pretrain_num_spectra"]
            else: self.num_spectra = self.dataset.get_num_validation_spectra()
        elif self.mode == "codebook_pretrain":
            self.num_spectra = self.dataset.get_num_supervision_spectra()
        else: raise ValueError("Invalid mode!")

        # quantization setups
        self.qtz_latent = self.space_dim == 3 and self.extra_args["quantize_latent"]
        self.qtz_spectra = self.space_dim == 3 and self.extra_args["quantize_spectra"]
        assert not (self.qtz_latent and self.qtz_spectra)
        self.qtz = self.qtz_latent or self.qtz_spectra
        self.qtz_n_embd = self.extra_args["qtz_num_embed"]
        self.qtz_strategy = self.extra_args["quantization_strategy"]

        # redshift setup
        self.apply_gt_redshift = self.extra_args["model_redshift"] and \
            self.extra_args["apply_gt_redshift"]
        self.classify_redshift = get_bool_classify_redshift(**self.extra_args)
        self.neg_sup_wrong_redshift = self.extra_args["negative_supervise_wrong_redshift"]
        self.neg_sup_optimize_alternately = self.extra_args["neg_sup_optimize_alternately"]
        self.calculate_binwise_spectra_loss = self.extra_args["calculate_binwise_spectra_loss"]
        self.use_binwise_spectra_loss_as_redshift_logits =  \
            self.extra_args["use_binwise_spectra_loss_as_redshift_logits"]
        self.optimize_spectra_for_each_redshift_bin = \
            self.extra_args["optimize_spectra_for_each_redshift_bin"]
        self.has_redshift_latents = get_bool_has_redshift_latents(**self.extra_args)

        assert not self.mode == "codebook_pretrain" or (
            self.apply_gt_redshift or self.neg_sup_wrong_redshift)
        # assert not self.apply_gt_redshift or \
        #      not self.calculate_binwise_spectra_loss
        assert not self.neg_sup_wrong_redshift or (
            self.mode == "codebook_pretrain" and self.classify_redshift and \
            self.calculate_binwise_spectra_loss)
        assert not self.neg_sup_optimize_alternately or \
            self.neg_sup_wrong_redshift
        assert not self.calculate_binwise_spectra_loss or \
            self.extra_args["spectra_batch_reduction_order"] == "qtz_first"
        assert not self.use_binwise_spectra_loss_as_redshift_logits or \
            (self.classify_redshift and self.calculate_binwise_spectra_loss)
        assert not self.optimize_spectra_for_each_redshift_bin or \
            self.calculate_binwise_spectra_loss
        assert not self.optimize_spectra_for_each_redshift_bin or \
            (self.classify_redshift and not self.use_binwise_spectra_loss_as_redshift_logits), \
            "For the brute force method, we keep spectra under all bins without averaging. \
            During inferrence however, we can calculate logits for visualization purposes."

        if self.classify_redshift:
            redshift_bins = init_redshift_bins(
                self.extra_args["redshift_lo"],
                self.extra_args["redshift_hi"],
                self.extra_args["redshift_bin_width"]
            )
            self.num_redshift_bins = len(redshift_bins)

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

        # all others
        self.plot_loss = self.extra_args["plot_loss"]
        self.index_latent = True # index latents as coords in model
        self.split_latent = self.mode == "redshift_pretrain" and \
            self.extra_args["split_latent"]

        self.regularize_redshift_logits = self.classify_redshift and \
            self.extra_args["regularize_redshift_logits"] and \
            self.extra_args["redshift_classification_method"] == "weighted_avg"
        self.redshift_logits_regu_method = self.extra_args["redshift_logits_regu_method"]

        self.regularize_codebook_logits = self.qtz_spectra and \
            self.extra_args["regularize_codebook_logits"]
        self.regularize_spectra_latents = self.extra_args["regularize_spectra_latents"]

        self.regularize_within_codebook_spectra = self.qtz_spectra and \
            self.extra_args["regularize_within_codebook_spectra"]
        self.regularize_across_codebook_spectra = self.qtz_spectra and \
            self.extra_args["regularize_across_codebook_spectra"]
        self.regularize_codebook_spectra = self.mode == "codebook_pretrain" and \
            (self.regularize_within_codebook_spectra or self.regularize_across_codebook_spectra)
        assert self.regularize_within_codebook_spectra + \
            self.regularize_across_codebook_spectra <= 1

        self.optimize_spectra_latents = self.extra_args["optimize_spectra_latents"]
        self.optimize_redshift_latents = self.mode == "redshift_pretrain" and \
            self.extra_args["optimize_redshift_latents"]
        self.optimize_spectra_latents_as_logits = \
            self.extra_args["optimize_spectra_latents_as_logits"]
        self.optimize_redshift_latents_as_logits = self.mode == "redshift_pretrain" and \
            self.extra_args["optimize_redshift_latents_as_logits"]
        # no pretrained latents to load
        assert not self.optimize_redshift_latents_as_logits or \
            self.mode == "redshift_pretrain" and self.optimize_redshift_latents

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
            if self.neg_sup_wrong_redshift:
                self.gt_bin_loss_fname = join(self.loss_dir, "gt_bin_loss")
                self.wrong_bin_regu_fname = join(self.loss_dir, "wrong_bin_regu")
                self.wrong_bin_loss_fname = join(self.loss_dir, "wrong_bin_loss")

        if self.mode == "redshift_pretrain":
            # redshift pretrain use pretrained model from codebook pretrain
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

        if self.extra_args["plot_logits_for_gt_bin"]:
            self.gt_bin_logits_fname = join(self.log_dir, "gt_bin_logits")

    def init_net(self):
        self.train_pipeline = self.pipeline[0]
        self.train_pipeline.set_batch_reduction_order(
            self.extra_args["spectra_batch_reduction_order"])

        latents, redshift_latents = self.init_latents()
        self.train_pipeline.set_latents(latents)
        if redshift_latents is not None:
            self.train_pipeline.set_redshift_latents(redshift_latents)
        self.freeze_and_load()

        log.info(self.train_pipeline)
        log.info("Total number of parameters: {}".format(
            sum(p.numel() for p in self.train_pipeline.parameters())))

    def collect_model_params(self):
        # collect all parameters from network and trainable latents
        self.params_dict = {}
        for name, param in self.train_pipeline.named_parameters():
            self.params_dict[name] = param

    def init_data(self):
        self.save_data = False
        self.shuffle_dataloader = True
        self.dataloader_drop_last = False

        self.configure_dataset()
        self.set_num_batches()
        self.init_dataloader()

    def init_loss(self):
        if self.extra_args["spectra_loss_cho"] == "emd":
            self.spectra_loss = spectra_supervision_emd_loss
        else:
            loss = get_loss(self.extra_args["spectra_loss_cho"], self.cuda)
            self.spectra_loss = partial(spectra_supervision_loss, loss)

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
        if self.mode == "codebook_pretrain":
            latents, net_params = self._assign_codebook_pretrain_optimization_params()
        elif self.mode == "redshift_pretrain":
            if self.optimize_latents_alternately:
                spectra_latents, redshift_latents, net_params = \
                    self._assign_redshift_pretrain_optimization_params()
            else: latents, net_params = self._assign_redshift_pretrain_optimization_params()
        else: raise ValueError()

        if self.neg_sup_optimize_alternately:
            assert self.mode == "codebook_pretrain" and not self.use_lbfgs
            latents_optm = self.optim_cls(latents, **self.optim_params)
            net_optm = self.optim_cls(net_params, **self.optim_params)
            optms = { "latents": latents_optm, "net": net_optm }

        elif self.optimize_latents_alternately:
            assert self.mode == "redshift_pretrain"
            if self.use_lbfgs:
                raise NotImplementedError()
            else:
                redshift_latents_optm = self.optim_cls(redshift_latents, **self.optim_params)
                spectra_latents_optm = self.optim_cls(spectra_latents, **self.optim_params)
                optms = {
                    "redshift_latents": redshift_latents_optm,
                    "spectra_latents": spectra_latents_optm }
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

        self.optimizer = multi_optimizer(**optms)
        if self.verbose: log.info(self.optimizer)

    def configure_dataset(self):
        """ Configure dataset with selected fields and set length accordingly.
        """
        self.dataset.set_mode(self.mode)

        # set required fields from dataset
        fields = ["wave_data","spectra_source_data","spectra_masks","spectra_redshift"]
        # todo, codebook pretrain "coords" not handled
        if self.pixel_supervision:
            fields.append("spectra_pixels")
        if self.neg_sup_wrong_redshift:
            fields.extend(["gt_redshift_bin_ids","gt_redshift_bin_masks"])

        # use original spectra wave
        self.dataset.set_wave_source("spectra")

        # set spectra data source
        # tmp: in sanity check mode, we manually change sup_id and val_id in spectra_data
        if self.mode == "redshift_pretrain":
            self.dataset.set_spectra_source("val")
        else: self.dataset.set_spectra_source("sup")

        # set input latents for codebook net
        fields.append("idx")
        self.dataset.toggle_integration(self.pixel_supervision)
        self.dataset.toggle_wave_sampling(self.sample_wave)
        if self.sample_wave:
            self.dataset.set_num_wave_samples(self.extra_args["pretrain_num_wave_samples"])
            self.dataset.set_wave_sample_method(self.extra_args["pretrain_wave_sample_method"])
        # if self.train_within_wave_range:
        #     self.dataset.set_wave_range(
        #         self.extra_args["spectra_supervision_wave_lo"],
        #         self.extra_args["spectra_supervision_wave_hi"])

        self.dataset.set_length(self.num_spectra)
        if self.extra_args["infer_selected"]:
            self.selected_ids = select_inferrence_ids(
                self.num_spectra,
                self.extra_args["pretrain_num_infer_upper_bound"])
        else:
            self.selected_ids = np.arange(self.num_spectra)

        if self.classify_redshift:
            self.dataset.set_num_redshift_bins(self.num_redshift_bins)

        self.dataset.set_fields(fields)

    #############
    # Training logic
    #############

    def begin_train(self):
        self.total_steps = 0
        self.codebook_pretrain_total_steps = 0

        if self.plot_loss:
            self.losses = []
            if self.neg_sup_wrong_redshift:
                self.gt_bin_losses = []
                self.wrong_bin_regus = []
                self.wrong_bin_losses = []
        if self.extra_args["plot_individ_spectra_loss"]:
            self.spectra_individ_losses = []
        if self.extra_args["plot_logits_for_gt_bin"]:
            self.gt_bin_logits = []
        if self.extra_args["resume_train"]:
            self.resume_train()

        log.info(f"{self.num_iterations_cur_epoch} batches per epoch.")

    def end_train(self):
        self.writer.close()

        if self.extra_args["plot_logits_for_gt_bin"]:
            self._plot_logits_for_gt_bin()

        if self.plot_grad_every != -1:
            plt.savefig(self.grad_fname)
            plt.close()

        if self.plot_loss:
            self._plot_loss(self.losses, self.loss_fname)
            if self.neg_sup_wrong_redshift:
                self._plot_loss(self.gt_bin_losses, self.gt_bin_loss_fname)
                self._plot_loss(self.wrong_bin_regus, self.wrong_bin_regu_fname)
                self._plot_loss(self.wrong_bin_losses, self.wrong_bin_loss_fname)
            if self.extra_args["plot_individ_spectra_loss"]:
                self._plot_individ_spectra_loss()

        if self.extra_args["log_gpu_every"] != -1:
            nvidia_smi.nvmlShutdown()

    #############
    # Epoch begin and end
    #############

    def begin_epoch(self):
        self.iteration = 0
        self.reset_data_iterator()
        self.init_log_dict()
        self.pre_epoch()

        if self.extra_args["plot_logits_for_gt_bin"]:
            self.cur_gt_bin_logits = []
        if self.extra_args["plot_individ_spectra_loss"]:
            self.cur_spectra_individ_losses = []

        self._get_current_em_latents_target()
        self._get_current_neg_sup_target()
        self._toggle_grad(on_off="off")

        # for n,p in self.train_pipeline.named_parameters(): print(n, p.requires_grad)

        self.timer.check("begun epoch")

    def end_epoch(self):
        self.post_epoch()

        if self.epoch < self.num_epochs:
            self.iteration = 0
            self.epoch += 1

        if self.extra_args["plot_logits_for_gt_bin"]:
            self.gt_bin_logits.append(self.cur_gt_bin_logits)
        if self.extra_args["plot_individ_spectra_loss"]:
            self.spectra_individ_losses.append(self.cur_spectra_individ_losses)

        self._toggle_grad(on_off="on")
        freeze_layers_incl( # always freeze pe
            self.train_pipeline, incls=["wave_encoder"])

        self.timer.check("epoch ended")

    def reset_data_iterator(self):
        """ Rewind the iterator for the new epoch.
        """
        self.train_data_loader_iter = iter(self.train_data_loader)

    def init_log_dict(self):
        """ Custom log dict.
        """
        super().init_log_dict()
        self.log_dict["pixel_loss"] = 0.0
        self.log_dict["spectra_loss"] = 0.0
        self.log_dict["gt_bin_losses"] = 0.0
        self.log_dict["wrong_bin_regus"] = 0.0
        self.log_dict["wrong_bin_losses"] = 0.0
        self.log_dict["redshift_logits_regu"] = 0.0
        self.log_dict["spectra_latents_regu"] = 0.0
        self.log_dict["codebook_spectra_regu"] = 0.0

    #############
    # One epoch
    #############

    def pre_epoch(self):
        self.set_num_batches()

        if self.save_model_every > -1 and self.epoch % self.save_model_every == 0:
            self.save_model()

        if self.save_data_every > -1 and self.epoch % self.save_data_every == 0:
            self.save_data = True

            if self.save_redshift:
                self.gt_redshift = []
                self.est_redshift = []

                if self.classify_redshift:
                    self.redshift_logits = []
                    if self.calculate_binwise_spectra_loss:
                        self.binwise_loss = []

            if self.save_qtz_weights:
                self.qtz_weights = []
            if self.save_pixel_values:
                self.gt_pixel_vals = []
                self.recon_pixel_vals = []
            if self.recon_gt_spectra:
                self.gt_fluxes = []
                self.recon_fluxes = []
                self.spectra_wave = []
                self.spectra_masks = []
            if self.plot_codebook_logits:
                self.codebook_logits = []
            if self.recon_codebook_spectra_individ:
                self.spectra_wave_c = []
                self.spectra_masks_c = []
                self.codebook_spectra = []

            # re-init dataloader to make sure pixels are in order
            self.use_all_pixels = True
            self.shuffle_dataloader = False
            self.sample_wave = not self.extra_args["pretrain_use_all_wave"]
            self.dataset.toggle_wave_sampling(self.sample_wave)
            self.set_num_batches()
            self.init_dataloader()
            self.reset_data_iterator()
            warnings.warn("dataloader state is modified in codebook_trainer, ensure this is for validation purpose only!")

        self.train_pipeline.train()

    def post_epoch(self):
        """ By default, this function logs to Tensorboard, renders images to Tensorboard,
            saves the model, and resamples the dataset.
        """
        self.train_pipeline.eval()

        total_loss = self.log_dict["total_loss"] / len(self.train_data_loader)

        if self.plot_loss:
            self.losses.append(total_loss)
            if self.neg_sup_optimize_alternately:
                self.gt_bin_losses.append(
                    self.log_dict["gt_bin_losses"] / len(self.train_data_loader))
                self.wrong_bin_losses.append(
                    self.log_dict["wrong_bin_losses"] / len(self.train_data_loader))
                if self.cur_neg_sup_target == "net":
                    self.wrong_bin_regus.append(
                        self.log_dict["wrong_bin_regus"] / len(self.train_data_loader))

        if self.log_tb_every > -1 and self.epoch % self.log_tb_every == 0:
            self.log_tb()

        if self.log_cli_every > -1 and self.epoch % self.log_cli_every == 0:
            self.log_cli()

        if self.render_tb_every > -1 and self.epoch % self.render_tb_every == 0:
            self.render_tb()

        # save data locally and restore trainer state
        if self.save_data:
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
            warnings.warn("dataloader state is modified in codebook_trainer, ensure this is for validation purpose only!")

    #############
    # One step
    #############

    def pre_step(self):
        # since we are optimizing latents which are inputs for the pipeline
        # we should also update the input each time
        # however since dataset receives a reference to the latents,
        # we don't need to update dataset with updated latents manually
        # self.dataset.set_hardcode_data("spectra_latents", self.latents.weight)
        return self.next_batch()

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
            if self.recon_gt_spectra:
                self.recon_fluxes.extend(ret["spectra"])
                self.gt_fluxes.extend(data["spectra_source_data"][:,1])
                self.spectra_wave.extend(data["spectra_source_data"][:,0])
                self.spectra_masks.extend(data["spectra_masks"])
            if self.recon_codebook_spectra_individ:
                self.codebook_spectra.extend(ret["codebook_spectra"])
                self.spectra_wave_c.extend(data["spectra_source_data"][:,0])
                self.spectra_masks_c.extend(data["spectra_masks"])

        self.iteration += 1
        self.total_steps += 1
        self.timer.check("batch ended")

    #############
    # Model Helpers
    #############

    def init_latents(self):
        if self.mode == "codebook_pretrain":
            assert not self.split_latent
            spectra_latents = self.init_codebook_pretrain_spectra_latents()
            redshift_latents = None
        elif self.mode == "redshift_pretrain":
            spectra_latents = self.init_redshift_pretrain_spectra_latents()
            if not self.has_redshift_latents:
                redshift_latents = None
            else: redshift_latents = self.init_redshift_pretrain_redshift_latents()
        else:
            raise ValueError("Invalid pretrainer mode.")
        return spectra_latents, redshift_latents

    def freeze_and_load(self):
        """ For redshift pretrain (sanity check), we load part of the pretrained
            model (from codebook pretrain) and freeze.
        """
        if self.mode == "codebook_pretrain":
            pass
        elif self.mode == "redshift_pretrain":
            load_excls, freeze_excls = [], []

            # codebook coefficients (no softmax applied)
            if self.optimize_spectra_latents:
                freeze_excls.append("nef.latents")

            # when we want to load codebook latents, we do it during initialization
            # if not self.extra_args["load_pretrained_spectra_latents"]:
            load_excls.append("nef.latents")

            if self.optimize_spectra_latents_as_logits:
                # directly optimize latents as coefficients
                pass
            else:
                # optimize an autodecoder
                if self.extra_args["optimize_codebook_logits_mlp"]:
                    freeze_excls.append("spatial_decoder.decode")
                if not self.extra_args["load_pretrained_codebook_logits_mlp"]:
                    load_excls.append("spatial_decoder.decode")

            # redshift bin logits
            if self.classify_redshift:
                if self.optimize_redshift_latents_as_logits:
                    # directly optimize logits
                    freeze_excls.append("nef.redshift_latents")
                else:
                    # optimize an autodecoder
                    freeze_excls.append("redshift_decoder")
                    freeze_excls.append("nef.redshift_latents")

            freeze_layers_excl(self.train_pipeline, excls=freeze_excls)
            self.load_model(self.pretrained_model_fname, excls=load_excls)
        else:
            raise ValueError()

    def init_codebook_pretrain_spectra_latents(self):
        if self.optimize_spectra_latents_as_logits:
            dim = self.extra_args["qtz_num_embed"]
        else: dim = self.extra_args["spectra_latent_dim"]
        if self.optimize_spectra_for_each_redshift_bin:
            sp = (self.num_spectra, self.num_redshift_bins, dim)
        else: sp = (self.num_spectra, dim)
        latents = self.create_latents(
            sp, seed=self.extra_args["seed"] + 1,
            zero_init=self.extra_args["zero_init_spectra_latents"],
            freeze=not self.optimize_spectra_latents
        )
        return latents

    def init_redshift_pretrain_redshift_latents(self):
        if not self.apply_gt_redshift and self.split_latent:
            sp = (self.num_spectra, self.extra_args["redshift_logit_latent_dim"])
            latents = self.create_latents(
                sp, seed=self.extra_args["seed"] + 2,
                zero_init=self.extra_args["zero_init_redshift_latents"],
                freeze=not self.optimize_redshift_latents)
        else: latents = None
        return latents

    def init_redshift_pretrain_spectra_latents(self):
        latents = None
        if self.optimize_spectra_latents_as_logits:
            sp_z_dim = self.extra_args["qtz_num_embed"]
        else: sp_z_dim = self.extra_args["spectra_latent_dim"]

        if self.optimize_spectra_for_each_redshift_bin:
            sp = (self.num_spectra, self.num_redshift_bins, sp_z_dim)
        else: sp = (self.num_spectra, sp_z_dim)

        if self.extra_args["load_pretrained_spectra_latents"]:
            assert self.extra_args["sample_from_codebook_pretrain_spectra"]
            # checkpoint comes from codebook pretrain (use sup spectra)
            checkpoint = torch.load(self.pretrained_model_fname)
            # assert checkpoint["model_state_dict"]["nef.latents"].shape[1] == sp_z_dim
            assert checkpoint["model_state_dict"]["nef.latents"].shape[-1] == sp_z_dim

            # redshift pretrain use val spectra which is a permutation of sup spectra
            permute_ids = self.dataset.get_redshift_pretrain_spectra_ids()
            pretrained = checkpoint["model_state_dict"]["nef.latents"][permute_ids]
            if self.optimize_spectra_for_each_redshift_bin:
                if pretrained.ndim != 3:
                    pretrained = pretrained[:,None].tile(1, self.num_redshift_bins, 1)
                assert pretrained.shape == sp
        else:
            pretrained = None

        latents = self.create_latents(
            sp, seed=self.extra_args["seed"], pretrained=pretrained,
            zero_init=self.extra_args["zero_init_spectra_latents"],
            freeze=not self.extra_args["optimize_spectra_latents"])

        return latents

    def create_latents(self, sp, pretrained=None, zero_init=False, freeze=False, seed=0):
        if pretrained is not None:
            latents = pretrained.to(self.device)
        elif zero_init:
            latents = torch.zeros(sp).to(self.device)
            # latents = 0.01 * torch.ones(n,m)
        else:
            torch.manual_seed(seed)
            latents = torch.rand(sp).to(self.device)

        latents = nn.Parameter(latents, requires_grad=not freeze)
        return latents

    def load_model(self, model_fname, excls=[]):
        assert(exists(model_fname))
        log.info(f"saved model found, loading {model_fname}")
        checkpoint = torch.load(model_fname)
        load_pretrained_model_weights(
            self.train_pipeline, checkpoint["model_state_dict"], excls=excls)
        if self.mode == "redshift_pretrain":
            self.codebook_pretrain_total_steps = checkpoint["iterations"]
        else: self.codebook_pretrain_total_steps = 0
        self.train_pipeline.train()
        return checkpoint

    def save_model(self):
        fname = f"model-ep{self.epoch}-it{self.iteration}.pth"
        model_fname = os.path.join(self.model_dir, fname)
        if self.verbose: log.info(f"Saving model checkpoint to: {model_fname}")

        checkpoint = {
            "iterations": self.total_steps,
            "epoch_trained": self.epoch,
            "model_state_dict": self.train_pipeline.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }
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

            self.total_steps = checkpoint["iterations"]
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if self.plot_loss:
                if exists(self.resume_loss_fname):
                    self.losses = list(np.load(self.resume_loss_fname))
                if self.neg_sup_wrong_redshift:
                    if exists(self.resume_gt_bin_loss_fname):
                        self.gt_bin_losses = list(np.load(self.resume_gt_bin_loss_fname))
                    if exists(self.resume_wrong_bin_regu_fname):
                        self.wrong_bin_regus = list(np.load(self.resume_wrong_bin_regu_fname))
                    if exists(self.resume_wrong_bin_loss_fname):
                        self.wrong_bin_losses = list(np.load(self.resume_wrong_bin_loss_fname))
                if self.extra_args["plot_individ_spectra_loss"]:
                    fname = self.resume_loss_fname[:-4] + "_individ.npy"
                    if exists(fname):
                        self.spectra_individ_losses = list(np.load(fname).T)

            log.info("resumed training")

        except Exception as e:
            log.info(e)
            log.info("start training from begining")

    #############
    # Data Helpers
    #############

    def set_num_batches(self):
        """ Set number of batches/iterations and batch size for each epoch.
            At certain epochs, we may not need all data and can break before
              iterating thru all data.
        """
        length = len(self.dataset)
        self.batch_size = min(self.extra_args["pretrain_batch_size"], length)

        if self.dataloader_drop_last:
            self.num_iterations_cur_epoch = int(length // self.batch_size)
        else:
            self.num_iterations_cur_epoch = int(np.ceil(length / self.batch_size))

    def _plot_grad_now(self):
        return self.plot_grad_every != -1 and \
            (self.epoch == 0 or self.epoch % self.plot_grad_every == 0)

    def save_local(self):
        if self.recon_gt_spectra:
            self._recon_gt_spectra()
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
                if self.calculate_binwise_spectra_loss:
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
        # fname = join(self.pixel_val_dir, f"model-ep{self.epoch}-it{self.iteration}.pth")
        # np.save(fname, vals)

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
        fname = join(self.qtz_weight_dir, f"model-ep{self.epoch}-it{self.iteration}.pth")
        np.save(fname, weights)
        np.set_printoptions(suppress=True)
        np.set_printoptions(precision=3)
        w = weights[self.selected_ids,0]
        log.info(f"Qtz weights {w}")

    def _recon_gt_spectra(self):
        log.info("reconstructing gt spectrum")

        self.gt_fluxes = torch.stack(self.gt_fluxes).view(
            self.num_spectra, -1).detach().cpu().numpy()[self.selected_ids]
        # [n_spectra,nsmpl]

        self.recon_fluxes = torch.stack(self.recon_fluxes).view(
            self.num_spectra, self.extra_args["spectra_neighbour_size"]**2, -1
        ).detach().cpu().numpy()[self.selected_ids]
        self.spectra_wave = torch.stack(self.spectra_wave).view(
            self.num_spectra, -1).detach().cpu().numpy()[self.selected_ids]
        self.spectra_masks = torch.stack(self.spectra_masks).bool().view(
            self.num_spectra, -1).detach().cpu().numpy()[self.selected_ids]

        # plot spectrum in multiple figures, each figure contains several spectrum
        n_spectrum = len(self.selected_ids)
        n_spectrum_per_fig = self.extra_args["num_spectrum_per_fig"]
        n_figs = int(np.ceil( n_spectrum / n_spectrum_per_fig ))

        for i in range(n_figs):
            fname = f"ep{self.epoch}-it{self.iteration}-plot{i}"
            lo = i * n_spectrum_per_fig
            hi = min(lo + n_spectrum_per_fig, n_spectrum)

            self.dataset.plot_spectrum(
                self.spectra_dir, fname,
                self.extra_args["flux_norm_cho"],
                self.spectra_wave[lo:hi], self.gt_fluxes[lo:hi],
                self.spectra_wave[lo:hi], self.recon_fluxes[lo:hi],
                clip=self.extra_args["plot_clipped_spectrum"],
                gt_masks=self.spectra_masks[lo:hi],
                recon_masks=self.spectra_masks[lo:hi]
            )

    def _recon_codebook_spectra_individ(self):
        """ Reconstruct codebook spectra for each spectra individually.
        """
        log.info("reconstructing codebook spectrum")

        self.codebook_spectra = torch.stack(self.codebook_spectra).view(
            self.num_spectra, self.qtz_n_embd, -1
        ).detach().cpu().numpy()[self.selected_ids]
        self.spectra_wave_c = torch.stack(self.spectra_wave_c).view(
            self.num_spectra, -1).detach().cpu().numpy()[self.selected_ids]
        self.spectra_masks_c = torch.stack(self.spectra_masks_c).bool().view(
            self.num_spectra, -1).detach().cpu().numpy()[self.selected_ids]

        if self.extra_args["infer_selected"]:
            input("plot codebook spectra for all spectra, press Enter to confirm...")

        prefix = "individ-"
        for i, (wave, masks, codebook_spectra) in enumerate(
                zip(self.spectra_wave_c, self.spectra_masks_c, self.codebook_spectra)
        ):
            cur_dir = join(self.codebook_spectra_dir, f"spectra-{i}")
            Path(cur_dir).mkdir(parents=True, exist_ok=True)

            fname = f"{prefix}ep{self.epoch}-it{self.iteration}"
            wave = np.tile(wave, self.qtz_n_embd).reshape(self.qtz_n_embd, -1)
            masks = np.tile(masks, self.qtz_n_embd).reshape(self.qtz_n_embd, -1)

            self.dataset.plot_spectrum(
                cur_dir, fname, self.extra_args["flux_norm_cho"],
                None, None, wave, codebook_spectra,
                is_codebook=True,
                save_spectra_together=True,
                clip=self.extra_args["plot_clipped_spectrum"],
                recon_masks=masks)

    def _plot_logits_for_gt_bin(self):
        logits = np.array(self.gt_bin_logits) # [nepochs,nspectra]
        plot_multiple(
            self.extra_args["num_spectrum_per_fig"],
            self.extra_args["num_spectrum_per_row"],
            logits.T, self.gt_bin_logits_fname)

    def _plot_codebook_logits(self):
        codebook_logits = torch.stack(self.codebook_logits).detach().cpu().numpy()
        fname = join(self.codebook_logits_dir, f"ep{self.epoch}-it{self.iteration}_logits")
        np.save(fname, codebook_logits)

        plot_multiple(
            self.extra_args["num_spectrum_per_fig"],
            self.extra_args["num_spectrum_per_row"],
            codebook_logits, fname, hist=True)

    def _plot_individ_spectra_loss(self):
        losses = np.array(self.spectra_individ_losses).T
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
                     f"ep{self.epoch}-it{self.iteration}_redshift_residual.txt")
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
                     f"ep{self.epoch}-it{self.iteration}_redshift_outlier.txt")
        with open(fname, "w") as f: f.write(f"{to_save}")

    def _plot_binwise_spectra_loss(self):
        losses = torch.stack(self.binwise_loss).detach().cpu().numpy()
        bin_centers = init_redshift_bins(
            self.extra_args["redshift_lo"], self.extra_args["redshift_hi"],
            self.extra_args["redshift_bin_width"])

        fname = join(self.redshift_dir, f"ep{self.epoch}-it{self.iteration}_losses")
        np.save(fname, np.concatenate((bin_centers[None,:], losses), axis=0))

        plot_multiple(
            self.extra_args["num_spectrum_per_fig"],
            self.extra_args["num_spectrum_per_row"],
            losses, fname, x=bin_centers,vertical_xs=self.gt_redshift)

    def _plot_redshift_logits(self):
        redshift_logits = torch.stack(self.redshift_logits).detach().cpu().numpy()
        bin_centers = init_redshift_bins(
            self.extra_args["redshift_lo"], self.extra_args["redshift_hi"],
            self.extra_args["redshift_bin_width"])

        fname = join(self.redshift_dir, f"ep{self.epoch}-it{self.iteration}_logits")
        np.save(fname, np.concatenate((bin_centers[None,:], redshift_logits), axis=0))

        plot_multiple(
            self.extra_args["num_spectrum_per_fig"],
            self.extra_args["num_spectrum_per_row"],
            redshift_logits, fname, x=bin_centers,vertical_xs=self.gt_redshift
        )

        if self.extra_args["plot_redshift_precision_recall"]:
            plot_precision_recall_all(
                redshift_logits, self.gt_redshift, self.extra_args["redshift_lo"],
                self.extra_args["redshift_hi"], self.extra_args["redshift_bin_width"],
                self.extra_args["num_spectrum_per_row"], f"{fname}_precision_recall.png")

        if self.extra_args["plot_redshift_precision_recall_together"]:
            plot_precision_recall_single(
                redshift_logits, self.gt_redshift, self.extra_args["redshift_lo"],
                self.extra_args["redshift_hi"], self.extra_args["redshift_bin_width"],
                f"{fname}_precision_recall.png")

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

    #############
    # Loss Helpers
    #############

    def calculate_loss(self, data):
        total_loss = 0
        add_to_device(data, self.gpu_fields, self.device)
        self.timer.check("added to gpu")

        # if self.classify_redshift and \
        #    self.extra_args["redshift_classification_method"] == "bayesian_weighted_avg":
        #     self.train_pipeline.set_bayesian_redshift_logits_calculation(
        #         get_loss(self.extra_args["spectra_loss_cho"], self.cuda),
        #         data["spectra_masks"], data["spectra_source_data"])
        # if self.extra_args["add_redshift_logit_bias"]:
        #     init_redshift_prob = data["init_redshift_prob"]
        #     if self.epoch != 0:
        #         init_redshift_prob = torch.zeros(
        #             init_redshift_prob.shape, dtype=init_redshift_prob.dtype
        #         ).to(init_redshift_prob.device)
        # else: init_redshift_prob = None

        if self.calculate_binwise_spectra_loss:
            spectra_loss_func = self.spectra_loss
        else: spectra_loss_func=None

        steps = self.codebook_pretrain_total_steps \
            if self.mode == "redshift_pretrain" else self.total_steps

        ret = forward(
            data,
            self.train_pipeline,
            steps,
            self.space_dim,
            spectra_loss_func=spectra_loss_func,
            qtz=self.qtz,
            qtz_strategy=self.qtz_strategy,
            index_latent=self.index_latent,
            split_latent=self.split_latent,
            apply_gt_redshift=self.apply_gt_redshift,
            perform_integration=self.pixel_supervision,
            trans_sample_method=self.trans_sample_method,
            regularize_codebook_spectra=self.regularize_codebook_spectra,
            calculate_binwise_spectra_loss=self.calculate_binwise_spectra_loss,
            save_coords=self.regularize_spectra_latents,
            save_spectra=self.save_data and self.recon_gt_spectra,
            save_redshift=self.save_data and self.save_redshift,
            save_qtz_weights=self.save_data and self.save_qtz_weights,
            save_redshift_logits=self.regularize_redshift_logits or \
                                 (self.save_data and self.classify_redshift),
            save_codebook_logits=self.regularize_codebook_logits or \
                                 (self.save_data and self.plot_codebook_logits),
            save_codebook_spectra=self.save_data and self.recon_codebook_spectra_individ
        )
        self.timer.check("forwarded")

        spectra_loss = self._calculate_spectra_loss(ret, data)

        # ii) pixel supervision loss
        recon_loss = 0
        if self.pixel_supervision:
            gt_pixels = data["spectra_pixels"]
            recon_pixels = ret["intensity"]
            recon_loss = self.pixel_loss(gt_pixels, recon_pixels)
            recon_loss *= extra_args["pretrain_pixel_beta"]
            self.log_dict["pixel_loss"] += recon_loss.item()

        # iii)
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

        # iv)
        codebook_logits_regu= 0
        if self.regularize_codebook_logits:
            codebook_logits_regu = torch.abs(
                torch.mean(torch.sum(ret["codebook_logits"],dim=-1)))
            codebook_logits_regu *= self.extra_args["codebook_logits_regu_beta"]
            self.log_dict["codebook_logits_regu"] += codebook_logits_regu.item()

        # v)
        spectra_latents_regu = 0
        if self.regularize_spectra_latents:
            # spectra_latents_regu = torch.abs(torch.mean(torch.sum(ret["coords"],dim=-1)))
            spectra_latents_regu = torch.mean(ret["coords"]**2)
            spectra_latents_regu *= self.extra_args["spectra_latents_regu_beta"]
            self.log_dict["spectra_latents_regu"] += spectra_latents_regu.item()

        # vi)
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

        total_loss = spectra_loss + recon_loss + \
            spectra_latents_regu + redshift_logits_regu + codebook_spectra_regu

        self.log_dict["total_loss"] += total_loss.item()
        self.timer.check("loss calculated")
        return total_loss, ret

    def _calculate_spectra_loss(self, ret, data):
        if self.optimize_spectra_for_each_redshift_bin:
            if self.neg_sup_wrong_redshift:
                spectra_loss = self._calculate_neg_sup_loss(ret, data)
            else:
                spectra_loss = torch.mean(ret["spectra_binwise_loss"])
        else:
            recon_fluxes = ret["intensity"]
            spectra_masks = data["spectra_masks"]
            gt_spectra = data["spectra_source_data"]

            assert len(recon_fluxes) != 0
            spectra_loss = self.spectra_loss(
                spectra_masks, gt_spectra, recon_fluxes,
                weight_by_wave_coverage=self.extra_args["weight_by_wave_coverage"]
            )
            if self.extra_args["plot_individ_spectra_loss"]:
                self.cur_spectra_individ_losses.extend(spectra_loss.detach().cpu().numpy())
            spectra_loss = torch.mean(spectra_loss, dim=-1)

        # if spectra_loss.__class__.__name__ == "Tensor":
        #     self.log_dict["spectra_loss"] += spectra_loss.item()
        # else: self.log_dict["spectra_loss"] += spectra_loss
        self.log_dict["spectra_loss"] += spectra_loss.item()
        return spectra_loss

    def _calculate_neg_sup_loss(self, ret, data):
        """ Calculate spectra loss under negative supervision settings.
        """
        all_bin_losses = ret["spectra_binwise_loss"] # [bsz,n_bins]
        gt_bin_losses = torch.mean(all_bin_losses[data["gt_redshift_bin_masks"]])
        self.log_dict["gt_bin_losses"] += gt_bin_losses.item()

        if self.cur_neg_sup_target == "latents":
            spectra_loss = torch.mean(ret["spectra_binwise_loss"])
            _, optimal_wrong_bin_losses = get_optimal_wrong_bin_ids(ret, data)
            self.log_dict["wrong_bin_losses"] += torch.mean(optimal_wrong_bin_losses).item()

        elif self.cur_neg_sup_target == "net":
            if self.extra_args["neg_sup_with_optimal_wrong_bin"]:
                _, optimal_wrong_bin_losses = get_optimal_wrong_bin_ids(ret, data)
                self.log_dict["wrong_bin_losses"] += torch.mean(
                    optimal_wrong_bin_losses).item()
                wrong_bin_losses = self.extra_args["neg_sup_constant"] - optimal_wrong_bin_losses
                wrong_bin_losses[wrong_bin_losses < 0] = 0
            else:
                wrong_bin_losses = all_bin_losses[~data["gt_redshift_bin_masks"]] # [n,]
                self.log_dict["wrong_bin_losses"] += torch.mean(wrong_bin_losses).item()
                wrong_bin_losses = self.extra_args["neg_sup_constant"] - wrong_bin_losses
                wrong_bin_losses[wrong_bin_losses < 0] = 0
                wrong_bin_losses = torch.sum(wrong_bin_losses, dim=-1) / \
                    (self.num_redshift_bins-1)

            wrong_bin_losses = torch.mean(wrong_bin_losses)
            wrong_bin_regus = self.extra_args["neg_sup_beta"] * wrong_bin_losses
            self.log_dict["wrong_bin_regus"] += wrong_bin_regus.item()
            spectra_loss = gt_bin_losses + wrong_bin_regus
        else:
            raise ValueError()

        return spectra_loss

    def log_cli(self):
        """ Controls CLI logging.
            By default, this function only runs every epoch.
        """
        # Average over iterations
        n = len(self.train_data_loader)
        total_loss = self.log_dict["total_loss"] / n
        log_text = "EPOCH {}/{}".format(self.epoch, self.num_epochs)

        log_text += " | total loss: {:>.3E}".format(total_loss)
        log_text += " | spectra loss: {:>.3E}".format(self.log_dict["spectra_loss"] / n)
        if self.pixel_supervision:
            log_text += " | pixel loss: {:>.3E}".format(self.log_dict["pixel_loss"] / n)
        # if self.neg_sup_wrong_redshift:
        #     log_text += " | gt bin loss: {:>.3E}".format(
        #         self.log_dict["gt_bin_losses"] / n)
        #     log_text += " | wrong bin regu: {:>.3E}".format(
        #         self.log_dict["wrong_bin_regus"] / n)
        #     log_text += " | wrong bin loss: {:>.3E}".format(
        #         self.log_dict["wrong_bin_losses"] / n)

        if self.regularize_redshift_logits:
            log_text += " | redshift logits regu: {:>.3E}".format(
                self.log_dict["redshift_logits_regu"] / n)
        if self.regularize_codebook_logits:
            log_text += " | codebook logits regu: {:>.3E}".format(
                self.log_dict["codebook_logits_regu"] / n)
        if self.regularize_spectra_latents:
            log_text += " | codebook latents regu: {:>.3E}".format(
                self.log_dict["spectra_latents_regu"] / n)
        if self.regularize_codebook_spectra:
            log_text += " | codebook spectra regu: {:>.3E}".format(
                self.log_dict["codebook_spectra_regu"] / n)

        log.info(log_text)

    ###################
    # Optimizer Helpers
    ###################

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
        for n,p in self.train_pipeline.named_parameters():
            if n != "nef.latents": p.requires_grad = on_off == "on"

    def _toggle_spectra_latents_grad(self, on_off):
        for n,p in self.train_pipeline.named_parameters():
            if n == "nef.latents": p.requires_grad = on_off == "on"

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
        residu = self.epoch % (sum(self.em_alternation_steps))
        if self.em_alternation_starts_with == "spectra_latents":
           self.cur_optm_target = "spectra_latents" \
               if residu < self.em_alternation_steps[0] else "redshift_latents"
        elif self.em_alternation_starts_with == "redshift_latents":
            self.cur_optm_target = "redshift_latents" \
                if residu < self.em_alternation_steps[0] else "spectra_latents"
        else: raise ValueError()

    def _get_current_neg_sup_target(self):
        if not self.neg_sup_optimize_alternately: return
        residu = self.epoch % (sum(self.neg_sup_alternation_steps))
        if self.neg_sup_alternation_starts_with == "net":
            self.cur_neg_sup_target = "net" \
                if residu < self.neg_sup_alternation_steps[0] else "latents"
        elif self.neg_sup_alternation_starts_with == "latents":
            self.cur_neg_sup_target = "latents" \
                if residu < self.neg_sup_alternation_steps[0] else "net"
        else: raise ValueError()

    def _assign_codebook_pretrain_optimization_params(self):
        """ Configure optimization parameters for codebook pretrain.
        """
        net_params, spectra_latents = [], None
        for name in self.params_dict:
            if name == "nef.latents":
                spectra_latents = self.params_dict[name]
            else: net_params.append(self.params_dict[name])

        latents_group, net_params_group = [], []
        self._add_spectra_latents(spectra_latents, latents_group)
        net_params_group.append({"params": net_params,
                                 "lr": self.extra_args["codebook_pretrain_lr"]})
        return latents_group, net_params_group

    def _assign_redshift_pretrain_optimization_params(self):
        spectra_latents, redshift_latents = None, None
        codebook_logit_params, redshift_logit_params = [], []

        for name in self.params_dict:
            if name == "nef.latents":
                spectra_latents = self.params_dict[name]
            elif name == "nef.redshift_latents":
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

        # codebook latents
        if self.optimize_spectra_latents:
            if self.optimize_latents_alternately:
                self._add_spectra_latents(spectra_latents, spectra_latents_group)
            else:
                self._add_spectra_latents(spectra_latents, latents_group)

        # codebook logits parameters
        if self.optimize_spectra_latents_as_logits:
            pass
        else:
            if self.extra_args["optimize_codebook_logits_mlp"]:
                net_params_group.append({"params": codebook_logit_params,
                                         "lr": self.extra_args["codebook_pretrain_lr"]})

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
                                         "lr": self.extra_args["codebook_pretrain_lr"]})

    def validate(self):
        pass
