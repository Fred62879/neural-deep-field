import os
import time
import torch
import shutil
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
from wisp.utils.plot import plot_grad_flow, plot_multiple
from wisp.optimizers import multi_optimizer
from wisp.loss import spectra_supervision_loss, \
    spectra_supervision_emd_loss, pretrain_pixel_loss
from wisp.utils.common import get_gpu_info, add_to_device, sort_alphanumeric, \
    select_inferrence_ids, load_embed, load_model_weights, forward, \
    load_pretrained_model_weights, get_pretrained_model_fname, \
    get_bool_classify_redshift, freeze_layers, get_loss, create_latent_mask, set_seed


class CodebookTrainer(BaseTrainer):
    """ Trainer class for codebook pretraining.
    """
    def __init__(self, pipeline, dataset, optim_cls, optim_params, device, **extra_args):
        super().__init__(pipeline, dataset, optim_cls, optim_params, device, **extra_args)

        assert(
            extra_args["space_dim"] == 3 and \
            extra_args["pretrain_codebook"]
        )

        # save config file to log directory
        dst = join(self.log_dir, "config.yaml")
        shutil.copyfile(extra_args["config"], dst)

        self.cuda = "cuda" in str(device)
        self.verbose = extra_args["verbose"]
        self.space_dim = extra_args["space_dim"]
        self.gpu_fields = extra_args["gpu_data"]
        self.recon_beta = extra_args["pretrain_pixel_beta"]
        self.redshift_logits_regu_method = extra_args["redshift_logits_regu_method"]

        self.check_configs()
        self.summarize_training_tasks()
        self.set_path()

        self.set_num_spectra()
        self.init_net()
        self.collect_model_params()

        self.init_data()
        self.init_loss()
        self.init_optimizer()

    #############
    # Initializations
    #############

    def check_configs(self):
        tasks = set(self.extra_args["tasks"])
        if "redshift_pretrain" in tasks:
            assert not self.extra_args["optimize_spectra_latents"] or \
                not self.extra_args["load_pretrained_latents_and_freeze"]

    def summarize_training_tasks(self):
        tasks = set(self.extra_args["tasks"])

        if "codebook_pretrain" in tasks:
            self.mode = "codebook_pretrain"
        elif "redshift_pretrain" in tasks:
            self.mode = "redshift_pretrain"
        else: raise ValueError()

        # quantization setups
        self.qtz_latent = self.space_dim == 3 and self.extra_args["quantize_latent"]
        self.qtz_spectra = self.space_dim == 3 and self.extra_args["quantize_spectra"]
        assert not (self.qtz_latent and self.qtz_spectra)
        self.qtz = self.qtz_latent or self.qtz_spectra
        self.qtz_n_embd = self.extra_args["qtz_num_embed"]
        self.qtz_strategy = self.extra_args["quantization_strategy"]

        self.pixel_supervision = self.extra_args["pretrain_pixel_supervision"]
        self.trans_sample_method = self.extra_args["trans_sample_method"]

        # redshift setup
        self.classify_redshift = get_bool_classify_redshift(**self.extra_args)
        self.apply_gt_redshift = self.extra_args["model_redshift"] and self.extra_args["apply_gt_redshift"]
        if self.mode == "codebook_pretrain": assert self.apply_gt_redshift

        self.sample_wave = not self.extra_args["pretrain_use_all_wave"] # True
        self.train_within_wave_range = not self.pixel_supervision and \
            self.extra_args["learn_spectra_within_wave_range"]

        self.save_redshift = "save_redshift_during_train" in tasks
        self.save_qtz_weights = "save_qtz_weights_during_train" in tasks
        self.save_pixel_values = "save_pixel_values_during_train" in tasks and self.pixel_supervision
        self.recon_gt_spectra = "recon_gt_spectra_during_train" in tasks
        self.recon_codebook_spectra_individ = "recon_codebook_spectra_individ_during_train" in tasks

        # all others
        self.plot_loss = self.extra_args["plot_loss"]
        self.index_latent = True # index latents as coords in model
        self.split_latent = self.mode == "redshift_pretrain" and \
            self.extra_args["split_latent"]
        self.regu_within_codebook_spectra = self.qtz_spectra and \
            self.extra_args["regu_within_codebook_spectra"]
        self.regu_across_codebook_spectra = self.qtz_spectra and \
            self.extra_args["regu_across_codebook_spectra"]
        self.regu_codebook_spectra = self.regu_within_codebook_spectra or \
            self.regu_across_codebook_spectra
        assert self.regu_within_codebook_spectra + \
            self.regu_across_codebook_spectra <= 1
        self.optimize_spectra_latents = self.extra_args["direct_optimize_codebook_logits"] or \
            (self.extra_args["optimize_spectra_latents"] and \
             not self.extra_args["load_pretrained_latents_and_freeze"])

    def set_path(self):
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        log.info(f"logging to {self.log_dir}")

        for cur_path, cur_pname, in zip(
                ["loss_dir","model_dir","spectra_dir","codebook_spectra_dir","qtz_weight_dir"],
                ["losses","models","train_spectra","train_codebook_spectra","qtz_weights"]):
            path = join(self.log_dir, cur_pname)
            setattr(self, cur_path, path)
            Path(path).mkdir(parents=True, exist_ok=True)

        self.grad_fname = join(self.log_dir, "grad.png")

        if self.plot_loss:
            self.loss_fname = join(self.loss_dir, "loss")

        if self.mode == "redshift_pretrain":
            # redshift pretrain use pretrained model from codebook pretrain
            self.pretrained_model_fname, _ = get_pretrained_model_fname(
                self.log_dir,
                self.extra_args["pretrain_log_dir"],
                self.extra_args["pretrained_model_name"])

        if self.extra_args["resume_train"]:
            self.resume_train_model_fname, self.resume_loss_fname = get_pretrained_model_fname(
                self.log_dir,
                self.extra_args["resume_log_dir"],
                self.extra_args["resume_model_fname"])

        if self.extra_args["plot_logits_for_gt_bin"]:
            self.gt_bin_logits_fname = join(self.log_dir, "gt_bin_logits")

    def set_num_spectra(self):
        if self.mode == "redshift_pretrain":
            if self.extra_args["sample_from_codebook_pretrain_spectra"]:
                self.num_spectra = self.extra_args["redshift_pretrain_num_spectra"]
            else: self.num_spectra = self.dataset.get_num_validation_spectra()
        elif self.mode == "codebook_pretrain":
            self.num_spectra = self.dataset.get_num_supervision_spectra()
        else: raise ValueError("Invalid mode!")

    def init_net(self):
        self.train_pipeline = self.pipeline[0]
        self.train_pipeline.set_batch_reduction_order("bin_avg_first")

        latents, redshift_latents = self.init_latents()
        self.train_pipeline.set_latents(latents.weight)
        if redshift_latents is not None:
            self.train_pipeline.set_redshift_latents(redshift_latents.weight)

        # for n,p in self.train_pipeline.named_parameters(): print(n, p.requires_grad)
        self.freeze_and_load()
        # for n,p in self.train_pipeline.named_parameters(): print(n, p.requires_grad)

        log.info(self.train_pipeline)
        log.info("Total number of parameters: {}".format(
            sum(p.numel() for p in self.train_pipeline.parameters())))

    def collect_model_params(self):
        # collect all parameters from network and trainable latents
        self.params_dict = {}
        # self.params_dict["latents"] = self.latents.weight
        # if not self.apply_gt_redshift and self.split_latent:
        #     self.params_dict["redshift_latents"] = self.redshift_latents.weight
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
            latents, net_params = self.assign_codebook_pretrain_optimization_params()
        elif self.mode == "redshift_pretrain":
            latents, net_params = self.assign_redshift_pretrain_optimization_params()
        else: raise ValueError()

        if self.extra_args["optimize_latents_use_lbfgs"]:
            # latents_optimizer = torch.optim.LBFGS(latents, **params)
            # if len(net_params) != 0:
            #     net_optimizer = self.optim_cls(net_params, **self.optim_params)
            #     self.optimizer = multi_optimizer(latents_optimizer, net_optimizer)
            # else:
            #     self.optimizer = multi_optimizer(latents_optimizer)

            params = {
                "lr": self.extra_args["redshift_latents_lr"],
                "max_iter": 4,
                "history_size": 10
            }
            self.optimizer = torch.optim.LBFGS(latents, **params)
            # for n,p in self.train_pipeline.named_parameters():
            #     print(n, p.requires_grad)
            # self.optimizer = torch.optim.LBFGS(self.train_pipeline.parameters(), **params)
        else:
            net_params.extend(latents)
            optimizer = self.optim_cls(net_params, **self.optim_params)
            self.optimizer = multi_optimizer(optimizer)

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

        # use original spectra wave
        self.dataset.set_wave_source("spectra")

        # set spectra data source
        # tmp: in sanity check mode, we manually change sup_id and val_id in spectra_data
        if self.mode == "redshift_pretrain":
            self.dataset.set_spectra_source("val")
        else: self.dataset.set_spectra_source("sup")

        # set input latents for codebook net
        fields.append("idx")
        # self.dataset.set_coords_source("spectra_latents")
        # self.dataset.set_hardcode_data("spectra_latents", self.latents.weight)
        # if not self.apply_gt_redshift and self.split_latent:
        #     fields.append("redshift_latents")
        #     self.dataset.set_hardcode_data("redshift_latents", self.redshift_latents.weight)

        self.dataset.toggle_integration(self.pixel_supervision)

        self.dataset.toggle_wave_sampling(self.sample_wave)
        if self.sample_wave:
            self.dataset.set_num_wave_samples(self.extra_args["pretrain_num_wave_samples"])
            self.dataset.set_wave_sample_method(self.extra_args["pretrain_wave_sample_method"])
        if self.train_within_wave_range:
            self.dataset.set_wave_range(
                self.extra_args["spectra_supervision_wave_lo"],
                self.extra_args["spectra_supervision_wave_hi"])

        self.dataset.set_length(self.num_spectra)
        if self.extra_args["infer_selected"]:
            self.selected_ids = select_inferrence_ids(
                self.num_spectra,
                self.extra_args["pretrain_num_infer_upper_bound"])
        else:
            self.selected_ids = np.arange(self.num_spectra)

        self.dataset.set_fields(fields)

    #############
    # Training logic
    #############

    def begin_train(self):
        self.total_steps = 0
        self.codebook_pretrain_total_steps = 0

        if self.plot_loss:
            self.losses = []
        if self.extra_args["plot_individ_spectra_loss"]:
            self.spectra_individ_losses = []
        if self.extra_args["plot_logits_for_gt_bin"]:
            self.gt_bin_logits = []
        if self.extra_args["resume_train"]:
            self.resume_train()

        log.info(f"{self.num_iterations_cur_epoch} batches per epoch.")

    def train(self):
        self.begin_train()

        # for epoch in tqdm(range(self.num_epochs + 1)):
        for self.epoch in range(self.num_epochs + 1):
            self.begin_epoch()
            self.timer.check("begun epoch")

            for batch in range(self.num_iterations_cur_epoch):
                iter_start_time = time.time()

                data = self.next_batch()
                self.timer.check("got data")

                self.pre_step()
                ret = self.step(data)
                self.post_step(data, ret)

                self.iteration += 1
                self.total_steps += 1

                self.timer.check("batch ended")

            self.end_epoch()
            self.timer.check("epoch ended")

        if self.extra_args["plot_logits_for_gt_bin"]:
            self._plot_logits_for_gt_bin()

        self.end_train()

    def end_train(self):
        self.writer.close()

        if self.plot_grad_every != -1:
            plt.savefig(self.grad_fname)
            plt.close()

        if self.plot_loss:
            x = np.arange(len(self.losses))
            plt.plot(x, self.losses); plt.title("Loss")
            plt.savefig(self.loss_fname + ".png")
            plt.close()
            np.save(self.loss_fname + ".npy", np.array(self.losses))

            plt.plot(x, np.log10(np.array(self.losses)))
            plt.title("Log10 loss")
            plt.savefig(self.loss_fname + "_log10.png")
            plt.close()

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
        self.pre_epoch()
        self.init_log_dict()

        if self.extra_args["plot_logits_for_gt_bin"]:
            self.cur_gt_bin_logits = []
        if self.extra_args["plot_individ_spectra_loss"]:
            self.cur_spectra_individ_losses = []

    def end_epoch(self):
        self.post_epoch()

        if self.epoch < self.num_epochs:
            self.iteration = 0
            self.epoch += 1
        else:
            self.scene_state.optimization.running = False

        if self.extra_args["plot_logits_for_gt_bin"]:
            self.gt_bin_logits.append(self.cur_gt_bin_logits)
        if self.extra_args["plot_individ_spectra_loss"]:
            self.spectra_individ_losses.append(self.cur_spectra_individ_losses)

    def reset_data_iterator(self):
        """ Rewind the iterator for the new epoch.
        """
        self.scene_state.optimization.iterations_per_epoch = len(self.train_data_loader)
        self.train_data_loader_iter = iter(self.train_data_loader)

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
        self.scene_state.optimization.losses["total_loss"].append(total_loss)

        if self.plot_loss:
            self.losses.append(total_loss)

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

    def init_log_dict(self):
        """ Custom log dict. """
        super().init_log_dict()
        self.log_dict["pixel_loss"] = 0.0
        self.log_dict["spectra_loss"] = 0.0
        self.log_dict["spectra_latents_regu"] = 0.0
        self.log_dict["redshift_logits_regu"] = 0.0
        self.log_dict["codebook_spectra_regu"] = 0.0

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
        # if self.extra_args["optimize_latents_use_lbfgs"]:
        #     # self.optimizer.set_loss_func(partial(self.calculate_loss, self))
        #     self.optimizer.set_loss_func(self.calculate_loss)
        #     self.optimizer.set_current_data(data)

        def closure(keep_ret=False):
            self.optimizer.zero_grad()
            self.timer.check("zero grad")
            loss, ret = self.calculate_loss(data)
            loss.backward()
            self.timer.check("backward done")
            if keep_ret: return loss, ret
            return loss

        if self.extra_args["optimize_latents_use_lbfgs"]:
            self.optimizer.step(closure)
            self.timer.check("stepped")

        loss, ret = closure(True) # forward backward without step
        if self.plot_grad_every != -1 and \
           (self.epoch == 0 or (self.epoch % self.plot_grad_every == 0)):
            plot_grad_flow(self.params_dict.items(), self.grad_fname)

        if not self.extra_args["optimize_latents_use_lbfgs"]:
            self.optimizer.step()
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
                if self.classify_redshift:
                    ids = torch.argmax(ret["redshift_logits"], dim=-1)
                    argmax_redshift = ret["redshift"][ids]
                    self.est_redshift.extend(argmax_redshift)
                else:
                    self.redshift.extend(ret["redshift"])
            if self.save_pixel_values:
                self.recon_pixel_vals.extend(ret["intensity"])
                self.gt_pixel_vals.extend(data["spectra_pixels"])
            if self.save_qtz_weights:
                self.qtz_weights.extend(ret["qtz_weights"])

            if self.recon_gt_spectra:
                self.recon_fluxes.extend(ret["spectra"])
                self.gt_fluxes.extend(data["spectra_source_data"][:,1])
                self.spectra_wave.extend(data["spectra_source_data"][:,0])
                self.spectra_masks.extend(data["spectra_masks"])

            if self.recon_codebook_spectra_individ:
                self.codebook_spectra.extend(ret["codebook_spectra"])
                self.spectra_wave_c.extend(data["spectra_source_data"][:,0])
                self.spectra_masks_c.extend(data["spectra_masks"])

    #############
    # Model Helpers
    #############

    def init_latents(self):
        if self.mode == "codebook_pretrain":
            assert not self.split_latent
            latents = self.init_codebook_pretrain_spectra_latents()
            redshift_latents = None
        elif self.mode == "redshift_pretrain":
            latents = self.init_redshift_pretrain_spectra_latents()
            redshift_latents = self.init_redshift_pretrain_redshift_latents()
        else:
            raise ValueError("Invalid pretrainer mode.")
        return latents, redshift_latents

    def freeze_and_load(self):
        """ For redshift pretrain (sanity check), we load part of the pretrained
            model (from codebook pretrain) and freeze.
        """
        if self.mode == "codebook_pretrain":
            pass
        elif self.mode == "redshift_pretrain":
            load_excls, freeze_excls = [], []

            # codebook logits
            if self.extra_args["direct_optimize_codebook_logits"]:
                # directly optimize logits
                load_excls.append("nef.latents")
                freeze_excls.append("nef.latents")
            else:
                # optimize an autodecoder
                if not self.extra_args["load_pretrained_codebook_logits_mlp"]:
                    load_excls.append("spatial_decoder.decode")
                if self.extra_args["optimize_codebook_logits_mlp"]:
                    freeze_excls.append("nef.latents")
                if self.extra_args["optimize_spectra_latents"]:
                    freeze_excls.append("spatial_decoder.decode")

            # redshift bin logits
            if self.classify_redshift:
                if self.extra_args["optimize_redshift_latents_as_logits"]:
                    # directly optimize logits
                    freeze_excls.append("nef.redshift_latents")
                else:
                    # optimize an autodecoder
                    freeze_excls.append("redshift_decoder")
                    freeze_excls.append("nef.redshift_latents")

            freeze_layers(self.train_pipeline, excls=freeze_excls)
            self.load_model(self.pretrained_model_fname, excls=load_excls)
        else:
            raise ValueError()

    def init_codebook_pretrain_spectra_latents(self):
        latents = None
        set_seed(self.extra_args["seed"] + 1)
        if self.extra_args["direct_optimize_codebook_logits"]:
            dim = self.extra_args["qtz_num_embed"]
        else: dim = self.extra_args["spectra_latent_dim"]
        latents = nn.Embedding(self.num_spectra, dim, device=self.device)
        return latents

    def init_redshift_pretrain_redshift_latents(self):
        redshift_latents = None
        set_seed(self.extra_args["seed"] + 2)
        red_z_dim = self.extra_args["redshift_logit_latent_dim"]

        if self.apply_gt_redshift:
            pass
        elif self.split_latent:
            if self.extra_args["zero_init_redshift_latents"]:
                one_latents = 0.01 * torch.ones(self.num_spectra, red_z_dim)
                # zero_latents = torch.zeros(self.num_spectra, red_z_dim)
                redshift_latents = nn.Embedding.from_pretrained(
                    one_latents, device=self.device,
                    freeze=not self.extra_args["optimize_redshift_latents"])
            else:
                redshift_latents = nn.Embedding(
                    self.num_spectra, red_z_dim, device=self.device
                ).requires_grad_(
                    self.extra_args["optimize_redshift_latents_as_logits"] or \
                    self.extra_args["optimize_redshift_latents"])

        return redshift_latents

    def init_redshift_pretrain_spectra_latents(self):
        latents = None
        set_seed(self.extra_args["seed"] + 1)
        if self.extra_args["direct_optimize_codebook_logits"]:
            latents = nn.Embedding(
                self.num_spectra, self.extra_args["qtz_num_embed"], device=self.device)
        else:
            sp_z_dim = self.extra_args["spectra_latent_dim"]
            if self.extra_args["sample_from_codebook_pretrain_spectra"]:
                if self.extra_args["load_pretrained_latents_and_freeze"]:
                    # checkpoint comes from codebook pretrain (use sup spectra)
                    checkpoint = torch.load(self.pretrained_model_fname)
                    assert checkpoint["latents"].shape[1] == sp_z_dim

                    # redshift pretrain use val spectra which is a permutation of sup spectra
                    permute_ids = self.dataset.get_redshift_pretrain_spectra_ids()
                    latents = nn.Embedding.from_pretrained(
                        checkpoint["latents"][permute_ids], freeze=True, device=self.device)
                else:
                    latents = nn.Embedding(self.num_spectra, sp_z_dim, device=self.device)
            else:
                latents = nn.Embedding(self.num_spectra, sp_z_dim, device=self.device)
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
        # checkpoint["latents"] = self.latents.weight
        # if not self.apply_gt_redshift and self.split_latent:
        #     checkpoint["redshift_latents"] = self.redshift_latents.weight

        torch.save(checkpoint, model_fname)
        return checkpoint

    def resume_train(self):
        try:
            checkpoint = self.load_model(self.resume_train_model_fname)
            # a = checkpoint["optimizer_state_dict"]
            # b = a["state"];c = a["param_groups"];print(b[0])

            # TODO: can we load latents as part of the model
            # self.latents = nn.Embedding.from_pretrained(
            #     checkpoint["latents"],
            #     freeze=not self.optimize_spectra_latents)
            # if not self.apply_gt_redshift and self.split_latent:
            #     self.redshift_latents = nn.Embedding.from_pretrained(
            #         checkpoint["redshift_latents"],
            #         freeze=self.extra_args["optimize_redshift_latents"])

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

    def save_local(self):
        if self.recon_gt_spectra:
            self._recon_gt_spectra()

        if self.recon_codebook_spectra_individ:
            self._recon_codebook_spectra_individ()

        if self.save_redshift:
            self._save_redshift()

        if self.save_qtz_weights:
            self._save_qtz_weights()

        if self.save_pixel_values:
            self._save_pixel_values()

    def _save_redshift(self):
        # if type(self.redshift) == list:
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

    def validate(self):
        pass

    def _plot_logits_for_gt_bin(self):
        logits = np.array(self.gt_bin_logits) # [nepochs,nspectra]
        plot_multiple(
            self.extra_args["num_spectrum_per_fig"],
            self.extra_args["num_spectrum_per_row"],
            logits.T, self.gt_bin_logits_fname)

    def _plot_individ_spectra_loss(self):
        losses = np.array(self.spectra_individ_losses).T
        np.save(self.loss_fname + "_individ.npy", losses)
        plot_multiple(
            self.extra_args["num_spectrum_per_fig"],
            self.extra_args["num_spectrum_per_row"],
            losses, self.loss_fname + "_individ"
        )
        plot_multiple(
            self.extra_args["num_spectrum_per_fig"],
            self.extra_args["num_spectrum_per_row"],
            np.log10(losses), self.loss_fname + "_individ_log10"
        )

    #############
    # Loss Helpers
    #############

    def calculate_loss(self, data):
        total_loss = 0
        add_to_device(data, self.gpu_fields, self.device)
        self.timer.check("added to gpu")

        if self.classify_redshift and \
           self.extra_args["redshift_classification_method"] == "bayesian_weighted_avg":
            self.train_pipeline.set_bayesian_redshift_logits_calculation(
                get_loss(self.extra_args["spectra_loss_cho"], self.cuda),
                data["spectra_masks"], data["spectra_source_data"]
            )

        if self.extra_args["add_redshift_logit_bias"]:
            init_redshift_prob = data["init_redshift_prob"]
            # if self.epoch != 0:
            #     init_redshift_prob = torch.zeros(
            #         init_redshift_prob.shape, dtype=init_redshift_prob.dtype
            #     ).to(init_redshift_prob.device)
        else: init_redshift_prob = None

        steps = self.codebook_pretrain_total_steps \
            if self.mode == "redshift_pretrain" else self.total_steps

        ret = forward(
            data,
            self.train_pipeline,
            steps,
            self.space_dim,
            qtz=self.qtz,
            qtz_strategy=self.qtz_strategy,
            index_latent=self.index_latent,
            split_latent=self.split_latent,
            apply_gt_redshift=self.apply_gt_redshift,
            perform_integration=self.pixel_supervision,
            trans_sample_method=self.trans_sample_method,
            regu_codebook_spectra=self.regu_codebook_spectra,
            save_spectra=self.save_data and self.recon_gt_spectra,
            save_redshift=self.save_data and self.save_redshift,
            save_redshift_logits=self.classify_redshift,
            save_qtz_weights=self.save_data and self.save_qtz_weights,
            save_codebook_spectra=self.save_data and \
                                  self.recon_codebook_spectra_individ,
            save_spectra_all_bins=True, # debug
            init_redshift_prob=init_redshift_prob, # debug
        )
        self.timer.check("forwarded")

        # i) spectra supervision loss
        spectra_loss = 0
        recon_fluxes = ret["intensity"]
        spectra_masks = data["spectra_masks"]
        gt_spectra = data["spectra_source_data"]

        if len(recon_fluxes) == 0:
            spectra_loss = 0
        else:
            spectra_loss = self.spectra_loss(
                spectra_masks, gt_spectra, recon_fluxes,
                weight_by_wave_coverage=self.extra_args["weight_by_wave_coverage"]
            )
            if self.extra_args["plot_individ_spectra_loss"]:
                self.cur_spectra_individ_losses.extend(spectra_loss.detach().cpu().numpy())
            spectra_loss = torch.mean(spectra_loss, dim=-1)
            self.log_dict["spectra_loss"] += spectra_loss.item()

        # ii) pixel supervision loss
        recon_loss = 0
        if self.pixel_supervision:
            gt_pixels = data["spectra_pixels"]
            recon_pixels = ret["intensity"]
            recon_loss = self.pixel_loss(gt_pixels, recon_pixels)
            recon_loss *= self.recon_beta
            self.log_dict["pixel_loss"] += recon_loss.item()

        # iii)
        spectra_latents_regu = 0
        if self.extra_args["regularize_pretrain_spectra_latents"]:
            spectra_latents_regu = torch.abs(torch.mean(torch.sum(data["coords"],dim=-1)))
            spectra_latents_regu *= self.extra_args["spectra_latents_regu_beta"]
            self.log_dict["spectra_latents_regu"] += spectra_latents_regu.item()

        # iv)
        redshift_logits_regu = 0
        if self.classify_redshift and self.extra_args["regu_redshift_logits"] and \
           self.extra_args["redshift_classification_method"] == "weighted_avg":

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

        # v)
        codebook_regu = 0
        if self.regu_within_codebook_spectra:
            sp = ret["full_range_codebook_spectra"].shape # [num_embed,nsmpl]
            dtp = ret["full_range_codebook_spectra"].device
            codebook_regu = F.l1_loss(
                ret["full_range_codebook_spectra"], torch.zeros(sp).to(dtp))
        elif self.regu_across_codebook_spectra:
            codebook_regu = torch.mean(
                torch.sum(ret["full_range_codebook_spectra"], dim=0))
        codebook_regu *= self.extra_args["codebook_spectra_regu_beta"]
        self.log_dict["codebook_spectra_regu"] += codebook_regu

        total_loss = spectra_loss + recon_loss + \
            spectra_latents_regu + redshift_logits_regu + codebook_regu

        self.log_dict["total_loss"] += total_loss.item()
        self.timer.check("loss calculated")
        return total_loss, ret

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
        if self.extra_args["regularize_pretrain_spectra_latents"]:
            log_text += " | spectra latents regu: {:>.3E}".format(
                self.log_dict["spectra_latents_regu"] / n)
        if self.classify_redshift and \
           self.extra_args["redshift_classification_method"] == "weighted_avg":
            log_text += " | redshift logits regu: {:>.3E}".format(
                self.log_dict["redshift_logits_regu"] / n)
        if self.regu_codebook_spectra:
            log_text += " | codebook spectra regu: {:>.3E}".format(
                self.log_dict["codebook_spectra_regu"] / n)

        log.info(log_text)

    ###################
    # Optimizer Helpers
    ###################

    def assign_codebook_pretrain_optimization_params(self):
        latents, net_params = [], []

        spectra_latents = None
        other_params, spectra_logit_params = [], []
        for name in self.params_dict:
            if name == "nef.latents":
                spectra_latents = self.params_dict[name]
            elif "spatial_decoder.decode" in name:
                spectra_logit_params.append(self.params_dict[name])
            else:
                other_params.append(self.params_dict[name])

        net_params.append({"params": other_params,
                           "lr": self.extra_args["codebook_pretrain_lr"]})

        if self.extra_args["direct_optimize_codebook_logits"]:
            latents.append({"params": spectra_latents,
                            "lr": self.extra_args["spectra_latents_lr"]})
        else:
            latents.append({"params": spectra_latents,
                            "lr": self.extra_args["codebook_pretrain_lr"]})
            net_params.append({"params": spectra_logit_params,
                               "lr": self.extra_args["codebook_pretrain_lr"]})
        return latents, net_params

    def assign_redshift_pretrain_optimization_params(self):
        latents, net_params = [], []

        spectra_latents, redshift_latents = None, None
        spectra_logit_params, redshift_logit_params = [], []

        for name in self.params_dict:
            if name == "nef.latents":
                spectra_latents = self.params_dict[name]
            elif name == "nef.redshift_latents":
                redshift_latents = self.params_dict[name]
            elif "redshift_decoder" in name:
                redshift_logit_params.append(self.params_dict[name])
            elif "spatial_decoder.decode" in name:
                spectra_logit_params.append(self.params_dict[name])

        # redshift latents & parameters
        if self.apply_gt_redshift:
            pass
        elif self.split_latent:
            if self.extra_args["optimize_redshift_latents_as_logits"]:
                if self.extra_args["optimize_latents_use_lbfgs"]:
                    latents.append(redshift_latents)
                else:
                    latents.append({"params": redshift_latents,
                                    "lr": self.extra_args["redshift_latents_lr"]})
            elif self.extra_args["optimize_redshift_latents"]:
                latents.append({"params": redshift_latents,
                                "lr": self.extra_args["redshift_latents_lr"]})
                net_params.append({"params": redshift_logit_params,
                                   "lr": self.extra_args["codebook_pretrain_lr"]})
            else:
                raise ValueError("Nothing to do.")
        else:
            raise ValueError("Must split latents.")

        # spectra latents
        if self.optimize_spectra_latents:
            if self.extra_args["optimize_latents_use_lbfgs"]:
                latents.append(spectra_latents)
            else:
                latents.append({"params": spectra_latents,
                                "lr": self.extra_args["spectra_latents_lr"]})

        # spectra decoder parameters
        if self.extra_args["direct_optimize_codebook_logits"]:
            pass
        else:
            if self.extra_args["optimize_codebook_logits_mlp"] or \
               not self.extra_args["load_pretrained_latents_and_freeze"]:
                net_params.append({"params": spectra_logit_params,
                                   "lr": self.extra_args["codebook_pretrain_lr"]})
        return latents, net_params
