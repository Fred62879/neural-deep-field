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


class CodebookTrainerDebug(BaseTrainer):
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

        self.plot_loss = self.extra_args["plot_loss"]
        self.split_latent = self.mode == "redshift_pretrain" and \
            self.extra_args["split_latent"]

        self.optimize_spectra_latents = self.extra_args["direct_optimize_codebook_logits"] or \
            (self.extra_args["optimize_spectra_latents"] and \
             not self.extra_args["load_pretrained_latents_and_freeze"])

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

    def init_net(self):
        self.train_pipeline = self.pipeline[0]
        log.info(self.train_pipeline)
        log.info("Total number of parameters: {}".format(
            sum(p.numel() for p in self.train_pipeline.parameters()))
        )

        self.latents = nn.Embedding(
            self.num_spectra, self.extra_args["qtz_num_embed"])
        self.latents.weight.requires_grad = False
        self.redshift_latents = nn.Embedding(
            self.num_spectra, self.extra_args["redshift_logit_latent_dim"])
        self.redshift_latents.weight.requires_grad = False

    def init_optimizer(self):
        params = {
            # "lr": self.extra_args["redshift_latents_lr"],
            "lr": self.extra_args["codebook_pretrain_lr"],
            "max_iter": 4,
            "history_size": 10
        }
        # latents = [self.latents.weight, self.redshift_latents.weight]
        # self.optimizer = torch.optim.LBFGS(latents, **params)
        # self.optimizer = torch.optim.LBFGS([self.redshift_latents.weight], **params)
        # self.optimizer = torch.optim.LBFGS(self.train_pipeline.parameters(), **params)

        net_params = []
        for n,p in self.train_pipeline.named_parameters():
            if n == "nef.latents.weight":
                net_params.append(p)
            elif n == "nef.redshift_latents.weight":
                net_params.append(p)

        self.optimizer = torch.optim.LBFGS(net_params, **params)

    def set_num_spectra(self):
        if self.mode == "redshift_pretrain":
            if self.extra_args["sample_from_codebook_pretrain_spectra"]:
                self.num_spectra = self.extra_args["redshift_pretrain_num_spectra"]
            else: self.num_spectra = self.dataset.get_num_validation_spectra()
        elif self.mode == "codebook_pretrain":
            self.num_spectra = self.dataset.get_num_supervision_spectra()
        else: raise ValueError("Invalid mode!")

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
        # if self.shuffle_dataloader: sampler_cls = RandomSampler
        # else: sampler_cls = SequentialSampler
        sampler_cls = SequentialSampler

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

    def configure_dataset(self):
        """ Configure dataset with selected fields and set length accordingly.
        """
        self.dataset.set_mode(self.mode)

        # set required fields from dataset
        fields = ["coords","wave_data","spectra_source_data","spectra_masks","spectra_redshift"]

        # use original spectra wave
        self.dataset.set_wave_source("spectra")

        # set spectra data source
        # tmp: in sanity check mode, we manually change sup_id and val_id in spectra_data
        if self.mode == "redshift_pretrain":
            self.dataset.set_spectra_source("val")
        else: self.dataset.set_spectra_source("sup")

        # set input latents for codebook net
        self.dataset.set_coords_source("spectra_latents")
        self.dataset.set_hardcode_data("spectra_latents", self.latents.weight)

        fields.append("redshift_latents")
        self.dataset.set_hardcode_data("redshift_latents", self.redshift_latents.weight)

        self.dataset.toggle_wave_sampling(self.sample_wave)
        if self.sample_wave:
            self.dataset.set_num_wave_samples(self.extra_args["pretrain_num_wave_samples"])
            self.dataset.set_wave_sample_method(self.extra_args["pretrain_wave_sample_method"])
        self.dataset.toggle_integration(self.pixel_supervision)
        if self.train_within_wave_range:
            self.dataset.set_wave_range(
                self.extra_args["spectra_supervision_wave_lo"],
                self.extra_args["spectra_supervision_wave_hi"])

        self.dataset.set_length(self.num_spectra)
        if self.extra_args["infer_selected"]:
            self.selected_ids = select_inferrence_ids(
                self.num_spectra,
                self.extra_args["pretrain_num_infer_upper_bound"]
            )
        else: self.selected_ids = np.arange(self.num_spectra)

        self.dataset.set_fields(fields)

    #############
    # Training logic
    #############

    def begin_train(self):
        self.total_steps = 0
        self.codebook_pretrain_total_steps = 0
        if self.plot_loss: self.losses = []
        log.info(f"{self.num_iterations_cur_epoch} batches per epoch.")

    def train(self):
        self.begin_train()
        for self.epoch in range(self.num_epochs + 1):
            self.begin_epoch()
            for batch in range(self.num_iterations_cur_epoch):
                data = self.next_batch()
                self.step(data)
                self.iteration += 1
                self.total_steps += 1
            self.end_epoch()
        self.end_train()

    def end_train(self):
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

    #############
    # Epoch begin and end
    #############

    def begin_epoch(self):
        self.iteration = 0
        self.reset_data_iterator()
        self.pre_epoch()
        self.init_log_dict()

    def end_epoch(self):
        self.post_epoch()
        if self.epoch < self.num_epochs:
            self.iteration = 0
            self.epoch += 1

    def reset_data_iterator(self):
        """ Rewind the iterator for the new epoch.
        """
        self.train_data_loader_iter = iter(self.train_data_loader)

    #############
    # One epoch
    #############

    def pre_epoch(self):
        self.set_num_batches()
        if self.save_model_every > -1 and self.epoch % self.save_model_every == 0:
            self.save_model()
        self.train_pipeline.train()

    def post_epoch(self):
        """ By default, this function logs to Tensorboard, renders images to Tensorboard,
            saves the model, and resamples the dataset.
        """
        self.train_pipeline.eval()

        total_loss = self.log_dict["total_loss"] / len(self.train_data_loader)
        if self.plot_loss:
            self.losses.append(total_loss)

        if self.log_cli_every > -1 and self.epoch % self.log_cli_every == 0:
            self.log_cli()

    #############
    # One step
    #############

    def step(self, data):
        # print('step')
        def closure():
            # print('closure')
            self.optimizer.zero_grad()
            loss = self.calculate_loss(data)
            loss.backward()
            # print('backward')
            return loss

        self.optimizer.step(closure)
        loss = closure()

        if self.plot_grad_every != -1 and \
           (self.epoch == 0 or (self.epoch % self.plot_grad_every == 0)):
            plot_grad_flow(self.train_pipeline.named_parameters(), self.grad_fname)

    def init_log_dict(self):
        """ Custom log dict. """
        super().init_log_dict()
        self.log_dict["spectra_loss"] = 0.0

    #############
    # Helper methods
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

    def calculate_loss(self, data):
        add_to_device(data, self.gpu_fields, self.device)
        ret = forward(
            data,
            self.train_pipeline,
            self.codebook_pretrain_total_steps,
            self.space_dim,
            qtz=self.qtz,
            qtz_strategy=self.qtz_strategy,
            split_latent=self.split_latent,
            apply_gt_redshift=self.apply_gt_redshift,
            save_redshift_logits=self.classify_redshift,
            trans_sample_method=self.trans_sample_method)

        total_loss = self.spectra_loss(
            data["spectra_masks"], data["spectra_source_data"], ret["intensity"],
            weight_by_wave_coverage=self.extra_args["weight_by_wave_coverage"])
        total_loss = torch.mean(total_loss, dim=-1)
        self.log_dict["total_loss"] += total_loss.item()
        return total_loss

    def log_cli(self):
        """ Controls CLI logging.
            By default, this function only runs every epoch.
        """
        # Average over iterations
        n = len(self.train_data_loader)
        total_loss = self.log_dict["total_loss"] / n
        log_text = "EPOCH {}/{}".format(self.epoch, self.num_epochs)
        log_text += " | total loss: {:>.3E}".format(total_loss)
        log.info(log_text)

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
        checkpoint["latents"] = self.latents.weight
        if not self.apply_gt_redshift and self.split_latent:
            checkpoint["redshift_latents"] = self.redshift_latents.weight

        torch.save(checkpoint, model_fname)
        return checkpoint

    def validate(self):
        pass
