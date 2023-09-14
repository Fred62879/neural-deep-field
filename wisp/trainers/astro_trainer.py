
import os
import time
import torch
import shutil
import nvidia_smi
import numpy as np
import torch.nn as nn
import logging as log
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from functools import partial
from os.path import exists, join
from torch.utils.data import BatchSampler, SequentialSampler, \
    RandomSampler, DataLoader

from wisp.datasets import default_collate
from wisp.datasets.patch_data import PatchData
from wisp.datasets.data_utils import get_neighbourhood_center_pixel_id
from wisp.utils.plot import plot_horizontally, plot_embed_map, plot_grad_flow
from wisp.utils.common import get_gpu_info, add_to_device, sort_alphanumeric, \
    load_pretrained_model_weights, forward, print_shape, create_patch_uid
from wisp.trainers import BaseTrainer, log_metric_to_wandb, log_images_to_wandb
from wisp.loss import spectra_supervision_loss, spectral_masking_loss, redshift_supervision_loss


class AstroTrainer(BaseTrainer):
    """ Trainer class for astro dataset.
        This trainer is used only in three cases:
          i)   main train (without pretrain)
          ii)  main train (following pretrain) supervised with only test spectra pixels
          iii) main train (following pretrain) supervised with all pixels
    """
    def __init__(self, pipeline, dataset, optim_cls, optim_params, device, **extra_args):
        super().__init__(pipeline, dataset, optim_cls, optim_params, device, **extra_args)

        # save config file to log directory
        dst = join(self.log_dir, "config.yaml")
        shutil.copyfile(extra_args["config"], dst)

        self.cuda = "cuda" in str(self.device)
        self.verbose = self.extra_args["verbose"]
        self.space_dim = self.extra_args["space_dim"]
        self.gpu_fields = self.extra_args["gpu_data"]
        self.weight_train = self.extra_args["weight_train"]
        self.spectra_beta = self.extra_args["spectra_beta"]
        self.redshift_beta = self.extra_args["redshift_beta"]

        self.save_data = False
        self.shuffle_dataloader = True
        self.load_spectra = extra_args["space_dim"] == 3
        self.dataloader_drop_last = extra_args["dataloader_drop_last"]

        self.pretrain_codebook = extra_args["pretrain_codebook"]
        self.use_all_pixels = extra_args["train_with_all_pixels"]
        self.spectra_n_neighb = extra_args["spectra_neighbour_size"]**2
        self.redshift_classification = extra_args["model_redshift"] and extra_args["redshift_model_method"] == "classification"

        assert self.use_all_pixels and extra_args["train_pixel_ratio"] == 1

        self.summarize_training_tasks()
        self.set_path()

        self.init_net()
        self.init_loss()
        self.init_optimizer()

    #############
    # Initializations
    #############

    def init_net(self):
        if self.pretrain_codebook:
            self.load_pretrained_model()
        log.info(self.pipeline)
        log.info("Total number of parameters: {}".format(
            sum(p.numel() for p in self.pipeline.parameters()))
        )

    def summarize_training_tasks(self):
        tasks = set(self.extra_args["tasks"])

        self.qtz_latent = self.space_dim == 3 and self.extra_args["quantize_latent"]
        self.qtz_spectra = self.space_dim == 3 and self.extra_args["quantize_spectra"]
        assert not (self.qtz_latent and self.qtz_spectra)
        self.qtz = self.qtz_latent or self.qtz_spectra
        self.qtz_strategy = self.extra_args["quantization_strategy"]
        self.cal_codebook_loss = self.qtz_latent and self.qtz_strategy == "hard"
        self.save_qtz_weights = "save_qtz_weights_during_train" in tasks and \
            (self.qtz_spectra or (self.qtz_latent and qtz_strategy == "soft"))

        # sample only pixels with GT spectra
        self.train_spectra_pixels_only = self.extra_args["train_spectra_pixels_only"]
        if self.train_spectra_pixels_only:
            assert self.extra_args["use_full_patch"]

        self.pixel_supervision = self.extra_args["pixel_supervision"]
        self.spectra_supervision = self.space_dim == 3 and \
            self.extra_args["spectra_supervision"]
        assert self.pixel_supervision + self.spectra_supervision >= 1

        self.spectral_inpaint = self.pixel_supervision and \
            self.space_dim == 3 and "spectral_inpaint" in tasks
        self.perform_integration = self.pixel_supervision
        self.trans_sample_method = self.extra_args["trans_sample_method"]

        if self.space_dim == 3 and self.qtz and self.extra_args["model_redshift"]:
            self.apply_gt_redshift = self.extra_args["apply_gt_redshift"]
            self.redshift_unsupervision = self.extra_args["redshift_unsupervision"]
            self.redshift_semi_supervision = self.extra_args["redshift_semi_supervision"]
            # assert self.redshift_semi_supervision
            assert sum([
                self.apply_gt_redshift, self.redshift_unsupervision,
                self.redshift_semi_supervision]) <= 1 # at most one of these three can be True
        else:
            self.apply_gt_redshift, self.redshift_unsupervision, \
                self.redshift_semi_supervision = False, False, False

        self.plot_loss = self.extra_args["plot_loss"]

        # save intensity of intereset for full train img
        self.recon_img = "recon_img_during_train" in tasks
        self.recon_crop = self.pixel_supervision and "recon_crop_during_train" in tasks
        self.save_codebook = self.qtz and \
            "save_codebook_during_train" in tasks
        self.plot_embed_map = self.qtz and \
            "plot_embed_map_during_train" in tasks
        self.save_latents = self.qtz and \
            ("save_latent_during_train" in tasks or "plot_latent_embed" in tasks)
        self.save_redshift =  self.qtz and self.extra_args["model_redshift"] and \
            "save_redshift_during_train" in tasks
        self.save_scaler = self.pixel_supervision and self.qtz and \
            self.extra_args["decode_scaler"] and "save_scaler_during_train" in tasks

        # recon spectra
        self.recon_gt_spectra = self.space_dim == 3 and \
            "recon_gt_spectra_during_train" in tasks
        self.recon_codebook_spectra = self.space_dim == 3 and self.qtz and \
            "recon_codebook_spectra_during_train" in tasks
        self.recon_codebook_spectra_individ = self.space_dim == 3 and self.qtz and \
            "recon_codebook_spectra_individ_during_train" in tasks

        if self.recon_crop:
            # save selected-cropped train image reconstruction
            self.recon_cutout_sizes = self.extra_args["recon_cutout_sizes"]
            self.recon_cutout_fits_uids = self.extra_args["recon_cutout_fits_uids"]
            self.recon_cutout_start_pos = self.extra_args["recon_cutout_start_pos"]

        # if self.recon_gt_spectra:
        # self.selected_spectra_ids = self.dataset.get_spectra_coord_ids()

        if self.spectra_supervision:
            self.num_supervision_spectra_upper_bound = self.extra_args["num_supervision_spectra_upper_bound"]

    def set_path(self):
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        log.info(f"logging to {self.log_dir}")

        for cur_path, cur_pname, in zip(
                ["model_dir","recon_dir","spectra_dir","codebook_spectra_dir",
                 "embed_map_dir","latent_dir","scaler_dir","redshift_dir",
                 "codebook_dir","qtz_weights_dir"],
                ["models","train_recons","train_spectra","train_codebook_spectra",
                 "train_embed_maps","latents","scaler","train_redshift",
                 "codebook","train_qtz_weights"]
        ):
            path = join(self.log_dir, cur_pname)
            setattr(self, cur_path, path)
            Path(path).mkdir(parents=True, exist_ok=True)

        self.grad_fname = join(self.log_dir, "grad.png")

        if self.plot_loss:
            self.loss_fname = join(self.log_dir, "loss")

        if self.pretrain_codebook:
            self.pretrained_model_fname = self.get_checkpoint_fname(
                self.extra_args["pretrain_log_dir"])

        if self.extra_args["resume_train"]:
            self.resume_model_fname = self.get_checkpoint_fname(
                self.extra_args["resume_log_dir"])
            self.resume_loss_fname = join(self.log_dir, "..", exp_dir, "loss.npy")

    def init_loss(self):
        if self.spectra_supervision:
            loss = self.get_loss(self.extra_args["spectra_loss_cho"])
            self.spectra_loss = partial(spectra_supervision_loss, loss)

        if self.redshift_semi_supervision:
            loss = self.get_loss(self.extra_args["redshift_loss_cho"])
            self.redshift_loss = partial(redshift_supervision_loss, loss)

        if self.pixel_supervision:
            loss = self.get_loss(self.extra_args["pixel_loss_cho"])
            if self.spectral_inpaint:
                loss = partial(spectral_masking_loss, loss,
                               self.extra_args["relative_train_bands"],
                               self.extra_args["relative_inpaint_bands"])
            self.pixel_loss = loss

    def init_optimizer(self):
        if self.pretrain_codebook:
            self.init_optimizer_off_pretrain()
        else:
            self.init_optimizer_no_pretrain()

    #############
    # Training logic
    #############

    def begin_train(self, i, tract, patch):
        self.get_cur_patch_data(i, tract, patch)
        self.configure_dataset()
        self.set_num_batches()
        self.init_dataloader()
        self.total_steps = 0

        if self.plot_loss:
            self.losses = []

        if self.extra_args["resume_train"]:
            self.resume_train()

    def train(self):
        self.timer.reset()

        for i, (tract, patch) in enumerate(zip(
                self.extra_args["tracts"], self.extra_args["patches"]
        )):
            self.begin_train(i, tract, patch)
            self.timer.check("train begun for current patch")

            for epoch in range(self.num_epochs + 1):
                self.epoch = epoch
                self.begin_epoch()

                # for batch in tqdm(range(self.num_iterations_cur_epoch)):
                for batch in range(self.num_iterations_cur_epoch):
                    data = self.next_batch()
                    self.timer.check("got data")
                    self.pre_step()
                    self.step(data)
                    self.post_step()
                    self.iteration += 1
                    self.total_steps += 1

                self.end_epoch()
            self.end_train()

    def end_train(self):
        self.writer.close()

        if self.plot_loss:
            x = np.arange(len(self.losses))
            plt.plot(x, self.losses); plt.title("Loss")
            plt.savefig(self.loss_fname + ".png")
            plt.close()

            plt.plot(x, np.log10(self.losses)); plt.title("Log10 loss")
            plt.savefig(self.loss_fname + "_log10.png")
            plt.close()

            np.save(self.loss_fname + ".npy", self.losses)

        if self.extra_args["log_gpu_every"] != -1:
            nvidia_smi.nvmlShutdown()

    #############
    # Epoch begin and end
    #############

    def begin_epoch(self):
        self.iteration = 0
        self.reset_data_iterator()
        self.timer.check("reset data iterator")
        self.pre_epoch()
        self.timer.check("pre_epoch done")
        self.init_log_dict()

    def end_epoch(self):
        self.post_epoch()

        if self.extra_args["valid_every"] > -1 and \
                self.epoch % self.extra_args["valid_every"] == 0 and \
                self.epoch != 0:
            self.validate()
            self.timer.check('validate')

    def reset_data_iterator(self):
        """ Rewind the iterator for the new epoch.
        """
        self.scene_state.optimization.iterations_per_epoch = len(self.train_data_loader)
        self.train_data_loader_iter = iter(self.train_data_loader)

    #############
    # One epoch
    #############

    def pre_epoch(self):
        if self.save_data_every > -1 and self.epoch % self.save_data_every == 0:
            self.save_data = True
            if self.save_scaler: self.scalers = []
            if self.save_latents: self.latents = []
            if self.save_redshift: self.redshift = []
            if self.save_codebook: self.codebook = None
            if self.plot_embed_map: self.embed_ids = []
            if self.save_qtz_weights: self.qtz_weights = []
            if self.recon_img or self.recon_crop: self.recon_pixels = []
            if self.recon_gt_spectra:
                self.recon_wave = []
                self.recon_masks = []
                self.recon_fluxes = []
            if self.recon_codebook_spectra_individ:
                self.codebook_spectra = []

            # re-init dataloader to make sure pixels are in order
            self.use_all_pixels = True
            self.shuffle_dataloader = False
            # self.set_num_batches(max_bsz=512)
            # self.dataset.toggle_wave_sampling(False)

            self.init_dataloader()
            self.reset_data_iterator()

        # note, if we want to save data during training and compare with inferrence data
        # we need to save model in `pre_epoch` (before step) so data save during training
        # is generated using exactly the same model as data generated during inferrence
        # otherwise, we can move save model to `post_epoch`
        if self.save_model_every > -1 and self.epoch % self.save_model_every == 0:
            self.save_model()

        self.pipeline.train()

    def post_epoch(self):
        """ By default, this function logs to Tensorboard, renders images to Tensorboard,
            saves the model, and resamples the dataset.
        """
        self.pipeline.eval()

        total_loss = self.log_dict["total_loss"] / len(self.train_data_loader)
        # print(total_loss, self.log_dict["total_loss"], len(self.train_data_loader))

        if self.plot_loss:
            self.losses.append(total_loss)

        if self.log_tb_every > -1 and self.epoch % self.log_tb_every == 0:
            self.log_tb()

        if self.log_cli_every > -1 and self.epoch % self.log_cli_every == 0:
            self.log_cli()

        if self.render_tb_every > -1 and self.epoch % self.render_tb_every == 0:
            self.render_tb()

        if self.save_data:
            self.save_local()

            # restore trainer state
            self.shuffle_dataloader = True
            self.save_data = False
            self.use_all_pixels = self.extra_args["train_with_all_pixels"]
            # self.set_num_batches(self.extra_args["batch_size"])
            # self.dataset.toggle_wave_sampling(True)

            self.init_dataloader()
            self.reset_data_iterator()

        self.timer.check("post_epoch done")

    #############
    # One step
    #############

    def init_log_dict(self):
        """ Custom log dict.
        """
        super().init_log_dict()
        self.log_dict["recon_loss"] = 0.0
        self.log_dict["spectra_loss"] = 0.0
        self.log_dict["codebook_loss"] = 0.0
        self.log_dict["n_gt_redshift"] = 0
        self.log_dict["redshift_loss"] = 0.0

    def pre_step(self):
        # if self.epoch == 0 and self.extra_args["log_gpu_every"] > -1 \
        #    and self.epoch % self.extra_args["log_gpu_every"] == 0:
        #     gpu_info = get_gpu_info()
        #     free = gpu_info.free / 1e9
        #     used = gpu_info.used / 1e9
        #     log.info(f"Free/Used GPU memory: ~{free}GB / ~{used}GB")
        pass

    def step(self, data):
        """ Advance the training by one step using the batched data supplied.
            @Param:
              data (dict): Dictionary of the input batch from the DataLoader.
        """
        self.optimizer.zero_grad(set_to_none=True)
        self.timer.check("zero grad")

        total_loss, ret = self.calculate_loss(data)
        self.timer.check("loss")

        total_loss.backward()
        if self.epoch != 0 and self.plot_grad_every != -1 \
           and self.plot_grad_every % self.epoch == 0:
            plot_grad_flow(self.pipeline.named_parameters(), self.grad_fname)
        self.optimizer.step()
        self.timer.check("backward and step")

        if self.save_data:
            if self.save_scaler:
                self.scalers.extend(ret["scaler"])
            if self.save_latents:
                self.latents.extend(ret["latents"])
            if self.save_redshift:
                if self.extra_args["redshift_model_method"] == "regression":
                    self.redshift.extend(ret["redshift"])
                else:
                    redshift = torch.sum(ret["redshift"] * ret["redshift_logits"], dim=-1)
                    self.redshift.extend(redshift)
            if self.plot_embed_map:
                self.embed_ids.extend(ret["embed_ids"])
            if self.save_qtz_weights:
                self.qtz_weights.extend(ret["qtz_weights"])
            if self.recon_img or self.recon_crop:
                self.recon_pixels.extend(ret["intensity"])
            if self.save_codebook and self.codebook is None:
                self.codebook = ret["codebook"]
            if self.recon_codebook_spectra_individ:
                self.codebook_spectra.extend(ret["codebook_spectra"])

            if self.recon_gt_spectra:
                if self.spectra_supervision:
                    self.recon_fluxes.append(ret["sup_spectra"])
                    self.recon_masks.append(data["sup_spectra_masks"])
                    self.recon_wave.append(data["sup_spectra_wave"][...,0])
                else:
                    self.recon_fluxes.extend(ret["spectra"])

    def post_step(self):
        pass

    #############
    # Data Helpers
    #############

    def get_cur_patch_data(self, i, tract, patch):
        self.cur_patch = PatchData(
            tract, patch,
            load_spectra=self.load_spectra,
            cutout_num_rows=self.extra_args["patch_cutout_num_rows"][i],
            cutout_num_cols=self.extra_args["patch_cutout_num_cols"][i],
            cutout_start_pos=self.extra_args["patch_cutout_start_pos"][i],
            full_patch=self.extra_args["use_full_patch"],
            spectra_obj=self.dataset.get_spectra_data_obj(),
            **self.extra_args
        )
        self.dataset.set_patch(self.cur_patch)

        self.cur_patch_uid = create_patch_uid(tract, patch)
        if self.load_spectra:
            self.val_spectra_map = self.cur_patch.get_spectra_bin_map()

    def init_dataloader(self):
        """ (Re-)Initialize dataloader.
        """
        if self.shuffle_dataloader: sampler_cls = RandomSampler
        else: sampler_cls = SequentialSampler
        # sampler_cls = SequentialSampler
        # sampler_cls = RandomSampler

        self.train_data_loader = DataLoader(
            self.dataset,
            batch_size=None,
            sampler=BatchSampler(
                sampler_cls(self.dataset),
                batch_size=self.batch_size,
                drop_last=self.dataloader_drop_last
            ),
            pin_memory=True,
            num_workers=self.extra_args["dataset_num_workers"]
        )

    def set_num_batches(self, max_bsz=None):
        """ Set number of batches/iterations and batch size for each epoch.
            At certain epochs, we may not need all data and/or all wave
              and can break before iterating thru all data.
        """
        length = self.get_dataset_length()
        if not self.use_all_pixels:
            length = int(length * self.extra_args["train_pixel_ratio"])

        self.batch_size = min(self.extra_args["batch_size"], length)
        if max_bsz is not None: # when we use all wave, we need to control bsz
            self.batch_size = min(self.batch_size, max_bsz)

        if self.dataloader_drop_last:
            self.num_iterations_cur_epoch = int(length // self.batch_size)
        else:
            self.num_iterations_cur_epoch = int(np.ceil(length / self.batch_size))
        log.info(f"num batches updated to: {self.num_iterations_cur_epoch}.")

    def configure_dataset(self):
        """ Configure dataset with selected fields and set length accordingly.
        """
        fields = ["coords"]

        if self.space_dim == 3:
            fields.append("wave_data")
            self.dataset.set_wave_source("trans")

        if self.pixel_supervision:
            if self.weight_train:
                fields.append("weights")
            if self.train_spectra_pixels_only:
                fields.append("spectra_val_pixels")
            else: fields.append("pixels")

        if self.spectra_supervision:
            fields.append("spectra_data")

        if self.redshift_semi_supervision:
            if self.train_spectra_pixels_only:
                fields.append("spectra_semi_sup_redshift")
            else:
                fields.extend([
                    "spectra_id_map","spectra_bin_map","redshift_data"])

        length = self.get_dataset_length()
        self.dataset.set_length(length)
        self.dataset.set_fields(fields)
        self.dataset.set_mode("main_train")
        self.dataset.toggle_wave_sampling(
            sample_wave=not self.extra_args["train_use_all_wave"]
        )
        self.set_coords()

    def set_coords(self):
        if self.train_spectra_pixels_only:
            pixel_id = get_neighbourhood_center_pixel_id(
                self.extra_args["spectra_neighbour_size"])
            coords = self.dataset.get_validation_spectra_coords()[:,pixel_id:pixel_id+1]
            self.dataset.set_coords_source("spectra_coords")
            self.dataset.set_hardcode_data("spectra_coords", coords)
        else:
            self.dataset.set_coords_source("fits")

    def get_dataset_length(self):
        """ Get length of dataset based on training tasks.
            If we do pixel supervision, we use #coords as length and don't
              count #spectra coords as they are included every batch.
            Otherwise, when we do spectra supervision only, we need to
              set #spectra coords as dataset length.
            (TODO: we assume that #spectra coords is far less
                   than batch size, which may not be the case soon)
        """
        if self.train_spectra_pixels_only:
            length = self.dataset.get_num_validation_spectra()
        else:
            length = self.dataset.get_num_coords()
        return length

    #############
    # Train Helpers
    #############

    def init_optimizer_no_pretrain(self):
        """ Init optimizer in case where no pretraining is utilized.
        """
        params, grid_params, codebook_params, rest_params = [], [], [], []
        params_dict = { name : param for name, param
                        in self.pipeline.named_parameters() }

        for name in params_dict.keys():
            if "spatial_encoder.grid" in name:
                grid_params.append(params_dict[name])
            elif "codebook" in name:
                codebook_params.append(params_dict[name])
            else:
                rest_params.append(params_dict[name])

        params.append({"params": rest_params,
                       "lr": self.extra_args["lr"]})
        params.append({"params": codebook_params,
                       "lr": self.extra_args["codebook_lr"]})
        params.append({"params": grid_params,
                       "lr": self.extra_args["grid_lr"] * self.grid_lr_weight})
        self.optimizer = self.optim_cls(params, **self.optim_params)

        if self.verbose:
            log.info(f"init codebook values {qtz_params}")
            log.info(self.optimizer)

    def init_optimizer_off_pretrain(self):
        """ Optimize parts of the model, freeze other parts shared with pretrain.
        """
        params, grid_params, scaler_params, redshift_params, logit_params = [],[],[],[],[]
        params_dict = { name : param for name, param
                        in self.pipeline.named_parameters() }

        for name in params_dict.keys():
            if "spatial_encoder.grid" in name:
                grid_params.append(params_dict[name])
            elif "scaler_decoder" in name:
                scaler_params.append(params_dict[name])
            elif "redshift_decoder" in name:
                redshift_params.append(params_dict[name])
            elif "spatial_decoder.decode" in name:
                logit_params.append(params_dict[name])
            else: pass

        params.append({"params": grid_params,
                       "lr": self.extra_args["grid_lr"] * self.grid_lr_weight})
        params.append({"params": logit_params, "lr": self.extra_args["lr"]})
        params.append({"params": scaler_params, "lr": self.extra_args["lr"]})
        params.append({"params": redshift_params, "lr": self.extra_args["lr"]})

        self.optimizer = self.optim_cls(params, **self.optim_params)
        if self.verbose:
            log.info(self.optimizer)

    def resume_train(self):
        """ Resume training from saved model.
        """
        try:
            assert(exists(self.resume_model_fname))
            log.info(f"resume model found, loading {self.resume_model_fname}")
            checkpoint = torch.load(self.resume_model_fname)

            if self.plot_loss:
                self.losses = list(np.load(self.resume_loss_fname))

            self.pipeline.load_state_dict(checkpoint["model_state_dict"])
            self.pipeline.eval()

            # if "cuda" in str(self.device):
            #     self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            #     for state in self.optimizer.state.values():
            #         for k, v in state.items():
            #             if torch.is_tensor(v):
            #                 state[k] = v.cuda()
            # else:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if self.verbose: log.info("resume training")

        except Exception as e:
            if self.verbose:
                log.info(e)
                log.info("start training from begining")

    def load_pretrained_model(self):
        """ Load weights from pretrained model.
        """
        try:
            log.info(f"pretrained model found, loading {self.pretrained_model_fname}")

            checkpoint = torch.load(self.pretrained_model_fname)
            load_pretrained_model_weights(
                self.pipeline, checkpoint["model_state_dict"],
                set(self.extra_args["main_train_frozen_layer"])
            )

            self.pipeline.train()
            log.info("loaded pretrained model")

        except Exception as e:
            log.info(e)
            log.info("pretrained model loading FAILED")

    def add_to_device(self, data):
        for field in self.gpu_fields:
            if field in data:
                data[field] = data[field].to(self.device)

    def calculate_loss(self, data):
        total_loss = 0
        add_to_device(data, self.extra_args["gpu_data"], self.device)

        self.timer.reset()
        ret = forward(data,
                      self.pipeline,
                      self.total_steps,
                      self.space_dim,
                      qtz=self.qtz,
                      qtz_strategy=self.qtz_strategy,
                      apply_gt_redshift=self.apply_gt_redshift,
                      perform_integration=self.perform_integration,
                      trans_sample_method=self.trans_sample_method,
                      spectra_supervision=self.spectra_supervision,
                      redshift_classification=self.redshift_classification,
                      save_scaler=self.save_data and self.save_scaler,
                      save_latents=self.save_data and self.save_latents,
                      save_codebook=self.save_data and self.save_codebook,
                      save_redshift=self.redshift_semi_supervision or \
                                    self.save_data and self.save_redshift,
                      save_embed_ids=self.save_data and self.plot_embed_map,
                      save_spectra=self.save_data and self.recon_gt_spectra,
                      save_codebook_loss=self.save_data and self.cal_codebook_loss,
                      save_qtz_weights=self.save_data and self.save_qtz_weights,
                      save_codebook_spectra=self.save_data and \
                                            self.recon_codebook_spectra_individ
        )
        self.timer.check("forward")

        # i) reconstruction loss (taking inpaint into account)
        recon_loss, recon_pixels = 0, None
        if self.pixel_supervision:
            if self.train_spectra_pixels_only:
                gt_pixels = data["spectra_val_pixels"]
            else: gt_pixels = data["pixels"]

            recon_pixels = ret["intensity"]

            if self.extra_args["weight_train"]:
                weights = data["weights"]
                gt_pixels = gt_pixels * weights
                recon_pixels = recon_pixels * weights

            if self.spectral_inpaint:
                mask = data["masks"]
                recon_loss = self.pixel_loss(gt_pixels, recon_pixels, mask)
            else:
                recon_loss = self.pixel_loss(gt_pixels, recon_pixels)
            self.log_dict["recon_loss"] += recon_loss.item()
            self.timer.check("recon loss")

        # ii) spectra supervision loss
        spectra_loss, recon_spectra = 0, None
        if self.spectra_supervision:
            recon_fluxes = ret["sup_spectra"]
            gt_spectra = data["sup_spectra_data"]
            spectra_masks = data["sup_spectra_masks"]

            if len(recon_fluxes) == 0:
                spectra_loss = 0
            else:
                spectra_loss = self.spectra_loss(
                    spectra_masks, gt_spectra, recon_fluxes,
                    weight_by_wave_coverage=self.extra_args["weight_by_wave_coverage"]
                ) * self.spectra_beta
                self.log_dict["spectra_loss"] += spectra_loss.item()

            self.timer.check("spectra loss")

        # iii) redshift loss
        redshift_loss = 0
        if self.redshift_semi_supervision:
            gt_redshift = data["spectra_semi_sup_redshift"]

            if len(gt_redshift) > 0:
                pred_redshift = ret["redshift"]
                if (self.pretrain_codebook or self.redshift_semi_supervision) and \
                   not self.train_spectra_pixels_only:
                    mask= data["spectra_bin_map"]
                else: mask = None

                redshift_loss = self.redshift_loss(
                    gt_redshift, pred_redshift, mask=mask) * self.redshift_beta
                self.log_dict["n_gt_redshift"] += len(gt_redshift)
                self.log_dict["redshift_loss"] += redshift_loss.item()

            self.timer.check("redshift loss")

        # iv) latent quantization codebook loss
        codebook_loss = 0
        if self.qtz_latent and self.qtz_strategy == "hard":
            codebook_loss = ret["codebook_loss"]
            self.log_dict["codebook_loss"] += codebook_loss.item()
            self.timer.check("codebook loss")

        torch.autograd.set_detect_anomaly(True)
        total_loss = redshift_loss + spectra_loss + codebook_loss
        # total_loss = redshift_loss + recon_loss + spectra_loss + codebook_loss
        self.log_dict["total_loss"] += total_loss.item()
        return total_loss, ret

    def log_cli(self):
        """ Controls CLI logging.
            By default, this function only runs every epoch.
        """
        # Average over iterations
        n = len(self.train_data_loader)

        log_text = "EPOCH {}/{}".format(self.epoch, self.num_epochs)
        log_text += " | total loss: {:>.3E}".format(self.log_dict["total_loss"] / n)

        if self.pixel_supervision:
            log_text += " | recon loss: {:>.3E}".format(self.log_dict["recon_loss"] / n)

        if self.spectra_supervision:
            log_text += " | spectra loss: {:>.3E}".format(self.log_dict["spectra_loss"] / n)

        if self.qtz_latent and self.qtz_strategy == "hard":
            log_text += " | codebook loss: {:>.3E}".format(self.log_dict["codebook_loss"] / n)

        if self.redshift_semi_supervision:
            # since we do semi-supervision for redshift, at some batches there may
            #  be no redshift sampled at all
            if self.log_dict["n_gt_redshift"] > 0:
                redshift_loss = self.log_dict["redshift_loss"] / n
                log_text += " | redshift loss: {:>.3E}".format(redshift_loss)
            else:
                log_text += " | redshift loss: no_sample"

        log.info(log_text)

    def save_model(self):
        fname = f"model-ep{self.epoch}-it{self.iteration}.pth"
        model_fname = os.path.join(self.model_dir, fname)

        if self.verbose: log.info(f"Saving model checkpoint to: {model_fname}")

        checkpoint = {
            "iterations": self.total_steps,
            "epoch_trained": self.epoch,
            "model_state_dict": self.pipeline.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }

        torch.save(checkpoint, model_fname)
        return checkpoint

    #############
    # Save helpers
    #############

    def save_local(self):
        if self.save_latents:
            fname = join(self.latent_dir, str(self.epoch))
            self.latents = torch.stack(self.latents).detach().cpu().numpy()
            np.save(fname, self.latents)

        if self.save_scaler: self._save_scaler()
        if self.save_redshift: self._save_redshift()
        if self.plot_embed_map: self._plot_embed_map()
        if self.save_qtz_weights: self.log_qtz_weights()
        if self.recon_gt_spectra: self._recon_gt_spectra()
        if self.recon_img or self.recon_crop:
            self.restore_evaluate_tiles()
        if self.recon_codebook_spectra_individ:
            self._recon_codebook_spectra_individ()

    def _save_scaler(self):
        """ Plot scaler values generated by the decoder before qtz for each pixel.
        """
        scalers = torch.stack(self.scalers).detach().cpu().numpy()
        re_args = {
            "fname": self.epoch,
            "dir": self.scaler_dir,
            "verbose": self.verbose,
            "num_bands": 1,
            "log_max": False,
            "save_locally": True,
            "plot_func": plot_horizontally,
            "zscale": True,
            "to_HDU": False,
            "recon_flat_trans": False,
            "calculate_metrics": False
        }
        _, _ = self.dataset.restore_evaluate_tiles(scalers, **re_args)

        scalers = scalers[self.val_spectra_map]
        np.set_printoptions(suppress=True)
        np.set_printoptions(precision=3)
        log.info(f"scaler: {scalers}")

    def _save_redshift(self):
        redshift = torch.stack(self.redshift).detach().cpu().numpy()
        if self.train_spectra_pixels_only:
            gt_redshift = self.dataset.get_semi_supervision_spectra_redshift().numpy()
        else:
            redshift = redshift[self.val_spectra_map]
            gt_redshift = self.cur_patch.get_spectra_pixel_redshift().numpy()
        np.set_printoptions(suppress=True)
        np.set_printoptions(precision=3)
        log.info(f"est redshift: {redshift}")
        log.info(f"GT. redshift: {gt_redshift}")

    def _plot_embed_map(self):
        """ Plot embed ids and save locally.
            Needs to subtract embed ids for spectra coords, if any.
              (During training, we add extra spectra coords, refer to
               astro_datasampler for details)
        """
        self.embed_ids = torch.stack(self.embed_ids).detach().cpu().numpy()
        re_args = {
            "fname": self.epoch,
            "dir": self.embed_map_dir,
            "verbose": self.verbose,
            "num_bands": 1,
            "log_max": False,
            "save_locally": False,
            "plot_func": plot_embed_map,
            "zscale": False,
            "to_HDU": False,
            "recon_flat_trans": False,
            "calculate_metrics": False
        }
        _, _ = self.dataset.restore_evaluate_tiles(self.embed_ids, **re_args)
        if self.recon_crop:
            self.save_cropped_recon_locally(**re_args)

    def log_qtz_weights(self):
        self.qtz_weights = torch.stack(self.qtz_weights).detach().cpu().numpy()[:,0]
        if not self.train_spectra_pixels_only:
            self.qtz_weights = self.qtz_weights[self.val_spectra_map]
        log.info(f"est qtz weights: {self.qtz_weights}")

    def _recon_gt_spectra(self):
        # if self.spectra_supervision:
        #     num_spectra = self.cur_patch.get_num_spectra()
        #     self.gt_wave = self.cur_patch.get_spectra_pixel_wave()
        #     self.gt_masks = self.cur_patch.get_spectra_pixel_masks()
        #     self.gt_fluxes = self.cur_patch.get_spectra_pixel_fluxes()
        # elif self.train_spectra_pixels_only:
        #     # todo, use spectra within current patch
        #     num_spectra = self.dataset.get_num_validation_spectra()
        #     val_spectra = self.dataset.get_validation_spectra()
        #     self.gt_wave = val_spectra[:,0]
        #     self.gt_fluxes = val_spectra[:,1]
        #     self.gt_masks = self.dataset.get_validation_spectra_masks()
        # else:
        #     num_spectra = self.cur_patch.get_num_spectra()
        #     self.gt_wave = self.cur_patch.get_spectra_pixel_wave()
        #     self.gt_masks = self.cur_patch.get_spectra_pixel_masks()
        #     self.gt_fluxes = self.cur_patch.get_spectra_pixel_fluxes()

        num_spectra = self.cur_patch.get_num_spectra()
        self.gt_wave = self.cur_patch.get_spectra_pixel_wave()
        self.gt_masks = self.cur_patch.get_spectra_pixel_masks()
        self.gt_fluxes = self.cur_patch.get_spectra_pixel_fluxes()

        self.recon_fluxes = torch.stack(self.recon_fluxes)

        if self.spectra_supervision:
            self.recon_wave = torch.stack(self.recon_wave)
            self.recon_masks = torch.stack(self.recon_masks)
            # recon_xxx [num_batches,num_spectra_coords,num_sampeles]
            self.recon_masks = self.recon_masks[0].detach().cpu().numpy()
            # we get all spectra at each batch (duplications), thus average over batches
            self.recon_wave = torch.mean(self.recon_wave, dim=0).detach().cpu().numpy()
            self.recon_fluxes = torch.mean(self.recon_fluxes, dim=0)
        else:
            self.recon_wave = np.tile(
                self.dataset.get_full_wave(), num_spectra).reshape(num_spectra, -1)
            self.recon_masks = np.tile(
                self.dataset.get_full_wave_masks(), num_spectra).reshape(num_spectra, -1)
            # all spectra are collected (no duplications) and we need
            #   only selected ones (incl. neighbours)
            self.recon_fluxes = self.recon_fluxes.view(
                -1, self.recon_fluxes.shape[-1]) # [bsz,num_samples]
            if not self.train_spectra_pixels_only:
                self.recon_fluxes = self.recon_fluxes[self.val_spectra_map]

        self.recon_fluxes = self.recon_fluxes.view(num_spectra, -1).detach().cpu().numpy()

        metrics = self.dataset.plot_spectrum(
            self.spectra_dir, self.epoch,
            self.extra_args["flux_norm_cho"],
            self.gt_wave, self.gt_fluxes,
            self.recon_wave, self.recon_fluxes,
            save_spectra=True,
            gt_masks=self.gt_masks,
            recon_masks=self.recon_masks,
            clip=self.extra_args["plot_clipped_spectrum"]
        )

        if metrics is not None:
            metric_options = metrics[0].keys()
            metrics = np.array([
                [ v for k,v in cur_spectra_metrics.items() ]
                for cur_spectra_metrics in metrics
            ]) # [n_spectra,n_metrics]

            for i, metric_option in enumerate(metric_options):
                cur_metrics = metrics[:,i]
                avg = np.mean(cur_metrics)
                log.info(f"avg_{metric_option}: {np.round(avg, 3)}")
                log.info(f"{metric_option}: {np.round(cur_metrics, 3)}")

    def _recon_codebook_spectra_individ(self):
        self.codebook_spectra = torch.stack(self.codebook_spectra).detach().cpu().numpy()
        if not self.train_spectra_pixels_only:
            self.codebook_spectra = self.codebook_spectra[self.val_spectra_map]
        num_spectra = self.extra_args["qtz_num_embed"]

        recon_wave = np.tile(
            self.dataset.get_full_wave(), num_spectra).reshape(num_spectra, -1)
        recon_masks = np.tile(
            self.dataset.get_full_wave_masks(), num_spectra).reshape(num_spectra, -1)

        for i, codebook_spectra in enumerate(self.codebook_spectra):
            cur_dir = join(self.codebook_spectra_dir, f"spectra-{i}")
            Path(cur_dir).mkdir(parents=True, exist_ok=True)

            fname = f"individ-ep{self.epoch}-it{self.iteration}"
            self.dataset.plot_spectrum(
                cur_dir, fname,
                self.extra_args["flux_norm_cho"],
                None, None,
                recon_wave, codebook_spectra,
                recon_masks=recon_masks,
                is_codebook=True,
                save_spectra_together=True,
                clip=self.extra_args["plot_clipped_spectrum"]
            )

    def restore_evaluate_tiles(self):
        recon_pixels = torch.stack(self.recon_pixels).detach().cpu().numpy()

        if self.train_spectra_pixels_only:
            gt_pixels = self.dataset.get_validation_spectra_pixels().numpy()
            np.save(join(self.recon_dir, f"{self.epoch}_gt.npy"), gt_pixels)
            np.save(join(self.recon_dir, f"{self.epoch}_recon.npy"), recon_pixels)
            np.set_printoptions(suppress=True)
            np.set_printoptions(precision=3)
            log.info(f"GT pixels: {gt_pixels}")
            log.info(f"Recon pixels: {recon_pixels}")
        else:
            re_args = {
                "fname": self.epoch,
                "dir": self.recon_dir,
                "verbose": self.verbose,
                "num_bands": self.extra_args["num_bands"],
                "log_max": True,
                "save_locally": True,
                "plot_func": plot_horizontally,
                "zscale": True,
                "to_HDU": False,
                "recon_flat_trans": False,
                "calculate_metrics": False
            }
            _, _ = self.dataset.restore_evaluate_tiles(recon_pixels, **re_args)

            if self.recon_crop:
                self.save_cropped_recon_locally(**re_args)

    def save_cropped_recon_locally(self, **re_args):
        for index, fits_uid in enumerate(self.extra_args["recon_cutout_fits_uids"]):
            if re_args["zscale_ranges"]:
                zscale_ranges = self.dataset.get_zscale_ranges(fits_uid)

            np_fname = join(re_args["dir"], f"{fits_uid}_{self.epoch}.npy")
            recon = np.load(np_fname) # [nbands,sz,sz]

            for (size, (r,c)) in zip(self.extra_args["recon_cutout_sizes"][index],
                                     self.extra_args["recon_cutout_start_pos"][index]):
                fname = join(re_args["dir"], f"{fits_uid}_{r}_{c}_{size}")
                np.save(fname, recon[:,r:r+size,c:c+size])
                if re_args["zscale"]:
                    plot_horizontally(recon[:,r:r+size,c:c+size], fname, zscale_ranges=zscale_ranges)
                else: re_args["plot_func"](recon[:,r:r+size,c:c+size], fname)

    def get_checkpoint_fname(self, exp_dir):
        """ Format checkpoint fname from given experiemnt directory.
        """
        if exp_dir is not None:
            pretrained_model_dir = join(self.log_dir, "..", exp_dir)
        else:
            # if log dir not specified, use last directory (exclude newly created one)
            dnames = os.listdir(join(self.log_dir, ".."))
            assert(len(dnames) > 1)
            dnames.sort()
            pretrained_model_dir = join(self.log_dir, "..", dnames[-2])

        pretrained_model_dir = join(pretrained_model_dir, "models")

        if self.extra_args["pretrained_model_name"] is not None:
            fname = join(pretrained_model_dir, self.extra_args["pretrained_model_name"])
        else:
            fnames = os.listdir(pretrained_model_dir)
            assert(len(fnames) > 0)
            fnames = sort_alphanumeric(fnames)
            fname = join(pretrained_model_dir, fnames[-1])
        return fname

    def log_codebook(self):
        if not self.verbose: return
        for n,p in self.pipeline.named_parameters():
            if "grid" not in n and "codebook" in n: print(p);break

    def validate(self):
        pass
