# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.  #
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import os
import time
import torch
import shutil
import nvidia_smi
import numpy as np
import torch.nn as nn
import logging as log
import matplotlib.pyplot as plt

from pathlib import Path
from functools import partial
from os.path import exists, join
from torch.utils.data import BatchSampler, SequentialSampler, \
    RandomSampler, DataLoader

from wisp.datasets import default_collate
from wisp.loss import spectra_supervision_loss
from wisp.utils.plot import plot_horizontally, plot_embed_map
from wisp.loss import spectra_supervision_loss, spectral_masking_loss
from wisp.trainers import BaseTrainer, log_metric_to_wandb, log_images_to_wandb
from wisp.utils.common import get_gpu_info, add_to_device, sorted_nicely, forward


class AstroTrainer(BaseTrainer):
    """ Trainer class for astro dataset.
    """
    def __init__(self, pipeline, dataset, num_epochs, batch_size,
                 optim_cls, lr, weight_decay, grid_lr_weight, optim_params, log_dir, device,
                 exp_name=None, info=None, scene_state=None, extra_args=None,
                 render_tb_every=-1, save_every=-1, using_wandb=False):

        self.shuffle_dataloader = True
        self.dataloader_drop_last = extra_args["dataloader_drop_last"]

        super().__init__(pipeline, dataset, num_epochs, batch_size, optim_cls,
                         lr, weight_decay, grid_lr_weight, optim_params, log_dir,
                         device, exp_name, info, scene_state, extra_args,
                         render_tb_every, save_every, using_wandb)

        # save config file to log directory
        dst = join(self.log_dir, "config.yaml")
        shutil.copyfile(extra_args["config"], dst)

        self.cuda = "cuda" in str(self.device)
        self.verbose = self.extra_args["verbose"]
        self.space_dim = self.extra_args["space_dim"]
        self.gpu_fields = self.extra_args["gpu_data"]
        self.weight_train = self.extra_args["weight_train"]

        self.use_all_pixels = True
        self.set_num_batches()

        self.set_log_path()
        self.summarize_training_tasks()

        self.configure_dataset()
        self.init_loss()
        self.init_dataloader()

    #############
    # Initializations
    #############

    def configure_dataset(self):
        """ Configure dataset with selected fields and set length accordingly.
        """
        fields = ["coords","pixels"]
        if self.weight_train:
            fields.append("weights")
        if self.space_dim == 3:
            fields.append("trans_data")
        if self.spectral_inpaint:
            pass
        if self.spectra_supervision:
            fields.append("spectra_supervision_data")

        length = self.dataset.get_num_coords()

        self.dataset.set_dataset_mode("train")
        self.dataset.set_dataset_length(length)
        self.dataset.set_dataset_fields(fields)
        self.dataset.set_dataset_coords_source("fits")

    def summarize_training_tasks(self):
        tasks = set(self.extra_args["tasks"])

        self.save_recon = "save_recon_during_train" in tasks
        self.spectral_inpaint = self.space_dim == 3 and "spectral_inpaint" in tasks
        self.plot_spectra = self.space_dim == 3 and "plot_spectra_during_train" in tasks
        self.spectra_supervision = self.space_dim == 3 and "spectra_supervision" in tasks
        self.quantize_latent = self.space_dim == 3 and self.extra_args["quantize_latent"]
        self.plot_embed_map = self.quantize_latent and "plot_embed_map_during_train" in tasks
        self.save_latents =  self.quantize_latent and ("save_latent_during_train" in tasks or "plot_latent_embed" in tasks)

        if self.save_recon:
            # save selected-cropped train image reconstruction
            self.recon_cutout_fits_ids = self.extra_args["recon_cutout_fits_ids"]
            self.recon_cutout_sizes = self.extra_args["recon_cutout_sizes"]
            self.recon_cutout_start_pos = self.extra_args["recon_cutout_start_pos"]
            self.save_cropped_recon = self.recon_cutout_fits_ids is not None and \
                self.recon_cutout_sizes is not None and self.recon_cutout_start_pos is not None
        else: self.save_cropped_recon = False

        if self.plot_spectra:
            self.selected_spectra_ids = self.dataset.get_spectra_coord_ids()

        if self.spectra_supervision:
            self.num_supervision_spectra = self.extra_args["num_supervision_spectra"]

    def set_log_path(self):
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        if self.verbose: log.info(f"logging to {self.log_dir}")

        for cur_path, cur_pname, in zip(
                ["model_dir","recon_dir","spectra_dir","embed_map_dir","latent_dir"],
                ["models","train_recons","train_spectra","train_embed_maps","latents"]):
            path = join(self.log_dir, cur_pname)
            setattr(self, cur_path, path)
            Path(path).mkdir(parents=True, exist_ok=True)

        if self.extra_args["resume_train"]:
            if self.extra_args["resume_log_dir"] is not None:
                pretrained_model_dir = join(self.log_dir, "..", self.extra_args["resume_log_dir"])
            else:
                # if log dir not specified, use last directory (exclude newly created one)
                dnames = os.listdir(join(self.log_dir, ".."))
                assert(len(dnames) > 1)
                dnames.sort()
                pretrained_model_dir = join(self.log_dir, "..", dnames[-2])

            pretrained_model_dir = join(pretrained_model_dir, "models")

            if self.extra_args["pretrained_model_name"] is not None:
                self.pretrained_model_fname = join(
                    pretrained_model_dir, self.extra_args["pretrained_model_name"])
            else:
                fnames = os.listdir(pretrained_model_dir)
                assert(len(fnames) > 0)
                fnames = sorted_nicely(fnames)
                self.pretrained_model_fname = join(pretrained_model_dir, fnames[-1])

    def init_loss(self):
        cho = self.extra_args["loss_cho"]
        if cho == "l1":
            loss = nn.L1Loss() if not self.cuda else nn.L1Loss().cuda()
        elif cho == "l2":
            loss = nn.MSELoss() if not self.cuda else nn.MSELoss().cuda()
        else: raise Exception("Unsupported loss choice")

        if self.spectra_supervision:
            self.spectra_loss = partial(spectra_supervision_loss, loss)

        if self.spectral_inpaint:
            loss = partial(spectral_masking_loss, loss,
                           self.extra_args["relative_train_bands"],
                           self.extra_args["relative_inpaint_bands"])
        self.loss = loss

    def init_dataloader(self):
        """ (Re-)Initialize dataloader.
        """
        if self.shuffle_dataloader: sampler_cls = RandomSampler
        else: sampler_cls = SequentialSampler

        self.train_data_loader = DataLoader(
            self.dataset,
            batch_size=None,
            #collate_fn=default_collate,
            sampler=BatchSampler(
                sampler_cls(self.dataset), batch_size=self.batch_size, drop_last=self.dataloader_drop_last),
            #pin_memory=True,
            num_workers=0
        )

    def init_optimizer(self):
        params_dict = { name : param for name, param
                        in self.pipeline.named_parameters() }
        params = []
        hps_params, decoder_params, grid_params, rest_params = [],[],[],[]

        # for name in params_dict:
        #     if "hyper_decod" in name:
        #         hps_params.append(params_dict[name])
        #     elif "decoder" in name:
        #         decoder_params.append(params_dict[name])
        #     elif "grid" in name:
        #         grid_params.append(params_dict[name])
        #     else:
        #         rest_params.append(params_dict[name])

        # params.append({"params": hps_params,
        #                "lr": self.hps_lr})
        # params.append({"params" : decoder_params,
        #                "lr": self.lr,
        #                "weight_decay": self.weight_decay})
        # params.append({"params" : grid_params,
        #                "lr": self.lr * self.grid_lr_weight})
        # params.append({"params" : rest_params,
        #                "lr": self.lr})

        for name in params_dict:
            if "grid" in name:
                grid_params.append(params_dict[name])
            else:
                rest_params.append(params_dict[name])

        params.append({"params" : grid_params,
                       "lr": self.lr * self.grid_lr_weight})
        params.append({"params" : rest_params,
                       "lr": self.extra_args["hps_lr"]})
        self.optimizer = self.optim_cls(params, **self.optim_params)

        print(self.optimizer)

    #############
    # training iterations
    #############

    def train(self):
        if self.extra_args["resume_train"]:
            self.resume_train()

        super().train()

        if self.extra_args["log_gpu_every"] != -1:
            nvidia_smi.nvmlShutdown()

    def iterate(self):
        """ Advances the training by one training step (batch).
        """
        if self.scene_state.optimization.running:
            iter_start_time = time.time()
            self.scene_state.optimization.iteration = self.iteration
            try:
                if self.train_data_loader_iter is None:
                    self.begin_epoch()

                # we may not need to iterate thru all data at certain epochs
                if self.iteration > self.num_iterations_cur_epoch:
                    raise StopIteration

                data = self.next_batch()
                self.iteration += 1

            except StopIteration:
                self.end_epoch()
                self.begin_epoch()
                data = self.next_batch()

            self.pre_step()
            self.step(data)
            self.post_step()
            iter_end_time = time.time()
            self.scene_state.optimization.elapsed_time += iter_end_time - iter_start_time

    #############
    # One epoch
    #############

    def pre_epoch(self):
        self.loss_lods = list(range(0, self.extra_args["num_lods"]))

        if self.extra_args["grow_every"] > 0:
            self.grow()

        if self.extra_args["only_last"]:
            self.loss_lods = self.loss_lods[-1:]

        if self.extra_args["resample"] and self.epoch % self.extra_args["resample_every"] == 0:
            self.resample_dataset()

        if self.extra_args["save_local_every"] > -1 and self.epoch % self.extra_args["save_local_every"] == 0:
            self.save_data_to_local = True
            if self.save_latents: self.latents = []
            if self.plot_embed_map: self.embed_ids = []
            if self.plot_spectra: self.smpl_spectra = []
            if self.save_recon or self.save_cropped_recon: self.smpl_pixels = []

            # re-init dataloader to make sure pixels are in order
            self.shuffle_dataloader = False
            self.use_all_pixels = True
            self.set_num_batches()
            self.init_dataloader()
            self.reset_data_iterator()

        self.pipeline.train()
        self.timer.check("pre_epoch done")

    def post_epoch(self):
        """ By default, this function logs to Tensorboard, renders images to Tensorboard,
            saves the model, and resamples the dataset.
        """
        self.pipeline.eval()

        total_loss = self.log_dict["total_loss"] / len(self.train_data_loader)
        self.scene_state.optimization.losses["total_loss"].append(total_loss)

        # log to cli and tensorboard
        if self.extra_args["log_tb_every"] > -1 and self.epoch % self.extra_args["log_tb_every"] == 0:
            self.log_cli(); self.log_tb()

        # render visualizations to tensorboard
        if self.render_tb_every > -1 and self.epoch % self.render_tb_every == 0:
            self.render_tb()

        # save data locally and restore trainer state
        if self.save_data_to_local:
            self.save_local()
            self.use_all_pixels = False
            self.shuffle_dataloader = True
            self.save_data_to_local = False
            self.set_num_batches()
            self.init_dataloader()
            self.reset_data_iterator()

        # save model
        if self.save_every > -1 and self.epoch % self.save_every == 0:
            self.save_model()

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

        with torch.cuda.amp.autocast():
            total_loss, recon_pixels, ret = self.calculate_loss(data)

        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.timer.check("backward and step")

        #plot_grad_flow(self.pipeline.named_parameters(), self.grad_fname)
        if self.save_data_to_local:
            recon_spectra, embed_ids, latents = self.get_data_to_save(ret)
            if self.save_latents: self.latents.extend(latents)
            if self.plot_embed_map: self.embed_ids.extend(embed_ids)
            if self.plot_spectra: self.smpl_spectra.append(recon_spectra)
            if self.save_recon or self.save_cropped_recon:
                self.smpl_pixels.extend(recon_pixels)

    def post_step(self):
        pass

    #############
    # Helper methods
    #############

    def set_num_batches(self):
        """ Set number of batches/iterations and batch size for each epoch.
            At certain epochs, we may not need all data and can break before
              iterating thru all data.
        """
        length = self.dataset.get_num_coords()
        if not self.use_all_pixels:
            length = int(length * self.extra_args["train_pixel_ratio"])

        self.batch_size = min(self.extra_args["batch_size"], length)

        if self.dataloader_drop_last:
            self.num_iterations_cur_epoch = int(length // self.batch_size)
        else:
            self.num_iterations_cur_epoch = int(np.ceil(length / self.batch_size))

    def resume_train(self):
        try:
            assert(exists(self.pretrained_model_fname))
            if self.verbose:
                log.info(f"saved model found, loading {self.pretrained_model_fname}")
            checkpoint = torch.load(self.pretrained_model_fname)

            self.pipeline.load_state_dict(checkpoint["model_state_dict"])
            self.pipeline.eval()

            if "cuda" in str(self.device):
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()
            else:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if self.verbose: log.info("resume training")

        except Exception as e:
            if self.verbose:
                log.info(e)
                log.info("start training from begining")

    def add_to_device(self, data):
        for field in self.gpu_fields:
            if field in data:
                data[field] = data[field].to(self.device)

    def calculate_loss(self, data):
        total_loss = 0
        add_to_device(data, self.extra_args["gpu_data"], self.device)

        ret = forward(self, self.pipeline, data,
                      quantize_latent=self.quantize_latent,
                      calculate_codebook_loss=self.quantize_latent,
                      spectra_supervision_train=self.spectra_supervision,
                      save_spectra=self.save_data_to_local and self.plot_spectra,
                      save_latents=self.save_data_to_local and self.save_latents,
                      save_embed_ids=self.save_data_to_local and self.plot_embed_map)

        recon_pixels = ret["intensity"]
        gt_pixels = data["pixels"]

        if self.extra_args["weight_train"]:
            weights = data["weights"]
            gt_pixels *= weights
            recon_pixels *= weights

        # i) reconstruction loss (taking inpaint into account)
        if self.spectral_inpaint:
            mask = data["masks"]
            recon_loss = self.loss(gt_pixels, recon_pixels, mask)
        else:
            recon_loss = self.loss(gt_pixels, recon_pixels)
        self.log_dict["recon_loss"] += recon_loss.item()

        # ii) spectra loss
        spectra_loss, recon_spectra = 0, None
        if self.spectra_supervision:
            gt_spectra = data["gt_spectra"]

            # todo: efficiently slice spectra with different bound
            (lo, hi) = data["recon_wave_bound_ids"][0]
            recon_spectra = ret["spectra"][:self.num_supervision_spectra,lo:hi,0]

            spectra_loss = self.spectra_loss(gt_spectra, recon_spectra)
            self.log_dict["spectra_loss"] += spectra_loss.item()

        # iii) latent quantization codebook loss
        if self.quantize_latent:
            cdbk_loss = ret["codebook_loss"]
            self.log_dict["codebook_loss"] += cdbk_loss.item()
        else: cdbk_loss = 0

        total_loss = recon_loss + spectra_loss + cdbk_loss
        self.log_dict["total_loss"] += total_loss.item()
        self.timer.check("loss")
        return total_loss, recon_pixels, ret

    def log_cli(self):
        """ Controls CLI logging.
            By default, this function only runs every epoch.
        """
        # Average over iterations
        n = len(self.train_data_loader)

        log_text = "EPOCH {}/{}".format(self.epoch, self.num_epochs)
        log_text += " | total loss: {:>.3E}".format(self.log_dict["total_loss"] / n)
        log_text += " | recon loss: {:>.3E}".format(self.log_dict["recon_loss"] / n)
        if self.quantize_latent:
            log_text += " | codebook loss: {:>.3E}".format(self.log_dict["codebook_loss"] / n)
        if self.spectra_supervision:
            log_text += " | spectra loss: {:>.3E}".format(self.log_dict["spectra_loss"] / n)
        log.info(log_text)

    def save_model(self):
        if self.extra_args["save_as_new"]:
            fname = f"model-ep{self.epoch}-it{self.iteration}.pth"
            model_fname = os.path.join(self.model_dir, fname)
        else: model_fname = os.path.join(self.model_dir, f"model.pth")

        if self.verbose: log.info(f"Saving model checkpoint to: {model_fname}")

        checkpoint = {
            "epoch_trained": self.epoch,
            "model_state_dict": self.pipeline.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }
        torch.save(checkpoint, model_fname)
        return checkpoint

    def get_data_to_save(self, ret):
        latents = None if not self.save_latents else ret["latents_to_save"]
        embed_ids = None if not self.plot_embed_map else ret["min_embed_ids"]
        recon_spectra = None if not self.plot_spectra else ret["spectra"][...,0]
        return recon_spectra, embed_ids, latents

    def save_local(self):
        if self.save_latents:
            fname = join(self.latent_dir, str(self.epoch))
            self.latents = torch.stack(self.latents).detach().cpu().numpy()
            np.save(fname, self.latents)

        if self.plot_embed_map:
            self.plot_save_embed_map()

        if self.save_recon or self.save_cropped_recon:
            self.restore_evaluate_tiles()

        if self.plot_spectra:
            self.plot_spectrum()

    def plot_save_embed_map(self):
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
        if self.save_cropped_recon:
            self.save_cropped_recon_locally(**re_args)

    def restore_evaluate_tiles(self):
        self.smpl_pixels = torch.stack(self.smpl_pixels).detach().cpu().numpy()
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
        _, _ = self.dataset.restore_evaluate_tiles(self.smpl_pixels, **re_args)
        if self.save_cropped_recon:
            self.save_cropped_recon_locally(**re_args)

    def plot_spectrum(self):
        self.smpl_spectra = torch.stack(self.smpl_spectra)

        if not self.spectra_supervision:
            # smpl_spectra [bsz,num_samples]
            # all spectra are collected (no duplications) and we need
            #   only selected ones (incl. neighbours)
            self.smpl_spectra = self.smpl_spectra[self.selected_spectra_ids]
        else:
            # smpl_spectra [bsz,num_spectra_coords,num_sampeles]
            # we get all spectra at each batch (duplications), thus average over batches
            self.smpl_spectra = torch.mean(self.smpl_spectra, dim=0)

        self.smpl_spectra = self.smpl_spectra.detach().cpu().numpy().reshape((
            self.dataset.get_num_gt_spectra(),
            self.extra_args["spectra_neighbour_size"]**2, -1))

        self.dataset.plot_spectrum(self.spectra_dir, self.epoch, self.smpl_spectra)

    def save_cropped_recon_locally(self, **re_args):
        for i, fits_id in enumerate(self.extra_args["recon_cutout_fits_ids"]):
            if re_args["zscale_ranges"]:
                zscale_ranges = self.dataset.get_zscale_ranges(fits_id)

            np_fname = join(re_args["dir"], f"{fits_id}_{self.epoch}.npy")
            recon = np.load(np_fname) # [nbands,sz,sz]

            for (size, (r,c)) in zip(self.extra_args["recon_cutout_sizes"][i],
                                     self.extra_args["recon_cutout_start_pos"][i]):
                fname = join(re_args["dir"], f"{fits_id}_{r}_{c}_{size}")
                np.save(fname, recon[:,r:r+size,c:c+size])
                if re_args["zscale"]:
                    plot_horizontally(recon[:,r:r+size,c:c+size], fname, zscale_ranges=zscale_ranges)
                else: re_args["plot_func"](recon[:,r:r+size,c:c+size], fname)

    def validate(self):
        pass
