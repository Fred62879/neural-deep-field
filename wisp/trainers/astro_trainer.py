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

from pathlib import Path
from functools import partial
from os.path import exists, join
from torch.utils.data import BatchSampler, SequentialSampler, \
    RandomSampler, DataLoader

from wisp.datasets import default_collate
from wisp.loss import spectra_supervision_loss
from wisp.utils.common import get_gpu_info, forward
from wisp.utils.plot import plot_gt_recon, plot_horizontally
from wisp.loss import spectra_supervision_loss, spectral_masking_loss
from wisp.trainers import BaseTrainer, log_metric_to_wandb, log_images_to_wandb


class AstroTrainer(BaseTrainer):
    """ Trainer class for astro dataset.
        The default overall flow:

        init()
        |- set_renderer()
        |- set_logger()

        train():
            for every epoch:
                pre_epoch()

                iterate()
                    pre_step()
                    step()
                    post_step()

                post_epoch()
                |- log_tb()
                |- save_model()
                |- render_tb()
                |- resample_dataset()

        validate()
    """

    def __init__(self, pipeline, dataset, num_epochs, batch_size,
                 optim_cls, lr, weight_decay, grid_lr_weight, optim_params, log_dir, device,
                 exp_name=None, info=None, scene_state=None, extra_args=None,
                 render_tb_every=-1, save_every=-1, using_wandb=False):

        self.use_all_pixels = False
        self.hps_lr = extra_args["hps_lr"]
        self.shuffle_dataloader = True # pre-epoch requires this

        super().__init__(pipeline, dataset, num_epochs, batch_size, optim_cls,
                         lr, weight_decay, grid_lr_weight, optim_params, log_dir,
                         device, exp_name, info, scene_state, extra_args,
                         render_tb_every, save_every, using_wandb)

        # save config file to log directory
        dst = join(self.log_dir, "config.yaml")
        shutil.copyfile(extra_args["config"], dst)

        self.set_training_mechanism()
        self.configure_dataset()
        self.set_log_path()
        self.init_loss()
        if self.extra_args["resume_train"]:
            self.resume_train()

        self.num_batches = int(np.ceil(len(self.dataset) / batch_size))
        if self.verbose:
            log.info(f"{self.num_batches} batches per epoch")

    #############
    # Initializations
    #############

    def set_training_mechanism(self):
        """ Set training mechanism. """
        tasks = set(self.extra_args["tasks"])

        self.verbose = self.extra_args["verbose"]
        self.space_dim = self.extra_args["space_dim"]
        self.cuda = "cuda" in str(self.device)
        self.weight_train = self.extra_args["weight_train"]

        self.spectral_inpaint = self.space_dim == 3 and 'spectral_inpaint' in tasks
        self.spectra_supervision = self.space_dim == 3 and "recon_gt_spectra_w_supervision" in tasks

        # save all train image reconstruction in original size
        self.save_recon = "save_recon_during_train" in tasks
        if self.save_recon:
            # save selected-cropped train image reconstruction
            self.recon_cutout_fits_ids = self.extra_args["recon_cutout_fits_ids"]
            self.recon_cutout_sizes = self.extra_args["recon_cutout_sizes"]
            self.recon_cutout_start_pos = self.extra_args["recon_cutout_start_pos"]
            self.save_cropped_recon = self.recon_cutout_fits_ids is not None and \
                self.recon_cutout_sizes is not None and self.recon_cutout_start_pos is not None
        else: self.save_cropped_recon = False

        # latent quantization
        self.quantize_latent = self.extra_args["quantize_latent"]
        if self.quantize_latent:
            self.plot_embed_map = "plot_embed_map_during_train" in tasks
            self.save_latent =  "save_latent_during_train" in tasks or "plot_latent_embed" in task
        else:
            self.plot_embed_map, self.save_latent = False, False

    def configure_dataset(self):
        """ Configure dataset with selected fields and set length accordingly. """
        fields = ['coords','pixels']
        if self.weight_train:
            fields.append('weights')
        if self.space_dim == 3:
            fields.append('trans_data')
        if self.spectral_inpaint:
            pass
        if self.spectra_supervision:
            fields.append("gt_spectra_data_w_supervision")

        length = self.dataset.get_num_coords()

        self.dataset.set_state("fits")
        self.dataset.set_dataset_length(length)
        self.dataset.set_dataset_fields(fields)

    def set_log_path(self):
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        if self.verbose: log.info(f'logging to {self.log_dir}')

        for cur_path, cur_pname, in zip(['recon_dir','model_dir'], ['train_recons','models']):
            path = join(self.log_dir, cur_pname)
            setattr(self, cur_path, path)
            Path(path).mkdir(parents=True, exist_ok=True)

        self.grad_fn = join(self.log_dir, 'gradient.png')
        self.train_loss_fn = join(self.log_dir, 'loss.npy')
        self.embed_map_fn = join(self.log_dir, 'embed_map.png')

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
                fnames.sort()
                self.pretrained_model_fname = join(pretrained_model_dir, fnames[-1])

    def init_loss(self):
        cho = self.extra_args['loss_cho']
        if cho == 'l1':
            loss = nn.L1Loss() if not self.cuda else nn.L1Loss().cuda()
        elif cho == 'l2':
            loss = nn.MSELoss() if not self.cuda else nn.MSELoss().cuda()
        else: raise Exception('Unsupported loss choice')

        if self.spectra_supervision:
            self.spectra_loss = partial(spectra_supervision_loss, loss)

        if self.spectral_inpaint:
            loss = partial(spectral_masking_loss, loss,
                           self.extra_args['relative_train_bands'],
                           self.extra_args['relative_inpaint_bands'])
        self.loss = loss

    #############
    # Train loops
    #############

    def train(self):
        super().train()
        if self.extra_args["log_gpu_every"] != -1:
            nvidia_smi.nvmlShutdown()

    def iterate(self):
        """ Advances the training by one training step (batch). """
        if self.scene_state.optimization.running:
            iter_start_time = time.time()
            self.scene_state.optimization.iteration = self.iteration
            try:
                if self.train_data_loader_iter is None:
                    self.begin_epoch()
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

    def next_batch(self):
        """ Actually iterate the data loader. """
        return next(self.train_data_loader_iter)

    #############
    # begin epoch
    #############

    def begin_epoch(self):
        self.reset_data_iterator()
        self.pre_epoch()
        self.init_log_dict()

    def reset_data_iterator(self):
        """ Rewind the iterator for the new epoch """
        self.scene_state.optimization.iterations_per_epoch = len(self.train_data_loader)
        self.train_data_loader_iter = iter(self.train_data_loader)

    def pre_epoch(self):
        self.loss_lods = list(range(0, self.extra_args["num_lods"]))

        if self.extra_args["grow_every"] > 0:
            self.grow()

        if self.extra_args["only_last"]:
            self.loss_lods = self.loss_lods[-1:]

        if self.extra_args["resample"] and self.epoch % self.extra_args["resample_every"] == 0:
            self.resample_dataset()

        if self.extra_args["save_local_every"] > -1 and self.epoch % self.extra_args["save_local_every"] == 0:
            #if self.epoch == 0 or (self.epoch + 1) % args.loss_smpl_intvl == 0
            self.save_data_to_local = True
            self.latents, self.embed_ids, self.smpl_pixels = [], [], []

            # re-init dataloader to make sure pixels are in order
            self.shuffle_dataloader = False
            self.use_all_pixels = True
            self.init_dataloader()

        self.pipeline.train()
        self.timer.check("pre_epoch done")

    def init_log_dict(self):
        """ Custom log dict. """
        super().init_log_dict()
        self.log_dict["recon_loss"] = 0.0
        self.log_dict["spectra_loss"] = 0.0
        self.log_dict["codebook_loss"] = 0.0

    def init_dataloader(self):
        """ (Re-)Initialize dataloader.
            When need to save data locally, use all coords in original order.
            Otherwise, randomly select 10% coords.
        """
        length = self.dataset.get_num_coords()
        if self.use_all_pixels:
            self.dataset.set_dataset_length(length)
        else:
            self.dataset.set_dataset_length(int(length*0.1))

        if self.shuffle_dataloader: sampler_cls = RandomSampler
        else: sampler_cls = SequentialSampler

        self.train_data_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            #collate_fn=default_collate,
            sampler=BatchSampler(
                sampler_cls(self.dataset), batch_size=self.batch_size, drop_last=False),
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
                       "lr": self.hps_lr})
        self.optimizer = self.optim_cls(params, **self.optim_params)
        print(self.optimizer)

    #############
    # end epoch
    #############

    def end_epoch(self):
        self.post_epoch()

        if self.extra_args["valid_every"] > -1 and \
           self.epoch % self.extra_args["valid_every"] == 0 and \
           self.epoch != 0:
            self.validate()
            self.timer.check("validate")

        if self.epoch < self.num_epochs:
            self.epoch += 1
        else:
            self.scene_state.optimization.running = False

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

        if self.save_data_to_local: # set in pre-epoch
            self.save_local()
            self.use_all_pixels = False
            self.shuffle_dataloader = True
            self.save_data_to_local = False

        # save model
        if self.save_every > -1 and self.epoch % self.save_every == 0: # and self.epoch != 0:
            self.save_model()

        self.timer.check("post_epoch done")

    #############
    # Core training step
    #############

    def pre_step(self):
        """ Sample lambda (and transmission) at the start of each iteration.
            Append to current batch data.
            @Param
              data: current batch data
        """
        # if self.epoch == 0 and self.extra_args["log_gpu_every"] > -1 \
        #    and self.epoch % self.extra_args["log_gpu_every"] == 0:
        #     gpu_info = get_gpu_info()
        #     free = gpu_info.free / 1e9
        #     used = gpu_info.used / 1e9
        #     log.info(f"Free/Used GPU memory: ~{free}GB / ~{used}GB")
        pass

    def calculate_loss(self, data):
        total_loss = 0

        ret = forward(self, self.pipeline, data, self.quantize_latent,
                      self.plot_embed_map, self.spectra_supervision)
        recon_pixels = ret["intensity"]
        gt_pixels = data["pixels"][0] #.to(self.device)

        if self.extra_args["weight_train"]:
            weights = data["weights"] #.to(self.device)
            gt_pixels *= weights
            recon_pixels *= weights

        # i) reconstruction loss (taking inpaint into account)
        if self.spectral_inpaint:
            mask = data["cur_mask"].to(self.device)
            recon_loss = self.loss(gt_pixels, recon_pixels, mask)
        else:
            recon_loss = self.loss(gt_pixels, recon_pixels)
        self.log_dict["recon_loss"] += recon_loss.item()

        # ii) spectra loss
        if self.spectra_supervision:
            (lo, hi) = data["trusted_wave_range_id"]
            recon_spectra = ret["spectra"][:,lo:hi]
            spectra_loss = self.spectra_loss(data["gt_spectra"], recon_spectra)
            self.log_dict["spectra_loss"] += spectra_loss.item()
        else:
            spectra_loss, recon_spectra = 0, None

        # iii) latent quantization codebook loss
        if self.quantize_latent:
            cdbk_loss = ret["codebook_loss"]
            self.log_dict["codebook_loss"] += cdbk_loss.item()
        else: cdbk_loss = 0

        total_loss = recon_loss + spectra_loss + cdbk_loss
        self.log_dict["total_loss"] += total_loss.item()

        embed_ids, latents = None, None
        if self.quantize_latent:
            if self.plot_embed_map:
                embed_ids = ret["embed_ids"]
            if self.save_latent:
                latents = ret["latents"]

        self.timer.check("loss")
        return total_loss, recon_pixels, recon_spectra, embed_ids, latents

    def step(self, data):
        """ Advance the training by one step using the batched data supplied.
            @Param:
              data (dict): Dictionary of the input batch from the DataLoader.
        """
        self.optimizer.zero_grad(set_to_none=True)
        self.timer.check("zero grad")

        with torch.cuda.amp.autocast():
            total_loss, recon_pixels, recon_spectra, embed_ids, latents = self.calculate_loss(data)

        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.timer.check("backward and step")

        #plot_grad_flow(self.model.named_parameters(), self.args.grad_fn)
        if self.save_data_to_local:
            self.latents = latents
            self.embed_ids = embed_ids
            self.recon_pixels = recon_pixels

    def post_step(self):
        if self.save_latent:
            self.latents.extend(self.latents.detach().cpu().numpy())

        if self.plot_embed_map:
            self.embed_ids.extend(self.embed_ids.detach().cpu().numpy())

        if self.save_recon or self.save_cropped_recon:
            self.smpl_pixels.extend(self.recon_pixels) #.detach().cpu().numpy())

    #############
    # Helper methods
    #############

    def resume_train(self):
        try:
            assert(exists(self.pretrained_model_fname))
            if self.verbose:
                log.info(f'saved model found, loading {self.pretrained_model_fname}')
            checkpoint = torch.load(self.pretrained_model_fname)

            self.pipeline.load_state_dict(checkpoint['model_state_dict'])
            self.pipeline.eval()

            if "cuda" in str(self.device):
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()
            else:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if self.verbose: log.info("resume training")

        except Exception as e:
            if self.verbose:
                log.info(e)
                log.info("start training from begining")

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

    def log_tb(self):
        for key in self.log_dict:
            if "loss" in key:
                self.writer.add_scalar(f"Loss/{key}", self.log_dict[key] / len(self.train_data_loader), self.epoch)
                if self.using_wandb:
                    log_metric_to_wandb(f"Loss/{key}", self.log_dict[key] / len(self.train_data_loader), self.epoch)

    def render_tb(self):
        self.pipeline.eval()
        for d in [self.extra_args["num_lods"] - 1]:
            out = self.renderer.shade_images(
                self.pipeline, f=self.extra_args["camera_origin"],
                t=self.extra_args["camera_lookat"], fov=self.extra_args["camera_fov"],
                lod_idx=d, camera_clamp=self.extra_args["camera_clamp"])

            # Premultiply the alphas since we"re writing to PNG (technically they"re already premultiplied)
            if self.extra_args["bg_color"] == "black" and out.rgb.shape[-1] > 3:
                bg = torch.ones_like(out.rgb[..., :3])
                out.rgb[..., :3] += bg * (1.0 - out.rgb[..., 3:4])

            out = out.image().byte().numpy_dict()

            log_buffers = ["depth", "hit", "normal", "rgb", "alpha"]

            for key in log_buffers:
                if out.get(key) is not None:
                    self.writer.add_image(f"{key}/{d}", out[key].T, self.epoch)
                    if self.using_wandb:
                        log_images_to_wandb(f"{key}/{d}", out[key].T, self.epoch)

    def save_local(self):
        if self.save_latent:
            fname = join(self.latent_dir, str(self.epoch))
            np.save(fname, np.array(self.latents))

        if self.plot_embed_map:
            fname = join(self.embed_map_dir, str(self.epoch))
            np.save(fname, np.array(self.embed_ids))

        if self.save_recon or self.save_cropped_recon:
            re_args = {
                "fname": str(self.epoch),
                "dir": self.recon_dir,
                "verbose": self.verbose,
                "to_HDU": False,
                "recon_HSI": False,
                "recon_norm": False,
                "recon_flat_trans": False,
                "calculate_metrics": False
            }
            _, _ = self.dataset.restore_evaluate_tiles(self.smpl_pixels, **re_args)

            if self.save_cropped_recon:
                for i, fits_id in enumerate(self.extra_args["recon_cutout_fits_ids"]):
                    zscale_ranges = self.dataset.get_zscale_ranges(fits_id)

                    np_fname = join(self.recon_dir, f"{fits_id}_{self.epoch}.npy")
                    recon = np.load(np_fname) # [nbands,sz,sz]

                    for (size, (r,c)) in zip(self.extra_args["recon_cutout_sizes"][i],
                                             self.extra_args["recon_cutout_start_pos"][i]):
                        fname = join(self.recon_dir, f"{fits_id}_{r}_{c}_{size}")
                        np.save(fname, recon[:,r:r+size,c:c+size])
                        plot_horizontally(recon[:,r:r+size,c:c+size], fname, zscale_ranges=zscale_ranges)

    def save_model(self):
        if self.extra_args["save_as_new"]:
            fname = f"model-ep{self.epoch}-it{self.iteration}.pth"
            model_fname = os.path.join(self.model_dir, fname)
        else: model_fname = os.path.join(self.model_dir, f"model.pth")

        #model_fname = self.args.model_fns[model_id]
        if self.verbose: log.info(f"Saving model checkpoint to: {model_fname}")

        checkpoint = {
            "epoch_trained": self.epoch - 1, # 0 based
            "model_state_dict": self.pipeline.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }
        torch.save(checkpoint, model_fname)
        return checkpoint


    def validate(self):
        pass
