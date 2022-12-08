# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.  #
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import os
import time
import torch
import wandb
import numpy as np
import torch.nn as nn
import logging as log

from pathlib import Path
from os.path import exists, join
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

        super().__init__(pipeline, dataset, num_epochs, batch_size, optim_cls,
                         lr, weight_decay, grid_lr_weight, optim_params, log_dir,
                         device, exp_name, info, scene_state, extra_args,
                         render_tb_every, save_every, using_wandb)

        self.verbose = self.extra_args["verbose"]

        self.space_dim = self.extra_args["space_dim"]
        self.quantize_latent = self.extra_args['quantize_latent']
        self.spectra_supervision = self.extra_args['spectra_supervision']
        self.spectral_inpaint = self.extra_args['inpaint_cho'] == 'spectral_inpaint'

        self.save_latent_during_train = self.space_dim == 3 and \
            (self.extra_args["use_ngp"] or self.extra_args["encode"]) and \
            self.extra_args["save_latent_during_train"]

        self.plot_embd_map = self.space_dim == 3 and \
           (self.extra_args["use_ngp"] or self.extra_args["encode"]) and \
           self.extra_args["quantize_latent"] and \
           self.extra_args["plot_embd_map_during_train"]

        self.init_loss()
        self.set_log_path()

    def init_loss(self):
        cho = self.extra_args['loss_cho']

        if cho == 'l1':
            loss = nn.L1Loss() if not self.extra_args['cuda'] else nn.L1Loss().cuda()
        elif cho == 'l2':
            loss = nn.MSELoss() if not self.extra_args['cuda'] else nn.MSELoss().cuda()
        else:
            raise Exception('Unsupported loss choice')

        if self.spectra_supervision:
            self.spectra_loss = partial(spectra_supervision_loss, loss)

        if self.spectral_inpaint:
            loss = partial(spectral_masking_loss, loss,
                           self.extra_args['relative_train_bands'],
                           self.extra_args['relative_inpaint_bands'])
        self.loss = loss

    def set_log_path(self):
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        if self.verbose: log.info('logging to', self.log_dir)

        for cur_path, cur_pname, in zip(
                ['model_dir','recon_dir','metric_dir', 'spectrum_dir',
                 'cdbk_spectrum_dir', 'cutout_dir','embd_map_dir','latent_dir',
                 'latent_embd_dir'],
                ['models','recons','metrics','spectrum','cdbk_spectrum',
                 'cutout','embd_map','latent','latent_embd_dir']
        ):
            path = join(self.log_dir, cur_pname)
            setattr(self, cur_path, path)
            Path(path).mkdir(parents=True, exist_ok=True)

        self.grad_fn = join(self.log_dir, 'gradient.png')
        self.train_loss_fn = join(self.log_dir, 'loss.npy')
        self.embd_map_fn = join(self.recon_dir, 'embd_map')
        self.cutout_png_fn = join(self.recon_dir, 'cutout')
        #self.model_fns = [join(config['model_dir'], str(i) + '.pth')
        #                       for i in range(config['num_model_smpls'])]

    ##############
    # One epoch loop
    ##############

    def iterate(self):
        """ Advances the training by one training step (batch) """
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
        """ Actually iterate the data loader """
        return next(self.train_data_loader_iter)

    ##########
    # begin epoch
    ##########

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
            #module.shuffle_pixls() # shuffle pixels at each epoch
            self.resample_dataset()

        if self.extra_args["save_local_every"] > -1 and self.epoch % self.extra_args["save_local_every"] == 0:
            #if self.epoch == 0 or (self.epoch + 1) % args.loss_smpl_intvl == 0
            # if save data, use pixels in orig order (incl. spectra supervision pixls)
            # redo batch division as total number of pixels changed
            #num_batches = module.reinit_pixl_ids()
            self.save_data_to_local = True
            self.latents, self.embd_ids, self.smpl_pixels = [], [], []

        self.pipeline.train()
        self.timer.check("pre_epoch done")

    def init_log_dict(self):
        """ Custom log dict """
        super().init_log_dict()
        self.log_dict["recon_loss"] = 0.0
        self.log_dict["spectra_loss"] = 0.0
        self.log_dict["codebook_loss"] = 0.0

    def resample_dataset(self):
        if hasattr(self.dataset, "resample"):
            if self.verbose: log.info("Reset DataLoader")
            self.dataset.resample()
            self.init_dataloader()
            self.timer.check("create_dataloader")
        else:
            raise ValueError("resample=True but the dataset doesn't have a resample method")

    ###########
    # end epoch
    ###########

    def end_epoch(self):
        self.post_epoch()

        #if self.extra_args["infer_during_train"]:
        #    infer(args, model_id, checkpoint)

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
            self.save_data_to_local = False

        # save model
        if self.save_every > -1 and self.epoch % self.save_every == 0: # and self.epoch != 0:
            #epoch in set(extra_args["smpl_epochs"])
            self.save_model()

        self.timer.check("post_epoch done")

    ###############
    # Core training step
    ###############

    def forward(self, data):
        dim = self.space_dim

        if dim == 2:
            requested_channels = {"density"}
            net_args = {"coords": data["coords"].to(self.device), "covar": None}

        elif dim == 3:
            requested_channels = ["density"]
            if self.quantize_latent:
                requested_channels.append("cdbk_loss")
                if self.plot_embd_map:
                    requested_channels.append("embd_ids")
                    requested_channels.append("latents")

            if self.spectra_supervision:
                requested_channels.append("recon_spectra")
            requested_channels = set(requested_channels)

            mc_cho = self.extra_args["mc_cho"]

            if mc_cho == "mc_hardcode":
                net_args = {"coords": self.cur_coords, "covar": self.covar, "trans": self.smpl_trans}
            elif mc_cho == "mc_bandwise":
                net_args = [self.cur_coords, self.covar, self.smpl_wave, self.smpl_trans]
            elif mc_cho == "mc_mixture":
                net_args = [self.cur_coords, self.covar, self.smpl_wave,
                            self.smpl_trans, self.nsmpl_within_each_band_mixture]
            else:
                raise Exception("Unsupported monte carlo choice")
        else:
            raise Exception("Unsupported space dim")

        return self.pipeline(channels=requested_channels, **net_args)

    def calculate_loss(self, data):
        total_loss = 0

        ret = self.forward(data)
        recon_pixels = ret["density"]
        gt_pixels = data["pixels"].to(self.device)

        if self.extra_args["weight_train"]:
            weights = data["weights"].to(self.device)
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
            lo = self.extra_args["trusted_spectra_wave_id_lo"]
            hi = self.extra_args["trusted_spectra_wave_id_hi"] + 1
            recon_spectra = ret["spectra"][:,lo:hi]
            spectra_loss = self.spectra_loss(self.gt_spectra, recon_spectra)
            self.log_dict["spectra_loss"] += spectra_loss.item()
        else:
            spectra_loss, recon_spectra = 0, None

        # iii) latent quantization codebook loss
        if self.quantize_latent:
            cdbk_loss = ret["cdbk_loss"]
            self.log_dict["codebook_loss"] += cdbk_loss.item()
        else: cdbk_loss = 0

        total_loss = recon_loss + spectra_loss + cdbk_loss
        #print(recon_loss, spectra_loss, cdbk_loss, total_loss, total_loss.dtype)
        self.log_dict["total_loss"] += total_loss.item()

        if self.quantize_latent:
            if self.plot_embd_map:
                embd_ids = ret["embd_ids"]
            if self.plot_latent_embd:
                latents = ret["latents"]
        else:
            embd_ids, latents = None, None

        self.timer.check("loss")
        return total_loss, recon_pixels, recon_spectra, embd_ids, latents

    def step(self, data):
        """ Advance the training by one step using the batched data supplied.
            @Param:
              data (dict): Dictionary of the input batch from the DataLoader.
        """
        #cur_batch_sz = self.select_cur_batch_data(batch, save_data)
        #if self.dim == 3: self.sample_wave_trans(cur_batch_sz)

        self.optimizer.zero_grad(set_to_none=True)
        self.timer.check("zero grad")

        with torch.cuda.amp.autocast():
            total_loss, recon_pixels, recon_spectra, embd_ids, latents = self.calculate_loss(data)

        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.timer.check("backward and step")

        #plot_grad_flow(self.model.named_parameters(), self.args.grad_fn)
        if self.save_data_to_local:
            # return {"recon_pixels": recon_pixels,
            #         "recon_spectra": recon_spectra,
            #         "embd_ids": embd_ids,
            #         "latents": latents }
            self.latents = latents
            self.embd_ids = embd_ids
            self.recon_pixels = recon_pixels

    def pre_step(self):
        """ Sample lambda (and corrpesponding transmission) before each iteration. """
        if self.space_dim == 3:
            self.dataset.sample_wave()

    def post_step(self):
        if self.save_latent_during_train:
            self.latents.extend(self.latents.detach().cpu().numpy())

        if self.plot_embd_map:
            self.embd_ids.extend(self.embd_ids.detach().cpu().numpy())

        if self.extra_args["save_cutout_during_train"]:
            self.smpl_pixels.extend(self.recon_pixels.detach().cpu().numpy())

    ############
    # Helper methods
    #############

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
            log_text += " | codebook loss: {:>.3E}".format(self.log_dict["cdbk_loss"] / n)
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
        # after data saving is done, init pixel ids again to only use fraction of pixels
        #num_batches = module.init_pixel_ids()
        #if save_model: model_id += 1
        if self.save_latent_during_train:
            fname = join(self.latent_dir, str(self.epoch))
            np.save(fname, np.array(self.latents))

        if self.plot_embd_map:
            fname = join(self.embd_map_dir, str(self.epoch))
            np.save(fname, np.array(self.embd_ids))

        if self.extra_args["save_cutout_during_train"]:
            smpl_cutout = np.array(self.smpl_pixels).reshape((-1, self.num_bands)) \
                [self.smpl_cutout_pixel_ids].T.\
                reshape((-1, self.extra_args['cutout_sz'], self.extra_args['cutout_sz']))
            fname = join(self.cutout_dir, str(self.epoch))
            np.save(fname, smpl_cutout)
            plot_gt_recon(self.gt_cutout, smpl_cutout, fn)

    def save_model(self):
        if self.extra_args["save_as_new"]:
            fname = f"model-ep{self.epoch}-it{self.iteration}.pth"
            model_fname = os.path.join(self.log_dir, fname)
        else: model_fname = os.path.join(self.log_dir, f"model.pth")

        #model_fname = self.args.model_fns[model_id]
        if self.verbose: log.info(f"Saving model checkpoint to: {model_fname}")

        checkpoint = {
            "epoch_trained": self.epoch - 1, # 0 based
            "model_state_dict": self.pipeline.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }
        torch.save(checkpoint, model_fname)
        return checkpoint

    ############
    # Inference
    ############

    def validate(self):
        self.pipeline.eval()

        record_dict = self.extra_args
        dataset_name = os.path.splitext(os.path.basename(self.extra_args["dataset_path"]))[0]
        model_fname = os.path.abspath(os.path.join(self.log_dir, f"model.pth"))
        record_dict.update({"dataset_name" : dataset_name, "epoch": self.epoch,
                            "log_fname" : self.log_fname, "model_fname": model_fname})
        parent_log_dir = os.path.dirname(self.log_dir)

        if self.verbose: log.info("Beginning validation...")

        data = self.dataset.get_images(split=self.extra_args["valid_split"], mip=self.extra_args["mip"])
        imgs = list(data["imgs"])

        img_shape = imgs[0].shape
        if self.verbose: log.info(f"Loaded validation dataset with {len(imgs)} images at resolution {img_shape[0]}x{img_shape[1]}")

        self.valid_log_dir = os.path.join(self.log_dir, "val")
        if self.verbose: log.info(f"Saving validation result to {self.valid_log_dir}")
        if not os.path.exists(self.valid_log_dir):
            os.makedirs(self.valid_log_dir)

        lods = list(range(self.pipeline.nef.num_lods))
        evaluation_results = self.evaluate_metrics(data["rays"], imgs, lods[-1], f"lod{lods[-1]}")
        record_dict.update(evaluation_results)
        if self.using_wandb:
            log_metric_to_wandb("Validation/psnr", evaluation_results["psnr"], self.epoch)
            log_metric_to_wandb("Validation/lpips", evaluation_results["lpips"], self.epoch)
            log_metric_to_wandb("Validation/ssim", evaluation_results["ssim"], self.epoch)

        df = pd.DataFrame.from_records([record_dict])
        df["lod"] = lods[-1]
        fname = os.path.join(parent_log_dir, f"logs.parquet")
        if os.path.exists(fname):
            df_ = pd.read_parquet(fname)
            df = pd.concat([df_, df])
        df.to_parquet(fname, index=False)
