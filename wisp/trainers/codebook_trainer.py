
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
from torch.utils.data import BatchSampler, \
    SequentialSampler, RandomSampler, DataLoader

from wisp.trainers import BaseTrainer
from wisp.utils.plot import plot_grad_flow
from wisp.loss import spectra_supervision_loss, pretrain_pixel_loss, \
    redshift_supervision_loss
from wisp.utils.common import get_gpu_info, add_to_device, sort_alphanumeric, \
    load_embed, load_model_weights, forward


class CodebookTrainer(BaseTrainer):
    """ Trainer class for codebook pretraining.
    """
    def __init__(self, pipeline, dataset, optim_cls, optim_params, device, **extra_args):
        super().__init__(pipeline, dataset, optim_cls, optim_params, device, **extra_args)

        assert(
            extra_args["space_dim"] == 3 and \
            extra_args["pretrain_codebook"] and \
            not extra_args["spectra_supervision"]
        )

        # save config file to log directory
        dst = join(self.log_dir, "config.yaml")
        shutil.copyfile(extra_args["config"], dst)

        self.cuda = "cuda" in str(device)
        self.verbose = extra_args["verbose"]
        self.space_dim = extra_args["space_dim"]
        self.gpu_fields = extra_args["gpu_data"]
        self.z_beta = extra_args["pretrain_redshift_beta"]
        self.recon_beta = extra_args["pretrain_pixel_beta"]
        self.num_sup_spectra = dataset.get_num_supervision_spectra()

        self.total_steps = 0
        self.save_data_to_local = False
        self.shuffle_dataloader = False
        self.dataloader_drop_last = False

        self.set_log_path()
        self.summarize_training_tasks()

        self.init_net()
        self.init_loss()
        self.init_optimizer()

        self.configure_dataset()
        self.set_num_batches()
        self.init_dataloader()

    #############
    # Initializations
    #############

    def init_net(self):
        self.train_pipeline = self.pipeline[0]
        self.infer_pipeline = self.pipeline[1]
        self.latents = nn.Embedding(
            self.num_sup_spectra,
            self.extra_args["codebook_pretrain_latent_dim"]
        )
        log.info("Total number of parameters: {}".format(
            sum(p.numel() for p in self.train_pipeline.parameters()))
        )

    def configure_dataset(self):
        """ Configure dataset with selected fields and set length accordingly.
        """
        self.dataset.set_mode("codebook_pretrain")

        # set required fields from dataset
        fields = ["trans_data","spectra_data","redshift_data"]
        self.dataset.set_fields(fields)

        # set input latents for codebook net
        self.dataset.set_coords_source("spectra_latents")
        self.dataset.set_hardcode_data("spectra_latents", self.latents.weight)

        self.dataset.toggle_wave_sampling(True) # TODO: False if we have too many spectra
        self.dataset.toggle_integration(self.pixel_supervision)
        self.dataset.set_length(self.num_sup_spectra)

    def summarize_training_tasks(self):
        tasks = set(self.extra_args["tasks"])

        self.pixel_supervision = self.extra_args["codebook_pretrain_pixel_supervision"]
        self.redshift_supervision = self.extra_args["generate_redshift"] and self.extra_args["redshift_supervision"] and not self.extra_args["use_gt_redshift"]

        self.save_soft_qtz_weights = "save_soft_qtz_weights_during_train" in tasks
        self.plot_spectra = self.space_dim == 3 and "recon_gt_spectra_during_train" in tasks
        self.save_redshift =  "save_redshift_during_train" in tasks and \
            self.extra_args["generate_redshift"]
        self.save_pixel_values = "save_pixel_values_during_train" in tasks and \
            self.extra_args["codebook_pretrain_pixel_supervision"]

    def set_log_path(self):
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        if self.verbose: log.info(f"logging to {self.log_dir}")

        for cur_path, cur_pname, in zip(
                ["model_dir","spectra_dir","codebook_spectra_dir",
                 "redshift_dir","soft_qtz_weight_dir"],
                ["models","train_spectra","codebook_spectra",
                 "redshift","soft_qtz_weights"]
        ):
            path = join(self.log_dir, cur_pname)
            setattr(self, cur_path, path)
            Path(path).mkdir(parents=True, exist_ok=True)

        self.grad_fname = join(self.log_dir, "grad.png")

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
                fnames = sort_alphanumeric(fnames)
                self.pretrained_model_fname = join(pretrained_model_dir, fnames[-1])

    def get_loss(self, cho):
        if cho == "l1":
            loss = nn.L1Loss() if not self.cuda else nn.L1Loss().cuda()
        elif cho == "l2":
            loss = nn.MSELoss() if not self.cuda else nn.MSELoss().cuda()
        else: raise Exception("Unsupported loss choice")
        return loss

    def init_loss(self):
        loss = self.get_loss(self.extra_args["spectra_loss_cho"])
        self.spectra_loss = partial(spectra_supervision_loss, loss)

        if self.pixel_supervision:
            loss = self.get_loss(self.extra_args["pixel_loss_cho"])
            self.pixel_loss = partial(pretrain_pixel_loss, loss)

        if self.redshift_supervision:
            loss = self.get_loss(self.extra_args["redshift_loss_cho"])
            self.redshift_loss = partial(redshift_supervision_loss, loss)

    def init_dataloader(self):
        """ (Re-)Initialize dataloader.
        """
        #if self.shuffle_dataloader: sampler_cls = RandomSampler
        #else: sampler_cls = SequentialSampler
        # sampler_cls = SequentialSampler
        sampler_cls = RandomSampler

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
        # collect all parameters from network and trainable latents
        self.params_dict = {
            "spectra_latents": param for _, param in self.latents.named_parameters()
        }
        for name, param in self.train_pipeline.named_parameters():
            self.params_dict[name] = param

        params, net_params, latents = [], [], []
        for name in self.params_dict:
            if "spectra_latents" in name:
                latents.append(self.params_dict[name])
            else:
                net_params.append(self.params_dict[name])

        params.append({"params": latents,
                       "lr": self.extra_args["codebook_pretrain_lr"]})
        params.append({"params": net_params,
                       "lr": self.extra_args["codebook_pretrain_lr"]})
        self.optimizer = self.optim_cls(params, **self.optim_params)
        if self.verbose: log.info(self.optimizer)

    #############
    # Training logic
    #############

    def train(self):
        if self.extra_args["resume_train"]:
            self.resume_train()

        log.info(f"{self.num_iterations_cur_epoch} batches per epoch.")

        for epoch in range(self.num_epochs + 1):
            self.begin_epoch()

            for batch in range(self.num_iterations_cur_epoch):
                iter_start_time = time.time()

                data = self.next_batch()

                self.pre_step()
                self.step(data)
                self.post_step()

                self.iteration += 1
                self.total_steps += 1

            self.end_epoch()

        self.writer.close()
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

    def end_epoch(self):
        self.post_epoch()

        if self.epoch < self.num_epochs:
            self.iteration = 0
            self.epoch += 1
        else:
            self.scene_state.optimization.running = False

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

        self.loss_lods = list(range(0, self.extra_args["grid_num_lods"]))

        if self.extra_args["grow_every"] > 0:
            self.grow()

        if self.extra_args["only_last"]:
            self.loss_lods = self.loss_lods[-1:]

        if self.save_local_every > -1 and self.epoch % self.save_local_every == 0:
            self.save_data_to_local = True

            if self.save_redshift: self.redshifts = []
            if self.plot_spectra: self.smpl_spectra = []
            if self.save_soft_qtz_weights: self.qtz_weights = []
            if self.save_pixel_values:
                self.gt_pixel_vals = []
                self.recon_pixel_vals = []

            # re-init dataloader to make sure pixels are in order
            self.shuffle_dataloader = False
            self.use_all_pixels = True
            self.set_num_batches()
            self.init_dataloader()
            self.reset_data_iterator()

        self.train_pipeline.train()
        self.timer.check("pre_epoch done")

    def post_epoch(self):
        """ By default, this function logs to Tensorboard, renders images to Tensorboard,
            saves the model, and resamples the dataset.
        """
        self.train_pipeline.eval()

        total_loss = self.log_dict["total_loss"] / len(self.train_data_loader)
        self.scene_state.optimization.losses["total_loss"].append(total_loss)

        if self.save_every > -1 and self.epoch % self.save_every == 0:
            self.save_model()

        if self.valid_every > -1 and self.epoch % self.valid_every == 0:
            self.validate()

        if self.log_tb_every > -1 and self.epoch % self.log_tb_every == 0:
            self.log_tb()

        if self.log_cli_every > -1 and self.epoch % self.log_cli_every == 0:
            self.log_cli()

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

        self.timer.check("post_epoch done")

    #############
    # One step
    #############

    def init_log_dict(self):
        """ Custom log dict. """
        super().init_log_dict()
        self.log_dict["pixel_loss"] = 0.0
        self.log_dict["spectra_loss"] = 0.0
        self.log_dict["redshift_loss"] = 0.0

    def pre_step(self):
        self.dataset.set_hardcode_data("spectra_latents", self.latents.weight)

    def step(self, data):
        """ Advance the training by one step using the batched data supplied.
            @Param:
              data (dict): Dictionary of the input batch from the DataLoader.
        """
        self.optimizer.zero_grad(set_to_none=True)
        self.timer.check("zero grad")

        total_loss, ret = self.calculate_loss(data)
        total_loss.backward()
        if self.plot_grad_every != -1 and (self.epoch == 0 or \
           (self.plot_grad_every % self.epoch == 0)):
            plot_grad_flow(self.params_dict.items(), self.grad_fname)
        self.optimizer.step()

        self.timer.check("backward and step")

        if self.save_data_to_local:
            if self.save_redshift: self.redshifts.extend(ret["redshift"])
            if self.plot_spectra: self.smpl_spectra.append(ret["spectra"])
            if self.save_soft_qtz_weights: self.qtz_weights.extend(ret["soft_qtz_weights"])
            if self.save_pixel_values:
                self.recon_pixel_vals.extend(ret["intensity"])
                self.gt_pixel_vals.extend(data["spectra_sup_pixels"])

    def post_step(self):
        pass

    #############
    # Validation
    #############

    def validate(self):
        """ Perform validation (recon gt spectra, codebook spectra etc.).
        """
        load_model_weights(self.infer_pipeline, self.train_pipeline.state_dict())
        self.infer_pipeline.eval()

        data = self.get_valid_data()
        ret = forward(data,
                      self.infer_pipeline,
                      self.total_steps,
                      self.space_dim,
                      self.extra_args["trans_sample_method"],
                      recon_codebook_spectra=True)

        codebook_spectra = ret["intensity"].detach().cpu().numpy()

        fname = f"ep{self.epoch}-it{self.iteration}"
        self.dataset.plot_spectrum(
            self.codebook_spectra_dir, fname, codebook_spectra,
            spectra_norm_cho=self.extra_args["spectra_norm_cho"],
            save_spectra=True, codebook=True,
            clip=self.extra_args["plot_clipped_spectrum"])

    def get_valid_data(self):
        """ Get data for codebook spectra recon.
        """
        bsz = self.extra_args["qtz_num_embed"]
        latents = load_embed(self.train_pipeline.state_dict(),
                             transpose=False, tensor=True)[:,None]

        wave = torch.FloatTensor(self.dataset.get_full_wave())[None,:,None].tile(bsz,1,1)
        data = {
            "coords": latents.to(self.device), # [bsz,nsmpl,latent_dim]
            "wave": wave.to(self.device),      # [bsz,nsmpl,1]
            "full_wave_bound": self.dataset.get_full_wave_bound(),
        }
        return data

    #############
    # Helper methods
    #############

    def set_num_batches(self):
        """ Set number of batches/iterations and batch size for each epoch.
            At certain epochs, we may not need all data and can break before
              iterating thru all data.
        """
        length = len(self.dataset)
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

            self.train_pipeline.load_state_dict(checkpoint["model_state_dict"])
            self.train_pipeline.eval()
            self.latents.load_state_dict(checkpoint["latents"])

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

    def calculate_loss(self, data):
        total_loss = 0
        add_to_device(data, self.gpu_fields, self.device)

        ret = forward(data,
                      self.train_pipeline,
                      self.total_steps,
                      self.space_dim,
                      self.extra_args["trans_sample_method"],
                      codebook_pretrain=True,
                      pixel_supervision_train=self.pixel_supervision,
                      redshift_supervision_train=self.redshift_supervision,
                      use_gt_redshift=self.extra_args["use_gt_redshift"],
                      quantize_spectra=True,
                      quantization_strategy="soft",
                      save_spectra=self.plot_spectra,
                      save_redshift=self.save_redshift,
                      save_soft_qtz_weights=self.save_soft_qtz_weights)

        # i) spectra supervision loss
        spectra_loss, recon_spectra = 0, None
        gt_spectra = data["spectra_sup_fluxes"]

        (lo, hi) = data["spectra_sup_wave_bound_ids"]
        recon_spectra = ret["spectra"][:,lo:hi]

        if len(recon_spectra) == 0:
            spectra_loss = 0
        else:
            spectra_loss = self.spectra_loss(gt_spectra, recon_spectra)
            self.log_dict["spectra_loss"] += spectra_loss.item()

        # ii) pixel supervision loss
        recon_loss = 0
        if self.pixel_supervision:
            gt_pixels = data["spectra_sup_pixels"]
            recon_pixels = ret["intensity"]

            recon_loss = self.pixel_loss(gt_pixels, recon_pixels)
            self.log_dict["pixel_loss"] += recon_loss.item()

        # iii) redshift loss
        redshift_loss = 0
        if self.redshift_supervision:
            gt_redshift = data["gt_spectra_redshift"]

            # ids = gt_redshift != -1
            pred_redshift = ret["redshift"]
            # if torch.count_nonzero(ids) != 0:
            # redshift_loss = self.redshift_loss(gt_redshift[ids], pred_redshift[ids]) * self.redshift_beta
            redshift_loss = self.redshift_loss(gt_redshift, pred_redshift) \
                * self.extra_args["redshift_beta"]
            self.log_dict["redshift_loss"] += redshift_loss.item()

        # total_loss = recon_loss*self.recon_beta + redshift_loss*self.z_beta
        total_loss = spectra_loss + recon_loss*self.recon_beta + redshift_loss*self.z_beta

        self.log_dict["total_loss"] += total_loss.item()
        self.timer.check("loss")
        return total_loss, ret

    def log_cli(self):
        """ Controls CLI logging.
            By default, this function only runs every epoch.
        """
        # Average over iterations
        n = len(self.train_data_loader)

        log_text = "EPOCH {}/{}".format(self.epoch, self.num_epochs)
        log_text += " | total loss: {:>.3E}".format(self.log_dict["total_loss"] / n)
        log_text += " | spectra loss: {:>.3E}".format(self.log_dict["spectra_loss"] / n)
        if self.pixel_supervision:
            log_text += " | pixel loss: {:>.3E}".format(self.log_dict["pixel_loss"] / n)
        if self.redshift_supervision:
            log_text += " | redshift loss: {:>.3E}".format(self.log_dict["redshift_loss"] / n)
        log.info(log_text)

    def save_model(self):
        fname = f"model-ep{self.epoch}-it{self.iteration}.pth"
        model_fname = os.path.join(self.model_dir, fname)
        if self.verbose: log.info(f"Saving model checkpoint to: {model_fname}")

        checkpoint = {
            "iterations": self.total_steps,
            "epoch_trained": self.epoch,
            "latents": self.latents,
            "model_state_dict": self.train_pipeline.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }

        torch.save(checkpoint, model_fname)
        return checkpoint

    def save_local(self):
        if self.plot_spectra:
            self._plot_spectrum()

        if self.save_redshift:
            self._save_redshift()

        if self.save_pixel_values:
            self._save_pixel_values()

        if self.save_soft_qtz_weights:
            self._save_soft_qtz_weights()

    def _save_pixel_values(self):
        gt_vals = torch.stack(self.gt_pixel_vals).detach().cpu().numpy()
        recon_vals = torch.stack(self.recon_pixel_vals).detach().cpu().numpy()
        # fname = join(self.pixel_val_dir, f"model-ep{self.epoch}-it{self.iteration}.pth")
        # np.save(fname, vals)
        np.set_printoptions(suppress=True)
        np.set_printoptions(precision=3)
        # log.info(f"Pixel vals gt/recon {gt_vals} / {recon_vals}")
        ratio = gt_vals / recon_vals
        log.info(f"gt/recon ratio: {ratio}")

    def _save_redshift(self):
        redshifts = torch.stack(self.redshifts).detach().cpu().numpy()
        fname = join(self.redshift_dir, f"model-ep{self.epoch}-it{self.iteration}.pth")
        np.save(fname, redshifts)
        np.set_printoptions(precision=3)
        log.info(f"Est. redshift {redshifts}")

        gt_redshift = self.dataset.get_supervision_spectra_redshift()
        log.info(f"GT redshift {gt_redshift}")

    def _save_soft_qtz_weights(self):
        weights = torch.stack(self.qtz_weights).detach().cpu().numpy()
        fname = join(self.soft_qtz_weight_dir, f"model-ep{self.epoch}-it{self.iteration}.pth")
        np.save(fname, weights)
        np.set_printoptions(suppress=True)
        np.set_printoptions(precision=3)
        log.info(f"Qtz weights {weights[:,0]}")

    def _plot_spectrum(self):
        self.smpl_spectra = torch.stack(self.smpl_spectra).view(
            self.num_sup_spectra, -1
        ).detach().cpu().numpy() # [num_sup_spectra,num_samples]

        fname = f"ep{self.epoch}-it{self.iteration}"
        self.dataset.plot_spectrum(self.spectra_dir, fname, self.smpl_spectra,
                                   self.extra_args["spectra_norm_cho"], clip=True,
                                   save_spectra=True)
