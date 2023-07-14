
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
from torch.utils.data import BatchSampler, \
    SequentialSampler, RandomSampler, DataLoader

from wisp.utils import PerfTimer
from wisp.trainers import BaseTrainer
from wisp.utils.plot import plot_grad_flow
from wisp.loss import spectra_supervision_loss, pretrain_pixel_loss
from wisp.utils.common import get_gpu_info, add_to_device, sort_alphanumeric, \
    select_inferrence_ids, load_embed, load_model_weights, forward


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
        self.recon_beta = extra_args["pretrain_pixel_beta"]
        self.num_sup_spectra = dataset.get_num_supervision_spectra()

        self.total_steps = 0
        self.save_data_to_local = False
        self.shuffle_dataloader = False
        self.dataloader_drop_last = False

        self.summarize_training_tasks()
        self.set_path()

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
        # log.info(self.train_pipeline)
        # log.info(self.infer_pipeline)
        log.info("Total number of parameters: {}".format(
            sum(p.numel() for p in self.train_pipeline.parameters()))
        )

    def configure_dataset(self):
        """ Configure dataset with selected fields and set length accordingly.
        """
        self.dataset.set_mode("pre_train")

        # set required fields from dataset
        fields = ["coords","wave_data"]
        if self.extra_args["batched_pretrain"]:
            fields.extend([
                "spectra_sup_data",
                "spectra_sup_mask",
                "spectra_sup_redshift",
            ])
            if self.pixel_supervision:
                fields.append("spectra_sup_pixels")
        else:
            fields.append("spectra_data")

        self.dataset.set_fields(fields)

        # use original spectra wave
        self.dataset.set_wave_source("spectra")

        # set input latents for codebook net
        self.dataset.set_coords_source("spectra_latents")
        self.dataset.set_hardcode_data("spectra_latents", self.latents.weight)

        self.dataset.toggle_wave_sampling(self.sample_wave)
        self.dataset.toggle_integration(self.pixel_supervision)
        if self.train_within_wave_range:
            self.dataset.set_wave_range(
                self.extra_args["spectra_supervision_wave_lo"],
                self.extra_args["spectra_supervision_wave_hi"])

        self.dataset.set_length(self.num_sup_spectra)

        self.selected_ids = select_inferrence_ids(
            self.num_sup_spectra,
            self.extra_args["pretrain_num_infer"]
        )
        # print(self.selected_ids)

    def summarize_training_tasks(self):
        tasks = set(self.extra_args["tasks"])

        self.plot_loss = self.extra_args["plot_loss"]

        # quantization setups
        self.qtz_latent = self.space_dim == 3 and self.extra_args["quantize_latent"]
        self.qtz_spectra = self.space_dim == 3 and self.extra_args["quantize_spectra"]
        assert not (self.qtz_latent and self.qtz_spectra)
        self.qtz = self.qtz_latent or self.qtz_spectra
        self.qtz_n_embd = self.extra_args["qtz_num_embed"]
        self.qtz_strategy = self.extra_args["quantization_strategy"]

        self.pixel_supervision = self.extra_args["codebook_pretrain_pixel_supervision"]
        self.trans_sample_method = self.extra_args["trans_sample_method"]

        # as long as we model redshift, we should apply gt redshift to spectra during pretrain
        #  and we should never do redshift supervision during pretrain
        self.apply_gt_redshift = self.extra_args["model_redshift"]

        self.sample_wave = True
        self.train_within_wave_range = not self.pixel_supervision and \
            self.extra_args["codebook_pretrain_within_wave_range"]

        self.save_redshift = "save_redshift_during_train" in tasks
        self.save_qtz_weights = "save_qtz_weights_during_train" in tasks
        self.save_pixel_values = "save_pixel_values_during_train" in tasks and self.pixel_supervision
        self.recon_gt_spectra = "recon_gt_spectra_during_train" in tasks
        self.recon_codebook_spectra_individ = "recon_codebook_spectra_individ_during_train" in tasks

    def set_path(self):
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        log.info(f"logging to {self.log_dir}")

        for cur_path, cur_pname, in zip(
                ["model_dir","spectra_dir","codebook_spectra_dir","qtz_weight_dir"],
                ["models","train_spectra","train_codebook_spectra","qtz_weights"]):
            path = join(self.log_dir, cur_pname)
            setattr(self, cur_path, path)
            Path(path).mkdir(parents=True, exist_ok=True)

        self.grad_fname = join(self.log_dir, "grad.png")

        if self.plot_loss:
            self.loss_fname = join(self.log_dir, "loss")

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

    def init_dataloader(self):
        """ (Re-)Initialize dataloader.
        """
        if self.shuffle_dataloader: sampler_cls = RandomSampler
        else: sampler_cls = SequentialSampler
        sampler_cls = SequentialSampler
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

    def begin_train(self):
        if self.plot_loss:
            self.losses = []

        if self.extra_args["resume_train"]:
            self.resume_train()

        log.info(f"{self.num_iterations_cur_epoch} batches per epoch.")

    def train(self):
        self.begin_train()

        # for epoch in tqdm(range(self.num_epochs + 1)):
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

        self.end_train()

    def end_train(self):
        self.writer.close()

        if self.plot_loss:
            x = np.arange(len(self.losses))
            plt.plot(x, self.losses)
            plt.savefig(self.loss_fname + ".png")
            np.save(self.loss_fname + ".npy", np.array(self.losses))

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

            if self.save_redshift:
                self.redshift = []
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
            self.shuffle_dataloader = False
            self.use_all_pixels = True
            self.sample_wave = False
            self.dataset.toggle_wave_sampling(self.sample_wave)
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

        if self.plot_loss:
            self.losses.append(total_loss)

        if self.save_every > -1 and self.epoch % self.save_every == 0:
            self.save_model()

        if self.log_tb_every > -1 and self.epoch % self.log_tb_every == 0:
            self.log_tb()

        if self.log_cli_every > -1 and self.epoch % self.log_cli_every == 0:
            self.log_cli()

        if self.render_tb_every > -1 and self.epoch % self.render_tb_every == 0:
            self.render_tb()

        # save data locally and restore trainer state
        if self.save_data_to_local:
            self.save_local()
            self.sample_wave = True
            self.use_all_pixels = False
            self.shuffle_dataloader = True
            self.save_data_to_local = False
            self.dataset.toggle_wave_sampling(self.sample_wave)
            self.configure_dataset()
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
        self.timer.check("backward done")

        if self.plot_grad_every != -1 and (self.epoch == 0 or \
           (self.plot_grad_every % self.epoch == 0)):
            plot_grad_flow(self.params_dict.items(), self.grad_fname)
        self.optimizer.step()

        self.timer.check("backward and step")

        if self.save_data_to_local:
            if self.save_redshift:
                self.redshift.extend(data["spectra_sup_redshift"])
            if self.save_pixel_values:
                self.recon_pixel_vals.extend(ret["intensity"])
                self.gt_pixel_vals.extend(data["spectra_sup_pixels"])
            if self.save_qtz_weights:
                self.qtz_weights.extend(ret["qtz_weights"])

            if self.recon_gt_spectra:
                self.recon_fluxes.extend(ret["spectra"])
                self.gt_fluxes.extend(data["spectra_sup_data"][:,1])
                self.spectra_masks.extend(data["spectra_sup_mask"])
                self.spectra_wave.extend(data["spectra_sup_data"][:,0])

            if self.recon_codebook_spectra_individ:
                self.codebook_spectra.extend(ret["codebook_spectra"])
                self.spectra_masks_c.extend(data["spectra_sup_mask"])
                self.spectra_wave_c.extend(data["spectra_sup_data"][:,0])

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

        ret = forward(
            data,
            self.train_pipeline,
            self.total_steps,
            self.space_dim,
            qtz=self.qtz,
            qtz_strategy=self.qtz_strategy,
            apply_gt_redshift=self.apply_gt_redshift,
            perform_integration=self.pixel_supervision,
            trans_sample_method=self.trans_sample_method,
            save_spectra=True, # we always need recon flux to calculate loss
            save_redshift=self.save_data_to_local and self.save_redshift,
            save_qtz_weights=self.save_data_to_local and self.save_qtz_weights,
            save_codebook_spectra=self.save_data_to_local and \
                                  self.recon_codebook_spectra_individ
        )

        # i) spectra supervision loss
        spectra_loss = 0
        recon_flux = ret["spectra"]
        gt_spectra = data["spectra_sup_data"]
        spectra_masks = data["spectra_sup_mask"]

        if len(recon_flux) == 0:
            spectra_loss = 0
        else:
            spectra_loss = self.spectra_loss(
                spectra_masks, gt_spectra, recon_flux)
            self.log_dict["spectra_loss"] += spectra_loss.item()

        # ii) pixel supervision loss
        recon_loss = 0
        if self.pixel_supervision:
            gt_pixels = data["spectra_sup_pixels"]
            recon_pixels = ret["intensity"]

            recon_loss = self.pixel_loss(gt_pixels, recon_pixels)
            self.log_dict["pixel_loss"] += recon_loss.item()

        total_loss = spectra_loss + recon_loss*self.recon_beta

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
        if type(self.redshift) == list:
            self.redshift = torch.stack(self.redshift)[self.selected_ids]
        np.set_printoptions(suppress=True)
        np.set_printoptions(precision=3)
        log.info(f"gt redshift values: {self.redshift}")

    def _save_pixel_values(self):
        gt_vals = torch.stack(self.gt_pixel_vals).detach().cpu().numpy()[self.selected_ids,0]
        recon_vals = torch.stack(self.recon_pixel_vals).detach().cpu().numpy()[self.selected_ids]
        # fname = join(self.pixel_val_dir, f"model-ep{self.epoch}-it{self.iteration}.pth")
        # np.save(fname, vals)
        np.set_printoptions(suppress=True)
        np.set_printoptions(precision=3)
        ratio = gt_vals / recon_vals
        log.info(f"gt/recon ratio: {ratio}")

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
            self.num_sup_spectra, -1).detach().cpu().numpy() # [n_spectra,nsmpl]
        self.recon_fluxes = torch.stack(self.recon_fluxes).view(
            self.num_sup_spectra, self.extra_args["spectra_neighbour_size"]**2, -1
        ).detach().cpu().numpy()
        self.spectra_wave = torch.stack(self.spectra_wave).view(
            self.num_sup_spectra, -1).detach().cpu().numpy()
        self.spectra_masks = torch.stack(self.spectra_masks).bool().view(
            self.num_sup_spectra, -1).detach().cpu().numpy()

        # print(self.gt_fluxes.shape, self.recon_fluxes.shape,
        #       self.spectra_masks.shape, self.spectra_wave.shape)
        # print(self.spectra_masks[0])

        fname = f"ep{self.epoch}-it{self.iteration}"
        self.dataset.plot_spectrum(
            self.spectra_dir, fname,
            self.extra_args["flux_norm_cho"],
            self.spectra_wave,
            self.gt_fluxes, self.recon_fluxes,
            save_spectra=True,
            spectra_ids=self.selected_ids,
            #gt_spectra_ids=self.selected_ids,
            #recon_spectra_ids=self.selected_ids,
            clip=self.extra_args["plot_clipped_spectrum"],
            masks=self.spectra_masks,
            spectra_clipped=False,
        )

    def _recon_codebook_spectra_individ(self):
        """ Reconstruct codebook spectra for each spectra individually.
        """
        log.info("reconstructing codebook spectrum")

        self.codebook_spectra = torch.stack(self.codebook_spectra).view(
            self.num_sup_spectra, self.qtz_n_embd, -1
        ).detach().cpu().numpy()[self.selected_ids]
        self.spectra_wave_c = torch.stack(self.spectra_wave_c).view(
            self.num_sup_spectra, -1).detach().cpu().numpy()[self.selected_ids]
        self.spectra_masks_c = torch.stack(self.spectra_masks_c).bool().view(
            self.num_sup_spectra, -1).detach().cpu().numpy()[self.selected_ids]

        prefix = "individ-"
        for i, (wave, masks, codebook_spectra) in enumerate(
                zip(self.spectra_wave_c, self.spectra_masks_c, self.codebook_spectra)
        ):
            cur_dir = join(self.codebook_spectra_dir, f"spectra-{i}")
            Path(cur_dir).mkdir(parents=True, exist_ok=True)

            fname = f"{prefix}ep{self.epoch}-it{self.iteration}"
            wave = np.tile(wave[None,:], self.qtz_n_embd).reshape(-1, self.qtz_n_embd).T
            masks = np.tile(masks[None,:], self.qtz_n_embd).reshape(-1, self.qtz_n_embd).T

            self.dataset.plot_spectrum(
                cur_dir, fname, self.extra_args["flux_norm_cho"],
                wave, None, codebook_spectra,
                is_codebook=True,
                save_spectra_together=True,
                clip=self.extra_args["plot_clipped_spectrum"],
                masks=masks
            )

    def validate(self):
        pass
