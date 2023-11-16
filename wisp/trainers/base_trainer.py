
import os
import time
import torch
#import wandb
import shutil
import numpy as np
import torch.nn as nn
import logging as log

from os.path import join
from datetime import datetime
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from wisp.utils import PerfTimer
#from wisp.datasets import default_collate
#from wisp.offline_renderer import OfflineRenderer
#from wisp.framework import WispState, BottomLevelRendererState


def log_metric_to_wandb(key, _object, step):
    wandb.log({key: _object}, step=step, commit=False)

def log_images_to_wandb(key, image, step):
    wandb.log({key: wandb.Image(np.moveaxis(image, 0, -1))}, step=step, commit=False)

class BaseTrainer(ABC):
    """ Base class for the trainer.
        The default overall flow of things:

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
    def __init__(self, pipeline, dataset, optim_cls, optim_params, device, scene_state=None, **extra_args):
        """ @Params
            pipeline (wisp.core.Pipeline): The pipeline with tracer and neural field to train.
            dataset (torch.Dataset): The dataset to use for training.
            optim_cls (torch.optim): The Optimizer object to use
            device (device): The device to run the training on.
            scene_state (wisp.core.State): Use this to inject a scene state from the
                                           outside to be synced elsewhere.
            extra_args (dict): Optional dict of extra_args for easy prototyping.
        """
        log.info(f'Training on {extra_args["dataset_path"]}')

        self.extra_args = extra_args

        self.cuda = "cuda" in str(device)
        self.verbose = extra_args["verbose"]
        self.use_gpu = extra_args["use_gpu"]
        self.use_tqdm = extra_args["use_tqdm"]
        self.space_dim = extra_args["space_dim"]
        self.gpu_fields = extra_args["gpu_data"]

        self.pipeline = pipeline

        self.device = device
        if device == "gpu":
            device_name = torch.cuda.get_device_name(device=self.device)
            log.info(f'Using {device_name} with CUDA v{torch.version.cuda}')

        self.dataset = dataset

        # Optimizer params
        self.optim_cls = optim_cls
        self.optim_params = optim_params
        self.lr = extra_args["grid_lr"]
        self.weight_decay = extra_args["weight_decay"]
        self.grid_lr_weight = extra_args["grid_lr_weight"]

        # Training params
        self.exp_name = extra_args["exp_name"]
        self.num_epochs = extra_args["num_epochs"]
        self.batch_size = extra_args["batch_size"]

        self.timer = PerfTimer(
            activate=extra_args["activate_trainer_timer"],
            show_memory=extra_args["show_memory"]
        )
        self.timer.reset()

        # In-training variables
        self.train_data_loader_iter = None
        self.val_data_loader = None
        self.dataset_size = None
        self.log_dict = {}

        self.log_fname = f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        if self.extra_args["log_fname"] is not None:
            self.log_fname += "-" + self.extra_args["log_fname"]

        if extra_args["on_cedar"] or extra_args["on_graham"]:
            log_dir = extra_args["cedar_log_dir"]
        else: log_dir = extra_args["log_dir"]
        self.log_dir = os.path.join(log_dir, self.exp_name, self.log_fname)

        # Default TensorBoard Logging
        self.writer = SummaryWriter(self.log_dir, purge_step=0)
        # self.writer.add_text('Info', self.info)

        self.valid_every = extra_args["valid_every"]
        self.log_tb_every = extra_args["log_tb_every"]
        self.log_cli_every = extra_args["log_cli_every"]
        self.plot_grad_every = extra_args["plot_grad_every"]
        self.render_tb_every = extra_args["render_tb_every"]
        self.save_data_every = extra_args["save_data_every"]
        self.save_model_every = extra_args["save_model_every"]
        # self.timer.check('set_logger')

        self.iteration = 1

        # save config file to log directory
        dst = join(self.log_dir, "config.yaml")
        shutil.copyfile(extra_args["config"], dst)

    def init_dataloader(self):
        self.train_data_loader = DataLoader(self.dataset,
                                            batch_size=self.batch_size,
                                            collate_fn=default_collate,
                                            shuffle=True, pin_memory=True, num_workers=0)

    def init_optimizer(self):
        """ Default initialization for the optimizer.
        """
        params_dict = { name : param
                        for name, param in self.pipeline.nef.named_parameters() }
        params = []
        decoder_params = []
        grid_params = []
        rest_params = []

        for name in params_dict:

            if 'decoder' in name:
                # If "decoder" is in the name, there's a good chance it is in fact a decoder,
                # so use weight_decay
                decoder_params.append(params_dict[name])

            elif 'grid' in name:
                # If "grid" is in the name, there's a good chance it is in fact a grid,
                # so use grid_lr_weight
                grid_params.append(params_dict[name])

            else:
                rest_params.append(params_dict[name])

        params.append({"params" : decoder_params,
                       "lr": self.lr,
                       "weight_decay": self.weight_decay})

        params.append({"params" : grid_params,
                       "lr": self.lr * self.grid_lr_weight})

        params.append({"params" : rest_params,
                       "lr": self.lr})

        self.optimizer = self.optim_cls(params, **self.optim_params)

    def init_renderer(self):
        """Default initalization for the renderer.
        """
        self.renderer = OfflineRenderer(**self.extra_args)

    def resample_dataset(self):
        """
        Override this function if some custom logic is needed.

        Args:
            (torch.utils.data.Dataset): Training dataset.
        """
        if hasattr(self.dataset, 'resample'):
            log.info("Reset DataLoader")
            self.dataset.resample()
            self.init_dataloader()
            self.timer.check('create_dataloader')
        else:
            raise ValueError("resample=True but the dataset doesn't have a resample method")

    def init_log_dict(self):
        """
        Override this function to use custom logs.
        """
        self.log_dict['total_loss'] = 0.0
        self.log_dict['total_iter_count'] = 0

    def pre_epoch(self):
        """
        Override this function to change the pre-epoch preprocessing.
        This function runs once before the epoch.
        """
        # The DataLoader is refreshed befored every epoch, because by default, the dataset refreshes
        # (resamples) after every epoch.

        self.loss_lods = list(range(0, self.extra_args["num_lods"]))
        if self.extra_args["grow_every"] > 0:
            self.grow()

        if self.extra_args["only_last"]:
            self.loss_lods = self.loss_lods[-1:]

        if self.extra_args["resample"] and self.epoch % self.extra_args["resample_every"] == 0:
            self.resample_dataset()

        self.pipeline.train()

        self.timer.check('pre_epoch done')

    def post_epoch(self):
        """ Override this function to change the post-epoch post processing.
            By default, this function logs to Tensorboard, renders images to Tensorboard,
              saves the model, and resamples the dataset.
            To keep default behaviour but also augment with other features, do
            super().post_epoch() in the derived method.
        """
        self.pipeline.eval()

        total_loss = self.log_dict['total_loss'] / len(self.train_data_loader)
        self.scene_state.optimization.losses['total_loss'].append(total_loss)

        self.log_cli()
        self.log_tb()

        # Render visualizations to tensorboard
        if self.render_tb_every > -1 and self.epoch % self.render_tb_every == 0:
            self.render_tb()

       # Save model
        if self.save_every > -1 and self.epoch % self.save_every == 0 and self.epoch != 0:
            self.save_model()

        self.timer.check('post_epoch done')

    def pre_step(self):
        """ Override this function to change the pre-step preprocessing (runs per iteration).
        """
        return

    def post_step(self):
        """ Override this function to change the pre-step preprocessing (runs per iteration).
        """
        return

    def grow(self):
        stage = min(self.extra_args["num_lods"],
                    (self.epoch // self.extra_args["grow_every"]) + 1) # 1 indexed
        if self.extra_args["growth_strategy"] == 'onebyone':
            self.loss_lods = [stage-1]
        elif self.extra_args["growth_strategy"] == 'increase':
            self.loss_lods = list(range(0, stage))
        elif self.extra_args["growth_strategy"] == 'shrink':
            self.loss_lods = list(range(0, self.extra_args["num_lods"]))[stage-1:]
        elif self.extra_args["growth_strategy"] == 'finetocoarse':
            self.loss_lods = list(range(
                0, self.extra_args["num_lods"]
            ))[self.extra_args["num_lods"] - stage:]
        elif self.extra_args["growth_strategy"] == 'onlylast':
            self.loss_lods = list(range(0, self.extra_args["num_lods"]))[-1:]
        else:
            raise NotImplementedError

    def begin_epoch(self):
        """Begin epoch.
        """
        self.reset_data_iterator()
        self.pre_epoch()
        self.init_log_dict()

    def end_epoch(self):
        """End epoch.
        """
        self.post_epoch()

        if self.extra_args["valid_every"] > -1 and \
                self.epoch % self.extra_args["valid_every"] == 0 and \
                self.epoch != 0:
            self.validate()
            self.timer.check('validate')

        if self.epoch < self.num_epochs:
            self.iteration = 1
            self.epoch += 1
        else:
            self.scene_state.optimization.running = False

    def reset_data_iterator(self):
        """Rewind the iterator for the new epoch.
        """
        self.scene_state.optimization.iterations_per_epoch = len(self.train_data_loader)
        self.train_data_loader_iter = iter(self.train_data_loader)

    def next_batch(self):
        """Actually iterate the data loader.
        """
        data = next(self.train_data_loader_iter)
        self.timer.check("got data")
        return data

    def iterate(self):
        """Advances the training by one training step (batch).
        """
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

    @abstractmethod
    def step(self, data):
        """Advance the training by one step using the batched data supplied.

        data (dict): Dictionary of the input batch from the DataLoader.
        """
        pass

    def log_cli(self):
        """
        Override this function to change CLI logging.

        By default, this function only runs every epoch.
        """
        # Average over iterations
        log_text = 'EPOCH {}/{}'.format(self.epoch, self.num_epochs)
        log_text += ' | total loss: {:>.3E}'.format(self.log_dict['total_loss'] / len(self.train_data_loader))

    def log_tb(self):
        """
        Override this function to change loss / other numeric logging to TensorBoard / Wandb.
        """
        for key in self.log_dict:
            if 'loss' in key:
                self.writer.add_scalar(f'Loss/{key}', self.log_dict[key] / len(self.train_data_loader), self.epoch)
                if self.using_wandb:
                    log_metric_to_wandb(f'Loss/{key}', self.log_dict[key] / len(self.train_data_loader), self.epoch)

    def render_tb(self):
        """
        Override this function to change render logging to TensorBoard / Wandb.
        """
        self.pipeline.eval()
        for d in [self.extra_args["num_lods"] - 1]:
            out = self.renderer.shade_images(self.pipeline,
                                             f=self.extra_args["camera_origin"],
                                             t=self.extra_args["camera_lookat"],
                                             fov=self.extra_args["camera_fov"],
                                             lod_idx=d,
                                             camera_clamp=self.extra_args["camera_clamp"])

            # Premultiply the alphas since we're writing to PNG (technically they're already premultiplied)
            if self.extra_args["bg_color"] == 'black' and out.rgb.shape[-1] > 3:
                bg = torch.ones_like(out.rgb[..., :3])
                out.rgb[..., :3] += bg * (1.0 - out.rgb[..., 3:4])

            out = out.image().byte().numpy_dict()

            log_buffers = ['depth', 'hit', 'normal', 'rgb', 'alpha']

            for key in log_buffers:
                if out.get(key) is not None:
                    self.writer.add_image(f'{key}/{d}', out[key].T, self.epoch)
                    if self.using_wandb:
                        log_images_to_wandb(f'{key}/{d}', out[key].T, self.epoch)

    def save_model(self):
        """
        Override this function to change model saving.
        """

        model_fname = os.path.join(self.log_dir, f'model-ep{self.epoch}-it{self.iteration}.pth')

        log.info(f'Saving model checkpoint to: {model_fname}')
        if self.extra_args["model_format"] == "full":
            torch.save(self.pipeline, model_fname)
        else:
            torch.save(self.pipeline.state_dict(), model_fname)

        if self.using_wandb:
            name = wandb.util.make_artifact_name_safe(f"{wandb.run.name}-model")
            model_artifact = wandb.Artifact(name, type="model")
            model_artifact.add_file(model_fname)
            wandb.run.log_artifact(model_artifact, aliases=["latest", f"ep{self.epoch}_it{self.iteration}"])

    def train(self):
        """
        Override this if some very specific training procedure is needed.
        """
        self.scene_state.optimization.running = True

        while self.scene_state.optimization.running:
            self.iterate()

        self.writer.close()

    @abstractmethod
    def validate(self):
        pass

    #######################
    # Properties
    #######################

    # @property
    # def epoch(self) -> int:
    #     return self.scene_state.optimization.epoch

    # @epoch.setter
    # def epoch(self, epoch: int) -> int:
    #     self.scene_state.optimization.epoch = epoch

    # @property
    # def iteration(self) -> int:
    #     return self.scene_state.optimization.iteration

    # @iteration.setter
    # def iteration(self, iteration: int) -> int:
    #     self.scene_state.optimization.iteration = iteration
