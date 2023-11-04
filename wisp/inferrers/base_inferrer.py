
import os
import torch
import logging as log

from wisp.utils import PerfTimer
from wisp.datasets.patch_data import PatchData
from wisp.utils.common import create_patch_uid, get_bool_classify_redshift

from os.path import join
from datetime import datetime
from abc import ABC, abstractmethod
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import BatchSampler, SequentialSampler, DataLoader

class BaseInferrer(ABC):
    """ Base class for inferrence.
    """
    def __init__(self, pipelines, dataset, device, mode, **extra_args):
        self.extra_args = extra_args

        self.mode = mode
        self.device = device
        self.dataset = dataset
        self.pipelines = pipelines
        self.run_model = True
        self.cuda = "cuda" in str(device)

        self.space_dim = extra_args["space_dim"]
        self.num_bands = extra_args["num_bands"]
        self.batch_size = extra_args["infer_batch_size"]
        self.log_pixel_ratio = extra_args["log_pixel_ratio"]
        self.neighbour_size = extra_args["spectra_neighbour_size"]
        self.trans_sample_method = extra_args["trans_sample_method"]

        self.verbose = extra_args["verbose"]
        self.infer_selected = extra_args["infer_selected"]
        self.plot_residual_map = extra_args["plot_residual_map"]
        self.infer_last_model_only = extra_args["infer_last_model_only"]
        self.recon_spectra_pixels_only = extra_args["train_spectra_pixels_only"]
        self.use_pretrained_latents_as_coords = \
            extra_args["main_train_with_pretrained_latents"]
        self.optimize_codebook_latents_for_each_redshift_bin = \
            self.extra_args["optimize_codebook_latents_for_each_redshift_bin"]

        self.index_latent = True
        self.split_latent = self.mode == "redshift_pretrain_infer" and \
            self.extra_args["split_latent"]

        self.timer = PerfTimer(activate=extra_args["perf"])
        self.timer.reset()

        self.infer_data_loader_iter = None
        self.init_dataloader()
        self.reset_data_iterator()

        # set log dir
        if extra_args["infer_log_dir"] is not None:
            infer_log_dir = extra_args["infer_log_dir"]
        else:
            dnames = os.listdir(join(extra_args["log_dir"],extra_args["exp_name"]))
            dnames.sort()
            infer_log_dir = dnames[-1]
        self.log_dir = join(extra_args["log_dir"], extra_args["exp_name"], infer_log_dir)

        # Default TensorBoard Logging
        self.writer = SummaryWriter(self.log_dir, purge_step=0)
        self.timer.check('set_logger')

        # initialization
        self.model_dir = "/"
        self.selected_model_fnames = []
        self.group_tasks = []

        self.summarize_inferrence_tasks()
        self.set_inferrence_funcs()

    def set_checkpoint(self, model_id, checkpoint):
        self.model_id = model_id
        self.checkpoint = checkpoint

    @abstractmethod
    def summarize_inferrence_tasks(self):
        self.group_tasks = []

    @abstractmethod
    def set_inferrence_funcs(self):
        self.infer_funcs = {}

    #############
    # Dataloader
    #############

    def init_dataloader(self, drop_last=False):
        self.infer_data_loader = DataLoader(
            self.dataset,
            batch_size=None,
            sampler=BatchSampler(
                SequentialSampler(self.dataset),
                batch_size=self.batch_size,
                drop_last=drop_last
            ),
            pin_memory=True,
            num_workers=self.extra_args["dataset_num_workers"]
        )

    def reset_data_iterator(self):
        """ Rewind the iterator. """
        self.infer_data_loader_iter = iter(self.infer_data_loader)

    def next_batch(self):
        """ Actually iterate the data loader. """
        return next(self.infer_data_loader_iter)

    def reset_dataloader(self, drop_last=False):
        """ Configure dataset based on current inferrence task.
            Then re-init dataloader.
        """
        self._configure_dataset()
        self.init_dataloader(drop_last=drop_last)
        self.reset_data_iterator()

    #############
    # Inferrence steps
    #############

    def infer(self):
        """ Perform each inferrence task (one at a time) using all selected models.
        """
        for i, (tract, patch) in enumerate(zip(
                self.extra_args["tracts"], self.extra_args["patches"]
        )):
            self.get_cur_patch_data(i, tract, patch)

            for group_task in self.group_tasks:
                self._toggle(group_task)
                self._register_inferrence_func(group_task)

                if self.verbose:
                    log.info(f"inferring for {group_task}")
                if self.run_model:
                    self.pre_inferrence()
                    self.inferrence_run_model()
                    self.post_inferrence()
                else:
                    self.inferrence_no_model_run()

    def pre_inferrence(self):
        pass

    def inferrence_run_model(self):
        """ Perform current inferrence task using each selected checkpoints.
            Override if needed.
        """
        for model_id, model_fname in enumerate(self.selected_model_fnames):
            model_fname = join(self.model_dir, model_fname)
            log.info(f"infer with {model_fname}")
            checkpoint = torch.load(model_fname)
            self.infer_with_checkpoint(model_id, checkpoint)

    def inferrence_no_model_run(self):
        """ Perform inferrence loading data from local. No model running.
        """
        pass

    def post_inferrence(self):
        pass

    #############
    # Get data for cur Patch
    #############

    def get_cur_patch_data(self, i, tract, patch):
        self.cur_patch = PatchData(
            tract, patch,
            load_spectra=self.extra_args["space_dim"] == 3,
            cutout_num_rows=self.extra_args["patch_cutout_num_rows"][i],
            cutout_num_cols=self.extra_args["patch_cutout_num_cols"][i],
            cutout_start_pos=self.extra_args["patch_cutout_start_pos"][i],
            full_patch=self.extra_args["use_full_patch"],
            spectra_obj=self.dataset.get_spectra_data_obj(),
            **self.extra_args
        )
        self.cur_patch_uid = create_patch_uid(tract, patch)
        if self.extra_args["space_dim"] == 3:
            self.val_spectra_map = self.cur_patch.get_spectra_bin_map()

    #############
    # Infer w/ one checkpoint
    #############

    def infer_with_checkpoint(self, model_id, checkpoint):
        self.pre_checkpoint(model_id)
        self.run_checkpoint(model_id, checkpoint)
        self.post_checkpoint(model_id)

    def pre_checkpoint(self, model_id):
        pass

    def run_checkpoint(self, model_id, checkpoint):
        pass

    def post_checkpoint(self, model_id):
        pass

    #############
    # Helpers
    #############

    def _toggle(self, task):
        # Toggle among all inferrence tasks s.t. only one task is performed at each time.
        assert(task in set(self.group_tasks))
        for group_task in self.group_tasks:
            setattr(self, group_task, False)
        setattr(self, task, True)

    def _register_inferrence_func(self, group_task):
        ( self.pre_inferrence,
          self.post_inferrence,
          self.pre_checkpoint,
          self.run_checkpoint,
          self.post_checkpoint ) = self.infer_funcs[group_task]

    @abstractmethod
    def _configure_dataset(self):
        """ Configure data fields and dataset size based on current task.
        """
        pass
