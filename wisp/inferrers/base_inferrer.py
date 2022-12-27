
import os
import torch
import logging as log

from os.path import join
from datetime import datetime
from wisp.utils import PerfTimer
from abc import ABC, abstractmethod
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import BatchSampler, SequentialSampler, DataLoader

class BaseInferrer(ABC):
    """ Base class for inferrence.
    """

    def __init__(self, pipelines, dataset, device, extra_args, info=None):

        # inferrence tasks to perform
        self.tasks = set(extra_args["tasks"])

        self.device = device
        self.dataset = dataset
        self.pipelines = pipelines

        self.verbose = extra_args["verbose"]
        self.space_dim = extra_args["space_dim"]
        self.num_bands = extra_args["num_bands"]
        self.batch_size = extra_args["infer_batch_size"]

        self.info = info
        self.extra_args = extra_args

        self.timer = PerfTimer(activate=extra_args["perf"])
        self.timer.reset()

        self.infer_data_loader_iter = None
        self.init_dataloader()
        self.reset_data_iterator()

        # set log dir
        if extra_args["infer_log_fname"] is not None:
            infer_log_fname = extra_args["infer_log_fname"]
        else:
            fnames = os.listdir(join(extra_args["log_dir"],extra_args["exp_name"]))
            fnames.sort()
            infer_log_fname = fnames[-1]

        self.log_dir = join(extra_args["log_dir"], extra_args["exp_name"], infer_log_fname)

        # Default TensorBoard Logging
        self.writer = SummaryWriter(self.log_dir, purge_step=0)
        self.writer.add_text('Info', self.info)
        self.timer.check('set_logger')

        # initialization
        self.model_dir = "/"
        self.selected_model_fnames = []
        self.group_tasks = []

    #############
    # Dataloader
    #############

    def init_dataloader(self):
        self.infer_data_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=BatchSampler(
                SequentialSampler(self.dataset), batch_size=self.batch_size, drop_last=False),
            #pin_memory=True,
            num_workers=0
        )

    def reset_data_iterator(self):
        """ Rewind the iterator. """
        self.infer_data_loader_iter = iter(self.infer_data_loader)

    def next_batch(self):
        """ Actually iterate the data loader. """
        return next(self.infer_data_loader_iter)

    def reset_dataloader(self):
        """ Select dataset based on current inferrence task.
            Then re-init dataloader.
        """
        self.configure_dataset()
        self.init_dataloader()
        self.reset_data_iterator()
        self.fits_ids = self.dataset.get_fits_ids()

    #############
    # Inferrence steps
    #############

    def infer(self):
        """ Perform each inferrence task (one at a time) using all selected models.
        """
        for group_task in self.group_tasks:
            self._toggle(group_task)
            self._register_inferrence_func(group_task)

            if self.verbose:
                log.info(f"inferring for {group_task}")
            self.pre_inferrence()
            self.inferrence()
            self.post_inferrence()

    def pre_inferrence(self):
        pass

    def inferrence(self):
        """ Perform current inferrence task using each selected checkpoints.
            Override if needed.
        """
        for model_id, model_fname in enumerate(self.selected_model_fnames):
            model_fname = join(self.model_dir, model_fname)
            checkpoint = torch.load(model_fname)["model_state_dict"]
            self.infer_with_checkpoint(model_id, checkpoint)

    def post_inferrence(self):
        pass

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
