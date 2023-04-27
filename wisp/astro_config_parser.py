
import os
import sys
import yaml
import torch
import pprint
import argparse

from wisp.datasets import *
from wisp.models.nefs import *
from wisp.models import AstroPipeline
from wisp.datasets.transforms import *


str2optim = {m.lower(): getattr(torch.optim, m) for m in dir(torch.optim) if m[0].isupper()}

def register_class(cls, name):
    globals()[name] = cls

def get_optimizer_from_config(args):
    """ Utility function to get the optimizer from the parsed config.
    """
    optim_cls = str2optim[args.optimizer_type]
    if args.optimizer_type == 'adam':
        #optim_params = {'eps': 1e-15}
        optim_params = {'lr': 1e-5, 'eps': 1e-8, 'betas': (args.b1, args.b2),
                        'weight_decay':  args.weight_decay}
    elif args.optimizer_type == 'sgd':
        optim_params = {'momentum': 0.8}
    else:
        optim_params = {}

    return optim_cls, optim_params

def get_dataset_from_config(args):
    """ Utility function to get the dataset from the parsed config.
    """
    if args.use_gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else: device = "cpu"

    transform = None #AddToDevice(device)
    if args.dataset_type == 'astro':
        dataset = AstroDataset(device=device, transform=transform, **vars(args))
        dataset.init()
    else:
        raise ValueError(f'"{args.dataset_type}" unrecognized dataset_type')
    return dataset

def get_pipelines_from_config(args, tasks=[]):
    """ Utility function to get the pipelines from the parsed config.
    """
    pipelines = {}
    tasks = set(tasks)
    if args.use_gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else: device = "cpu"

    if args.dataset_type == 'astro':
        nef_train = globals()[args.nef_type](**vars(args))
        pipelines["full"] = AstroPipeline(nef_train)
        # log.info(pipelines["full"])

        # pipeline for spectra inferrence
        if "recon_gt_spectra" in tasks or "recon_dummy_spectra" in tasks:
            nef_infer_spectra = globals()[args.nef_type](
                integrate=False, qtz_calculate_loss=False, **vars(args))
            pipelines["spectra_infer"] = AstroPipeline(nef_infer_spectra)

        # pipeline for codebook spectra inferrence
        if "recon_codebook_spectra" in tasks:
            codebook_nef = CodebookNef(integrate=False, **vars(args))
            pipelines["codebook"] = AstroPipeline(codebook_nef)
    else:
        raise ValueError(f"{args.dataset_type} unrecognized dataset_type")

    for _, pipeline in pipelines.items():
        pipeline.to(device)
    return device, pipelines
