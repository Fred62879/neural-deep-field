
import os
import sys
import yaml
import torch
import pprint
import argparse

from wisp.datasets import *
from wisp.models.nefs import *
from wisp.models.grids import *
from wisp.models.hypers import *
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
        optim_params = {'eps': 1e-15}
    elif args.optimizer_type == 'sgd':
        optim_params = {'momentum': 0.8}
    else:
        optim_params = {}
    return optim_cls, optim_params

def get_dataset_from_config(args):
    """ Utility function to get the dataset from the parsed config. """
    if args.dataset_type == 'astro':
        dataset = AstroDataset(**vars(args))
        dataset.init()
    else:
        raise ValueError(f'"{args.dataset_type}" unrecognized dataset_type')
    return dataset

def get_pipelines_from_config(args, tasks=[]):
    """ Utility function to get the pipelines from the parsed config.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.dataset_type == 'astro':
        nef, quantz, hyper_decod = None, None, None

        if args.space_dim == 2 or args.coords_embed_method in {"positional","grid"}:
            nef = globals()[args.nef_type](**vars(args))

        if args.space_dim == 3:
            if args.quantize_latent:
                quantz = LatentQuantizer(**vars(args))
            hyper_decod = HyperSpectralDecoder(**vars(args))

        pipeline = AstroPipeline(nef, quantz, hyper_decod)
    else:
        raise ValueError(f"{args.dataset_type} unrecognized dataset_type")

    log.info(pipeline)
    pipeline.to(device)
    return device, [pipeline]
