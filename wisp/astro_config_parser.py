
import os
import sys
import yaml
import torch
import pprint
import argparse

from wisp.datasets import *
from wisp.models.nefs import *
from wisp.models.grids import *
from wisp.models import Pipeline
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
    if args.dataset_type == 'astro2d' or args.dataset_type == 'astro3d':
        dataset = AstroDataset(**vars(args))
        dataset.init()
    else:
        raise ValueError(f'"{args.dataset_type}" unrecognized dataset_type')
    return dataset

def get_pipelines_from_config(args, tasks=[]):
    """ Utility function to get the pipelines from the parsed config. """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.dataset_type == 'astro2d':
        nef = globals()[args.nef_type](**vars(args))
        pipeline = Pipeline(nef)
        pipeline.nef.grid.init_from_geometric(
            args.min_grid_res, args.max_grid_res, args.num_lods)

    elif args.dataset_type == 'astro3d':
        nef = globals()[args.nef_type](**vars(args))
        spatial_pipeline = Pipeline(nef)
        spatial_pipeline.nef.grid.init_from_geometric(
            args.min_grid_res, args.max_grid_res, args.num_lods)
    else:
        raise ValueError(f'"{args.dataset_type}" unrecognized dataset_type')

    pipeline.to(device)
    return device, [pipeline]

def init_model(norm_cho, inte_cho, args, pe_coord=True, pe_wave=True,
               encode=True, encoder_output_scaler=True, calculate_loss=True,
               covr_rnge=None, nsmpl_within_each_band=None,
               full_wave=None, full_trans=None, num_coords_with_full_wave=-1):
    if args.dim == 2:
        model = MLP2D(norm_cho, args, calculate_loss)
    elif args.dim == 3:
        model = Monte_Carlo(norm_cho, inte_cho, pe_coord, pe_wave, encode,
                            encoder_output_scaler, calculate_loss,
                            args, covr_rnge=covr_rnge,
                            nsmpl_within_each_band=nsmpl_within_each_band,
                            full_wave=full_wave, full_trans=full_trans,
                            num_coords_with_full_wave=num_coords_with_full_wave)
    else:
        raise Exception('Unsupported monte carlo choice')
    print(model)
    return model if not args.cuda else model.cuda()

