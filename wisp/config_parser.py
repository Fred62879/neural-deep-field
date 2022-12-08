# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import os
import sys
import argparse
import pprint
import yaml
import torch
from wisp.datasets import *
from wisp.models import Pipeline
from wisp.models.nefs import *
from wisp.models.grids import *
from wisp.tracers import *
from wisp.datasets.transforms import *

str2optim = {m.lower(): getattr(torch.optim, m) for m in dir(torch.optim) if m[0].isupper()}

def register_class(cls, name):
    globals()[name] = cls

def get_optimizer_from_config(args):
    """Utility function to get the optimizer from the parsed config.
    """
    optim_cls = str2optim[args.optimizer_type]
    if args.optimizer_type == 'adam':
        optim_params = {'eps': 1e-15}
    elif args.optimizer_type == 'sgd':
        optim_params = {'momentum': 0.8}
    else:
        optim_params = {}
    return optim_cls, optim_params

def get_modules_from_config(args):
    """Utility function to get the modules for training from the parsed config.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nef = globals()[args.nef_type](**vars(args))
    if not args.trainer_type == 'astro_trainer': tracer = None
    else: tracer = globals()[args.tracer_type](**vars(args))
    #tracer = globals()[args.tracer_type](**vars(args))
    pipeline = Pipeline(nef, tracer)

    if args.pretrained:
        if args.model_format == "full":
            pipeline = torch.load(args.pretrained)
        else:
            pipeline.load_state_dict(torch.load(args.pretrained))
    pipeline.to(device)

    if args.dataset_type == 'astro2d':
        train_dataset = AstroDataset(**vars(args))
        train_dataset.init()

        pipeline.nef.grid.init_from_geometric(args.min_grid_res, args.max_grid_res, args.num_lods)
        pipeline.to(device)

    elif args.dataset_type == "multiview":
        transform = SampleRays(args.num_rays_sampled_per_img)
        train_dataset = MultiviewDataset(**vars(args), transform=transform)
        train_dataset.init()

        if pipeline.nef.grid is not None:
            if isinstance(pipeline.nef.grid, OctreeGrid):
                if not args.valid_only and not pipeline.nef.grid.blas_initialized():
                    if args.multiview_dataset_format in ['rtmv']:
                        pipeline.nef.grid.init_from_pointcloud(train_dataset.coords)
                    else:
                        pipeline.nef.grid.init_dense()
                    pipeline.to(device)
            if isinstance(pipeline.nef.grid, HashGrid):
                if not args.valid_only:
                    if args.tree_type == 'quad':
                        pipeline.nef.grid.init_from_octree(args.base_lod, args.num_lods)
                    elif args.tree_type == 'geometric':
                        pipeline.nef.grid.init_from_geometric(16, args.max_grid_res, args.num_lods)
                    else:
                        raise NotImplementedError
                    pipeline.to(device)

    elif args.dataset_type == "sdf":
        train_dataset = SDFDataset(args.sample_mode, args.num_samples,
                                   args.get_normals, args.sample_tex)

        if pipeline.nef.grid is not None:
            if isinstance(pipeline.nef.grid, OctreeGrid):

                if not args.valid_only and not pipeline.nef.grid.blas_initialized():
                    pipeline.nef.grid.init_from_mesh(
                        args.dataset_path, sample_tex=args.sample_tex, num_samples=args.num_samples_on_mesh)
                    pipeline.to(device)

                train_dataset.init_from_grid(pipeline.nef.grid, args.samples_per_voxel)
            else:
                train_dataset.init_from_mesh(args.dataset_path, args.mode_mesh_norm)
    else:
        raise ValueError(f'"{args.dataset_type}" unrecognized dataset_type')
    return pipeline, train_dataset, device
