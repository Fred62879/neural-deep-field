
import os
import sys
import yaml
import torch
import pprint
import argparse

from wisp.datasets import *
from wisp.trainers import *
from wisp.inferrers import *
from wisp.models.nefs import *
from wisp.datasets.transforms import *
from wisp.models import AstroPipeline
from wisp.models.layers import init_codebook


str2optim = {m.lower(): getattr(torch.optim, m) for m in dir(torch.optim) if m[0].isupper()}

def register_class(cls, name):
    globals()[name] = cls

def get_pretrain_pipelines(pipelines, tasks, args):
    if not args.pretrain_codebook: return

    if "codebook_pretrain" in tasks:
        assert(args.quantize_latent or args.quantize_spectra)
        pretrain_nef = CodebookPretrainNerf(
            args.codebook_pretrain_pixel_supervision, **vars(args))
        pipelines["codebook_net"] = AstroPipeline(pretrain_nef)

    if "pretrain_infer" in tasks:
        pretrain_nef = CodebookPretrainNerf(
            pretrain_pixel_supervision=args.codebook_pretrain_pixel_supervision,
            **vars(args)
        )
        pipelines["full"] = AstroPipeline(pretrain_nef)

        if "recon_gt_spectra" in tasks:
            spectra_nef = CodebookPretrainNerf(**vars(args))
            pipelines["spectra_infer"] = AstroPipeline(spectra_nef)

        if "recon_codebook_spectra" in tasks:
            codebook_nef = CodebookNef(**vars(args))
            pipelines["codebook_spectra_infer"] = AstroPipeline(codebook_nef)

        elif "recon_codebook_spectra_individ" in tasks:
            codebook_spectra_nef = CodebookPretrainNerf(**vars(args))
            pipelines["codebook_spectra_infer"] = AstroPipeline(codebook_spectra_nef)

    return pipelines

def get_main_train_pipelines(pipelines, tasks, args):
    if "train" in tasks:
        # full pipline for training
        # nef_train = globals()[args.nef_type](**vars(args))
        nef_train = AstroHyperSpectralNerf(**vars(args))
        pipelines["full"] = AstroPipeline(nef_train)

    if "infer" in tasks:
        # full pipline for img recon
        nef_train = AstroHyperSpectralNerf(**vars(args))
        pipelines["full"] = AstroPipeline(nef_train)

        # pipeline for spectra inferrence
        if "recon_gt_spectra" in tasks or "recon_dummy_spectra" in tasks:
            spectra_nef = AstroHyperSpectralNerf(integrate=False, **vars(args))
            pipelines["spectra_infer"] = AstroPipeline(spectra_nef)

        # pipeline for codebook spectra inferrence
        if "recon_codebook_spectra" in tasks:
            codebook_nef = CodebookNef(**vars(args))
            pipelines["codebook_spectra_infer"] = AstroPipeline(codebook_nef)

        elif "recon_codebook_spectra_individ" in tasks:
            codebook_nef = AstroHyperSpectralNerf(integrate=False, **vars(args))
            pipelines["codebook_spectra_infer"] = AstroPipeline(codebook_nef)

    return pipelines

def get_pipelines_from_config(args, tasks={}):
    """ Utility function to get all required pipelines from the parsed config.
    """
    pipelines = {}
    if args.use_gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else: device = "cpu"

    if args.dataset_type == 'astro':
        if args.pretrain_codebook:
            get_pretrain_pipelines(pipelines, tasks, args)
        get_main_train_pipelines(pipelines, tasks, args)
    else:
        raise ValueError(f"{args.dataset_type} unrecognized dataset_type")

    for _, pipeline in pipelines.items():
        pipeline.to(device)
    return device, pipelines

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

def get_trainer_from_config(trainer_cls, pipeline, dataset, optim_cls, optim_params, device, args):
    trainer = trainer_cls(
        pipeline, dataset, optim_cls, optim_params, device, **vars(args)
    )
    return trainer

def get_inferrer_from_config(pipelines, dataset, device, mode, args):
    inferrer = globals()[args.inferrer_type](
        pipelines, dataset, device, mode=mode, **vars(args))
    return inferrer
