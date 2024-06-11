
import os
import sys
import yaml
import torch
import pprint
import argparse

from wisp.models import *
from wisp.datasets import *
from wisp.trainers import *
from wisp.inferrers import *
from wisp.models.nefs import *
from wisp.datasets.transforms import *
from wisp.utils.common import has_common, find_common


str2optim = {m.lower(): getattr(torch.optim, m) for m in dir(torch.optim) if m[0].isupper()}

def register_class(cls, name):
    globals()[name] = cls

def get_device_from_config(args):
    if args.use_gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else: device = "cpu"
    return device

def get_dataset_from_config(device, args):
    if args.dataset_type == 'astro':
        dataset = AstroDataset(device=device, transform=None, **vars(args))
        dataset.init()
    else:
        raise ValueError(f'"{args.dataset_type}" unrecognized dataset_type')
    return dataset

def get_optimizer_from_config(args):
    optim_cls = str2optim[args.optimizer_type]
    if args.optimizer_type == 'adam':
        optim_params = {'lr': 1e-5, 'eps': 1e-8, 'betas': (args.b1, args.b2),
                        'weight_decay':  args.weight_decay}
    elif args.optimizer_type == 'adamw':
        optim_params = {'lr': 1e-5, 'eps': 1e-8, 'betas': (args.b1, args.b2),
                        'weight_decay':  args.weight_decay}
    elif args.optimizer_type == 'sgd':
        optim_params = {'momentum': args.sgd_momentum}
    else: optim_params = {}
    return optim_cls, optim_params

def get_trainer_from_config(pipeline, dataset, optim_cls, optim_params, device, tasks, args):
    trainer_cls = AstroTrainer if "main_train" in tasks else SpectraTrainer
    trainer = trainer_cls(
        pipeline, dataset, optim_cls, optim_params, device, **vars(args))
    return trainer

def get_inferrer_from_config(pipelines, dataset, device, tasks, args):
    infer_tasks = ["main_infer","test","spectra_pretrain_infer","sanity_check_infer",
                   "generalization_infer","redshift_classification_sc_infer",
                   "redshift_classification_genlz_infer",
                   "redshift_pretrain_infer","redshift_test_infer"]
    cur_infer_task = find_common(tasks, infer_tasks)[0]
    inferrer = globals()[args.inferrer_type](
        pipelines, dataset, device, mode=cur_infer_task, **vars(args))
    return inferrer

def get_train_pipeline_from_config(device, tasks, args):
    if "main_train" in tasks:
        if args.pixel_supervision:
            pipeline = AstroPipeline(globals()[args.nef_type](**vars(args)))
        elif args.spectra_supervision:
            pipeline = AstroPipeline(globals()[args.nef_type](integrate=False,**vars(args)))
        else: raise ValueError()
    elif "redshift_pretrain" in tasks:
        pipeline = AstroPipeline(SpectraBaseline(**vars(args)))
    elif has_common(tasks, ["spectra_pretrain","sanity_check","generalization"]):
        pipeline = AstroPipeline(SpectraNerf(**vars(args)))
    elif has_common(tasks, ["redshift_classification_train"]):
        pipeline = AstroPipeline(RedshiftClassifier(**vars(args)))
    else: raise ValueError()
    pipeline.to(device)
    return pipeline

def get_infer_pipelines_from_config(device, tasks, args):
    pipelines = {}
    if "main_infer" in tasks or "test" in tasks:
        pipelines["full"] = AstroPipeline(globals()[args.nef_type](**vars(args)))

        if "recon_gt_spectra" in tasks in tasks:
            pipelines["spectra_infer"] = AstroPipeline(
                AstroHyperSpectralNerf(integrate=False, **vars(args)))

        if "recon_codebook_spectra" in tasks:
            pipelines["codebook_spectra_infer"] = AstroPipeline(CodebookNef(**vars(args)))
        elif "recon_codebook_spectra_individ" in tasks:
            pipelines["codebook_spectra_infer"] = AstroPipeline(
                AstroHyperSpectralNerf(integrate=False, **vars(args)))

    elif "redshift_pretrain_infer" in tasks or "redshift_test_infer" in tasks:
        pipelines["spectra_baseline"] = AstroPipeline(SpectraBaseline(**vars(args)))

    elif has_common(
        tasks, ["spectra_pretrain_infer","sanity_check_infer","generalization_infer"]
    ):
        pipelines["full"] = AstroPipeline(SpectraNerf(**vars(args)))

        # if "recon_spectra" in tasks or "recon_spectra_all_bins" in tasks or \
        #    "save_redshift" in tasks or "plot_redshift_logits" in tasks or \
        #    "plot_codebook_coeff" in tasks or \
        #    "plot_binwise_spectra_loss" in tasks or \
        #    "plot_codebook_coeff_all_bins" in tasks or \
        #    "save_redshift_classification_data" in tasks or \
        #    "plot_global_lambdawise_spectra_loss" in tasks:
        pipelines["spectra_infer"] = AstroPipeline(SpectraNerf(**vars(args)))

        if "recon_codebook_spectra" in tasks:
            pipelines["codebook_spectra_infer"] = AstroPipeline(CodebookNef(**vars(args)))
        elif "recon_codebook_spectra_individ" in tasks:
            pipelines["codebook_spectra_infer"] = AstroPipeline(SpectraNerf(**vars(args)))

    elif has_common(
        tasks, ["redshift_classification_sc_infer","redshift_classification_genlz_infer"]
    ):
        pipelines["redshift_classifier"] = AstroPipeline(RedshiftClassifier(**vars(args)))

    for _, pipeline in pipelines.items():
        pipeline.to(device)
        # log.info(pipeline)
    return pipelines
