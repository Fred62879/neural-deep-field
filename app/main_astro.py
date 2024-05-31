
if __name__ == "__main__":


    import os
    from pathlib import Path
    import sys
    path_root = Path(__file__).parents[1]
    sys.path.append(str(path_root))

    import logging as log

    from wisp.configuration import *
    from wisp.parser import parse_args
    from wisp.utils.common import query_GPU_mem, has_common
    from wisp.trainers import AstroTrainer, CodebookTrainer

    args, args_str = parse_args()
    args.wave_embed_dim = args.spectra_latent_dim

    if args.use_gpu:
        query_GPU_mem()

    tasks = set(args.tasks)
    dataset = get_dataset_from_config(args)
    device, pipelines = get_pipelines_from_config(args, tasks=tasks)

    def infer(mode):
        inferrer = get_inferrer_from_config(pipelines, dataset, device, mode, args)
        inferrer.infer()

    if "codebook_pretrain" in tasks and args.pretrain_codebook:
        optim_cls, optim_params = get_optimizer_from_config(args)
        trainer = get_trainer_from_config(
            CodebookTrainer,
            [ pipelines["codebook_net"] ],
            dataset, optim_cls, optim_params, device, args
        )
        trainer.train()

    elif has_common(tasks, ["sanity_check","generalization"]) and args.pretrain_codebook:
        optim_cls, optim_params = get_optimizer_from_config(args)
        trainer_cls = CodebookTrainerDebug if args.debug_lbfgs else CodebookTrainer
        trainer = get_trainer_from_config(
            trainer_cls,
            [ pipelines["codebook_net"] ],
            dataset, optim_cls, optim_params, device, args
        )
        trainer.train()

    elif has_common(tasks, ["redshift_classification_train","redshift_classification_genlz"]):
        optim_cls, optim_params = get_optimizer_from_config(args)
        trainer = get_trainer_from_config(
            CodebookTrainer, [ pipelines["redshift_classifier"] ],
            dataset, optim_cls, optim_params, device, args
        )
        trainer.train()

    elif "train" in tasks:
        optim_cls, optim_params = get_optimizer_from_config(args)
        trainer = get_trainer_from_config(
            AstroTrainer, pipelines["full"], dataset, optim_cls, optim_params, device, args)
        trainer.train()

    elif "redshift_classification_sc_infer" in tasks:
        infer("redshift_classification_sc_infer")
    elif "redshift_classification_genlz_infer" in tasks:
        infer("redshift_classification_genlz_infer")
    elif "codebook_pretrain_infer" in tasks:
        infer("codebook_pretrain_infer")
    elif "sanity_check_infer" in tasks:
        infer("sanity_checl_infer")
    elif "generalization_infer" in tasks:
        infer("generalization_infer")
    elif "main_infer" in tasks:
        infer("main_infer")
    elif "test" in tasks:
        infer("test")
    else: raise ValueError("unsupported task")
