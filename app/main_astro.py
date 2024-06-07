
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
    from wisp.trainers import AstroTrainer, SpectraTrainer

    args, args_str = parse_args()
    # args.wave_embed_dim = args.spectra_latent_dim
    if args.use_gpu: query_GPU_mem()

    tasks = set(args.tasks)
    device = get_device_from_config(args)
    dataset = get_dataset_from_config(device, args)

    if "train" in tasks:
        optim_cls, optim_params = get_optimizer_from_config(args)
        pipeline = get_train_pipeline_from_config(device, tasks, args)
        trainer = get_trainer_from_config(
            pipeline, dataset, optim_cls, optim_params, device, tasks, args)
        trainer.train()
    elif "infer" in tasks:
        pipelines = get_infer_pipelines_from_config(device, tasks, args)
        inferrer = get_inferrer_from_config(pipelines, dataset, device, tasks, args)
        inferrer.infer()
    else:
        raise ValueError()
