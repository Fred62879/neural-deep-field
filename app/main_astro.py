
if __name__ == "__main__":

    import logging as log

    from wisp.astro_config_parser import *
    from wisp.parsers.parser import parse_args
    from wisp.trainers import AstroTrainer, CodebookTrainer
    from wisp.utils.common import set_seed, default_log_setup

    args, args_str = parse_args()
    tasks = set(args.tasks)
    default_log_setup(args.log_level)

    set_seed(args.seed)
    log.info(f"set seed as {args.seed}")

    dataset = get_dataset_from_config(args)
    device, pipelines = get_pipelines_from_config(args, tasks=tasks)

    if "codebook_pretrain" in tasks and args.pretrain_codebook:
        optim_cls, optim_params = get_optimizer_from_config(args)
        trainer = get_trainer_from_config(
            CodebookTrainer,
            [ pipelines["codebook_net"] ],
            dataset, optim_cls, optim_params, device, args
        )
        trainer.train()

    if "redshift_pretrain" in tasks and args.pretrain_codebook:
        optim_cls, optim_params = get_optimizer_from_config(args)
        trainer = get_trainer_from_config(
            CodebookTrainer,
            [ pipelines["codebook_net"] ],
            dataset, optim_cls, optim_params, device, args
        )
        trainer.train()

    if "train" in tasks:
        optim_cls, optim_params = get_optimizer_from_config(args)
        trainer = get_trainer_from_config(
            AstroTrainer, pipelines["full"], dataset, optim_cls, optim_params, device, args)
        trainer.train()

    if "codebook_pretrain_infer" in tasks and args.pretrain_codebook:
        # infer for pretrained model (recon gt spectra & codebook spectra ect.)
        inferrer = get_inferrer_from_config(
            pipelines, dataset, device, "codebook_pretrain_infer", args)
        inferrer.infer()

    if "redshift_pretrain_infer" in tasks and args.pretrain_codebook:
        # infer for pretrained model (recon gt spectra & codebook spectra ect.)
        inferrer = get_inferrer_from_config(
            pipelines, dataset, device, "redshift_pretrain_infer", args)
        inferrer.infer()

    if "main_infer" in tasks:
        inferrer = get_inferrer_from_config(pipelines, dataset, device, "main_infer", args)
        inferrer.infer()

    if "test" in tasks:
        inferrer = get_inferrer_from_config(pipelines, dataset, device, "test", args)
        inferrer.infer()
