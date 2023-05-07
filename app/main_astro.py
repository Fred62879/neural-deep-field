
if __name__ == "__main__":

    from app_utils import *
    from wisp.astro_config_parser import *
    from wisp.parsers.parser import parse_args
    from wisp.trainers import AstroTrainer, CodebookTrainer

    args, args_str = parse_args()
    tasks = set(args.tasks)
    default_log_setup(args.log_level)

    set_seed()
    dataset = get_dataset_from_config(args)
    device, pipelines = get_pipelines_from_config(args, tasks=tasks)

    # perform codebook_train, train, and infer in order
    if "codebook_pretrain" in tasks and args.pretrain_codebook:
        optim_cls, optim_params = get_optimizer_from_config(args)
        trainer = get_trainer_from_config(
            CodebookTrainer,
            [ pipelines["codebook_net"], pipelines["codebook"] ],
            dataset, optim_cls, optim_params, device, args
        )
        trainer.train()

    if "train" in tasks:
        optim_cls, optim_params = get_optimizer_from_config(args)
        trainer = get_trainer_from_config(
            AstroTrainer, pipelines["full"], dataset, optim_cls, optim_params, device, args)
        trainer.train()

    if "infer" in tasks:
        inferrer = get_inferrer_from_config(pipelines, dataset, device, args)
        inferrer.infer()
