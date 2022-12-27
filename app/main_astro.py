

if __name__ == "__main__":
    import os
    import app_utils

    from wisp.trainers import *
    from wisp.inferrers import *
    from wisp.framework import WispState
    from wisp.parsers.parser import parse_args
    from wisp.astro_config_parser import get_dataset_from_config, \
        get_pipelines_from_config, get_optimizer_from_config

    # Usual boilerplate
    args, args_str = parse_args()
    app_utils.default_log_setup(args.log_level)

    dataset = get_dataset_from_config(args)
    device, pipelines = get_pipelines_from_config(args, tasks=args.tasks)

    def train():
        optim_cls, optim_params = get_optimizer_from_config(args)
        trainer = globals()[args.trainer_type](
            pipelines["full"], dataset, args.num_epochs, args.batch_size,
            optim_cls, args.lr, args.weight_decay,
            args.grid_lr_weight, optim_params, args.log_dir, device,
            exp_name=args.exp_name, info=args_str, extra_args=vars(args),
            render_tb_every=args.render_tb_every, save_every=args.save_every)
        trainer.train()

    def infer():
        inferrer = globals()[args.inferrer_type](
            pipelines, dataset, device, vars(args), info=args_str)
        inferrer.infer()

    if "train" in set(args.tasks):
        train()
    if "infer" in set(args.tasks):
        infer()
