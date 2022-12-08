# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


if __name__ == "__main__":
    import os
    import wandb
    import app_utils

    from wisp.trainers import *
    from wisp.framework import WispState
    from wisp.parsers.parser import parse_args
    from wisp.config_parser import get_modules_from_config, get_optimizer_from_config

    # Usual boilerplate
    args, args_str = parse_args()

    using_wandb = args.wandb_project is not None
    if using_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.exp_name if args.wandb_run_name is None else args.wandb_run_name,
            entity=args.wandb_entity,
            job_type="validate" if args.valid_only else "train",
            config=vars(args),
            sync_tensorboard=True
        )

    app_utils.default_log_setup(args.log_level)

    optim_cls, optim_params = get_optimizer_from_config(args)
    pipeline, train_dataset, device = get_modules_from_config(args)
    print(pipeline)

    trainer = globals()[args.trainer_type](pipeline, train_dataset, args.num_epochs, args.batch_size,
                                           optim_cls, args.lr, args.weight_decay,
                                           args.grid_lr_weight, optim_params, args.log_dir, device,
                                           exp_name=args.exp_name, info=args_str, extra_args=vars(args),
                                           render_tb_every=args.render_tb_every, save_every=args.save_every, using_wandb=using_wandb)
    if args.valid_only:
        trainer.validate()
    else:
        trainer.train()

    if args.trainer_type == "MultiviewTrainer" and using_wandb and args.wandb_viz_nerf_angles != 0:
        trainer.render_final_view(
            num_angles=args.wandb_viz_nerf_angles,
            camera_distance=args.wandb_viz_nerf_distance
        )

    if using_wandb:
        wandb.finish()
