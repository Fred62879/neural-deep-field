
import yaml
import torch
import pprint
import logging
import argparse

str2optim = {m.lower(): getattr(torch.optim, m) for m in dir(torch.optim) if m[0].isupper()}

def parse_input_config():
    parser = define_cmd_line_args()
    args = parser.parse_args()

    # parse yaml config file
    if args.config is not None:
        parse_yaml_config(args.config, parser)
    args = parser.parse_args()

    config = vars(args)
    args_str = argparse_to_str(parser, args)
    return config, args_str

def argparse_to_str(parser, args):
    """Convert parser to string representation for Tensorboard logging.
    Args:
        args : The parsed arguments.
    Returns:
        arg_str : The string to be printed.
    """
    args_dict = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        args_dict[group.title] = vars(argparse.Namespace(**group_dict))
    pp = pprint.PrettyPrinter(indent=2)
    args_str = pp.pformat(args_dict)
    args_str = f"```{args_str}```"
    return args_str

def parse_yaml_config(config_path, parser):
    """Parses and sets the parser defaults with a yaml config file.

    Args:
        config_path : path to the yaml config file.
        parser : The parser for which the defaults will be set.
        parent : True if parsing the parent yaml. Should never be set to True by the user.
    """
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    list_of_valid_fields = []
    for group in parser._action_groups:
        group_dict = {list_of_valid_fields.append(a.dest) for a in group._group_actions}
    list_of_valid_fields = set(list_of_valid_fields)

    defaults_dict = {}

    # Load the parent config if it exists
    parent_config_path = config_dict.pop("parent", None)

    if parent_config_path is not None:
        if not os.path.isabs(parent_config_path):
            parent_config_path = os.path.join(os.path.split(config_path)[0], parent_config_path)
        with open(parent_config_path) as f:
            parent_config_dict = yaml.safe_load(f)
        if "parent" in parent_config_dict.keys():
            raise Exception("Hierarchical configs of more than 1 level deep are not allowed.")
        for key in parent_config_dict:
            for field in parent_config_dict[key]:
                if field not in list_of_valid_fields:
                    raise ValueError(
                        f"ERROR: {field} is not a valid option. Check for typos in the config."
                    )
                defaults_dict[field] = parent_config_dict[key][field]

    # Loads child parent and overwrite the parent configs
    # The yaml files assumes the argument groups, which aren"t actually nested.
    for key in config_dict:
        for field in config_dict[key]:
            if field not in list_of_valid_fields:
                raise ValueError(
                    f"ERROR: {field} is not a valid option. Check for typos in the config."
                )
            defaults_dict[field] = config_dict[key][field]

    parser.set_defaults(**defaults_dict)

def define_cmd_line_args():
    """ Define all command line arguments
    """
    parser = argparse.ArgumentParser(
        description="ArgumentParser for implicit universe based on kaolin-wisp.")

    ###################
    # Global arguments
    ###################
    global_group = parser.add_argument_group("global")

    global_group.add_argument("--config", type=str, help="Path to config file to replace defaults.")

    global_group.add_argument("--use_gpu", action="store_true")
    global_group.add_argument("--verbose", action="store_true")
    global_group.add_argument("--print-shape", action="store_true")
    global_group.add_argument("--activate_timer", action="store_true")
    global_group.add_argument("--dataloader-drop-last", action="store_true")
    global_group.add_argument("--perf", action="store_true",
                              help="Use high-level profiling for the trainer.")

    global_group.add_argument("--exp-name", type=str, default="unnamed_experiment",
                              help="Experiment name.")
    global_group.add_argument("--tasks", nargs="+", type=str,
                              choices=["train",
                                       "plot_embed_map_during_train","save_codebook",
                                       "save_latent_during_train","save_recon_during_train",
                                       "recon_gt_spectra_during_train","infer_during_train",
                                       "infer",
                                       "recon_img","recon_synthetic","recon_gt_spectra",
                                       "recon_dummy_spectra","recon_codebook_spectra",
                                       "plot_embed_map","plot_latent_embed",
                                       "plot_redshift","plot_save_scaler"])

    ###################
    # General global network things
    ###################
    net_group = parser.add_argument_group("net")

    net_group.add_argument("--nef-type", type=str,
                           help="The neural field class to be used.")
    net_group.add_argument("--mlp-output-norm-method", type=str,
                           choices=["identity","arcsinh","sinh"])

    net_group.add_argument("--model-redshift", action="store_true",
                           help="whether the arch model redshift or not.")

    net_group.add_argument("--encode-coords", action="store_true")
    net_group.add_argument("--coords-encode-method", type=str,
                           choices=["positional","grid"],
                           help="ra/dec coordinate encoding method.")

    net_group.add_argument("--encode-wave", action="store_true")
    net_group.add_argument("--wave-encode-method", type=str,
                           choices=["positional"],
                           help="lambda encoding method.")

    ###################
    # Embedder arguments
    ###################
    embedder_group = parser.add_argument_group("embedder")

    embedder_group.add_argument("--coords-embed-dim", type=int,
                                help="ra/dec coordinate embedding dimension, if use pe.")
    embedder_group.add_argument("--coords-embed-bias", action="store_true")
    embedder_group.add_argument("--coords-embed-seed", type=int, default=0)
    embedder_group.add_argument("--coords-embed-omega", type=int, default=1,
                                help="frequency of sinusoidal functaions.")
    embedder_group.add_argument("--coords-embed-sigma", type=int, default=1,
                                help="variance to intialize pe weights.")

    embedder_group.add_argument("--wave-embed-dim", type=int,
                                help="wave embedding dimension, if use pe.")
    embedder_group.add_argument("--wave-embed-bias", action="store_true")
    embedder_group.add_argument("--wave-embed-seed", type=int, default=0)
    embedder_group.add_argument("--wave-embed-omega", type=int, default=1,
                                help="frequency of sinusoidal functaions.")
    embedder_group.add_argument("--wave-embed-sigma", type=int, default=1,
                                help="variance to intialize pe weights.")

    embedder_group.add_argument("--wave-encoder-num-hidden-layers", type=int, default=1)
    embedder_group.add_argument("--wave-encoder-hidden-dim", type=int, default=64)
    embedder_group.add_argument("--wave-encoder-siren-first-w0", type=int, default=30)
    embedder_group.add_argument("--wave-encoder-siren-hidden-w0", type=int, default=30)
    embedder_group.add_argument("--wave-encoder-siren-seed", type=int, default=0)
    embedder_group.add_argument("--wave-encoder-siren-coords-scaler", type=int, default=1)
    embedder_group.add_argument("--wave-encoder-siren-last-linear", action="store_true")

    ###################
    # Grid arguments
    ###################
    grid_group = parser.add_argument_group("grid")

    grid_group.add_argument("--grid-dim", type=int, help="Dimension of grid (2 for image, 3 for shape).")
    grid_group.add_argument("--grid-type", type=str, default="OctreeGrid",
                            choices=["None", "OctreeGrid", "CodebookOctreeGrid", "TriplanarGrid", "HashGrid"],
                            help="Type of grid to use.")
    grid_group.add_argument("--grid-interpolation-type", type=str, default="linear",
                            choices=["linear", "closest"], help="SPC interpolation mode.")
    grid_group.add_argument("--grid-multiscale-type", type=str,
                            default="sum", choices=["cat", "sum"],
                            help="Type of multiscale aggregation function to use.")
    grid_group.add_argument("--grid-feature-dim", type=int,
                            default=32, help="Feature map dimension")
    grid_group.add_argument("--grid-feature-std", type=float,
                            default=0.0, help="Feature map std")
    grid_group.add_argument("--grid-feature-bias", type=float,
                            default=0.0, help="Feature map bias")
    grid_group.add_argument("--grid-noise-std", type=float,
                            default=0.0, help="Added noise to features in training.")
    grid_group.add_argument("--grid-num-lods", type=int,
                            default=1, help="Number of LODs")
    grid_group.add_argument("--grid-base-lod", type=int,
                            default=2, help="Base level LOD")
    grid_group.add_argument("--min-grid-res", type=int, default=16,
                            help="The minimum grid resolution. Used only in geometric initialization.")
    grid_group.add_argument("--max-grid-res", type=int, default=2048,
                            help="The maximum grid resolution. Used only in geometric initialization.")
    grid_group.add_argument("--tree-type", type=str, default="quad", choices=["quad", "geometric"],
                            help="What type of tree to use. `quad` is a quadtree or octree-like growing \
                                  scheme, whereas geometric is the Instant-NGP growing scheme.")
    grid_group.add_argument("--codebook-bitwidth", type=int, default=8,
                            help="Bitwidth to use for the codebook. The number of vectors will be 2^bitwidth.")

    ###################
    # Decoder arguments
    ###################
    decoder_group = parser.add_argument_group("decoder")

    decoder_group.add_argument("--decoder-layer-type", type=str, default="none",
                            choices=["none", "spectral_norm", "frobenius_norm", "l_1_norm", "l_inf_norm"])
    decoder_group.add_argument("--decoder-activation-type", type=str,
                           default="relu", choices=["relu", "sin"])
    decoder_group.add_argument("--decoder-decoder-type", type=str,
                           default="basic", choices=["none", "basic"])

    decoder_group.add_argument("--decoder-num-layers", type=int, default=1,
                           help="Number of layers for the decoder")
    decoder_group.add_argument("--decoder-hidden-dim", type=int, default=128,
                           help="Network width")
    decoder_group.add_argument("--decoder-skip", type=int, default=None,
                           help="Layer to have skip connection.")

    decoder_group.add_argument("--siren-seed", type=int, default=1)
    decoder_group.add_argument("--siren-first-w0", type=int, default=30)
    decoder_group.add_argument("--siren-hidden-w0", type=int, default=30)
    decoder_group.add_argument("--siren-coords-scaler", type=int, default=1)
    decoder_group.add_argument("--siren-last-linear", action="store_true")

    ###################
    # Quantization arguments
    ###################
    qtz_group = parser.add_argument_group("quantization")

    qtz_group.add_argument("--quantization-strategy", type=str)
    qtz_group.add_argument("--qtz-latent-dim", type=int)
    qtz_group.add_argument("--qtz-num-embed", type=int)
    qtz_group.add_argument("--qtz-beta", type=float, help="codebook loss weight")
    qtz_group.add_argument("--qtz-seed", type=int)
    qtz_group.add_argument("--qtz-soft-temperature", type=int)
    qtz_group.add_argument("--qtz-temperature-scale", type=int)
    qtz_group.add_argument("--codebook-pretrain-latent-dim", type=int)

    ###################
    # Spatial Decoder arguments
    ###################
    spatial_decod_group = parser.add_argument_group("quantization")

    spatial_decod_group.add_argument("--generate-scaler", action="store_true")
    # spatial_decod_group.add_argument("--generate-redshift", action="store_true")

    spatial_decod_group.add_argument("--spatial-decod-hidden-dim", type=int)
    spatial_decod_group.add_argument("--spatial-decod-num-hidden-layers", type=int)
    spatial_decod_group.add_argument("--spatial-decod-output-dim", type=int)
    spatial_decod_group.add_argument("--spatial-decod-layer-type", type=str, default='none',
                                     choices=["none", "spectral_norm", "frobenius_norm",
                                              "l_1_norm", "l_inf_norm"])
    spatial_decod_group.add_argument("--spatial-decod-activation-type", type=str,
                                     default="relu", choices=["relu", "sin"])

    spatial_decod_group.add_argument("--scaler-decod-hidden-dim", type=int)
    spatial_decod_group.add_argument("--scaler-decod-num-hidden-layers", type=int)
    spatial_decod_group.add_argument("--scaler-decod-layer-type", type=str, default='none',
                                     choices=["none", "spectral_norm", "frobenius_norm",
                                              "l_1_norm", "l_inf_norm"])
    spatial_decod_group.add_argument("--scaler-decod-activation-type", type=str,
                                     default="relu", choices=["relu", "sin"])

    spatial_decod_group.add_argument("--redshift-decod-hidden-dim", type=int)
    spatial_decod_group.add_argument("--redshift-decod-num-hidden-layers", type=int)
    spatial_decod_group.add_argument("--redshift-decod-layer-type", type=str, default='none',
                                     choices=["none", "spectral_norm", "frobenius_norm",
                                              "l_1_norm", "l_inf_norm"])
    spatial_decod_group.add_argument("--redshift-decod-activation-type", type=str,
                                     default="relu", choices=["relu", "sin"])

    ###################
    # Hyperspectral arguments
    ###################
    hps_group = parser.add_argument_group("hyperspectral")

    hps_group.add_argument("--hps-combine-method", type=str, choices=["add","concat"],
                           help="method to combine ra/dec coordinate with lambda.")
    hps_group.add_argument("--integration-method", type=str,
                            choices=["identity","dot_prod","trapezoid","simpson"])

    ###################
    # Arguments for dataset
    ###################
    data_group = parser.add_argument_group("dataset")

    data_group.add_argument("--dataset-type", type=str, default=None,
                            choices=["sdf", "multiview", "astro"],
                            help="Dataset class to use")
    data_group.add_argument("--dataset-path", type=str, help="Path to the dataset")
    data_group.add_argument("--dataset-num-workers", type=int, default=-1,
                            help="Number of workers for dataset preprocessing, \
                            if it supports multiprocessing. -1 indicates no multiprocessing.")

    # Astro Dataset
    data_group.add_argument("--space-dim", type=int)
    data_group.add_argument("--coords-range-fname", type=str)

    # fits data
    data_group.add_argument("--patch-selection-cho", type=str)
    data_group.add_argument("--tracts", nargs="+", help="tracts of chose FITS files")
    data_group.add_argument("--patches", nargs="+", help="patch ids of chosen FITS files")

    data_group.add_argument("--use-full-patch", action="store_true")
    data_group.add_argument("--plot-img-distrib", action="store_true")
    data_group.add_argument("--load-patch-data-cache", action="store_true")
    data_group.add_argument("--patch-cutout-num-rows",nargs="+", type=int,
                            help="size of cutout from patch (if not using full patch)")
    data_group.add_argument("--patch-cutout-num-cols",nargs="+", type=int,
                            help="size of cutout from patch (if not using full patch)")
    data_group.add_argument("--patch-cutout-sizes",nargs="+", type=int,
                            help="size of cutout from patch (if not using full patch)")
    data_group.add_argument("--patch-cutout-start-pos", nargs="+", type=list,
                            help="starting row number of cutout from patch")

    data_group.add_argument("--num-bands", type=int)
    data_group.add_argument("--filters", nargs="+", type=str)
    data_group.add_argument("--filter-ids", nargs="+", type=int)
    data_group.add_argument("--sensors-full-name", nargs="+")
    data_group.add_argument("--sensor-collection-name", type=str)
    data_group.add_argument("--u-band-scale",type=float, default=10**((30-27)/2.5),
                            help="scale value for u band pixel values")

    data_group.add_argument("--gt-img-norm-cho", type=str, default="identity")
    data_group.add_argument("--train-pixels-norm", type=str,
                            choices=["identity","arcsinh","linear","clip","zscale"])
    data_group.add_argument("--infer-pixels-norm", type=str,
                            choices=["identity","arcsinh"])

    # trans data
    data_group.add_argument("--trans-sample-method", type=str,
                            choices=["hardcode","bandwise","mixture"])
    #data_group.add_argument("--trans-cho", type=str, default="orig_trans")
    data_group.add_argument("--wave-lo", type=int, default=3000,
                            help="smallest lambda value for transmission data (angstrom)")
    data_group.add_argument("--wave-hi", type=int, default=10900,
                            help="largest lambda value for transmission data (angstrom)")
    data_group.add_argument("--hardcode-num-trans-samples",type=int,
                            help="#samples of hardcoded selection of transmissions.")
    data_group.add_argument("--trans-threshold",type=float, default=1e-3,
                            help="smallest transmission value that we keep, \
                            range of transmission below this will be converted to 0.")
    data_group.add_argument("--trans-sample-interval",type=int, default=10,
                            help="discretization interval for transmission data, default 10.")
    data_group.add_argument("--plot-trans", action="store_true")

    # spectra data
    data_group.add_argument("--spectra-data-source", type=str)
    data_group.add_argument("--spectra-tracts", type=str, nargs='+')
    data_group.add_argument("--spectra-patches_r", type=str, nargs='+')
    data_group.add_argument("--spectra-patches_c", type=str, nargs='+')

    data_group.add_argument("--recon-spectra-clip-range", type=int, nargs='+')
    data_group.add_argument("--dummy-spectra-clip-range", type=int, nargs='+')
    data_group.add_argument("--gt-spectra-ids", type=int, nargs='+',
                            help="id of chosen gt spectra for supervision/recon etc.")
    data_group.add_argument("--spectra-markers", type=int, nargs='+',
                            help="marker to plot each spectra.")
    data_group.add_argument("--spectra-smooth-sigma",type=int, default=5)
    data_group.add_argument("--flux-norm-cho",type=str,
                            choices=["max","sum","scale_gt","scale_recon"],
                            help="0- divide with max, 1-divide with sum")
    data_group.add_argument("--trusted-range-only", action="store_true",
                            help="plot gt spectra within trusted range only")

    data_group.add_argument("--num-gt-spectra", type=int)
    data_group.add_argument("--processed-spectra-cho", type=str)
    data_group.add_argument("--spectra-supervision-wave-lo", type=int)
    data_group.add_argument("--spectra-supervision-wave-hi", type=int)
    data_group.add_argument("--load_spectra_data_from_cache", action="store_true")

    ###################
    # Arguments for optimizer
    ###################
    optim_group = parser.add_argument_group("optimizer")

    optim_group.add_argument("--optimizer-type", type=str, default="adam",
                             choices=list(str2optim.keys()), help="Optimizer to be used.")

    optim_group.add_argument("--lr", type=float, default=0.0001)
    optim_group.add_argument("--grid_lr", type=float, default=0.001)
    optim_group.add_argument("--codebook-lr", type=float, default=0.0001)
    optim_group.add_argument("--codebook-pretrain-lr", type=float, default=0.0001)

    optim_group.add_argument("--weight-decay", type=float, default=0, help="Weight decay.")
    optim_group.add_argument("--grid-lr-weight", type=float, default=100.0,
                             help="Relative LR weighting for the grid")
    optim_group.add_argument("--b1",type=float, default=0.5)
    optim_group.add_argument("--b2",type=float, default=0.999)

    ###################
    # Arguments for training
    ###################
    train_group = parser.add_argument_group("trainer")

    train_group.add_argument("--trainer-type", type=str, help="Trainer class to use")

    train_group.add_argument("--num-epochs", type=int, default=250,
                             help="Number of epochs to run the training.")
    train_group.add_argument("--batch-size", type=int, default=512,
                             help="Batch size for the training.")
    train_group.add_argument("--resample", action="store_true",
                             help="Resample the dataset after every epoch.")
    train_group.add_argument("--only-last", action="store_true",
                             help="Train only last LOD.")
    train_group.add_argument("--resample-every", type=int, default=1,
                             help="Resample every N epochs")
    train_group.add_argument("--model-format", type=str, default="full",
                             choices=["full", "state_dict"],
                             help="Format in which to save models.")
    train_group.add_argument("--save-as-new", action="store_true",
                             help="Save the model at every epoch (no overwrite).")
    train_group.add_argument("--save-every", type=int, default=200,
                             help="Save the model at every N epoch.")
    train_group.add_argument("--render-tb-every", type=int, default=100,
                                help="Render every N iterations")
    train_group.add_argument("--log-tb-every", type=int, default=100,
                             help="Log to tensorboard at every N epoch.")
    train_group.add_argument("--log-cli-every", type=int, default=100,
                             help="Log to command line at every N epoch.")
    train_group.add_argument("--log-gpu-every", type=int, default=100,
                             help="Log to cli gpu usage at every N epoch.")
    train_group.add_argument("--save-local-every", type=int, default=100,
                             help="Save data to local every N epoch.")
    train_group.add_argument("--plot-grad-every", type=int, default=20,
                             help="Plot gradient at every N epoch.")
    train_group.add_argument("--using-wandb", action="store_true")
    train_group.add_argument("--gpu-data", nargs="+", type=str,
                             help="data fields that can be added to gpu.")

    train_group.add_argument("--pixel-loss-cho",type=str, choices=["l1","l2"])
    train_group.add_argument("--spectra-loss-cho",type=str, choices=["l1","l2"])
    train_group.add_argument("--redshift-loss-cho",type=str, choices=["l1","l2"])

    train_group.add_argument("--train-with-all-pixels", action="store_true")
    train_group.add_argument("--train-pixel-ratio", type=float, default=1,
                             help="ratio of (unmasked) pixels used for training per epoch")
    # train_group.add_argument("--masked_pixl_ratio_per_epoch", type=float, default=1,
    #                          help="ratio of masked pixels used for spectral inpaint training per epoch")

    train_group.add_argument("--pretrain-codebook", action="store_true")
    train_group.add_argument("--codebook-pretrain-pixel-supervision", action="store_true")
    train_group.add_argument("--weight-train", action="store_true")
    train_group.add_argument("--train-use-all-wave", action="store_true")
    train_group.add_argument("--infer-during-train", action="store_true")
    train_group.add_argument("--train-spectra-pixels-only", action="store_true")
    train_group.add_argument("--pixel-supervision", action="store_true",
                             help="whether training supervised by pixel values or not.")
    train_group.add_argument("--spectra-supervision", action="store_true",
                             help="whether training supervised by spectra or not.")
    train_group.add_argument("--redshift-supervision", action="store_true",
                             help="whether training supervised by redshift or not.")
    train_group.add_argument("--apply-gt-redshift", action="store_true",
                             help="whether apply gt redshift directly to spectra or not.")

    train_group.add_argument("--spectra-supervision-start-epoch", type=int)
    train_group.add_argument("--spectra-beta", type=float, help="spectra loss weight scaler.")
    train_group.add_argument("--redshift-beta", type=float, help="redshift loss weight scaler.")
    train_group.add_argument("--pretrain-pixel-beta", type=float)
    train_group.add_argument("--pretrain-redshift-beta", type=float)

    train_group.add_argument("--quantize-latent", action="store_true")
    train_group.add_argument("--quantize-spectra", action="store_true")
    train_group.add_argument("--decode-spatial-embedding", action="store_true")

    train_group.add_argument("--resume-train", action="store_true")
    train_group.add_argument("--resume-log-dir", type=str)
    train_group.add_argument("--pretrain-log-dir", type=str)
    train_group.add_argument("--pretrained-model-name", type=str)

    train_group.add_argument("--num-trans-samples", type=int, default=40,
                             help="# transmission to sample at each training iteration.")
    train_group.add_argument("--uniform-sample-trans", action="store_true",
                             help="whether uniformly sample transmission or not.")
    train_group.add_argument("--mixture-avg-per-band", action="store_true",
                            help="for mixture sampling method, whether average pixel values \
                            with number of samples falling within each band")

    train_group.add_argument("--num-supervision-spectra", type=int,
                             help="# of gt spectra used for supervision \
                             (always select the first n spectra).")
    train_group.add_argument("--spectra-neighbour-size", type=int,
                             help="size of neighbourhood to average when calculating spectra.")

    train_group.add_argument("--log-dir", type=str, default="_results/logs/runs/",
                             help="Log file directory for checkpoints.")
    # TODO (ttakikawa): This is only really used in the SDF training but it should be useful for multiview too
    train_group.add_argument("--grow-every", type=int, default=-1,
                             help="Grow network every X epochs")
    train_group.add_argument("--prune-every", type=int, default=-1,
                             help="Prune every N iterations")
    # TODO (ttakikawa): Only used in multiview training, combine with the SDF growing schemes.
    train_group.add_argument("--random-lod", action="store_true",
                             help="Use random lods to train.")
    # One by one trains one level at a time.
    # Increase starts from [0] and ends up at [0,...,N]
    # Shrink strats from [0,...,N] and ends up at [N]
    # Fine to coarse starts from [N] and ends up at [0,...,N]
    # Only last starts and ends at [N]
    train_group.add_argument("--growth-strategy", type=str, default="increase",
                             choices=["onebyone","increase","shrink", "finetocoarse", "onlylast"],
                             help="Strategy for coarse-to-fine training")

    ###################
    # Arguments for validation
    ###################
    valid_group = parser.add_argument_group("validation")

    valid_group.add_argument("--valid-only", action="store_true",
                             help="Run validation only (and do not run training).")
    valid_group.add_argument("--valid-every", type=int, default=-1,
                             help="Frequency of running validation.")
    valid_group.add_argument("--valid-split", type=str, default="val",
                             help="Split to use for validation.")

    ##################
    # Arguments for inference
    ##################
    infer_group = parser.add_argument_group("inference")

    infer_group.add_argument("--inferrer-type", type=str, help="Inferrer class to use",
                             choices=["AstroInferrer"])

    infer_group.add_argument("--infer-log-dir", type=str)
    infer_group.add_argument("--infer-batch-size", type=int, default=4096)
    infer_group.add_argument("--infer-use-all-wave", action="store_true",
                             help="should set this to true, implementation assumes infer with all lambda")
    infer_group.add_argument("--infer-synthetic-band", action="store_true")
    infer_group.add_argument("--infer-last-model-only", action="store_true")

    infer_group.add_argument("--to-HDU", action="store_true", default=False,
                             help="generate HDU files for reconstructed image")
    infer_group.add_argument("--recon-norm", action="store_true", default=False)
    infer_group.add_argument("--recon-restore", action="store_true", default=False)
    infer_group.add_argument("--metric-options", nargs="+", choices=["mse","psnr","ssim"])

    # these several args, if specified, directs reconstructing smaller cutouts than train image
    # Note, if recon_img is included as inferrence tasks, we always reconstruct the full train image
    # regardless of whether these three are given or not
    infer_group.add_argument("--recon-zoomed", action="store_true",
                             help="whether reconstruct zoomed in cutouts or not. \
                             If true, the three below args needs to be specified")
    infer_group.add_argument("--recon-cutout-patch-uids", nargs="+", type=str,
                             help="id of tiles to generate reconstructed cutout")
    infer_group.add_argument("--recon-cutout-sizes", nargs="+", type=list,
                             help="list of sizes of each cutout for each tile")
    infer_group.add_argument("--recon-cutout-start-pos", nargs="+", type=list,
                             help="list of start (r/c) positions of each cutout for each tile")

    infer_group.add_argument("--mark-spectra", action="store_true")
    infer_group.add_argument("--average-neighbour-spectra", action="store_true")
    infer_group.add_argument("--plot-clipped-spectrum", action="store_true")
    infer_group.add_argument("--plot-spectrum-with-gt", action="store_true")
    infer_group.add_argument("--plot-spectrum-with-trans", action="store_true")
    infer_group.add_argument("--plot-spectrum-together", action="store_true")
    infer_group.add_argument("--infer-spectra-individually", action="store_true")
    infer_group.add_argument("--codebook-spectra-clip-range", nargs="+")
    infer_group.add_argument("--num-spectra-plot-per-row", type=int)

    infer_group.add_argument("--plot-labels", nargs="+", type=str)
    infer_group.add_argument("--plot-colors", nargs="+", type=str)
    infer_group.add_argument("--plot-styles", nargs="+", type=str)

    ###################
    # Argument for unit test
    ###################
    test_group = parser.add_argument_group("unit test")

    test_group.add_argument("--is_test", action="store_true", default=False)
    test_group.add_argument("--fake_coord", nargs="+", required=False)
    test_group.add_argument("--fake_spectra_cho", type=int, )

    ###################
    # Argument for inpainting
    ###################
    inpaint_group = parser.add_argument_group("inpaint")

    # inpainting args
    inpaint_group.add_argument("--mask-bandset-cho", type=str,default="None")
    inpaint_group.add_argument("--mask-mode", type=str, default="rand_diff")
    inpaint_group.add_argument("--inpaint-cho", type=str, default="no_inpaint",
                               choices=["no_inpaint","spatial_inpaint","spectral_inpaint"])

    inpaint_group.add_argument("--plot-masked-gt", action="store_true", default=False)

    inpaint_group.add_argument("--mask_sz", type=int, default=1)
    inpaint_group.add_argument("--mask_seed", type=int, default=0)
    inpaint_group.add_argument("--m_start_r", type=int, default=1)
    inpaint_group.add_argument("--m_start_c", type=int, default=1)
    inpaint_group.add_argument("--train-bands", nargs="+", type=int)
    inpaint_group.add_argument("--inpaint-bands", nargs="+", type=int)
    inpaint_group.add_argument("--inpaint-sample-ratio", type=float, default=1.0,
                               help="percent of pixels not masked")
    inpaint_group.add_argument("--relative-train-bands", nargs="+", type=int)
    inpaint_group.add_argument("--relative-inpaint-bands", nargs="+", type=int)

    ###############
    # Argument for experiment
    ###############
    exp_group = parser.add_argument_group("experiment")

    exp_group.add_argument("--para_nms", nargs="+")
    exp_group.add_argument("--trail_id", type=str, default="trail_dum")
    exp_group.add_argument("--experiment_id", type=str, default="exp_dum")
    exp_group.add_argument("--abl_cho", type=int)
    exp_group.add_argument("--gaus_cho", type=int)
    exp_group.add_argument("--gabor_mfn_cho", type=int)
    exp_group.add_argument("--fourier_mfn_cho", type=int)
    exp_group.add_argument("--sample_ratio_cho", type=int)
    exp_group.add_argument("--ipe_rand_schedule_cho", type=int)
    exp_group.add_argument("--ipe_global_schedule_cho", type=int)

    # add_log_level_flag(parser)
    parser.add_argument(
        '--log_level', action='store', type=int, default=logging.INFO,
        help='Logging level to use globally, DEBUG: 10, INFO: 20, WARN: 30, ERROR: 40.'
    )

    return parser
