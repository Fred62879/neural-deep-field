
import os
import sys
import json
import yaml
import torch
import pprint
import logging
import argparse
import argparse
import configargparse

from wisp.utils.common import set_seed, default_log_setup, \
    get_current_ablate_params_and_vals


str2optim = {m.lower(): getattr(torch.optim, m) for m in dir(torch.optim) if m[0].isupper()}

def write_config_updates(fname, args):
    """
    python main.py --arg1=v1 --arg2=v2 --boolarg1
    """
    def parse_func(arg):
        is_bool = False
        arg_val = arg.split("=")
        if len(arg_val) == 1:
            arg = arg_val[0][2:]
            val = True
            is_bool = True
        elif len(arg_val) == 2:
            arg, val = arg_val[0][2:], arg_val[1]
        else: raise ValueError()
        updates[arg] = val

    cmd_args = sys.argv[1:] # ['--arg1=v1','--arg2=v2','--boolarg1']
    updates = {}
    [parse_func(cmd_arg) for cmd_arg in cmd_args]
    if "ablat-id" in updates:
        params, vals = get_current_ablate_params_and_vals(args)
        for param, val in zip(params, vals):
            updates[param] = val
            print(param, val)

    with open(fname, "w") as fp:
        json.dump(updates, fp)

def parse_args():
    """ Parse all command arguments and generate all needed ones.
    """
    config, args_str = parse_input_config()
    args = argparse.Namespace(**config)

    default_log_setup(args.log_level)

    set_seed(args.seed)
    logging.info(f"set seed as {args.seed}")

    if args.perform_ablation:
        params, vals = get_current_ablate_params_and_vals(args)
        for param, val in zip(params, vals):
            logging.info(f"ablate {param} with val: {val}")

    return args, args_str

def parse_input_config():
    parser = define_cmd_line_args()
    args = parser.parse_args()

    # parse yaml config file
    if args.config is not None:
        parse_yaml_config(args.config, parser)
    args = parser.parse_args()

    config = setup_ablation(args)

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

def setup_ablation(args):
    config = vars(args)
    if args.perform_ablation:
        # param, val = get_current_ablate_param_and_val(args)
        # config[param] = val
        # if "log_fname" in config and config["log_fname"] is not None:
        #     config["log_fname"] += f"_{param}_{val}"
        # else: config["log_fname"] = f"{param}_{val}"

        # each param is combined with the following params
        params, vals = get_current_ablate_params_and_vals(args)
        for param, val in zip(params, vals):
            config[param] = val
            if "log_fname" in config and config["log_fname"] is not None:
                config["log_fname"] += f"_{param}_{val}"
            else: config["log_fname"] = f"{param}_{val}"

        if args.ablat_hardcode_params is not None:
            id = args.ablat_id
            for i, param in enumerate(args.ablat_hardcode_params):
                config[param] = args.ablat_hardcode_vals[i][id]
                print(param, args.ablat_hardcode_vals[i][id])

    return config

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

    global_group.add_argument("--verbose", action="store_true")
    global_group.add_argument("--use-gpu", action="store_true")
    global_group.add_argument("--use-tqdm", action="store_true")
    global_group.add_argument("--on-cc", action="store_true")
    global_group.add_argument("--on-cedar", action="store_true")
    global_group.add_argument("--on-graham", action="store_true")
    global_group.add_argument("--on-narval", action="store_true")
    global_group.add_argument("--on-sockeye", action="store_true")
    global_group.add_argument("--print-shape", action="store_true")
    global_group.add_argument("--show-memory", action="store_true")
    global_group.add_argument("--activate-model-timer", action="store_true")
    global_group.add_argument("--activate-dataset-timer", action="store_true")
    global_group.add_argument("--activate-trainer-timer", action="store_true")
    global_group.add_argument("--dataloader-drop-last", action="store_true")
    global_group.add_argument("--perf", action="store_true",
                              help="Use high-level profiling for the trainer.")

    global_group.add_argument("--seed", type=int, default=42, help="global random seed.")
    global_group.add_argument("--exp-name", type=str, default="unnamed_experiment",
                              help="Experiment name.")
    global_group.add_argument("--log-fname", type=str, help="Log dir name")
    global_group.add_argument("--tasks", nargs="+", type=str,
                              choices=["train",
                                       "plot_embed_map_during_train","save_codebook",
                                       "save_latent_during_train","save_recon_during_train",
                                       "recon_gt_spectra_during_train","infer_during_train",
                                       "infer",
                                       "recon_img","recon_synthetic","recon_gt_spectra",
                                       "recon_dummy_spectra","recon_codebook_spectra",
                                       "recon_codebook_spectra_individ",
                                       "plot_embed_map","plot_latent_embed",
                                       "plot_redshift","plot_save_scaler"])
    ###################
    # Ablation options
    ###################
    ablat_group = parser.add_argument_group("ablation")

    ablat_group.add_argument("--perform-ablation", action="store_true")
    ablat_group.add_argument("--ablat-id", type=int)
    ablat_group.add_argument("--ablat-params", nargs="+")
    ablat_group.add_argument("--ablat-vals", nargs="+")
    ablat_group.add_argument("--ablat-num-vals", nargs="+")

    ablat_group.add_argument("--ablat-hardcode-params", nargs="+")
    ablat_group.add_argument("--ablat-hardcode-vals", nargs="+")

    ###################
    # Debug options
    ###################
    debug_group = parser.add_argument_group("debug")

    debug_group.add_argument("--add-redshift-logit-bias", action="store_true")
    debug_group.add_argument("--plot-logits-for-gt-bin", action="store_true")
    debug_group.add_argument("--plot-individ-spectra-loss", action="store_true")

    debug_group.add_argument("--generalize-train-first-layer", action="store_true")
    debug_group.add_argument("--add-validation-spectra-not-in-supervision", action="store_true")
    debug_group.add_argument("--num-extra-validation-spectra", type=int)

    debug_group.add_argument("--debug-lbfgs", action="store_true")

    ###################
    # General global network things
    ###################
    net_group = parser.add_argument_group("net")

    net_group.add_argument("--nef-type", type=str,
                           help="The neural field class to be used.")
    net_group.add_argument("--model-redshift", action="store_true",
                           help="whether the arch model redshift or not.")
    net_group.add_argument("--redshift-model-method", type=str,
                           help="choice of redshift modeling",
                           choices=["none","regression","classification"])
    net_group.add_argument("--redshift-classification-strategy", type=str)
    net_group.add_argument("--redshift-classification-method", type=str,
                           choices=["argmax","weighted_avg"])
    net_group.add_argument("--encode-coords", action="store_true")
    net_group.add_argument("--coords-encode-method", type=str,
                           choices=["positional","grid"],
                           help="ra/dec coordinate encoding method.")
    net_group.add_argument("--encode-wave", action="store_true")
    net_group.add_argument("--wave-encode-method", type=str,
                           choices=["positional"],
                           help="lambda encoding method.")
    net_group.add_argument("--linear-norm-wave", action="store_true")
    net_group.add_argument("--wave-multiplier", type=int)

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

    embedder_group.add_argument("--wave-encoder-skip-layers", nargs="+")
    embedder_group.add_argument("--wave-encoder-skip-all-layers", action="store_true")
    embedder_group.add_argument("--wave-encoder-hidden-dim", type=int, default=64)
    embedder_group.add_argument("--wave-encoder-num-hidden-layers", type=int, default=1)
    embedder_group.add_argument("--wave-encoder-gaussian_sigma", type=int, default=1)

    embedder_group.add_argument("--wave-encoder-siren-seed", type=int, default=0)
    embedder_group.add_argument("--wave-encoder-siren-first-w0", type=int, default=30)
    embedder_group.add_argument("--wave-encoder-siren-hidden-w0", type=int, default=30)
    embedder_group.add_argument("--wave-encoder-siren-last-linear", action="store_true")
    embedder_group.add_argument("--wave-encoder-siren-coords-scaler", type=int, default=1)

    ###################
    # Grid arguments
    ###################
    grid_group = parser.add_argument_group("grid")

    grid_group.add_argument("--grid-dim", type=int,
                            help="Dimension of grid (2 for image, 3 for shape).")
    grid_group.add_argument("--grid-type", type=str, default="OctreeGrid",
                            choices=["None", "OctreeGrid", "CodebookOctreeGrid",
                                     "TriplanarGrid", "HashGrid"],
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
    grid_group.add_argument("--tree-type", type=str, default="quad",
                            choices=["quad", "geometric"],
                            help="What type of tree to use. `quad` is a quadtree or \
                            octree-like growing scheme, whereas geometric is the \
                            Instant-NGP growing scheme.")
    grid_group.add_argument("--codebook-bitwidth", type=int, default=8,
                            help="Bitwidth to use for the codebook. The number of vectors will be 2^bitwidth.")
    grid_group.add_argument("--interpolation-align-corners", action="store_true",
                            help="arg for dense grid interpolation, torch.align_sample")


    ###################
    # Decoder arguments
    ###################
    decoder_group = parser.add_argument_group("decoder")

    decoder_group.add_argument("--decoder-layer-type", type=str, default="none",
                               choices=["none", "spectral_norm", "frobenius_norm",
                                        "l_1_norm", "l_inf_norm"])
    decoder_group.add_argument("--decoder-activation-type", type=str,
                               default="relu", choices=["relu", "sin"])
    decoder_group.add_argument("--decoder-decoder-type", type=str,
                               default="basic", choices=["none", "basic"])
    decoder_group.add_argument("--decoder-num-hidden-layers", type=int, default=1,
                               help="Number of layers for the decoder")
    decoder_group.add_argument("--decoder-hidden-dim", type=int, default=128,
                               help="Network width")
    decoder_group.add_argument("--decoder-batch-norm", action="store_true")
    decoder_group.add_argument("--decoder-skip-layers", nargs="+", type=int,
                               help="Layer to have skip connection.")
    decoder_group.add_argument("--decoder-gaussian-sigma", type=int)

    decoder_group.add_argument("--decoder-latents-skip-all-layers", action="store_true",
                               help="pass latents as skip to each decoder layer.")
    decoder_group.add_argument("--decoder-activate-before-latents-skip", action="store_true")
    decoder_group.add_argument("--decoder-latents-skip-method", nargs="+")
    decoder_group.add_argument("--decoder-latents-skip-add-conversion-method", nargs="+")

    decoder_group.add_argument("--baseline-decoder-input-dim", type=int)
    decoder_group.add_argument("--baseline-decoder-num-hidden-layers", type=int, default=1,
                               help="Number of layers for the decoder")
    decoder_group.add_argument("--baseline-decoder-hidden-dim", type=int, default=128,
                               help="Network width")
    decoder_group.add_argument("--baseline-decoder-batch-norm", action="store_true")
    decoder_group.add_argument("--baseline-decoder-skip-all-layers", action="store_true",
                               help="pass latents as skip to each decoder layer.")
    decoder_group.add_argument("--baseline-decoder-activate-before-skip", action="store_true")
    decoder_group.add_argument("--baseline-decoder-skip-method", nargs="+")
    decoder_group.add_argument("--baseline-decoder-skip-add-conversion-method", nargs="+")

    decoder_group.add_argument("--classifier-decoder-input-dim", type=int)
    decoder_group.add_argument("--classifier-decoder-num-hidden-layers", type=int, default=1,
                               help="Number of layers for the decoder")
    decoder_group.add_argument("--classifier-decoder-hidden-dim", type=int, default=128,
                               help="Network width")
    decoder_group.add_argument("--classifier-decoder-batch-norm", action="store_true")

    decoder_group.add_argument("--classifier-decoder-skip-all-layers", action="store_true",
                               help="pass latents as skip to each decoder layer.")
    decoder_group.add_argument("--classifier-decoder-activate-before-skip", action="store_true")
    decoder_group.add_argument("--classifier-decoder-skip-method", nargs="+")
    decoder_group.add_argument("--classifier-decoder-skip-add-conversion-method", nargs="+")

    decoder_group.add_argument("--classify-based-on-loss", action="store_true")
    decoder_group.add_argument("--classify-based-on-concat-spectra", action="store_true")
    decoder_group.add_argument("--classify-based-on-concat-wave-loss", action="store_true")
    decoder_group.add_argument("--classify-based-on-concat-wave-spectra", action="store_true")
    decoder_group.add_argument("--redshift-classification-batched-data-fields", nargs="+")

    decoder_group.add_argument("--siren-seed", type=int, default=1)
    decoder_group.add_argument("--siren-first-w0", type=int, default=30)
    decoder_group.add_argument("--siren-hidden-w0", type=int, default=30)
    decoder_group.add_argument("--siren-coords-scaler", type=int, default=1)
    decoder_group.add_argument("--siren-last-linear", action="store_true")

    decoder_group.add_argument("--lambdawise-weights-decoder-num-layers", action="store_true")
    decoder_group.add_argument("--lambdawise-weights-decoder-hidden-dim", action="store_true")

    ###################
    # Quantization arguments
    ###################
    qtz_group = parser.add_argument_group("quantization")

    qtz_group.add_argument("--quantize-latent", action="store_true")
    qtz_group.add_argument("--quantize-spectra", action="store_true")

    qtz_group.add_argument("--quantization-strategy", type=str)
    qtz_group.add_argument("--quantization-calculate-loss", action="store_true")
    qtz_group.add_argument("--qtz-latent-dim", type=int)
    qtz_group.add_argument("--qtz-num-embed", type=int)
    qtz_group.add_argument("--qtz-beta", type=float, help="codebook loss weight")
    qtz_group.add_argument("--qtz-seed", type=int)


    qtz_group.add_argument("--temped-qtz", action="store_true")
    qtz_group.add_argument("--qtz-temperature-scale", type=float)

    ###################
    # Pretrain arguments
    ###################
    pretrain_group = parser.add_argument_group("pretrain")

    pretrain_group.add_argument("--pretrain-codebook", action="store_true")

    pretrain_group.add_argument("--pretrain-batch-size", type=int, default=512)
    pretrain_group.add_argument("--pretrain-pixel-beta", type=float)
    pretrain_group.add_argument("--pretrain-redshift-beta", type=float)

    pretrain_group.add_argument("--sanity-check-finetune", action="store_true")
    pretrain_group.add_argument("--spectra-wrong-bin-regu-beta", type=float)

    pretrain_group.add_argument("--pretrain-optimize-latents-alternately", action="store_true",
                                help="optimize codebook and redshift latents alternately \
                                in redshfit pretrain (EM).")
    pretrain_group.add_argument("--em-alternation-starts-with", type=str,
                                choices=["spectra_latents","redshift_latents"],
                                help="alternate optimization starts with.")
    pretrain_group.add_argument("--em-alternation-steps", nargs="+", type=int,
                                help="alternately optimize each for given steps.")

    pretrain_group.add_argument("--use-global-spectra-loss-as-lambdawise-weights",
                                action="store_true")
    pretrain_group.add_argument("--limit-redshift-to-pretrain-range", action="store_true",
                                help="during sanity check or generalization, we limit\
                                redshfit range to be within the redshift range of pretrain.")
    pretrain_group.add_argument("--global-redshift-fname", type=str)
    pretrain_group.add_argument("--global-restframe-spectra-loss-fname", type=str)
    pretrain_group.add_argument("--regress-lambdawise-weights", action="store_true")
    pretrain_group.add_argument("--regress-lambdawise-weights-share-latents",
                                action="store_true")
    pretrain_group.add_argument("--regress-lambdawise-weights-use-gt-bin-latent",
                                action="store_true")
    pretrain_group.add_argument("--wrong-bin-regu-beta", type=int)

    pretrain_group.add_argument("--apply-gt-redshift", action="store_true",
                                help="apply gt redshift instead of generating redshift.")
    pretrain_group.add_argument("--brute-force-redshift", action="store_true")

    pretrain_group.add_argument("--optimize-latents-for-each-redshift-bin",action="store_true",
                                help="if brute force, we may assign each bin with a latent.")

    pretrain_group.add_argument("--calculate-spectra-loss-based-on-optimal-bin",
                                action="store_true", help="when use single latent, calculate \
                                loss use best bin spectra loss.")
    pretrain_group.add_argument("--calculate-spectra-loss-based-on-top-n-bins",
                                action="store_true")
    pretrain_group.add_argument("--num-bins-to-calculate-spectra-loss", action="store_true")

    pretrain_group.add_argument("--negative-supervise-wrong-redshift", action="store_true",
                                help="discourage wrong redshift from generating high \
                                quality spectra during codebook pretrain.")
    pretrain_group.add_argument("--neg-sup-with-optimal-wrong-bin", action="store_true",
                                help="neg sup with spectra loss of optimal wrong bin")
    pretrain_group.add_argument("--neg-sup-beta", type=float)
    pretrain_group.add_argument("--neg-sup-constant", type=float)

    pretrain_group.add_argument("--neg-sup-optimize-alternately", action="store_true",
                                help="optimize network and latents alternately during \
                                negative supervision")
    pretrain_group.add_argument("--neg-sup-alternation-starts-with", type=str,
                                choices=["net","latents"],
                                help="alternate optimization starts with.")
    pretrain_group.add_argument("--neg-sup-alternation-steps", nargs="+", type=int,
                                help="alternately optimize each for given steps.")

    pretrain_group.add_argument("--zero-init-spectra-latents", action="store_true")
    pretrain_group.add_argument("--optimize-spectra-latents-as-logits", action="store_true",
                                help="optimize latents directly as logits without autodecoder.")
    pretrain_group.add_argument("--optimize-spectra-latents", action="store_true")
    pretrain_group.add_argument("--load-pretrained-spectra-latents", action="store_true")
    pretrain_group.add_argument("--load-pretrained-spectra-latents-to-gt-bin-only", action="store_true")

    pretrain_group.add_argument("--optimize-gt-bin-only", action="store_true")
    pretrain_group.add_argument("--dont-optimize-gt-bin", action="store_true")

    pretrain_group.add_argument("--optimize-codebook-logits-mlp", action="store_true")
    pretrain_group.add_argument("--load-pretrained-codebook-logits-mlp", action="store_true")

    pretrain_group.add_argument("--use-binwise-spectra-loss-as-redshift-logits",
                                action="store_true", help="calculate redshift logits based \
                                on recon loss for spectra corresponding to each redshift bin.")
    pretrain_group.add_argument("--binwise-loss-beta", type=float)

    pretrain_group.add_argument("--zero-init-redshift-latents", action="store_true")
    pretrain_group.add_argument("--optimize-redshift-latents-as-logits", action="store_true")
    pretrain_group.add_argument("--optimize-redshift-latents", action="store_true")
    pretrain_group.add_argument("--optimize-redshift-logits-mlp", action="store_true")

    pretrain_group.add_argument("--regularize-redshift-logits", action="store_true")
    pretrain_group.add_argument("--redshift-logits-regu-beta", type=float)
    pretrain_group.add_argument("--redshift-logits-regu-method",type=str,
                             choices=["l1","l1_excl_largest","laplace"])

    pretrain_group.add_argument("--regularize-codebook-logits", action="store_true",
                                help="Regularize codebook logits during pretrain.")
    pretrain_group.add_argument("--codebook-logits-regu-beta", type=float)

    pretrain_group.add_argument("--regularize-spectra-latents", action="store_true",
                                help="Regularize spectra latents using L2 loss s.t. the \
                                the latents are close to the manifold surface (close to 0).")
    pretrain_group.add_argument("--spectra-latents-regu-beta", type=float)
    pretrain_group.add_argument("--regularize-binwise-spectra-latents", action="store_true",
                                help="Regularize latents for each bin of the same spectra s.t. \
                                they are close in value.")
    pretrain_group.add_argument("--binwise-spectra-latents-regu-beta", type=float)

    pretrain_group.add_argument("--regularize-within-codebook-spectra", action="store_true")
    pretrain_group.add_argument("--regularize-across-codebook-spectra", action="store_true")
    pretrain_group.add_argument("--codebook-spectra-regu-beta", type=float)

    pretrain_group.add_argument("--pretrain-log-dir", type=str)
    pretrain_group.add_argument("--pretrained-model-name", type=str)

    pretrain_group.add_argument("--pretrain-use-all-wave", action="store_true")
    pretrain_group.add_argument("--pretrain-wave-sample-method", type=str, default="uniform")
    pretrain_group.add_argument("--pretrain-num-wave-samples", type=int, default=1000,
                                help="# wave to sample at each pretrain iteration.")

    pretrain_group.add_argument("--pretrain-pixel-supervision", action="store_true")
    pretrain_group.add_argument("--pretrain-with-coords", action="store_true",
                                help="performe pretraining with 2d coords instead of \
                                optimizing latent variables")

    pretrain_group.add_argument("--use-latents-as-coords", action="store_true",
                                help="pass latents as coords to model.")

    pretrain_group.add_argument("--sanity-check-sample-bins", action="store_true")
    pretrain_group.add_argument("--sanity-check-sample-bins-per-step", action="store_true")
    pretrain_group.add_argument("--sanity-check-num-bins-to-sample", type=int)

    pretrain_group.add_argument("--classifier-train-use-bin-sampled-data", action="store_true",
                                help="data gather from sanity check are sampled with bins")
    pretrain_group.add_argument("--classifier-train-sample-bins", action="store_true")
    pretrain_group.add_argument("--classifier-train-num-bins-to-sample", action="store_true")

    pretrain_group.add_argument("--classifier-train-add-baseline-logits", action="store_true")
    pretrain_group.add_argument("--classifier-test-add-baseline-logits", action="store_true")
    pretrain_group.add_argument("--baseline-logits-scale-weight", type=float, default=-1)
    pretrain_group.add_argument("--baseline-logits-path", type=str)
    pretrain_group.add_argument("--baseline-logits-fname-suffix", type=str)

    pretrain_group.add_argument("--load-classification-data-individually",
                                action="store_true")

    ###################
    # Spatial Decoder arguments
    ###################
    spatial_decod_group = parser.add_argument_group("quantization")

    spatial_decod_group.add_argument("--decode-bias", action="store_true")
    spatial_decod_group.add_argument("--decode-scaler", action="store_true")

    spatial_decod_group.add_argument("--spatial-decod-hidden-dim", type=int)
    spatial_decod_group.add_argument("--spatial-decod-num-hidden-layers", type=int)
    spatial_decod_group.add_argument("--spatial-decod-output-dim", type=int)
    spatial_decod_group.add_argument("--spatial-decod-layer-type", type=str, default='none',
                                     choices=["none", "spectral_norm", "frobenius_norm",
                                              "l_1_norm", "l_inf_norm"])
    spatial_decod_group.add_argument("--spatial-decod-activation-type", type=str,
                                     default="relu", choices=["relu", "sin"])
    spatial_decod_group.add_argument("--spatial-decod-skip-layers", nargs="+", type=int,
                                     help="Layer to have skip connection.")

    spatial_decod_group.add_argument("--scaler-decod-hidden-dim", type=int)
    spatial_decod_group.add_argument("--scaler-decod-num-hidden-layers", type=int)
    spatial_decod_group.add_argument("--scaler-decod-layer-type", type=str, default='none',
                                     choices=["none", "spectral_norm", "frobenius_norm",
                                              "l_1_norm", "l_inf_norm"])
    spatial_decod_group.add_argument("--scaler-decod-activation-type", type=str,
                                     default="relu", choices=["relu", "sin"])
    spatial_decod_group.add_argument("--scaler-decod-skip-layers", nargs="+", type=int,
                                     help="Layer to have skip connection.")

    spatial_decod_group.add_argument("--redshift-decod-hidden-dim", type=int)
    spatial_decod_group.add_argument("--redshift-decod-num-hidden-layers", type=int)
    spatial_decod_group.add_argument("--redshift-decod-layer-type", type=str, default='none',
                                     choices=["none", "spectral_norm", "frobenius_norm",
                                              "l_1_norm", "l_inf_norm"])
    spatial_decod_group.add_argument("--redshift-decod-activation-type", type=str,
                                     default="relu", choices=["relu", "sin"])
    spatial_decod_group.add_argument("--redshift-decod-skip-layers", nargs="+", type=int,
                                     help="Layer to have skip connection.")

    ###################
    # Hyperspectral arguments
    ###################
    hps_group = parser.add_argument_group("hyperspectral")

    hps_group.add_argument("--hps-combine-method", type=str, choices=["add","concat"],
                           help="method to combine ra/dec coordinate with lambda.")
    hps_group.add_argument("--integration-method", type=str,
                            choices=["identity","dot_prod","trapezoid","simpson"])
    hps_group.add_argument("--use-batched-hps-model", action="store_true")
    hps_group.add_argument("--intensify-intensity", action="store_true",
                           help="intensify pixel value with to capture high dynamic range.")
    hps_group.add_argument("--intensification-method", type=str,
                           choices=["identity","arcsinh","sinh"])
    hps_group.add_argument("--spectra-batch-reduction-order", type=str,
                           choices=["bin_avg_first","qtz_first"],
                           help="refer to hps_decoder_batched.py")

    ###################
    # Arguments for dataset
    ###################
    data_group = parser.add_argument_group("dataset")

    data_group.add_argument("--dataset-type", type=str, default=None,
                            choices=["sdf", "multiview", "astro"],
                            help="Dataset class to use")

    data_group.add_argument("--dataset-path", type=str, help="Path to the dataset")
    data_group.add_argument("--cc-dataset-path", type=str, help="Path to the dataset")
    data_group.add_argument("--cedar-dataset-path", type=str, help="Path to the dataset")
    data_group.add_argument("--sockeye-dataset-path", type=str, help="Path to the dataset")
    data_group.add_argument("--cc-input-fits-path", type=str,help="Path to the dataset")
    data_group.add_argument("--cedar-input-fits-path", type=str, help="Path to the dataset")
    data_group.add_argument("--narval-input-fits-path", type=str, help="Path to the dataset")
    data_group.add_argument("--graham-input-fits-path", type=str, help="Path to the dataset")
    data_group.add_argument("--sockeye-input-fits-path", type=str, help="Path to the dataset")

    data_group.add_argument("--dataset-num-workers", type=int, default=-1,
                            help="Number of workers for dataset preprocessing, \
                            if it supports multiprocessing. -1 indicates no multiprocessing.")

    # Astro Dataset
    data_group.add_argument("--space-dim", type=int)
    data_group.add_argument("--calculate-wave-range-based-on-trans", action="store_true")
    data_group.add_argument("--calculate-wave-range-based-on-spectra", action="store_true")
    data_group.add_argument("--wave-range-fname", type=str,
                            help="fname of lambda range used for linear normalization.")
    # fits data
    data_group.add_argument("--patch-selection-cho", type=str)
    data_group.add_argument("--use-full-patch", action="store_true")
    data_group.add_argument("--tracts", nargs="+", type=str, help="tracts of chose FITS files")
    data_group.add_argument("--patches", nargs="+", type=str, help="patch ids of chosen FITS files")

    data_group.add_argument("--use-full-test-patch", action="store_true")
    data_group.add_argument("--test-tracts", nargs="+", help="tracts of chose FITS files")
    data_group.add_argument("--test-patches", nargs="+", help="patch ids of chosen FITS files")

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

    data_group.add_argument("--coords-type", type=str, choices=["img","world"])
    data_group.add_argument("--normalize-coords", action="store_true")
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
    data_group.add_argument("--spectra-data-sources", nargs="+", type=str)
    data_group.add_argument("--random-permute-source-spectra", action="store_true")

    data_group.add_argument("--require-only-basic-spectra", action="store_true",
                            help="set to True if only need wave, flux, and ivar.")

    data_group.add_argument("--deimos-source-spectra-link", type=str)
    data_group.add_argument("--deimos-spectra-data-format", type=str)
    data_group.add_argument("--deimos-source-spectra-fname", type=str)
    data_group.add_argument("--deimos-processed-spectra-cho", type=str)
    data_group.add_argument("--download-deimos-source-spectra", action="store_true")

    data_group.add_argument("--zcosmos-source-spectra-link", type=str)
    data_group.add_argument("--zcosmos-spectra-data-format", type=str)
    data_group.add_argument("--zcosmos-source-spectra-fname", type=str)
    data_group.add_argument("--zcosmos-processed-spectra-cho", type=str)
    data_group.add_argument("--download-zcosmos-source-spectra", action="store_true")

    data_group.add_argument("--deep2-spectra-data-format", type=str)
    data_group.add_argument("--deep2-source-spectra-fname", type=str)
    data_group.add_argument("--deep2-processed-spectra-cho", type=str)

    data_group.add_argument("--deep3-spectra-data-format", type=str)
    data_group.add_argument("--deep3-source-spectra-fname", type=str)
    data_group.add_argument("--deep3-processed-spectra-cho", type=str)

    data_group.add_argument("--vipers-spectra-data-format", type=str)
    data_group.add_argument("--vipers-processed-spectra-cho", type=str)
    data_group.add_argument("--vipers-w1-source-spectra-fname", type=str)
    data_group.add_argument("--vipers-w4-source-spectra-fname", type=str)

    data_group.add_argument("--max-spectra-len", type=int)
    data_group.add_argument("--min-num-valid-samples", type=int)
    data_group.add_argument("--spectra-tracts", type=str, nargs='+')
    data_group.add_argument("--spectra-patches_r", type=str, nargs='+')
    data_group.add_argument("--spectra-patches_c", type=str, nargs='+')
    data_group.add_argument("--interpolate-trans", action="store_true")

    #data_group.add_argument("--recon-spectra-clip-range", type=int, nargs='+')
    #data_group.add_argument("--dummy-spectra-clip-range", type=int, nargs='+')
    data_group.add_argument("--gt-spectra-ids", type=int, nargs='+',
                            help="id of chosen gt spectra for supervision/recon etc.")
    data_group.add_argument("--spectra-markers", type=int, nargs='+',
                            help="marker to plot each spectra.")

    data_group.add_argument("--process-ivar", action="store_true")
    data_group.add_argument("--convolve-spectra", action="store_true")
    data_group.add_argument("--spectra-smooth-sigma",type=int, default=5,
                            help="width of smooth kernel in units of angstrom")
    data_group.add_argument("--spectra-upsample-scale",type=int, default=10)
    data_group.add_argument("--flux-norm-cho",type=str,
                            choices=["identity","max","sum","linr","scale_gt","scale_recon"],
                            help="0- divide with max, 1-divide with sum")
    data_group.add_argument("--trans-norm-cho",type=str)

    data_group.add_argument("--sample-from-pretrain-spectra", action="store_true",
                            help="sample spectra for sanity check from pretrain spectra.")
    data_group.add_argument("--sanity-check-max-num-spectra", type=int,
                            help="num of spectra used for redshift pretrain, used to \
                            sample a subset of spectra from codebook pretrain spectra.")
    data_group.add_argument("--generalization-max-num-spectra", type=int)
    data_group.add_argument("--num-gt-spectra-upper-bound", type=int)
    data_group.add_argument("--num-supervision-spectra-upper-bound", type=int,
                             help="upper bound# of gt spectra used for supervision \
                             (always select the first n spectra).")

    data_group.add_argument("--sup-spectra-ratio", nargs="+", type=float)
    data_group.add_argument("--spectra-split-ratios", nargs="+", type=float,
                            help="ratio of test,val,sup spectra.")
    data_group.add_argument("--val-spectra-ratio", type=float,
                            help="ratio of validation over validation plus test spectra \
                            (the first ration% are validation spectra)\
                            used when using full patch or train spectra pixels only.")

    data_group.add_argument("--spectra-cho", type=str)
    data_group.add_argument("--spectra-drop-not-in-patch", action="store_true")
    data_group.add_argument("--spectra-process-patch-info", action="store_true")
    data_group.add_argument("--spectra-supervision-wave-lo", type=int)
    data_group.add_argument("--spectra-supervision-wave-hi", type=int)
    data_group.add_argument("--codebook-spectra-plot-wave-lo", type=int)
    data_group.add_argument("--codebook-spectra-plot-wave-hi", type=int)
    data_group.add_argument("--load-spectra-data-from-cache", action="store_true")

    data_group.add_argument("--redshift-lo", type=float,
                            help="Min value of redshift interval.")
    data_group.add_argument("--redshift-hi", type=float)
    data_group.add_argument("--redshift-bin-width", type=float,
                            help="Width of each bin to divide the interval.")

    data_group.add_argument("--filter-redshift", action="store_true")
    data_group.add_argument("--filter-redshift-lo", type=float,
                            help="Min value of redshift interval.")
    data_group.add_argument("--filter-redshift-hi", type=float)

    data_group.add_argument("--redshift-classification-sc-data-dir", type=str)
    data_group.add_argument("--redshift-classification-genlz-data-dir", type=str)
    data_group.add_argument("--redshift-classification-data-fname-prefix", type=str)

    data_group.add_argument("--redshift-regress-spectra-sample-method", type=str)

    ###################
    # Arguments for optimizer
    ###################
    optim_group = parser.add_argument_group("optimizer")

    optim_group.add_argument("--optimizer-type", type=str, default="adam",
                             choices=list(str2optim.keys()), help="Optimizer to be used.")
    optim_group.add_argument("--optimize-latents-use-lbfgs", action="store_true")

    optim_group.add_argument("--lr", type=float, default=0.0001)
    optim_group.add_argument("--grid-lr", type=float, default=0.001)
    optim_group.add_argument("--codebook-lr", type=float, default=0.0001)
    optim_group.add_argument("--spectra-latents-lr", type=float, default=0.001)
    optim_group.add_argument("--redshift-latents-lr", type=float, default=0.001)
    optim_group.add_argument("--spectra-pretrain-lr", type=float, default=0.0001)
    optim_group.add_argument("--classifier-lr", type=float, default=0.0001)
    optim_group.add_argument("--baseline-lr", type=float, default=0.0001)

    optim_group.add_argument("--weight-decay", type=float, default=0, help="Weight decay.")
    optim_group.add_argument("--grid-lr-weight", type=float, default=100.0,
                             help="Relative LR weighting for the grid")
    optim_group.add_argument("--b1",type=float, default=0.5)
    optim_group.add_argument("--b2",type=float, default=0.999)

    optim_group.add_argument("--sgd-momentum", type=float, default=0)

    ###################
    # Arguments for training
    ###################
    train_group = parser.add_argument_group("trainer")

    train_group.add_argument("--trainer-type", type=str, help="Trainer class to use")
    train_group.add_argument("--log-dir", type=str, default="_results/logs/runs/",
                             help="Log file directory for checkpoints.")
    train_group.add_argument("--cc-log-dir", type=str)
    train_group.add_argument("--cedar-log-dir", type=str)
    train_group.add_argument("--sockeye-log-dir", type=str)

    train_group.add_argument("--plot-loss", action="store_true")
    train_group.add_argument("--plot-l2-loss", action="store_true")
    train_group.add_argument("--plot-gt-bin-loss", action="store_true")
    train_group.add_argument("--using-wandb", action="store_true")
    train_group.add_argument("--step-based-sample-w-replace", action="store_true")
    train_group.add_argument("--step-based-sample-wo-replace", action="store_true")

    train_group.add_argument("--train_based_on_epochs", action="store_true")
    train_group.add_argument("--num-steps", type=int, default=250)
    train_group.add_argument("--num-epochs", type=int, default=250,
                             help="Number of epochs to run the training.")
    train_group.add_argument("--batch-size", type=int, default=512,
                             help="Batch size for the training.")
    train_group.add_argument("--render-tb-every", type=int, default=100,
                                help="Render every N iterations")
    train_group.add_argument("--log-tb-every", type=int, default=100,
                             help="Log to tensorboard at every N epoch.")
    train_group.add_argument("--log-cli-every", type=int, default=100,
                             help="Log to command line at every N epoch.")
    train_group.add_argument("--log-gpu-every", type=int, default=100,
                             help="Log to cli gpu usage at every N epoch.")
    train_group.add_argument("--save-model-every", type=int, default=200,
                             help="Save the model at every N epoch.")
    train_group.add_argument("--save-data-every", type=int, default=100,
                             help="Save data locally every N epoch.")
    train_group.add_argument("--plot-grad-every", type=int, default=20,
                             help="Plot gradient at every N epoch.")

    train_group.add_argument("--gpu-data", nargs="+", type=str,
                             help="data fields that can be added to gpu.")
    train_group.add_argument("--main-train-frozen-layer", nargs="+", type=str)

    train_group.add_argument("--pixel-loss-cho",type=str, choices=["l1","l2"])
    train_group.add_argument("--pixel-loss-reduction",type=str, choices=["none","sum","mean"])
    train_group.add_argument("--spectra-loss-cho",type=str, choices=["l1","l2"])
    train_group.add_argument("--spectra-loss-reduction",type=str,
                             choices=["none","sum","mean"])
    train_group.add_argument("--spectra-lambdawise-loss-reduction",type=str,
                             choices=["none","sum","mean"])
    train_group.add_argument("--redshift-loss-cho",type=str, choices=["l1","l2"])
    train_group.add_argument("--redshift-loss-reduction",type=str, choices=["none","sum","mean"])

    train_group.add_argument("--spectra-ssim-loss-filter-size",type=int)
    train_group.add_argument("--spectra-ssim-loss-filter-sigma",type=int)

    # train_group.add_argument("--recon-full-range-codebook-spectra", action="store_true",
    #                          help="reconstruct codebook spectra under all lambda and \
    #                          all redshift values, then recon spectra via indexing")

    train_group.add_argument("--split-latent", action="store_true",
                             help="use different latents for each decoder.")

    train_group.add_argument("--scaler-latent-dim", type=int)
    train_group.add_argument("--spectra-latent-dim", type=int)
    train_group.add_argument("--redshift-logit-latent-dim", type=int)

    train_group.add_argument("--train-with-all-pixels", action="store_true")
    train_group.add_argument("--train-pixel-ratio", type=float, default=1,
                             help="ratio of (unmasked) pixels used for training per epoch")

    train_group.add_argument("--main-train-with-pretrained-latents", action="store_true",
                             help="use trainable latent variable of autodecoder from \
                             pretrain for main train as spatial embedding instead of \
                             encoding coordinates. ")

    train_group.add_argument("--learn-spectra-within-wave-range", action="store_true")
    train_group.add_argument("--weight-train", action="store_true")
    train_group.add_argument("--weight-by-wave-coverage", action="store_true",
                             help="spectra loss weighted by emitted lambda coverage or not.")

    train_group.add_argument("--train-spectra-pixels-only", action="store_true",
                             help="whether sample only spectra pixels for main training.")
    train_group.add_argument("--pixel-supervision", action="store_true",
                             help="whether main training supervised by pixel values.")
    train_group.add_argument("--spectra-supervision", action="store_true",
                             help="whether main training supervised by spectra.")

    train_group.add_argument("--redshift-unsupervision", action="store_true",
                             help="generate redshift w/o supervision.")
    train_group.add_argument("--redshift-semi-supervision", action="store_true",
                             help="generate redshift w. semi supervision.")

    train_group.add_argument("--correct-gt-redshift-based-on-redshift-bin", action="store_true")

    train_group.add_argument("--spectra-beta", type=float, help="spectra loss weight scaler.")
    train_group.add_argument("--redshift-beta", type=float, help="redshift loss weight scaler.")

    train_group.add_argument("--decode-spatial-embedding", action="store_true")

    train_group.add_argument("--resume-train", action="store_true")
    train_group.add_argument("--resume-log-dir", type=str)
    train_group.add_argument("--resume-model-fname", type=str)

    train_group.add_argument("--uniform-sample-wave", action="store_true",
                             help="whether uniformly sample source wave or not.")
    train_group.add_argument("--mixture-avg-per-band", action="store_true",
                            help="for mixture sampling method, whether average pixel values \
                            with number of samples falling within each band")

    train_group.add_argument("--train-use-all-wave", action="store_true")
    train_group.add_argument("--spectra-supervision-use-all-wave", action="store_true")


    train_group.add_argument("--train-num-wave-samples", type=int, default=40,
                             help="# wave to sample at each training iteration.")
    train_group.add_argument("--spectra-supervision-num-wave-samples", type=int, default=1000,
                             help="# wave to sample at each training iteration \
                             for spectra supervision.")

    train_group.add_argument("--spectra-neighbour-size", type=int,
                             help="size of neighbourhood to average when calculating spectra.")
    train_group.add_argument("--infer-with-multi-neighbours", action="store_true",
                             help="infer with multi pixels in the neighbourhood and average.")

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
    infer_group.add_argument("--pretrain-infer-batch-size", type=int, default=4096)

    infer_group.add_argument("--classify-redshift-based-on-l2", action="store_true",
                             help="when classify bins, use l2 loss")
    infer_group.add_argument("--classify-redshift-based-on-combined-ssim-l2",
                             action="store_true", help="when classify bins, use ssim first\
                             and then l2 loss")
    infer_group.add_argument("--infer-use-global-loss-as-lambdawise-weights",
                             action="store_true")

    infer_group.add_argument("--infer-use-all-wave", action="store_true",
                             help="should set this to true, implementation assumes \
                             infer with all lambda")
    infer_group.add_argument("--infer-num-wave", type=int)
    infer_group.add_argument("--pretrain-infer-use-all-wave", action="store_true")
    infer_group.add_argument("--pretrain-infer-num-wave", type=int)
    infer_group.add_argument("--pretrain-infer-wave-sample-method", type=str,
                             choices=["uniform_non_random"],
                             help="should only sample every several wave.")

    infer_group.add_argument("--plot-residual-map", action="store_true")
    infer_group.add_argument("--img-resid-lo", type=int, default=-10)
    infer_group.add_argument("--img-resid-hi", type=int, default=10)
    infer_group.add_argument("--infer-synthetic-band", action="store_true")
    infer_group.add_argument("--infer-last-model-only", action="store_true")

    infer_group.add_argument("--to-HDU", action="store_true", default=False,
                             help="generate HDU files for reconstructed image")
    infer_group.add_argument("--recon-norm", action="store_true", default=False)
    infer_group.add_argument("--recon-restore", action="store_true", default=False)
    infer_group.add_argument("--log-pixel-ratio", action="store_true", default=False,
                             help="print ratio of recon over GT pixel val")
    infer_group.add_argument("--metric-options", nargs="+", choices=["mse","psnr","ssim"])
    infer_group.add_argument("--spectra-metric-options", nargs="+", choices=["zncc"])
    infer_group.add_argument("--spectra-zncc-window-width", type=float, default=1)
    infer_group.add_argument("--calculate-sliding-zncc-above-threshold",
                             action="store_true", default=False)

    infer_group.add_argument("--infer-selected", action="store_true", default=False,
                             help="infer only selected coords/spectra.")
    infer_group.add_argument("--pretrain-num-infer-upper-bound", type=int, default=60,
                             help="max #spectra to infer for pretrain.")
    infer_group.add_argument("--test-num-infer-upper-bound", type=int, default=60,
                             help="max #spectra to infer for test.")
    infer_group.add_argument("--spectra-inferrence-id-fname", type=str,
                             help="during codebook pretrain, we can infer for given id only.")

    infer_group.add_argument("--infer-outlier-only", action="store_true", default=False,
                             help="during sanity check, we infer only for outlier.")

    # these several args, if specified, directs reconstructing smaller cutouts than train image.
    # Note, if recon_img is included as inferrence tasks, we always reconstruct the full
    #   train image regardless of whether these three are given or not.
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
    infer_group.add_argument("--plot-spectrum-with-loss", action="store_true")
    infer_group.add_argument("--plot-spectrum-with-ivar", action="store_true")
    infer_group.add_argument("--plot-spectrum-with-lines", action="store_true")
    infer_group.add_argument("--plot-spectrum-with-recon", action="store_true")
    infer_group.add_argument("--plot-spectrum-with-trans", action="store_true")
    infer_group.add_argument("--plot-spectrum-with-weights", action="store_true")
    infer_group.add_argument("--plot-spectrum-under-gt-bin", action="store_true")
    infer_group.add_argument("--plot-spectrum-color-based-on-loss", action="store_true")
    infer_group.add_argument("--plot-spectrum-under-optimal-wrong-bin", action="store_true")
    infer_group.add_argument("--plot-codebook-coeff-under-optimal-wrong-bin",
                             action="store_true")
    infer_group.add_argument("--plot-spectrum-with-sliding-zncc", action="store_true")
    infer_group.add_argument("--plot-spectrum-according-to-zncc", action="store_true")
    infer_group.add_argument("--plot-spectrum-together", action="store_true")
    infer_group.add_argument("--local-zncc-threshold", type=float,
                             help="if plot spectrum according to zncc, we highlight region of \
                             the spectra above this threshold with a different color.")

    infer_group.add_argument("--plot-residual-with-ivar", action="store_true",
                             help="plot spectra residual and ivar together")
    infer_group.add_argument("--plot-residual-times-ivar", action="store_true",
                             help="plot residual times ivar, should follow N(0,1)")
    infer_group.add_argument("--plot-ivar-region", action="store_true",
                             help="plot gt spectra +- sqrt(1/ivar) region.")

    infer_group.add_argument("--plot-global-lambdawise-spectra-loss-with-ivar",
                             action="store_true")
    infer_group.add_argument("--plot-global-lambdawise-spectra-loss-with-lines",
                             action="store_true")
    infer_group.add_argument("--emitted-wave-overlap-discretization-val", type=float)

    infer_group.add_argument("--infer-spectra-individually", action="store_true")
    infer_group.add_argument("--codebook-spectra-clip-range", nargs="+")

    infer_group.add_argument("--log-redshift-est-stats", action="store_true")
    infer_group.add_argument("--redshift-est-stats-cho", type=str,
                             choices=["accuracy","precision_recall"])
    infer_group.add_argument("--num-redshift-est-stats-residual-levels", type=int)
    infer_group.add_argument("--redshift-est-stats-residual-levels", nargs="+")
    infer_group.add_argument("--redshift-est-stats-max-residual",
                             type=float, default=-1)
    infer_group.add_argument("--plot-redshift-est-stats-individually",
                             action="store_true")
    infer_group.add_argument("--calculate-redshift-est-stats-based-on", type=str,
                             choices=["logits","residuals"])

    infer_group.add_argument("--save-classification-data-individually",
                             action="store_true")

    infer_group.add_argument("--num-spectrum-per-fig", type=int,
                             help="number of spectrum in each figure.")
    infer_group.add_argument("--num-spectrum-per-row", type=int,
                             help="number of spectrum in each row in each plot.")

    infer_group.add_argument("--plot-labels", nargs="+", type=str)
    infer_group.add_argument("--plot-colors", nargs="+", type=str)
    infer_group.add_argument("--plot-styles", nargs="+", type=str)

    infer_group.add_argument("--sanity-check-plot-same-pca-dim-as-pretrain",
                             action="store_true")
    infer_group.add_argument("--pretrain-pca-dim-fname", type=str)
    infer_group.add_argument("--spectra-latents-plot-pca-dim", type=int)

    infer_group.add_argument("--redshift-est-stats-labels", type=str, nargs="+")
    infer_group.add_argument("--redshift-est-stats-fnames", type=str, nargs="+")
    infer_group.add_argument("--overlay-redshift-est-stats-fname", type=str)

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
    inpaint_group.add_argument("--perform-inpainting", action="store_true")
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
