
import re
import os
import sys
import torch
import random
import logging
import nvidia_smi
import numpy as np
import logging as log
import torch.nn as nn

from os.path import join
from astropy.io import fits
from astropy.wcs import WCS
from functools import reduce
from itertools import accumulate
from collections import defaultdict
from astropy.coordinates import SkyCoord


def get_log_dir(**kwargs):
    if kwargs["on_cedar"] or kwargs["on_graham"] or kwargs["on_narval"]:
        log_dir = kwargs["cedar_log_dir"]
    elif kwargs["on_sockeye"]:
        log_dir = kwargs["sockeye_log_dir"]
    else: log_dir = kwargs["log_dir"]
    return log_dir

def get_current_ablate_param_and_val(args):
    id = args.ablat_id
    num_vals = args.ablat_num_vals
    acc = list(accumulate(num_vals))
    acc = np.array(acc) - id
    param_id = np.where(acc > 0)[0][0]
    val_id = num_vals[param_id] - acc[param_id]
    # print(id, acc, param_id, val_id)
    param = args.ablat_params[param_id]
    val = args.ablat_vals[param_id][val_id]
    return param, val

def get_bool_encode_coords(**kwargs):
    return kwargs["encode_coords"] and not \
        ( kwargs["pretrain_codebook"] and \
          kwargs["main_train_with_pretrained_latents"] )

def get_bool_classify_redshift(**kwargs):
    return kwargs["space_dim"] == 3 and \
        kwargs["model_redshift"] and \
        not kwargs["apply_gt_redshift"] and \
        kwargs["redshift_model_method"] == "classification"

def get_bool_has_redshift_latents(**kwargs):
    """ Check whether we need an independent redshift latents.
    """
    return kwargs["space_dim"] == 3 and \
        kwargs["model_redshift"] and \
        kwargs["split_latent"] and \
        not kwargs["apply_gt_redshift"] and \
        not kwargs["use_binwise_spectra_loss_as_redshift_logits"] and \
        not kwargs["optimize_codebook_latents_for_each_redshift_bin"]

def get_optimal_wrong_bin_ids(ret, data):
    """ Get id of the non-GT redshift bin that achieves the lowest spectra loss.
    """
    all_bin_losses = ret["spectra_binwise_loss"] # [bsz,nbins]
    all_bin_losses[data["gt_redshift_bin_masks"]] = float('inf')
    optimal_wrong_bin_losses, optimal_wrong_bin_ids = torch.min(
        all_bin_losses, dim=-1) # ids of optimal wrong bins
    return optimal_wrong_bin_ids, optimal_wrong_bin_losses

def get_bin_id(lo, bin_width, val):
    val = val - bin_width / 2
    n = (val - lo) / bin_width
    return int(np.rint(n))

def get_bin_ids(lo, bin_width, vals, add_batched_dim=False):
    vals = vals - bin_width / 2
    ids = (vals - lo) / bin_width
    ids = np.rint(ids).astype(int)
    if add_batched_dim:
        bsz = len(vals)
        indices = np.arange(bsz)[None,:]
        ids = np.concatenate((indices, ids[None,:]), axis=0)
    return ids

def get_loss(cho, cuda):
    if cho == "l1_mean":
        loss = nn.L1Loss()
    elif cho == "l1_sum":
        loss = nn.L1Loss(reduction="sum")
    elif cho == "l1_none":
        loss = nn.L1Loss(reduction="none")
    elif cho == "l2_mean":
        loss = nn.MSELoss()
    elif cho == "l2_sum":
        loss = nn.MSELoss(reduction="sum")
    elif cho == "l2_none":
        loss = nn.MSELoss(reduction="none")
    else:
        raise Exception("Unsupported loss choice")
    if cuda: loss = loss.cuda()
    return loss

def create_batch_ids(ids):
    """ Add batch dim id to a given list of ids.
        @Param
          ids: [n,] list of ids
        @Return
          ids: [2,n] ids with batch dim ([0,1,2,...,n])
    """
    n = len(ids)
    indices = np.arange(n)[None,:]
    ids = np.concatenate((indices, ids[None,:]), axis=0)
    return ids

def log_data(obj, field, fname=None, gt_field=None, mask=None,
              log_ratio=False, log_to_console=True):
    """ Log estimated and gt data is specified.
        If `fname` is not None, we save recon data locally.
        If `mask` is not None, we apply mask before logging.
        If `log_ratio` is True, we log ratio of recon over gt data.
    """
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=3)
    log_ratio = log_ratio and gt_field is not None

    if gt_field is not None:
        gt = to_numpy(getattr(obj, gt_field))
    recon = to_numpy(getattr(obj, field))
    if mask is not None:
        recon = recon[mask]

    if fname is not None and len(recon) > 0:
        if gt_field is not None:
            to_save = np.concatenate((gt[None,:], recon[None,:]), axis=0)
        else: to_save = recon
        if fname[-3:] == "npy":
            np.save(fname, to_save)
        elif fname[-3:] == "txt":
            with open(fname, "w") as f:
                f.write(f"{to_save}")

    if not log_to_console: return

    if gt_field is None:
        log.info(f"{field}: {recon}")
    elif log_ratio:
        ratio = recon/gt
        log.info(f"{field}/{gt_field}: {ratio}")
    else:
        log.info(f"{gt_field}: {gt}")
        log.info(f"recon {field}: {recon}")

def freeze_layers_incl(model, incls=[]):
    """ Freeze layers (in incls) in model.
    """
    for n, p in model.named_parameters():
        freeze = False
        for incl in incls:
            if incl in n: freeze = True; break
        if freeze: p.requires_grad = False

def freeze_layers_excl(model, excls=[]):
    """ Freeze layers in model (excluding those in `excls`).
    """
    for n, p in model.named_parameters():
        freeze = True
        for excl in excls:
            if excl in n: freeze = False; break
        if freeze: p.requires_grad = False

def init_redshift_bins(lo, hi, bin_width, init_np=False):
    if init_np: redshift_bin_center = np.arange(lo, hi, bin_width)
    else:       redshift_bin_center = torch.arange(lo, hi, bin_width)
    offset = bin_width / 2
    redshift_bin_center += offset
    return redshift_bin_center

def create_latent_mask(lo, hi, ndim):
    mask = torch.zeros(ndim)
    mask[lo:hi] = 1
    return mask.long()

def segment_bool_array(arr):
    """ Get segments of True from a boolean array.
    """
    arr = arr.astype(np.bool)
    segments = []
    n, lo = len(arr), -1
    for i in range(n):
        if arr[i]:
            if lo == -1: lo = i
        else:
            if lo != -1: segments.append([lo, i])
            lo = -1
    if lo != -1: segments.append([start, n])
    return segments

def tensor_to_numpy(tensor, detach=True):
    if detach: tensor = tensor.detach()
    if tensor.device != "cpu": tensor = tensor.cpu()
    tensor = tensor.numpy()
    return tensor

def to_numpy(tensor, detach=True):
    class_name = tensor.__class__.__name__
    if class_name == "list":
        sub_class_name = tensor[0].__class__.__name__
        if sub_class_name == "Tensor":
            tensor = tensor_to_numpy(torch.stack(tensor))
        elif sub_class_name == "ndarray" or sub_class_name == "list":
            tensor = np.array(tensor)
        else: raise ValueError()
    elif class_name == "Tensor":
        tensor = tensor_to_numpy(torch.stack(tensor))
    assert tensor.__class__.__name__ == "ndarray"
    return tensor

def get_pretrained_model_fname(log_dir, dname, model_fname):
    """ Format checkpoint fname from given experiemnt directory.
    """
    if dname is not None:
        pretrained_dir = join(log_dir, "..", dname)
    else:
        # if log dir not specified, use last directory (exclude newly created one)
        dnames = os.listdir(join(log_dir, ".."))
        assert(len(dnames) > 1)
        dnames.sort()
        pretrained_dir = join(log_dir, "..", dnames[-2])

    pretrained_model_dir = join(pretrained_dir, "models")
    if model_fname is not None:
        pretrained_model_fname = join(pretrained_model_dir, model_fname)
    else:
        fnames = os.listdir(pretrained_model_dir)
        assert(len(fnames) > 0)
        fnames = sort_alphanumeric(fnames)
        pretrained_model_fname = join(pretrained_model_dir, fnames[-1])

    return pretrained_dir, pretrained_model_fname

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    # logging.info(f"Random seed set as {seed}")

def default_log_setup(level=logging.INFO):
    """ Sets up default logging, always logging to stdout.
        :param level: logging level, e.g. logging.INFO
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    logging.basicConfig(
        level=level,
        format='%(asctime)s|%(levelname)8s| %(message)s',
        handlers=handlers
    )

def select_inferrence_ids(n, m):
    np.random.seed(20)
    ids = np.arange(n)
    np.random.shuffle(ids)
    ids = ids[:m]
    return ids

def create_patch_uid(tract, patch):
    patch = patch.replace(",", "")
    return f"{tract}{patch}"

def print_shape(data):
    for n,p in data.items():
        if p is None:
            print(f"{n} is None")
        elif type(p) == tuple or type(p) == list:
            print(n, len(p), p[0].__class__.__name__)
        elif type(p) == torch.Tensor:
            print(n, p.shape, p.dtype, p.device)
        elif type(p) == int:
            print(n, p)
        else: print(n, p.shape, p.dtype)

def get_input_latent_dim(**kwargs):
    """ Get the dimension of the input RA/DEC coordinate for MLP.
    """
    if kwargs["pretrain_codebook"] and \
       ("codebook_pretrain" in kwargs["tasks"] or \
        "codebook_pretrain_infer" in kwargs["tasks"] or \
        "redshift_pretrain" in kwargs["tasks"] or \
        "redshift_pretrain_infer" in kwargs["tasks"] or \
        kwargs["main_train_with_pretrained_latents"]
       ):
        latents_dim = kwargs["codebook_latent_dim"]
    elif kwargs["coords_encode_method"] == "positional_encoding":
        latents_dim = kwargs["coords_embed_dim"]
    elif kwargs["coords_encode_method"] == "grid":
        latents_dim = kwargs["grid_feature_dim"]
        if kwargs["grid_multiscale_type"] == 'cat':
            latents_dim *= kwargs["grid_num_lods"]
    else:
        latents_dim = 2
    return latents_dim

def add_to_device(data, valid_fields, device):
    for field in valid_fields:
        if field in data:
            data[field] = data[field].to(device)

def sort_alphanumeric(list):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(list, key = alphanum_key)

def get_grid(rlo, rhi, clo, chi):
    ''' Generates 2d grid. '''
    nr, nc = rhi-rlo, chi-clo
    r = np.repeat(np.arange(rlo, rhi), nc).reshape((nr,nc,1))
    c = np.tile(np.arange(clo, chi), nr).reshape((nr,nc,1))
    grid = np.concatenate((r, c), axis=2)
    return grid

def get_gpu_info():
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate
    return nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

def query_mem(a):
    return a.element_size() * a.nelement()

def query_GPU_mem():
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    print(f"Total memory: {info.total/1e9}GB")
    print(f"Free memory: {info.free/1e9}GB")
    print(f"Used memory: {info.used/1e9}GB")
    nvidia_smi.nvmlShutdown()

def generate_hdu(header, data, fname):
    hdu = fits.PrimaryHDU(data=data, header=header)
    hdu.writeto(fname, overwrite=True)

# def get_num_wave_smpl(infer, args):
#     ''' Get # of wave samples to use. '''

#     # all 3 mc can infer with all wave
#     if infer and args.infer_use_all_wave:
#         nsmpl = len(np.load(args.full_wave_fn))
#     elif args.mc_cho == 'mc_hardcode':
#         nsmpl = args.num_trans_smpl
#     elif args.mc_cho == 'mc_bandwise':
#         nsmpl = args.num_trans_smpl//args.num_bands

#     # only mixture mc can train with all wave
#     elif not infer and args.train_use_all_wave:
#         nsmpl = len(np.load(args.full_wave_fn))
#     elif args.mc_cho == 'mc_mixture':
#         nsmpl = args.num_trans_smpl
#     else:
#         raise Exception('Unsupported monte carlo choice')
#     return nsmpl

def restore_unmasked(recon, gt, mask):
    ''' Fill recon with unmasked pixels from gt
        @Param
          recon/gt/mask [nbands,sz,sz]
    '''
    (nbands,sz,sz) = recon.shape
    npixls = sz**2

    recon = recon.reshape((nbands, -1))
    gt = gt.reshape((nbands, -1))

    for i in range(nbands):
        cur_mask = mask[i]
        is_train_band = np.count_nonzero(cur_mask) == npixls
        if is_train_band:
            recon[i] = gt[i]
        else:
            ids = np.arange(npixls).reshape((sz,sz))
            unmasked_ids = (ids[cur_mask == 1]).flatten()
            recon[i][unmasked_ids] = gt[i][unmasked_ids]

    recon = recon.reshape((-1,sz,sz))
    return recon

def worldToPix(header, ra, dec):
    w = WCS(header)
    sc = SkyCoord(ra, dec, unit='deg')
    x, y = w.world_to_pixel(sc)
    return int(y), int(x) # r,c

def world2NormPix(coords, args, infer=True, spectrum=True, coord_wave=None):
    rnge = np.load(args.coords_rnge_fn)
    coords = normalize_coord(rnge, coords)
    coords = torch.tensor(coords).type(args.float_tensor)
    #coords = reshape_coords(coords, args, infer=infer, spectrum=spectrum, coord_wave=coord_wave)
    return coords

def forward(
        data,
        pipeline,
        step_num,
        space_dim,
        spectra_loss_func=None,
        qtz=False,
        qtz_strategy="none",
        index_latent=False,
        split_latent=False,
        apply_gt_redshift=False,
        codebook_pretrain=False,
        spectra_supervision=False,
        perform_integration=False,
        trans_sample_method="none",
        redshift_supervision_train=False,
        regularize_codebook_spectra=False,
        calculate_binwise_spectra_loss=False,
        save_scaler=False,
        save_spectra=False,
        save_latents=False,
        save_codebook=False,
        save_redshift=False,
        save_embed_ids=False,
        save_qtz_weights=False,
        save_codebook_loss=False,
        save_gt_bin_spectra=False,
        save_optimal_bin_ids=False,
        save_codebook_logits=False,
        save_redshift_logits=False,
        save_codebook_latents=False,
        save_codebook_spectra=False,
        save_spectra_all_bins=False,
        init_redshift_prob=None, # debug
):
    net_args, requested_channels = {}, []

    if "coords" in data:
        net_args["coords"] = data["coords"]
    else: net_args["coords"] = None

    net_args["init_redshift_prob"] = init_redshift_prob # debug

    if space_dim == 2:
        requested_channels = ["intensity"]

    elif space_dim == 3:
        requested_channels = ["intensity"]
        if save_scaler: requested_channels.append("scaler")
        if save_spectra: requested_channels.append("spectra")
        if save_latents: requested_channels.append("latents")
        if save_codebook: requested_channels.append("codebook")
        if save_redshift: requested_channels.append("redshift")
        if save_embed_ids: requested_channels.append("min_embed_ids")
        if save_qtz_weights: requested_channels.append("qtz_weights")
        if save_codebook_loss: requested_channels.append("codebook_loss")
        if save_gt_bin_spectra: requested_channels.append("gt_bin_spectra")
        if save_optimal_bin_ids: requested_channels.append("optimal_bin_ids")
        if save_codebook_logits: requested_channels.append("codebook_logits")
        if save_redshift_logits: requested_channels.append("redshift_logits")
        if save_codebook_latents: requested_channels.append("codebook_latents")
        if save_codebook_spectra: requested_channels.append("codebook_spectra")
        if save_spectra_all_bins: requested_channels.append("spectra_all_bins")

        net_args["wave"] = data["wave"]
        net_args["wave_range"] = data["wave_range"] # linear normalization

        if index_latent:
            if "idx" in data:
                net_args["idx"] = data["idx"]
            if "selected_ids" in data:
                net_args["selected_ids"] = data["selected_ids"]

        if split_latent:
            if "scaler_latents" in data:
                net_args["scaler_latents"] = data["scaler_latents"]
            if "redshift_latents" in data:
                net_args["redshift_latents"] = data["redshift_latents"]

        if apply_gt_redshift:
            net_args["specz"] = data["spectra_redshift"]
        if perform_integration:
            net_args["trans"] = data["trans"]
            net_args["nsmpl"] = data["nsmpl"]
            # net_args["trans_mask"] = data["trans_mask"]
        if spectra_supervision:
            net_args["num_sup_spectra"] = data["num_sup_spectra"]
            net_args["sup_spectra_wave"] = data["sup_spectra_wave"]
            requested_channels.append("sup_spectra")
        if qtz:
            qtz_args = defaultdict(lambda: False)
            if qtz_strategy == "soft":
                qtz_args["save_qtz_weights"] = save_qtz_weights
                qtz_args["temperature"] = step_num + 1
                if save_embed_ids:
                    qtz_args["find_embed_id"] = save_embed_ids
            qtz_args["save_codebook_spectra"] = save_codebook_spectra
            net_args["qtz_args"] = qtz_args
        if regularize_codebook_spectra:
            net_args["full_emitted_wave"] = data["full_emitted_wave"]
            requested_channels.append("full_range_codebook_spectra")
        if calculate_binwise_spectra_loss:
            net_args["spectra_masks"] = data["spectra_masks"]
            net_args["spectra_loss_func"] = spectra_loss_func
            net_args["spectra_source_data"] = data["spectra_source_data"]
            requested_channels.extend(
                ["spectra_binwise_loss","redshift_logits"])
        if save_gt_bin_spectra:
            net_args["gt_redshift_bin_ids"] = data["gt_redshift_bin_ids"]
    else:
        raise ValueError("Unsupported space dimension.")

    # print(requested_channels)
    requested_channels = set(requested_channels)
    return pipeline(channels=requested_channels, **net_args)

def load_layer_weights(checkpoint, layer_identifier):
    for n, p in checkpoint.items():
        if layer_identifier(n):
            return p
    assert(False)

def includes_layer(target_layers, source_layer):
    """ Determin if source_layer is present in target_layers.
        Note: target_layers are generally abbreviations of source_layer.
    """
    for target_layer in target_layers:
        if target_layer in source_layer: return True
    return False

def load_pretrained_model_weights(model, pretrained_state, shared_layer_names=None, excls=[]):
    """ Load weights from saved model.
        Loading is performed for only layers in both the given model and
          the pretrained state and in `shared_layer_names` if not None.
        Also, we don't load layers included in excls.
    """
    pretrained_dict = {}
    cur_state = model.state_dict()
    for n in cur_state.keys():
        to_exclude = False
        for excl in excls:
            if excl in n: to_exclude = True

        if not to_exclude and (n in pretrained_state and (
                shared_layer_names is None or includes_layer(shared_layer_names, n))
        ):
            pretrained_dict[n] = pretrained_state[n]
        else: pretrained_dict[n] = cur_state[n]

    model.load_state_dict(pretrained_dict)

def load_model_weights(model, pretrained_state):
    cur_state = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_state.items() if k in cur_state}
    model.load_state_dict(pretrained_dict)

def load_embed(pretrained_state, transpose=True, tensor=False):
    for n,p in pretrained_state.items():
        if "grid" not in n and "codebook" in n:
            if transpose:
                p = p.T
            break
    if tensor:
        return p
    return np.array(p.cpu())

# def load_partial_latent(model, pretrained_state, lo, hi):
#     cur_state = model.state_dict()
#     pretrained_dict = {}
#     for k, v in pretrained_state.items():
#         if 'latents' in k: v = v[lo:hi]
#         pretrained_dict[k] = v
#     model.load_state_dict(pretrained_dict)

# def load_model_weights_exact(model, pretrained_state, train_chnls):
#     cur_state = model.state_dict()
#     pretrained_dict = {}
#     for k, v in pretrained_state.items():
#         if k in cur_state and 'scale_layer' in k:
#             v = v[train_chnls]
#         pretrained_dict[k] = v
#     model.load_state_dict(pretrained_dict)

# def load_model(model, optimizer, modelDir, model_smpl_intvl, cuda, verbose):
#     try:
#         nmodels = len(os.listdir(modelDir))
#         if nmodels < 1: raise ValueError("No saved models found")

#         modelnm = join(modelDir, str(nmodels-1)+'.pth')
#         if verbose:
#             print(f'= Saved model found, loading {modelnm}')
#         checkpoint = torch.load(modelnm)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         model.train()

#         if cuda:
#             optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#             for state in optimizer.state.values():
#                 for k, v in state.items():
#                     if torch.is_tensor(v):
#                         state[k] = v.cuda()
#         else:
#             optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

#         epoch_trained = checkpoint['epoch_trained']
#         model_file_id = (epoch_trained+1)//model_smpl_intvl+1
#         if verbose: print("= resume training")
#         return model_file_id, epoch_trained, model, optimizer
#     except Exception as e:
#         if verbose:
#             print('!=', e)
#             print("= start training from begining")
#         return 0, -1, model, optimizer
