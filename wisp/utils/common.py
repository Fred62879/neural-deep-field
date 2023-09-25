
import re
import os
import sys
import torch
import random
import logging
import nvidia_smi
import numpy as np
import torch.nn as nn

from os.path import join
from astropy.io import fits
from astropy.wcs import WCS
from functools import reduce
from collections import defaultdict
from astropy.coordinates import SkyCoord


def get_bool_encode_coords(**kwargs):
    return kwargs["encode_coords"] and not \
        ( kwargs["pretrain_codebook"] and \
          kwargs["main_train_with_pretrained_latents"] )

def get_bool_classify_redshift(**kwargs):
    return kwargs["model_redshift"] and not kwargs["apply_gt_redshift"] \
        and kwargs["redshift_model_method"] == "classification"

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

def freeze_layers(model, excls):
    """ Freeze layers in model (excluding those in `excls`).
    """
    for n, p in model.named_parameters():
        to_exclude = False
        for excl in excls:
            if excl in n: to_exclude = True
        if not to_exclude:
            p.requires_grad = False

def init_redshift_bins(**kwargs):
    redshift_bin_center = torch.arange(
        kwargs["redshift_lo"],
        kwargs["redshift_hi"],
        kwargs["redshift_bin_width"])
    offset = kwargs["redshift_bin_width"] / 2
    redshift_bin_center += offset
    return redshift_bin_center

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

def to_numpy(tensor):
    class_name = tensor.__class__.__name__
    if class_name == "Tensor":
        tensor = tensor.numpy()
    assert tensor.__class__.__name__ == "ndarray"
    return tensor

def get_pretrained_model_fname(log_dir, dname, model_fname):
    """ Format checkpoint fname from given experiemnt directory.
    """
    if dname is not None:
        pretrained_model_dir = join(log_dir, "..", dname)
    else:
        # if log dir not specified, use last directory (exclude newly created one)
        dnames = os.listdir(join(log_dir, ".."))
        assert(len(dnames) > 1)
        dnames.sort()
        pretrained_model_dir = join(log_dir, "..", dnames[-2])

    loss_fname = join(pretrained_model_dir, "loss.npy")

    pretrained_model_dir = join(pretrained_model_dir, "models")

    if model_fname is not None:
        pretrained_model_fname = join(pretrained_model_dir, model_fname)
    else:
        fnames = os.listdir(pretrained_model_dir)
        assert(len(fnames) > 0)
        fnames = sort_alphanumeric(fnames)
        pretrained_model_fname = join(pretrained_model_dir, fnames[-1])

    return pretrained_model_fname, loss_fname

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
    set_seed()
    ids = np.arange(n)
    np.random.shuffle(ids)
    ids = ids[:m]
    # ids = np.array([26 ,86  ,2 ,55 ,75 ,93 ,16 ,73 ,54 ,95 ,53 ,92, 78 ,13  ,7 ,30 ,22 ,24 ,33  ,8])
    return ids

def create_patch_uid(tract, patch):
    patch = patch.replace(",", "")
    return f"{tract}{patch}"

def print_shape(data):
    for n,p in data.items():
        if p is None:
            print(f"{n} is None")
        elif type(p) == tuple or type(p) == list:
            print(n, len(p), p[0].dtype, p[0].device)
        elif type(p) == torch.Tensor:
            print(n, p.shape, p.dtype, p.device)
        elif type(p) == int:
            print(n, p)
        else: print(n, p.shape, p.dtype)

def get_input_latents_dim(**kwargs):
    """ Get the dimension of the input RA/DEC coordinate for MLP.
    """
    if kwargs["pretrain_codebook"] and \
       ("codebook_pretrain" in kwargs["tasks"] or \
        "codebook_pretrain_infer" in kwargs["tasks"] or \
        "redshift_pretrain" in kwargs["tasks"] or \
        "redshift_pretrain_infer" in kwargs["tasks"] or \
        kwargs["main_train_with_pretrained_latents"]):
        latents_dim = kwargs["codebook_pretrain_latent_dim"]
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
        qtz=False,
        qtz_strategy="none",
        apply_gt_redshift=False,
        codebook_pretrain=False,
        classify_redshift=False,
        spectra_supervision=False,
        perform_integration=False,
        trans_sample_method="none",
        redshift_supervision_train=False,
        save_scaler=False,
        save_spectra=False,
        save_latents=False,
        save_codebook=False,
        save_redshift=False,
        save_embed_ids=False,
        save_qtz_weights=False,
        save_codebook_loss=False,
        save_codebook_spectra=False,
        save_spectra_all_bins=False
):
    requested_channels = []
    net_args = {"coords": data["coords"] }

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
        if classify_redshift: requested_channels.append("redshift_logits")
        if save_codebook_spectra: requested_channels.append("codebook_spectra")
        if save_spectra_all_bins: requested_channels.append("spectra_all_bins")

        net_args["wave"] = data["wave"]
        net_args["wave_range"] = data["wave_range"] # linear normalization

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
    else:
        raise ValueError("Unsupported space dimension.")

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

def load_pretrained_model_weights(model, pretrained_state, shared_layer_names=None):
    """ Load weights from saved model.
        Loading is performed for only layers in both the given model and
          the pretrained state and in `shared_layer_names` if not None.
    """
    pretrained_dict = {}
    cur_state = model.state_dict()
    for n in cur_state.keys():
        if n in pretrained_state and (
                shared_layer_names is None or includes_layer(shared_layer_names, n)
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
