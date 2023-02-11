
import re
import os
import torch
import nvidia_smi
import numpy as np

from os.path import join
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord


def get_input_latents_dim(**kwargs):
    """ Infer the dimension of the input RA/DEC coordinate.
    """
    if kwargs["coords_encode_method"] == "positional":
        latents_dim = kwargs["coords_embed_dim"]
    elif kwargs["coords_encode_method"] == "grid":
        latents_dim = kwargs["grid_feature_dim"]
        if kwargs["grid_multiscale_type"] == 'cat':
            latents_dim *= kwargs["grid_num_lods"]
    else:
        latents_dim = 2
    # if self.kwargs["grid_multiscale_type"] == 'cat':
    #     self.effective_feature_dim = self.grid.feature_dim * self.kwargs["grid_num_lods"]
    # else: grid_effective_feature_dim = self.grid.feature_dim
    return latents_dim

def add_to_device(data, valid_fields, device):
    for field in valid_fields:
        if field in data:
            data[field] = data[field].to(device)

def sorted_nicely(list):
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

def get_num_wave_smpl(infer, args):
    ''' Get # of wave samples to use. '''

    # all 3 mc can infer with all wave
    if infer and args.infer_use_all_wave:
        nsmpl = len(np.load(args.full_wave_fn))
    elif args.mc_cho == 'mc_hardcode':
        nsmpl = args.num_trans_smpl
    elif args.mc_cho == 'mc_bandwise':
        nsmpl = args.num_trans_smpl//args.num_bands

    # only mixture mc can train with all wave
    elif not infer and args.train_use_all_wave:
        nsmpl = len(np.load(args.full_wave_fn))
    elif args.mc_cho == 'mc_mixture':
        nsmpl = args.num_trans_smpl
    else:
        raise Exception('Unsupported monte carlo choice')
    return nsmpl

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

def forward(class_obj, pipeline, data,
            pixel_supervision_train=True,
            spectra_supervision_train=False,
            redshift_supervision_train=False,
            quantize_latent=False,
            calculate_codebook_loss=False,
            infer=False,
            save_scaler=False,
            save_spectra=False,
            save_latents=False,
            save_redshift=False,
            save_embed_ids=False):

    # cannot in both train and infer mode
    train = pixel_supervision_train or spectra_supervision_train or redshift_supervision_train
    assert( train != infer )

    requested_channels = []
    net_args = {"coords": data["coords"] }

    if class_obj.space_dim == 2:
        requested_channels = ["intensity"]

    elif class_obj.space_dim == 3:
        requested_channels = ["intensity"]
        if quantize_latent and calculate_codebook_loss:
            requested_channels.append("codebook_loss")
        if save_scaler: requested_channels.append("scaler")
        if save_spectra: requested_channels.append("spectra")
        if save_latents: requested_channels.append("latents")
        if save_embed_ids: requested_channels.append("min_embed_ids")
        if spectra_supervision_train: requested_channels.append("spectra")
        if save_redshift or redshift_supervision_train: requested_channels.append("redshift")

        net_args["full_wave_bound"] = data["full_wave_bound"]

        if pixel_supervision_train or infer:
            sample_method = class_obj.extra_args["trans_sample_method"]
            if sample_method == "hardcode":
                net_args["wave"] = data["wave"]
                net_args["trans"] = data["trans"]
                net_args["nsmpl"] = data["nsmpl"]
            elif sample_method == "bandwise":
                pass
            elif sample_method == "mixture":
                net_args["wave"] = data["wave"]
                net_args["trans"] = data["trans"]
                net_args["nsmpl"] = data["nsmpl"]
            else: raise ValueError("Unrecognized transmission sampling method.")

        if spectra_supervision_train:
            net_args["full_wave"] = data["full_wave"]
            # num of coords for gt, dummy (incl. neighbours) spectra
            net_args["num_spectra_coords"] = data["num_spectra_coords"]

    else: raise ValueError("Unsupported space dimension.")

    requested_channels = set(requested_channels)
    return pipeline(channels=requested_channels, **net_args)

def load_embed(pretrained_state):
    for n,p in pretrained_state.items():
        if "grid" not in n and "codebook" in n:
            embed = p.T
            break
    return np.array(embed.cpu())

def load_partial_latent(model, pretrained_state, lo, hi):
    cur_state = model.state_dict()
    pretrained_dict = {}
    for k, v in pretrained_state.items():
        if 'latents' in k: v = v[lo:hi]
        pretrained_dict[k] = v
    model.load_state_dict(pretrained_dict)

def load_model_weights_exact(model, pretrained_state, train_chnls):
    cur_state = model.state_dict()
    pretrained_dict = {}
    for k, v in pretrained_state.items():
        if k in cur_state and 'scale_layer' in k:
            v = v[train_chnls]
        pretrained_dict[k] = v
    model.load_state_dict(pretrained_dict)

def load_model_weights(model, pretrained_state):
    cur_state = model.state_dict()
    #print(pretrained_state.keys())
    pretrained_dict = {k: v for k, v in pretrained_state.items() if k in cur_state}
    #print(pretrained_dict.keys())
    model.load_state_dict(pretrained_dict)

def load_model(model, optimizer, modelDir, model_smpl_intvl, cuda, verbose):
    try:
        nmodels = len(os.listdir(modelDir))
        if nmodels < 1: raise ValueError("No saved models found")

        modelnm = join(modelDir, str(nmodels-1)+'.pth')
        if verbose:
            print(f'= Saved model found, loading {modelnm}')
        checkpoint = torch.load(modelnm)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.train()

        if cuda:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
        else:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch_trained = checkpoint['epoch_trained']
        model_file_id = (epoch_trained+1)//model_smpl_intvl+1
        if verbose: print("= resume training")
        return model_file_id, epoch_trained, model, optimizer
    except Exception as e:
        if verbose:
            print('!=', e)
            print("= start training from begining")
        return 0, -1, model, optimizer

def load_layer_weights(checkpoint, layer_identifier):
    for n, p in checkpoint.items():
        #if layer_name in n:
        if layer_identifier(n):
            return p
    assert(False)
