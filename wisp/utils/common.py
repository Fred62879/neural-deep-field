
import os
import torch
import pickle
import random
import nvidia_smi
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from astropy.io import fits
from astropy.wcs import WCS
from functools import reduce
from os.path import exists, join
from astropy.nddata import Cutout2D
from astroquery.svo_fps import SvoFps
from astropy.coordinates import SkyCoord
#from unagi import filters as unagi_filters
from wisp.utils.plot import plot_embd_map
from wisp.utils.numerical import normalize, calculate_metrics


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
    print("Total memory:", info.total)
    print("Free memory:", info.free)
    print("Used memory:", info.used)
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

def forward(class_obj, pipeline, data, quantize_latent=False, plot_embd_map=False, spectra_supervision=False):
    if class_obj.space_dim == 2:
        requested_channels = {"intensity"}
        #print("forward", data["coords"].shape)
        net_args = {"coords": data["coords"].to(class_obj.device) }

    elif class_obj.space_dim == 3:
        # channels for nerf
        requested_channels = ["latents"]
        if class_obj.quantize_latent:
            requested_channels.append("scaler")
            requested_channels.append("redshift")
        requested_channels = set(requested_channels)

        trans_sample_method = class_obj.extra_args["trans_sample_method"]
        if trans_sample_method == "hardcode":
            net_args = {
                "coords": data["coords"],
                "trans": data["trans"]
            }
        elif trans_sample_method == "bandwise":
            net_args = {
                "coords": data["coords"].to(class_obj.device),
                "wave": data["wave"].to(class_obj.device),
                "trans": data["trans"].to(class_obj.device)
            }
        elif trans_sample_method == "mixture":
            net_args = {
                "coords": data["coords"], #.to(class_obj.device),
                "wave":   data["wave"], #.to(class_obj.device),
                "trans":  data["trans"], #.to(class_obj.device),
                "nsmpl":  data["nsmpl"] #.to(class_obj.device)
            }
        else: raise ValueError("Unrecognized transmission sampling method.")

        if spectra_supervision:
            net_args["full_wave"] = data["full_wave"]

    else: raise Exception("Unsupported space dimension.")
    return pipeline(channels=requested_channels, **net_args)

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
    pretrained_dict = {k: v for k, v in pretrained_state.items() if k in cur_state}
    model.load_state_dict(pretrained_dict)

def load_model(model, optimizer, modelDir, model_smpl_intvl, cuda, verbose):
    try:
        nmodels = len(os.listdir(modelDir))
        if nmodels < 1: raise ValueError("No saved models found")

        modelnm = os.path.join(modelDir, str(nmodels-1)+'.pth')
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

def load_layer_weights(checkpoint, layer_name):
    for n, p in checkpoint.items():
        if layer_name in n:
            return p
    assert(False)
