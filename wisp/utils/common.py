
import os
import torch
import pickle
import random
#import nvidia_smi
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

def generate_hdu(header, data, fn):
    hdu = fits.PrimaryHDU(data=data, header=header)
    hdu.writeto(fn, overwrite=True)

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

def get_gt_spectra_pix_pos(fn, ra, dec):
    #get_gt_spectra_pix_pos('../../data/pdr3_input/pdr3_dud/calexp-HSC-I-9812-1%2C7.fits',149.373750, 2.776508)
    id = 0 if 'Mega-u' in fn else 1
    header = fits.open(fn)[id].header
    return worldToPix(header, ra, dec)

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
