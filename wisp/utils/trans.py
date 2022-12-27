
import os
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt


from os.path import exists, join



########################
# I) wave & trans utils

"""
def batch_sample_trans(full_wave, trans, avg_distrib, nsmpl, bsz, sort=True, counts=None,
                       encd_ids=None, use_all_wave=False, avg_per_band=False):

    ''' Sample wave and trans for MC mixture estimate
        @Param
          full_wave   [nsmpl_full]
          trans       [nbands,nsmpl_full]
          avg_distrib [nsmpl_full]
          encd_ids    [nbands,nsmpl_full]
        @Return
          smpl_wave   [nsmpl,1]
          smpl_trans  [nbands,nsmpl]
          ids         [nsmpl]
          avg_nsmpl   [nbands]
    '''
    '''
    nbands = trans.shape[0]

    if use_all_wave: # use all lambda
        ids = torch.arange(avg_distrib.shape[-1], device=avg_distrib.device)
    else: # [nsmpl]
        ids = torch.multinomial(avg_distrib, nsmpl, replacement=True)

    if encd_ids is None:
        pass
    elif use_all_wave:
        full_nsmpl = len(full_wave)
        avg_nsmpl = torch.full((nbands), full_nsmpl, device=avg_distrib.device)
    elif avg_per_band:
        # count number of samples falling within response range of each band
        avg_nsmpl = [torch.sum(encd_id[ids]) for encd_id in encd_ids]
        avg_nsmpl = torch.tensor(avg_nsmpl, device=trans.device) # [nbands]
        avg_nsmpl[avg_nsmpl==0] = 1 # avoid dividing by 0
    else:
        avg_nsmpl = nsmpl

    # count sample to plot histogram
    if counts is not None: increment_repeat(counts, ids.flatten())

    # sort sampled waves (required only for spectrum plotting)
    if sort: ids, _ = torch.sort(ids) # [nsmpl]

    smpl_wave = full_wave[ids].unsqueeze(-1) # [nsmpl,1]
    smpl_trans = torch.stack([cur_trans[ids] for cur_trans in trans]) # [nbands,nsmpl]

    if encd_ids is not None:
        return smpl_wave, smpl_trans, ids, avg_nsmpl
    return smpl_wave, smpl_trans, ids
"""


def load_bandwise_trans(bandwise_wave_fn, bandwise_trans_fn, distrib_fn, lo=3000.0, hi=10900.0):
    ''' Load unprocessed wave and trans (list of lists) from unagi package
        used in MC estimate where sample individually for each band
        distribution can be same as unagi_trans or different
    '''
    if exists(bandwise_wave_fn) and exists(bandwise_trans_fn):
        with open(bandwise_wave_fn, 'rb') as fp:
            bandwise_wave = pickle.load(fp)
        with open(bandwise_trans_fn, 'rb') as fp:
            bandwise_trans = pickle.load(fp)
        with open(distrib_fn, 'rb') as fp:
            distrib = pickle.load(fp)

        bandwise_nm_wave = [(np.array(cur_wave)-lo)/(hi-lo) for cur_wave in bandwise_wave]

        return bandwise_wave, bandwise_nm_wave, bandwise_trans, distrib

    fs = ['g','r','i','z','y','nb387','nb816','nb921']
    waves, transs, _, _ = get_wave_trans(fs)
    norm_waves = [(np.array(wave)-lo)/(hi-lo) for wave in waves]

    with open(bandwise_wave_fn,'wb') as fp:
        pickle.dump(waves,fp)
    #with open(bandwise_nm_wave_fn,'wb') as fp:
    #    pickle.dump(norm_waves,fp)
    with open(bandwise_trans_fn,'wb') as fp:
        pickle.dump(trans,fp)
    lo = np.array(lo).astype(np.float64)
    hi = np.array(hi).astype(np.float64)
    norm_waves = [(np.array(cur_wave)-lo)/(hi-lo) for cur_wave in bandwise_wave]
    '''
    distrib = [len(a)*[1] for a in waves]
    with open(distrib_fn,'wb') as fp:
        pickle.dump(distrib, fp)
    '''
    return waves, norm_waves, trans, distrib

def load_trans(wave_fn, trans_fn, avg_distrib_fn=None, float_tensor=None, lo=3000, hi=10900, n_spaces=0):
    ''' Load processed wave, trans, and average
        distribution used in MC mixture estimation
        Assume wave is sorted
    '''
    avg_distrib = None
    avg = avg_distrib_fn is not None
    wave, trans = np.load(wave_fn), np.load(trans_fn)
    norm_wave = (wave - lo) / (hi - lo)

    if avg:
        avg_distrib = np.load(avg_distrib_fn)
        '''
        avg_distrib = np.load(avg_distrib_fn)
        if not avg_distrib[0] == 1:
            print('In utils_train::load_trans, distrib is not ones')
            trans /= (avg_distrib + 1e-6) # compensate for selection probability
        '''
    if float_tensor is not None:
        norm_wave = torch.tensor(norm_wave).type(float_tensor)
        trans = torch.tensor(trans).type(float_tensor)
        if avg:
            avg_distrib = torch.tensor(avg_distrib).type(float_tensor)
    return wave, norm_wave, trans, avg_distrib


###########################
# II) multi-band img recon

def load_hardcode(args):
    _, norm_wave, trans, _ = \
        load_trans(args.hdcd_wave_fn, args.hdcd_trans_fn,
                   float_tensor=args.float_tensor)
    coord_wave = torch.tile(norm_wave, (args.npixls,1))\
                      .unsqueeze(-1) # [npixls,nsmpl,1]
    return norm_wave, coord_wave, trans, None, None

def sample_hardcode(wave, trans, distrib, avg_nsmpl, args):
    return [None,trans,None]

def load_bandwise(args):
    _, norm_wave, trans, distrib = load_bandwise_trans\
        (args.bdws_wave_fn, args.bdws_trans_fn, args.distrib_fn)

    avg_nsmpl = torch.full((args.num_bands,1), args.nsmpl_per_band)
    avg_nsmpl = avg_nsmpl.squeeze().type(args.float_tensor)
    return norm_wave, None, trans, distrib, avg_nsmpl

def sample_bandwise(wave, trans, distrib, avg_nsmpl, args):
    smpl_wave, smpl_trans, _ = batch_sample_trans_bandwise \
        (wave, trans, distrib, args.recon_bsz, args)

    trans_args = [smpl_wave, smpl_trans, None]
    return trans_args

def load_mixture(args):
    _, norm_wave, trans, distrib = load_trans \
        (args.full_wave_fn, args.full_trans_fn,
         args.distrib_fn, args.float_tensor)
    return norm_wave, None, trans, distrib, None

def sample_mixture(wave, trans, distrib, avg_nsmpl, args):
    encd_ids = torch.tensor(np.load(args.encd_ids_fn)).type(args.float_tensor)
    smpl_wave, smpl_trans, _, avg_nsmpl = batch_sample_trans \
        (wave, trans, distrib, args.num_trans_smpl, args.recon_bsz,
         encd_ids=encd_ids, use_all_wave=args.infer_use_all_wave,
         avg_per_band=args.avg_per_band)

    # [recon_bsz,nsmpl,1]/[recon_bsz,nbands,nsmpl]/[recon_bsz,nbands]
    trans_args = [smpl_wave, smpl_trans, avg_nsmpl]
    return trans_args

def load_synthetic(args):
    _, norm_wave, trans, distrib = load_trans \
        (args.full_wave_fn, args.flat_trans_fn,
         args.synthetic_distrib_fn, args.float_tensor)

    coord_wave, avg_nsmpl = None, None
    if args.mc_cho == 'mc_hardcode':
        coord_wave = torch.tile(norm_wave, (args.npixls,1)) \
                          .unsqueeze(-1) # [npixls,nsmpl,1]
    else:
        s_nsmpl = len(norm_wave) if args.infer_use_all_wave \
            else args.num_trans_smpl
        avg_nsmpl = torch.tensor([s_nsmpl]).type(args.float_tensor)
    return norm_wave, coord_wave, trans, distrib, avg_nsmpl

def sample_synthetic(wave, trans, distrib, avg_nsmpl, args):
    if args.mc_cho == 'mc_hardcode':
        assert(False)
    elif args.mc_cho == 'mc_bandwise':
        assert(False)
    elif args.mc_cho == 'mc_mixture':
        if not args.infer_use_all_wave:
            ids = torch.multinomial(distrib, args.num_trans_smpl)
            smpl_wave, smpl_trans = wave[ids], trans[:,ids]
        else:
            smpl_wave, smpl_trans = wave, trans
        smpl_wave = smpl_wave.tile(args.recon_bsz,1).unsqueeze(-1)
        smpl_trans = smpl_trans.tile(args.recon_bsz,1,1)
    else:
        raise Exception('Unsupported mc choice for synthetic band')
    trans_args = [smpl_wave, smpl_trans, avg_nsmpl]
    return trans_args

def load_full(args):
    _, norm_wave, trans, _ = load_trans \
        (args.full_wave_fn, args.full_trans_fn,
         float_tensor=args.float_tensor)

    distrib, coord_wave, avg_nsmpl = None, None, None
    if args.mc_cho == 'mc_hardcode':
        coord_wave = torch.tile(norm_wave, (args.npixls,1)).unsqueeze(-1)
    elif args.avg_per_band:
        #encd_ids = np.load(args.encd_ids_fn)
        #avg_nsmpl = [torch.sum(encd_id, dim=1) for encd_id in encd_ids]
        #avg_nsmpl = torch.stack(avg_nsmpl).T.type(trans.dtype) # [bsz,nbands]

        # num wave samples within each band
        avg_nsmpl = np.load(args.nsmpl_within_bands_fn)
        avg_nsmpl = torch.tensor(avg_nsmpl, device=args.device)
    else:
        # total num of samples across full wave range
        #arr = np.load(args.avg_nsmpl_fn) # [nbands]
        #avg_nsmpl = torch.tensor(arr).type(args.float_tensor)
        avg_nsmpl = len(np.load(args.full_wave_fn))

    return norm_wave, coord_wave, trans, distrib, avg_nsmpl

def sample_full(wave, trans, distrib, avg_nsmpl, args):
    if args.mc_cho == 'mc_hardcode':
        smpl_wave, smpl_trans = None, trans

    elif args.mc_cho == 'mc_bandwise':
        ''' during train, since we sample each band independently,
              we end up with indep spectra for each band.
            however, when recon (img and spectra), since we use all wave
              the spectra is contiguous across bands and we can thus
              remove the 'band' dimension
        '''
        bsz, nbands = args.recon_bsz, args.num_bands
        smpl_wave = wave[None,:,None].tile(bsz,1,1)
        #smpl_wave = wave[None,None,:,None].tile(bsz,nbands,1,1)
        if not args.uniform_smpl: smpl_trans = None
        else: smpl_trans = trans #[None,...].tile(bsz, 1, 1)

    elif args.mc_cho == 'mc_mixture':
        assert(avg_nsmpl is not None)
        smpl_trans = trans
        smpl_wave = wave.tile(args.recon_bsz,1)[...,None] # [bsz,nsmpl,1]
    else:
        raise Exception('Unsupported mc choice for synthetic band')
    trans_args = [smpl_wave, smpl_trans, avg_nsmpl]
    return trans_args


# spectra load sample
def spectra_load_sample_trans(args):
    orig_wave  = np.load(args.full_wave_fn)
    orig_trans = np.load(args.full_trans_fn)
    trans_args, distrib, nposs = [], None, args.spectra_max_batch_sz

    if args.mc_cho == 'mc_hardcode':
        spectrum_wave, norm_wave, trans = load_trans \
            (args.hdcd_wave_fn, args.hdcd_trans_fn,
             float_tensor=args.float_tensor)[:3]
        #norm_wave = norm_wave.tile((nposs,1)).unsqueeze(-1)
        norm_wave = norm_wave.unsqueeze(-1)
        spectrum_nsmpl = nposs
        trans_args = [norm_wave, trans]

    elif args.infer_use_all_wave:
        spectrum_wave, norm_wave, trans = load_trans \
            (args.full_wave_fn, args.full_trans_fn,
             float_tensor=args.float_tensor)[:3]
        norm_wave = norm_wave[None,:,None].tile((nposs,1,1)) # [nposs,nsmpl,1]
        #norm_wave = norm_wave.unsqueeze(-1) # [sp_nsmpl,1]
        trans_args, spectrum_nsmpl = [norm_wave, trans], len(norm_wave)

    elif args.mc_cho == 'mc_bandwise':
        spectrum_wave, norm_wave, trans, distrib = load_bandwise_trans \
            (args.bdws_wave_fn, args.bdws_trans_fn, args.distrib_fn)
        smpl_norm_wave, smpl_trans, spectrum_wave = batch_sample_trans_bandwise \
            (norm_wave, trans, distrib, nposs, args, waves=spectrum_wave)
        trans_args, spectrum_nsmpl = [smpl_norm_wave, smpl_trans], args.num_trans_smpl

    elif args.mc_cho == 'mc_mixture':
        spectrum_wave, norm_wave, trans, distrib = load_trans \
            (args.full_wave_fn, args.full_trans_fn, args.distrib_fn, args.float_tensor)
        smpl_norm_wave, _, spectrum_wave = sample_trans\
            (norm_wave, trans, distrib, args.num_trans_smpl, spectrum_wave)
        smpl_norm_wave = smpl_norm_wave.tile((nposs,1))[...,None] # [nposs,nsmpl,1]
        #smpl_norm_wave = smpl_norm_wave.unsqueeze(-1) # [nsmpl,1]
        trans_args, spectrum_nsmpl = [smpl_norm_wave], len(norm_wave)

    else:
        raise Exception('! Unsupported choice for spectrum plotting')

    covr_rnge, nsmpl_within_each_band = None, None
    if args.mc_cho == 'mc_bandwise':
        covr_rnge = get_bandwise_covr_rnge \
            (args.uniform_smpl, args.bdws_wave_fn,
             args.bdws_trans_fn, args.trans_thresh, args.float_tensor)

        nsmpl_within_each_band = np.load(args.nsmpl_within_bands_fn)
        nsmpl_within_each_band = torch.tensor(nsmpl_within_each_band, device=args.device)

    return norm_wave, orig_wave, orig_trans, spectrum_wave, trans_args, covr_rnge, nsmpl_within_each_band
