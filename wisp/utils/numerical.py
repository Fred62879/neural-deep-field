
import torch
import numpy as np
import logging as log

from collections import defaultdict
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
from astropy.visualization import ZScaleInterval
from skimage.metrics import structural_similarity
from wisp.utils.common import to_numpy, init_redshift_bins, get_bin_id


def reduce_latents_dim_pca(all_latents, n, selected_axes=None):
    """
    Cast high-dim latetns to low dim and plot.
    @Params
       all_latents: [n_models,bsz,dim]
    """
    assert n <= 3
    if selected_axes is None:
        selected_axes = get_major_pca_axes(all_latents[-1], n)
    latents_dim_reduced = all_latents[...,selected_axes]
    return selected_axes, latents_dim_reduced

def get_major_pca_axes(latents, n):
    """
    Get the axis that contributes the most to each principal component.
    """
    selected_axes = set()
    pca = PCA(n_components=n)
    data_pca = pca.fit_transform(latents)
    for i in range(n):
        order = np.argsort(abs(pca.components_[i]))[::-1]
        j = 0
        while order[j] in selected_axes: j += 1
        selected_axes.add(order[j])
    selected_axes = np.array(list(selected_axes))
    return selected_axes

def calculate_emd(distrib1, distrib2, norm="l2", mask=None, weight=None, precision=None):
    """ Calculate (masked) earth mover's distance between two distributions.
        @Param
           distrib: [...,num_bins] (assume they sum to 1)
    """
    # assert(distrib1.shape == distrib2.shape)
    sub = distrib1 - distrib2
    n = sub.shape[-1]

    if mask is not None: sub *= mask
    if weight is not None: sub *= weight
    # if precision is not None: sub *= precision

    if norm == "l1":
        emd = torch.linalg.norm(sub, ord=1, dim=-1)
    elif norm == "l2":
        emd = torch.linalg.norm(sub, dim=-1)
    else:
        raise ValueError("Unsupported norm choice for emd calculation")
    return emd # / n

def find_closest_tensor(tensor1, tensor2):
    ''' Calculate L2-normalized distance between the each tensor from tensor1 and
          tensor2. Then find id of closest tensor in tensor2 for each tensor in tensor1
        Input:  tensor1 [n,d]
                tensor2 [m,d]
        Output: id_min_dist [n]
    '''
    tensor2 = tensor2.T
    similarity = torch.matmul(tensor1, tensor2)
    distances = (
        torch.sum(tensor1 ** 2, dim=-1, keepdims=True) +
        torch.sum(tensor2 ** 2, dim=0) - 2 * similarity) # [n,m]

    # Derive the indices for minimum distances
    ids = torch.argmin(distances, dim=-1)
    return ids

def get_coords_norm_range(coords, **kwargs):
    """ Get normalization range for coordinates.
        Note: when we are using grid for coord encoding, we make sure
              i) input coords are image coords (r/c)
              ii) the norm range is the same on both r and c dimensions
                  (normed coords range from -1 to 1 on the longer edge,
                   and range within a smaller range on the shorter edge)
             s.t. each pixel after normalization is still a square
    """
    coords = to_numpy(coords)

    if kwargs["coords_encode_method"] == "grid":
        assert kwargs["coords_type"] == "img"

    if kwargs["coords_type"] == "img":
        start_row, start_col = min(coords[...,0]), min(coords[...,1])
        num_rows = max(coords[...,0]) - start_row
        num_cols = max(coords[...,1]) - start_col
        edge_len = max(num_rows, num_cols) # norm range is the same
        norm_range = np.array([start_row, start_row + edge_len,
                               start_col, start_col + edge_len])
    elif kwargs["coords_type"] == "world":
        raise NotImplementedError("Temporarily not support world coords")
        min_ra, max_ra = np.min(coords[...,0]), np.max(coords[...,0])
        min_dec, max_dec = np.min(coords[...,1]), np.max(coords[...,1])
        norm_range = np.array([min_ra, max_ra, min_dec, max_dec])
    else:
        raise ValueError()
    return norm_range

def normalize_coords(coords, norm_range=None, **kwargs):
    """ Normalize given coords.
        Currently only supports linear normalization.
        @Param
          coords: [...,2] (float32)
    """
    if norm_range is None:
        norm_range = get_coords_norm_range(coords, **kwargs)
    (lo0, hi0, lo1, hi1) = norm_range
    coords[...,0] = 2 * (coords[...,0] - lo0) / (hi0 - lo0) - 1
    coords[...,1] = 2 * (coords[...,1] - lo1) / (hi1 - lo1) - 1
    return coords, norm_range

def normalize_data(data, norm_cho):
    if norm_cho == "identity":
        data = data
    elif norm_cho == "max":
        data = data / np.max(data)
    elif norm_cho == "sum":
        data = data / np.sum(data)
    elif norm_cho == "linr":
        lo, hi = min(data), max(data)
        data = (data - lo) / (hi - lo)
    else: raise ValueError()
    return data

# def calculate_redshift_estimation_stats_based_on_logits(
#         logits, gt_redshifts, lo, hi, bin_width, num_threshes
# ):
#     """
#     Calculate precision and recall based on logits of each bin.
#     @param
#         logits: [num_spectra,num_bins]
#         gt_redshifts: gt redshift value for each spectra [num_spectra,]
#         lo/hi: min and max redshift values.
#         bin_width: width of each redshift bin in terms of redshift values.
#     """
#     n_spectra = len(logits)
#     bins = init_redshift_bins(lo, hi, bin_width)
#     gt_ids = [get_bin_id(lo, bin_width, gt_redshift) for gt_redshift in gt_redshifts]
#     ids = np.arange(len(bins))

#     # get thresholds for precision recall calculation
#     if num_precision_recall_threshes <= 0:
#         threshes = np.array(list(set(logits.flatten())))
#         threshes = np.sort(threshes)
#     else:
#         mn, mx = np.min(logits), np.max(logits)
#         step = (mx - mn) / num_precision_recall_threshes
#         threshes = np.arange(mn, mx, step)

#     # def calculate_individually(thresh):
#     #     p = logits > thresh
#     #     # n = logits <= thresh
#     #     tp = ids[p] == gt_ids
#     #     # fp = ids[p] != gt_ids
#     #     # fn = ids[n] == gt_ids
#     #     # precision = sum(tp) / (sum(tp) + sum(fp))
#     #     # recall = sum(tp) / (sum(tp) + sum(fn))
#     #     recall.append(sum(tp))
#     #     precision.append(sum(tp) / sum(p))

#     def calculate(thresh):
#         ps = logits > thresh
#         ns = logits <= thresh
#         # if np.sum(ps) == 0: return
#         n_tps_each = [sum(ids[p] == gt_id) for p, gt_id in zip(ps, gt_ids)]
#         n_fns_each = [sum(ids[n] == gt_id) for n, gt_id in zip(ns, gt_ids)]
#         n_tps = sum(n_tps_each)
#         n_fns = sum(n_fns_each)
#         # recall.append(n_tps / n_spectra)
#         recall.append(n_tps / (n_tps + n_fns) )
#         precision.append(n_tps / np.sum(ps))
#         # recall.append(n_fns)
#         # precision.append(n_tps)

#     precision, recall = [], []
#     [ calculate(thresh) for thresh in threshes ]
#     return threshes, np.array(precision), np.array(recall)

def calculate_redshift_estimation_stats_based_on_residuals(
        residuals, num_residual_levels, cho="accuracy", residual_levels=None
):
    """ Calculate precision and recall based on result for all spectra.
        We didn't use threshold from 0-1 as in case of a multi-class classification.
        Instead, we calculate precision and recall at different values of redshift residual.
        Thus, when residual is 0, only absolutely correct samples are counted as positive,
          when residual is largest, all samples are counted as positive.
        @param
          logits: [num_spectra,num_bins]
    """
    n_spectra = len(residuals)
    residuals = np.abs(residuals)
    residuals = np.around(residuals, decimals=2)

    lo, hi = np.min(residuals), np.max(residuals)
    if lo == hi:
        assert lo == 0
        return None, None

    if residual_levels is not None:
        pass
    elif num_residual_levels <= 0:
        residual_levels = np.array(list(set(residuals.flatten())))
        residual_levels = np.sort(residual_levels)
    else:
        step = (hi - lo) / num_residual_levels
        residual_levels = np.arange(lo, hi + 1e-6, step)

    def calculate_accuracy(residual_level):
        ps = sum(residuals <= residual_level)
        stats.append(ps / n_spectra)

    def calculate_precision_recall(residual_level):
        tps = residuals <= residual_level
        fns = residuals > residual_level
        n_tps = sum(tps)
        n_fns = sum(fns)
        stats[0].append(n_tps / (n_tps + n_fns) ) # recall
        stats[1].append(n_tps / n_spectra) # precision

    if cho == "accuracy":
        stats = []
        func = calculate_accuracy
    elif cho == "precision_recall":
        stats = [[],[]]
        func = calculate_precision_recall
    [ func(residual_level) for residual_level in residual_levels ]
    return residual_levels, np.array(stats)

def calculate_zscale_ranges(pixels):
    """ Calculate zscale ranges based on given pixels for each bands separately.
        @Param
          pixels: [npixels,nbands]
        @Return
          zscale: [2,nbands] (vmin, vmax)
    """
    num_bands = pixels.shape[-1]
    zmins, zmaxs = [], []
    for i in range(num_bands):
        zmin, zmax = ZScaleInterval(contrast=.25).get_limits(pixels[...,i])
        zmins.append(zmin);zmaxs.append(zmax)
    return np.array([zmins, zmaxs])

def calculate_zscale_ranges_multiple_patches(pixels):
    """ Calculate zscale ranges based on given pixels for each bands separately.
        @Param
          pixels: [n,npixels,nbands]
        @Return
          zscale: [n,2,nbands] (vmin, vmax)
    """
    return np.array([calculate_zscale_ranges(pixels[i])
                     for i in range(len(pixels))])

def normalize(data, normcho, verbose=False, gt=None):
    ''' Normalize input img data
        @Param
           data: list of [nbands,sz,sz]
    '''
    # data [nr,nc,nbands] # all bands of a full tile
    if normcho == 'identity':
        if verbose: log.info('No pixel normlization')
    elif normcho == 'arcsinh':
        data = np.arcsinh(data)
        if verbose: log.info('Pixel normlization: arcsinh')
    elif normcho == 'linear':
        lo, hi = np.min(data), np.max(data)
        data = (data - lo) / (hi - lo)
    elif normcho == 'clip':
        data = np.clip(data, 0, 1)
    elif normcho == 'zscale':
        assert(gt is not None)
        data = zscale_img(data, gt)
    else:
        raise Exception('Unsupported pixel normalizaton choice')
    return data

def zscale_img(data, gt):
    ''' Normalize to [0,1] based on zscale value of gt.
        Doesn't change value of input recon. Return new img instead.
        Steps: i) get z1 and z2 of gt
              ii) Normalize recon to have min as z1 and max as z2
             iii) Clip recon to [0,1]
        @Param
          recon: image to normalize     [..]/[...,nabnds]
          gt: image to derive z1 and z2 [..]/[...,nbands]
    '''
    def zscale(a, b):
        z1, z2 = ZScaleInterval(contrast=.25).get_limits(b)
        a = (a - z1) / (z2 - z1)
        return np.clip(a, 0, 1)

    if data.ndim == 2:
        return zscale(data, gt)

    nbands = data.shape[-1]
    for i in range(nbands):
        data[...,i] = zscale(data[...,i], gt[...,i])
    return data

def calculate_zncc(s1, s2):
    """ Calculate zero-mean normalized cross correlation between two signals.
        @Ref
          https://stackoverflow.com/questions/13439718/how-to-interpret-the-values-returned-by-numpy-correlate-and-numpy-corrcoef
    """
    n = len(s1)
    m1, m2 = np.mean(s1), np.mean(s2)
    std1, std2 = np.std(s1), np.std(s2)
    # zncc = np.sum( (s1-m1)*(s2-m2) ) / (n*(std1*std2)) # i)
    zncc = np.correlate(s1-m1, s2-m2, mode='valid')[0] / (n*(std1*std2) + 1e-10) # ii)
    # zncc = np.corrcoef(s1, s2)[0,1] # iii)
    return zncc

def calculate_zncc_composite(s1, s2, window_width=1):
    """ Calcualte global and local sliding zncc between s1 and s2.
        @Param
          s1, s2: 1D signals of the same length
    """
    assert s1.shape == s2.shape and s1.ndim == 1 and window_width >= 1
    n = len(s1)
    zncc = calculate_zncc(s1, s2)

    zncc_sliding = []
    los = np.arange(0, n - window_width)
    #print(len(s1), window_width)
    #print(los)
    for lo in los:
        hi = min(lo + window_width, n)
        cur_zncc = calculate_zncc(s1[lo:hi], s2[lo:hi])
        zncc_sliding.append(cur_zncc)
    return (zncc, zncc_sliding)

def calculate_sam_spectrum(gen, gt, convert_to_degree=False):
    numerator = np.sum(np.multiply(gt, gen))
    denominator = np.linalg.norm(gt) * np.linalg.norm(gen)
    val = np.clip(numerator / denominator, -1, 1)
    sam_angles = np.arccos(val)
    if convert_to_degree:
        sam_angles = sam_angles * 180.0 / np.pi
    return sam_angles

def calculate_mse(gen, gt):
    return np.mean((gen - gt)**2)

def calculate_psnr(gen, gt):
    mse = calculate_mse(gen, gt)
    return 20 * np.log10(np.max(gt) / np.sqrt(mse) + 1e-10)

def calculate_ssim(gen, gt):
    rg = np.max(gt)-np.min(gt)
    return structural_similarity\
        (gt, gen, data_range=rg) #, win_size=gt.shape[1]-1)

def calculate_metric(recon, gt, band, option, **kwargs):
    if option == "zncc":
        metric = calculate_zncc_composite(recon, gt, **kwargs)
    elif option == 'mse':
        metric = calculate_mse(recon[band], gt[band])
    elif option == 'psnr':
        metric = calculate_psnr(recon[band], gt[band])
    elif option == 'ssim':

        metric = calculate_ssim(recon[band], gt[band])
    elif option == 'sam':
        metric = calculate_sam(recon[:,:,band:band+1], gt[:,:,band:band+1])
    elif option == 'abs':
        metric = np.abs(recon[band] - gt[band]).mean()
    elif option == 'min':
        metric = np.min(recon[band])
    elif option == 'max':
        metric = np.max(recon[band])
    elif option == 'mean':
        metric = np.mean(recon[band])
    elif option == 'median':
        metric = np.median(recon[band])
    return metric

def calculate_metrics(recon, gt, options, zscale=False, **kwargs):
    """ Calculate metrics and stats of recon w.r.t gt
        @Return
           metrics: [n_metrics(,n_bands)]
    """
    if len(options) == 0: return {}

    assert recon.shape == gt.shape
    if recon.ndim == 3: # [nbands,sz,sz]
        metrics = np.zeros((len(options), recon.shape[0]))
        num_bands = len(recon)
        if zscale: recon = zscale_img(recon, gt)
    else:
        metrics = {}
        assert recon.ndim == 1

    for i, option in enumerate(options):
        if recon.ndim == 3:
            for band in range(num_bands):
                cur_metrics = calculate_metric(recon, gt, band, option, **kwargs)
                metrics[i, band] = cur_metrics
        else:
            metrics[option] = calculate_metric(recon, gt, None, option, **kwargs)
    return metrics

'''
# calculate normalized cross correlation between given 2 imgs
def calculate_ncc(img1, img2):
    a, b = img1.flatten(), img2.flatten()
    return 1/len(a) * np.sum\
        ( (a-np.mean(a)) * (b-np.mean(b)) /
          np.sqrt(np.var(a)*np.var(b)) )

# image shape should be [sz,sz,nchls]
def calculate_sam(org_img, pred_img, convert_to_degree=False):
    numerator = np.sum(np.multiply(pred_img, org_img), axis=2)
    denominator = np.linalg.norm(org_img, axis=2) * np.linalg.norm(pred_img, axis=2)
    val = np.clip(numerator / denominator, -1, 1)
    sam_angles = np.arccos(val)
    if convert_to_degree:
        sam_angles = sam_angles * 180.0 / np.pi
    return np.mean(np.nan_to_num(sam_angles))


def normalize_all(pixls_all, args):
    for i, pixls_one in enumerate(pixls_all):
        pixls_all[i] = normalize(pixls_one, args)

def smooth(wave, trans, intvl):
    n = len(wave)
    m = n//intvl
    res_wave, res_trans = [], []
    for i in range(m):
        lo, hi = i*intvl, min((i+1)*intvl, n)
        res_wave.append(wave[(lo+hi)/2])
        res_trans.append(np.average(trans[lo:hi]))
    return np.array(res_wave), np.array(res_trans)

def double_sinh(data):
    return torch.sinh(torch.sinh(data))

def sigmoid_denorm(data):
    data[data <= -1] = 0
    data[data >= 1] = 0
    return torch.log(data+1) - torch.log(1-data)

def sigmoid_norm(data):
    return 2*torch.sigmoid(data)-1

def unnormalize(data, normcho):
    if normcho == 'identity':
        return data
    if normcho == 'arcsinh':
        return np.arcsinh(data)
    assert(False)

def scnd_moment(data):
    avg = np.mean(data**2, axis=(0,1))
    return np.sqrt(avg)

# data [npixls,nbands], args [nbands,2] - (lo,hi)
def normalize(data, normcho, args=None, verbose=False):
    if normcho == 0:
        print('--- no pixel normlization')
        return data/2

    if normcho == 1:
        print('--- pixel normlization: log')
        return np.log(data-np.min(data)+1e-2)

    if normcho == 2:
        print('--- pixel normlization: arcsinh')
        return np.arcsinh(data)
    if normcho == 3: # sigmoid
        print('--- pixel normlization: sigmoid')
        return 2/(1 + np.exp(-data)) - 1
    if normcho == 4:
        print('--- no pixel normlization, sinh output')
        return data

    if normcho == 5:
        print('--- pixel normlization: local min and max')
        lo, hi = np.min(data, axis=0), np.max(data, axis=0)
        res = (data - lo) / (hi - lo)
        if verbose:
            print('    post norm min and max', np.min(res, axis=0), np.max(res, axis=0))
        return res

    if normcho == 6:
        assert(args is not None)
        print('--- pixel normlization: global min and max')
        if verbose: print('norm args is ', args)

        res = (data-args[:,0])/(args[:,1] - args[:,0])
        if verbose:
            print('    post norm min and max ', np.min(res, axis=0), np.max(res, axis=0))
        return res

    if normcho == 7:
        print('--- pixel normlization: local same mean and std')
        mean, std = np.mean(data, axis=0), np.std(data, axis=0)
        res = (data - mean) / std
        if verbose:
            print('    post norm mean and std', np.mean(res, axis=0), np.std(res, axis=0))
        return res

    if normcho == 8:
        print('--- pixel normlization: global bandwise divide std')
        assert(args is not None)
        if verbose: print('norm args is ', args)

        if verbose:
            print('    pre norm mean and std', np.mean(data, axis=0), np.std(data, axis=0))
            print('    pre norm min and max ', np.min(data, axis=0), np.max(data, axis=0))
        res = data/args[:,1] #(data-args[:,0])/(args[:,1])
        if verbose:
            print('    post norm mean and std', np.mean(res, axis=0), np.std(res, axis=0))
            print('    post norm min and max ', np.min(res, axis=0), np.max(res, axis=0))
        return res

    if normcho == 9:
        print('--- pixel normlization: global across-band divide std')
        assert(args is not None)
        if verbose: print('norm args is ', args)

        if verbose:
            print('    pre norm mean and std', np.mean(data, axis=0), np.std(data, axis=0))
            print('    pre norm min and max ', np.min(data, axis=0), np.max(data, axis=0))
        #res = (data-args[:,0])/(args[:,1])
        res = data/args[:,1]
        if verbose:
            print('    post norm mean and std', np.mean(res, axis=0), np.std(res, axis=0))
            print('    post norm min and max ', np.min(res, axis=0), np.max(res, axis=0))
        return res

    if normcho == 10:
        print('--- pixel normlization: global u-band divide std')
        assert(args is not None)
        if verbose: print('norm args is ', args)

        if verbose:
            print('    pre norm mean and std', np.mean(data, axis=0), np.std(data, axis=0))
            print('    pre norm min and max ', np.min(data, axis=0), np.max(data, axis=0))
        #res = (data-args[:,0])/(args[:,1])
        res = data/args[:,1]
        if verbose:
            print('    post norm mean and std', np.mean(res, axis=0), np.std(res, axis=0))
            print('    post norm min and max ', np.min(res, axis=0), np.max(res, axis=0))
        return res
    if normcho == 'second_moment':
        assert(False)
        # pixls_one [orig_nr, orig_nc, nbands]
        img_sz = args.img_sz
        nbands = args.num_bands
        start_r, start_c = args.start_r, args.start_c
        assert(len(m) == data.shape[-1] == nbands)

        if os.path.exists(args.norm_args_fn):
            m = np.load(args.norm_args_fn)
        else:
            m = scnd_moment(data)
            np.save(args.norm_args_fn, m)

        pixls_one = pixls_one[start_r:start_r+img_sz,
                              start_c:start_c+img_sz]/m
        pixls_one = pixls_one.reshape((-1,nbands))


    if normcho == 11:
        print('--- pixel normlization: clip')
        return np.clip(data, -1, 1)
    else:
        raise Exception('Unsupported normalizaton choice')

# data [npixls,]
def get_norm_args(data, normcho):
    if normcho == 6:
        return [np.min(data), np.max(data)]
    if normcho == 8: # bandwise
        return [np.mean(data), np.std(data)]
    raise Exception('no norm args needed')

'''
