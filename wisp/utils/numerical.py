
import torch
import numpy as np
import logging as log

from astropy.visualization import ZScaleInterval
from skimage.metrics import structural_similarity


def calculate_emd(distrib1, distrib2, norm="l2"):
    """ Calculate earth mover's distance between two distributions.
        @Param
           distrib: [...,num_bins] (assume they sum to 1)
    """
    # assert(distrib1.shape == distrib2.shape)
    n = distrib1.shape[-1]
    sub = distrib1 - distrib2
    if norm == "l1":
        emd = torch.linalg.norm(sub, ord=1, dim=-1)
    elif norm == "l2":
        emd = torch.linalg.norm(sub, dim=-1)
    else:
        raise ValueError("Unsupported norm choice for emd calculation")
    return emd

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

def get_coords_range(coords):
    ''' Find min and max of given 2D coords. '''
    min_ra, max_ra = np.min(coords[...,0]), np.max(coords[...,0])
    min_dec, max_dec = np.min(coords[...,1]), np.max(coords[...,1])
    return np.array([min_ra, max_ra, min_dec, max_dec])

def normalize_coords(coords):
    ''' Normalize given coords.
        Currently only supports linear normalization.
        @Param
          coords: [...,2] (float32)
    '''
    coords_range = get_coords_range(coords)
    (min_x, max_x, min_y, max_y) = coords_range
    coords[...,0] = (coords[...,0] - min_x) / (max_x - min_x)
    coords[...,1] = (coords[...,1] - min_y) / (max_y - min_y)
    return np.float32(coords), np.float32(coords_range)

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

def calculate_metric(recon, gt, band, option):
    if option == 'mse':
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

def calculate_metrics(recon, gt, options, zscale=False):
    ''' Calculate metrics and stats of recon w.r.t gt '''
    num_bands = len(recon)
    metrics = np.zeros((len(options), num_bands))

    if zscale:
        recon = zscale_img(recon, gt)

    for i, option in enumerate(options):
        for band in range(num_bands):
            metrics[i, band] = calculate_metric(recon, gt, band, option)
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
