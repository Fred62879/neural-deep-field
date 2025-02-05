
import math
import torch
import torch.nn as nn
import warnings
import numpy as np

import torch.nn.functional as F

from wisp.utils.numerical import calculate_emd
from skimage.metrics import structural_similarity as ssim


def get_reduce(cho):
    if cho == "sum":
        reduce_func = torch.sum
    elif cho == "mean":
        reduce_func = torch.mean
    elif cho == "none":
        # reduce_func = torch.nn.identity
        assert 0
    else: raise ValueError()
    return reduce_func

def get_loss(cho, reduction, cuda, filter_size=-1, filter_sigma=-1):
    if cho == "l1":
        loss = nn.L1Loss(reduction=reduction)
    elif cho == "l2":
        loss = nn.MSELoss(reduction=reduction)
    elif cho == "l4":
        loss = lpnormloss(p=4, reduction=reduction)
    elif cho == "ssim1d":
        loss = ssim1d(filter_size, filter_sigma, reduction=reduction)
    else:
        raise Exception("Unsupported loss choice")
    if cuda: loss = loss.cuda()
    return loss

class lpnormloss(nn.Module):
    def __init__(self, p, dim=-1, reduction="none"):
        super(lpnormloss, self).__init__()
        self.p = p
        self.dim = dim
        self.reduction = reduction

    def forward(self, input, target):
        if self.reduction == "none":
            loss = torch.pow(input - target, self.p)
        else:
            loss = torch.linalg.norm(
                input - target, ord=self.p, dim=self.dim)
            if self.reduction == "sum":
                loss = loss
            elif self.reduction == "mean":
                loss = loss / input.shape[self.dim]
            else: raise ValueError("invalid reduction method")
        return loss

class ssim1d(nn.Module):
    def __init__(self, filter_size, filter_sigma, val_range=1,
                 reduction="none", size_average=True, full=False
    ):
        super(ssim1d, self).__init__()
        assert reduction == "none" and filter_size > 0 and filter_sigma > 0

        self.full = full
        assert filter_size % 2
        self.pad = (filter_size - 1) // 2
        self.val_range = val_range
        self.filter_size = filter_size
        self.size_average = size_average
        self.filter = self.gaussian(filter_size, filter_sigma)

    def forward(self, input, target, mask):
        ssim = self._forward(input, target, mask)
        return ssim

    def gaussian(self, size, sigma):
        """
        Generates a list of Tensor values drawn from a gaussian distribution with standard
        diviation = sigma and sum of all elements = 1.
        Length of list = filter_size
        """
        gauss =  torch.Tensor([
            math.exp(-(x - size//2)**2/float(2*sigma**2)) for x in range(size)
        ])
        gauss = gauss/gauss.sum()
        gauss = gauss[None,None,:].cuda()
        return gauss

    def _forward(self, s1, s2, mask):
        """
        Calculate the masked ssim loss between the given two spectra.
        @Param
          s1/2: spectra flux [(nbins,)bsz,nsmpl]
          mask: 0 indicates `nan`
                 [(nbins,)bsz,nsmpl] duplicate in dim0 if multi-bin
        @Note
          When processing spectra, we append `nan` to the end of each spectra to make
           them the same length. During inferrence, we use all samples and thus will
           involve these `nan` samples (during training, we sample only within supervision
           range and thus no `nan` will be sampled).
          When calculating ssim, valid flux values may turn to `nan` ssim if there are
           `nan` values within the kernel range.
          We can simply replace these `nan` with 0 which is essentially zero padding.
        """
        # assert valid region (indicated by mask) has no nan values
        assert not torch.isnan(s1[mask]).any() and not torch.isnan(s2[mask]).any()

        invalid, val1, val2 = self._replace_nan(s1, s2)

        dim3 = False
        if s1.ndim == 2:
            bsz,nsmpl = s1.shape
            s1, s2 = s1[:,None], s2[:,None] # [bsz,1,nsmpl]
        elif s1.ndim == 3:
            dim3 = True
            nbins,bsz,nsmpl = s1.shape
            s1 = s1.view(-1,1,nsmpl) # [nbins*bsz,1,nsmpl]
            s2 = s2.view(-1,1,nsmpl)
            # mask = mask.view(-1,1,nsmpl)
        else: raise ValueError()

        # calculating the mu parameter (locally) for both signals using a gaussian filter
        # calculates the luminosity params
        channels = s1.shape[1] # 1
        mu1 = F.conv1d(s1, self.filter, padding=self.pad, groups=channels)
        mu2 = F.conv1d(s2, self.filter, padding=self.pad, groups=channels)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu12 = mu1 * mu2

        # now we calculate the sigma square parameter
        # sigma deals with the contrast component
        sigma1_sq = F.conv1d(s1 * s1, self.filter, padding=self.pad, groups=channels) - mu1_sq
        sigma2_sq = F.conv1d(s2 * s2, self.filter, padding=self.pad, groups=channels) - mu2_sq
        sigma12 =  F.conv1d(s1 * s2, self.filter, padding=self.pad, groups=channels) - mu12

        # Some constants for stability
        C1 = (0.01 * self.val_range) ** 2
        C2 = (0.03 * self.val_range) ** 2
        # contrast_metric = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
        # contrast_metric = torch.mean(contrast_metric)
        numerator1 = 2 * mu12 + C1
        numerator2 = 2 * sigma12 + C2
        denominator1 = mu1_sq + mu2_sq + C1
        denominator2 = sigma1_sq + sigma2_sq + C2

        ssim_score = (numerator1 * numerator2) / (denominator1 * denominator2)
        ssim_score = ssim_score[:,0] # [(nbins*)bsz,nsmpl]
        ssim_score = (ssim_score + 1) / 2
        if torch.min(ssim_score) < 0 or torch.max(ssim_score) > 1.001:
            warnings.warn(f"min/max: {torch.min(ssim_score)}/{torch.max(ssim_score)}")
        ssim_score = 1 - ssim_score # we minimize 1 - ssim (maximize ssim)

        if not dim3:
            s1, s2 = s1[:,0], s2[:,0] # [bsz,nsmpl]
        else:
            s1 = s1.view(nbins,bsz,nsmpl)
            s2 = s2.view(nbins,bsz,nsmpl)
            ssim_score = ssim_score.view(nbins,bsz,nsmpl)

        self._restore_nan(invalid, s1, s2, val1, val2, ssim_score, mask)
        return ssim_score

    def _replace_nan(self, s1, s2):
        invalid = torch.isnan(s1)
        if torch.sum(invalid) == 0:
            return invalid, None, None
        val1 = s1[invalid]
        val2 = s2[invalid]
        s1[invalid] = 0
        s2[invalid] = 0
        return invalid, val1, val2

    def _restore_nan(self, invalid, s1, s2, val1, val2, ssim_score, mask):
        """
        Change ssim in invalid region to `nan`.
        Restore any replaced `nan` values in the two spectra.
        """
        ssim_score[mask == 0] = math.nan
        if torch.sum(invalid) > 0:
            ssim_score[invalid] = math.nan
            s1[invalid] = val1
            s2[invalid] = val2

class spectra_supervision_loss(nn.Module):
    def __init__(self, loss_func, weight_by_wave_coverage):
        """
        Loss function for spectra supervision
        @Param
          loss: l1/l2 as specified in config
        """
        super(spectra_supervision_loss, self).__init__()
        self.loss_func = loss_func
        self.weight_by_wave_coverage = weight_by_wave_coverage
        self.is_ssim = loss_func.__class__.__name__ == "ssim1d"

    def forward(self, gt_spectra, recon_fluxes, mask):
        """
        Calculate lambda-wise spectra loss
        @Param
          gt_spectra:   [(nbins,)bsz,4+2*nbanbds,num_smpls]
                         (wave/flux/ivar/weight/trans_mask/trans(nbands)/band_mask(nbands))
          recon_fluxes: [(nbins,)bsz,num_smpls]
        @Return
          lambda-wise spectra loss [(nbins,)bsz,num_smpls]
        """
        if self.weight_by_wave_coverage:
            if gt_spectra.ndim == 3:
                weight = gt_spectra[:,3]
                ret = self._forward(gt_spectra[:,1]*weight, recon_fluxes*weight, mask)
            elif gt_spectra.ndim == 4:
                weight = gt_spectra[:,:,3]
                ret = self._forward(gt_spectra[:,:,1]*weight, recon_fluxes*weight, mask)
            else: raise ValueError()
        else:
            if gt_spectra.ndim == 3:
                # print(torch.isnan(gt_spectra[:,1]).any(), torch.isnan(recon_fluxes).any())
                ret = self._forward(gt_spectra[:,1], recon_fluxes, mask)
                # print(torch.isnan(gt_spectra[:,1]).any(), torch.isnan(recon_fluxes).any())
            elif gt_spectra.ndim == 4:
                #print(torch.isnan(gt_spectra[:,:,1]).any(), torch.isnan(recon_fluxes).any())
                ret = self._forward(gt_spectra[:,:,1], recon_fluxes, mask)
                #print(torch.isnan(gt_spectra[:,:,1]).any(), torch.isnan(recon_fluxes).any())
            else: raise ValueError()

        assert recon_fluxes.shape == ret.shape
        return ret

    def _forward(self, s1, s2, mask):
        if self.is_ssim:
            ret = self.loss_func(s1, s2, mask)
        else: ret = self.loss_func(s1, s2)
        return ret

def pretrain_pixel_loss(loss, gt_pixels, recon_pixels):
    gt_pixels = gt_pixels / (torch.sum(gt_pixels, dim=-1)[...,None])
    recon_pixels = recon_pixels / (torch.sum(recon_pixels, dim=-1)[...,None])
    emd = calculate_emd(gt_pixels, recon_pixels)
    emd = torch.mean(torch.abs(emd))
    return emd

def redshift_supervision_loss(loss, gt_redshift, recon_redshift, mask=None):
    ''' Loss function for few-shot redshift supervision
        @Param
          loss: l1/l2 as specified in config
          wave_rnge: Trusted lambda range of redshift
          redshift_ids: ids of redshift to supervise
          redshift: [bsz, num_smpls]
    '''
    if mask is None:
        return loss(gt_redshift, recon_redshift)
    return loss(gt_redshift, recon_redshift[mask])

def spectral_masking_loss(
        loss, relative_train_bands, relative_inpaint_bands,
        gt_pixels, recon_pixels, mask
):
    """
    Loss function for spectral inpainting
    @Param
      loss: l1/l2 as specified in config
      relative_train/inpaint_bands:
          Train/inpaint bands may not form a continuous seq of ints.
          Relative bands are which the two are renumbered to a
          continuous seq starting from 0 while maintaing the relative order.
          e.g. before: [1,4]/[2,9]
               after:  [0,2]/[1,3]
      pixels: [bsz,num_bands]
      mask: sliced with only inpaint dim left
    """
    masked_gt = gt_pixels[:,relative_train_bands].flatten()
    masked_recon = recon_pixels[:,relative_train_bands].flatten()

    gt = torch.masked_select(gt_pixels[:,relative_inpaint_bands], mask)
    recon = torch.masked_select(recon_pixels[:,relative_inpaint_bands], mask)

    masked_gt = torch.cat((masked_gt, gt))
    masked_recon = torch.cat((masked_recon, recon))
    error = loss(masked_gt, masked_recon)
    return error

# def spectra_supervision_loss(loss, mask, gt_spectra, recon_fluxes, redshift_logits, weight_by_wave_coverage=True):
#     """ Loss function for spectra supervision
#         @Param
#           loss: l1/l2 as specified in config
#           mask:       [bsz,num_smpls]
#           gt_spectra: [bsz,4+2*nbanbds,num_smpls]
#                       (wave/flux/ivar/weight/trans_mask/trans(nbands)/band_mask(nbands))
#           recon_fluxes: [num_bins,bsz,num_smpls]
#           redshift_logits: [bsz,num_bins]
#     """
#     num_bins = len(recon_fluxes)
#     gt_fluxes = gt_spectra[:,1]*mask[None,...].tile(num_bins,1,1)
#     spectra_loss_bin_wise = loss(gt_fluxes, recon_fluxes*mask)
#     spectra_loss_bin_wise = torch.mean(spectra_loss_bin_wise, dim=-1)
#     logits = redshift_logits * spectra_loss_bin_wise.T
#     return spectra_supervision_loss(
#         loss, mask, gt_spectra, recon_fluxes, weight_by_wave_coverage=True)

# def spectra_supervision_emd_loss(mask, gt_spectra, recon_flux, weight_by_wave_coverage=True):
#     """ Loss function for spectra supervision
#         @Param
#           mask:       [bsz,num_smpls]
#           gt_spectra: [bsz,4+2*nbanbds,num_smpls]
#                       (wave/flux/ivar/weight/trans_mask/trans(nbands)/band_mask(nbands))
#           recon_flux: [bsz,num_smpls]
#     """
#     # norm spectra each so they sum to 1 (earth movers distance)
#     # DON'T use /= or spectra will be modified in place and
#     #   if we save spectra later on the spectra will be inaccurate

#     # conver to pmf
#     gt_flux = gt_spectra[:,1] / (torch.sum(gt_spectra[:,1]*mask, dim=-1)[...,None] + 1e-10)
#     recon_flux = recon_flux / (torch.sum(recon_flux*mask, dim=-1)[...,None] + 1e-10)

#     # sanity check, sum of unmasked flux should be same as bsz
#     # gt_flux = torch.masked_select(gt_flux, mask.bool())
#     # recon_flux = torch.masked_select(recon_flux, mask.bool())
#     # print(torch.sum(gt_flux), torch.sum(recon_flux))

#     if weight_by_wave_coverage:
#         weight = gt_spectra[:,3]
#     else: weight = None

#     emd = calculate_emd(gt_flux, recon_flux, mask=mask,
#                         weight=weight, precision=gt_spectra[:,2])
#     emd = torch.mean(torch.abs(emd))
#     return emd

# class sigmoid_denorm:

#     def __init__(self, lo, hi):
#         self.lo = lo
#         self.hi = hi

#     def __call__(self, data):
#         data[data <= -1] = lo
#         data[data >= 1]  = hi
#         return torch.log(data+1) - torch.log(1-data)

# class SAM:
#     def __init__(self, mx=2, convert_to_degree=False):
#         self.mx = mx
#         self.convert_to_degree = convert_to_degree

#     def __call__(self, org_img, pred_img):
#         # Spectral angles are first computed for each pair of pixels
#         numerator = np.sum(np.multiply(pred_img, org_img), axis=2)
#         denominator = np.linalg.norm(org_img, axis=2) * np.linalg.norm(pred_img, axis=2)
#         val = np.clip(numerator / denominator, -1, 1)
#         sam_angles = np.arccos(val)

#         if self.convert_to_degree:
#             sam_angles = sam_angles * 180.0 / np.pi
#         return np.mean(np.nan_to_num(sam_angles))

# class SSIM:
#     def __init__(self, mx=2):
#         self.mx = mx

#     def __call__(self, org_img, pred_img):
#         return structural_similarity(org_img, pred_img,
#                                      data_range=self.mx, multichannel=True)
# class PSNR:
#     def __init__(self, mx=2):
#         self.mx = mx

#     @staticmethod
#     def __call__(self, img1, img2):
#         mse = np.mean((img1 - img2) ** 2)
#         return 20 * np.log10(self.mx / np.sqrt(mse))

# class RawNerfLoss:
#     """
#     Random nerf loss
#     reduce: 0-sum, 1-mean
#     """
#     def __init__(self, eps, reduce=0):
#         self.eps = eps
#         self.reduce = reduce

#     def __call__(self, y, y_hat):
#         sg_y_hat = y_hat.clone().detach()
#         assert(sg_y_hat.requires_grad == False)
#         assert(y_hat.requires_grad)
#         if self.reduce == 0:
#             return torch.sum(( (y_hat-y)/(sg_y_hat+self.eps) )**2)
#         #return torch.mean(torch.abs(y_hat-y))
#         return torch.mean(( (y_hat-y)/(sg_y_hat+self.eps) )**2)

# def RandNerfLoss(y_hat, y, eps, reduce=0):
#     #sg_y_hat = y_hat.clone().detach()
#     sg_y_hat = 0
#     #assert(sg_y_hat.requires_grad == False)
#     #assert(y_hat.requires_grad)

#     if reduce == 0:
#         return torch.sum(( (y_hat-y)/(sg_y_hat+eps) )**2)
#     return torch.mean(( (y_hat-y)/(sg_y_hat+eps) )**2)
#     #return torch.mean(torch.abs(y_hat-y))
