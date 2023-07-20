import torch
import numpy as np

from wisp.utils.numerical import calculate_emd

def pretrain_pixel_loss(loss, gt_pixels, recon_pixels):
    gt_pixels = gt_pixels / (torch.sum(gt_pixels, dim=-1)[...,None])
    recon_pixels = recon_pixels / (torch.sum(recon_pixels, dim=-1)[...,None])
    emd = calculate_emd(gt_pixels, recon_pixels)
    emd = torch.mean(torch.abs(emd))
    return emd

def plotspectra(spectra):
    spectra = spectra.detach().cpu().numpy()
    import matplotlib.pyplot as plt
    n,m = spectra.shape
    x = np.arange(m)
    fig, axs = plt.subplots(2,10,figsize=(50,10))
    for i in range(n):
        axis = axs[i//10,i%10]
        axis.plot(x,spectra[i])
    fig.tight_layout()
    plt.savefig('tmp.png')
    plt.close()

def spectra_supervision_loss(loss, mask, gt_spectra, recon_flux):
    ''' Loss function for spectra supervision
        @Param
          loss: l1/l2 as specified in config
          mask:       [bsz,num_smpls]
          gt_spectra: [bsz,4+2*nbanbds,num_smpls]
                      (wave/flux/ivar/trans_mask/trans(nbands)/band_mask(nbands))
          recon_flux: [bsz,num_smpls]
    '''
    # norm spectra each so they sum to 1 (earth movers distance)
    # DON'T use /= or spectra will be modified in place and
    #   if we save spectra later on the spectra will be inaccurate
    gt_flux = gt_spectra[:,1] / (torch.sum(gt_spectra[:,1]*mask, dim=-1)[...,None] + 1e-10)
    recon_flux = recon_flux / (torch.sum(recon_flux*mask, dim=-1)[...,None] + 1e-10)
    # print(gt_flux.shape, recon_flux.shape)
    # plotspectra(gt_flux)

    # sanity check, sum of unmasked flux should be same as bsz
    # gt_flux = torch.masked_select(gt_flux, mask.bool())
    # recon_flux = torch.masked_select(recon_flux, mask.bool())
    # print(torch.sum(gt_flux), torch.sum(recon_flux))

    emd = calculate_emd(gt_flux, recon_flux, mask=mask, precision=gt_spectra[:,2])
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
    # print(gt_redshift, recon_redshift[mask])
    return loss(gt_redshift, recon_redshift[mask])

def spectral_masking_loss(loss, relative_train_bands, relative_inpaint_bands,
                          gt_pixels, recon_pixels, mask):
    ''' Loss function for spectral inpainting
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
    '''
    masked_gt = gt_pixls[:,relative_train_bands].flatten()
    masked_recon = recon_pixls[:,relative_train_bands].flatten()

    a = torch.masked_select(gt_pixels[:,relative_inpaint_bands], mask)
    b = torch.masked_select(recon_pixels[:,relative_inpaint_bands], mask)

    masked_gt = torch.cat((masked_gt, a))
    masked_recon = torch.cat((masked_recon, b))
    error = loss(masked_gt, masked_recon)
    return error

class sigmoid_denorm:

    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi

    def __call__(self, data):
        data[data <= -1] = lo
        data[data >= 1]  = hi
        return torch.log(data+1) - torch.log(1-data)

class SAM:
    def __init__(self, mx=2, convert_to_degree=False):
        self.mx = mx
        self.convert_to_degree = convert_to_degree

    def __call__(self, org_img, pred_img):
        # Spectral angles are first computed for each pair of pixels
        numerator = np.sum(np.multiply(pred_img, org_img), axis=2)
        denominator = np.linalg.norm(org_img, axis=2) * np.linalg.norm(pred_img, axis=2)
        val = np.clip(numerator / denominator, -1, 1)
        sam_angles = np.arccos(val)

        if self.convert_to_degree:
            sam_angles = sam_angles * 180.0 / np.pi
        return np.mean(np.nan_to_num(sam_angles))

class SSIM:
    def __init__(self, mx=2):
        self.mx = mx

    def __call__(self, org_img, pred_img):
        return structural_similarity(org_img, pred_img,
                                     data_range=self.mx, multichannel=True)
class PSNR:
    def __init__(self, mx=2):
        self.mx = mx

    @staticmethod
    def __call__(self, img1, img2):
        mse = np.mean((img1 - img2) ** 2)
        return 20 * np.log10(self.mx / np.sqrt(mse))

'''
Random nerf loss
reduce: 0-sum, 1-mean
'''
class RawNerfLoss:

    def __init__(self, eps, reduce=0):
        self.eps = eps
        self.reduce = reduce

    def __call__(self, y, y_hat):
        sg_y_hat = y_hat.clone().detach()
        assert(sg_y_hat.requires_grad == False)
        assert(y_hat.requires_grad)
        if self.reduce == 0:
            return torch.sum(( (y_hat-y)/(sg_y_hat+self.eps) )**2)
        #return torch.mean(torch.abs(y_hat-y))
        return torch.mean(( (y_hat-y)/(sg_y_hat+self.eps) )**2)

'''
def RandNerfLoss(y_hat, y, eps, reduce=0):
    #sg_y_hat = y_hat.clone().detach()
    sg_y_hat = 0
    #assert(sg_y_hat.requires_grad == False)
    #assert(y_hat.requires_grad)

    if reduce == 0:
        return torch.sum(( (y_hat-y)/(sg_y_hat+eps) )**2)
    return torch.mean(( (y_hat-y)/(sg_y_hat+eps) )**2)
    #return torch.mean(torch.abs(y_hat-y))
'''
