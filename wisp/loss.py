import torch
import numpy as np

from wisp.utils.numerical import calculate_emd


def spectra_supervision_loss(loss, gt_spectra, recon_spectra):
    ''' Loss function for few-shot spectra supervision
        @Param
          loss: l1/l2 as specified in config
          gt/recon_spectra: [bsz, num_smpls]
    '''
    #return loss(gt_spectra, recon_spectra)

    # norm spectra each so they sum to 1 (earth movers distance)
    # print(gt_spectra.shape, recon_spectra.shape)
    gt_spectra /= (torch.sum(gt_spectra, dim=-1)[...,None])
    recon_spectra /= (torch.sum(recon_spectra, dim=-1)[...,None])

    emd = calculate_emd(gt_spectra, recon_spectra)
    emd = torch.mean(torch.abs(emd))
    return emd

def redshift_supervision_loss(loss, gt_redshift, recon_redshift):
    ''' Loss function for few-shot redshift supervision
        @Param
          loss: l1/l2 as specified in config
          wave_rnge: Trusted lambda range of redshift
          redshift_ids: ids of redshift to supervise
          redshift: [bsz, num_smpls]
    '''
    return loss(gt_redshift, recon_redshift)

def spectral_masking_loss(loss, relative_train_bands, relative_inpaint_bands,
                          gt_pixls, recon_pixls, mask):
    ''' Loss function for spectral inpainting
        @Param
          loss: l1/l2 as specified in config
          relative_train/inpaint_bands:
              Train/inpaint bands may not form a continuous seq of ints.
              Relative bands are which the two are renumbered to a
              continuous seq starting from 0 while maintaing the relative order.
              e.g. before: [1,4]/[2,9]
                   after:  [0,2]/[1,3]
          pixls: [bsz,num_bands]
          mask: sliced with only inpaint dim left
    '''
    masked_gt = gt_pixls[:,relative_train_bands].flatten()
    masked_recon = recon_pixls[:,relative_train_bands].flatten()

    a = torch.masked_select(gt_pixls[:,relative_inpaint_bands],mask)
    b = torch.masked_select(recon_pixls[:,relative_inpaint_bands],mask)

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
