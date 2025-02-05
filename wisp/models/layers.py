
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.functional import one_hot
from wisp.utils.numerical import find_closest_tensor
from wisp.utils.common import set_seed, get_bool_classify_redshift, init_redshift_bins


def normalize_frobenius(x):
    """Normalizes the matrix according to the Frobenius norm.

    Args:
        x (torch.FloatTensor): A matrix.

    Returns:
        (torch.FloatTensor): A normalized matrix.
    """
    norm = torch.sqrt((torch.abs(x)**2).sum())
    return x / norm

def normalize_L_1(x):
    """
    Normalizes the matrix according to the L1 norm.
    @Params:
        x (torch.FloatTensor): A matrix.
    @Returns:
        (torch.FloatTensor): A normalized matrix.
    """
    abscolsum = torch.sum(torch.abs(x), dim=0)
    abscolsum = torch.min(torch.stack([1.0/abscolsum, torch.ones_like(abscolsum)], dim=0), dim=0)[0]
    return x * abscolsum[None,:]

def normalize_Linear(x):
    """
    Normalizes the matrix according to [0,1].
    @Params:
        x (torch.FloatTensor): A matrix.
    @Returns:
        (torch.FloatTensor): A normalized matrix.
    """
    lo, _ = torch.min(x, dim=-1, keepdim=True)
    hi, _ = torch.max(x, dim=-1, keepdim=True)
    ret = (x - lo) / (hi - lo)
    return ret

def normalize_L_inf(x):
    """Normalizes the matrix according to the Linf norm.

    Args:
        x (torch.FloatTensor): A matrix.

    Returns:
        (torch.FloatTensor): A normalized matrix.
    """
    absrowsum = torch.sum(torch.abs(x), axis=1)
    absrowsum = torch.min(torch.stack([1.0/absrowsum, torch.ones_like(absrowsum)], dim=0), dim=0)[0]
    return x * absrowsum[:,None]

class FrobeniusLinear(nn.Module):
    """A standard Linear layer which applies a Frobenius normalization in the forward pass.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = nn.Linear(*args, **kwargs)

    def forward(self, x):
        weight = normalize_frobenius(self.linear.weight)
        return F.linear(x, weight, self.linear.bias)

class L_1_Linear(nn.Module):
    """A standard Linear layer which applies a L1 normalization in the forward pass.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = nn.Linear(*args, **kwargs)

    def forward(self, x):
        weight = normalize_L_1(self.linear.weight)
        return F.linear(x, weight, self.linear.bias)

class L_inf_Linear(nn.Module):
    """A standard Linear layer which applies a Linf normalization in the forward pass.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = nn.Linear(*args, **kwargs)

    def forward(self, x):
        weight = normalize_L_inf(self.linear.weight)
        return F.linear(x, weight, self.linear.bias)

def spectral_norm_(*args, **kwargs):
    """Initializes a spectral norm layer.
    """
    return nn.utils.spectral_norm(nn.Linear(*args, **kwargs))

def get_layer_class(layer_type):
    """Convenience function to return the layer class name from text.

    Args:
        layer_type (str): Text name for the layer.

    Retunrs:
        (nn.Module): The layer to be used for the decoder.
    """
    if layer_type == 'none':
        return nn.Linear
    elif layer_type == 'spectral_norm':
        return spectral_norm_
    elif layer_type == 'frobenius_norm':
        return FrobeniusLinear
    elif layer_type == "l_1_norm":
        return L_1_Linear
    elif layer_type == "l_inf_norm":
        return L_inf_Linear
    else:
        assert(False and "layer type does not exist")

def init_codebook(seed, num_embed, latent_dim):
    # set_seed()
    qtz_codebook = nn.Embedding(num_embed, latent_dim)
    qtz_codebook.weight.data.uniform_(
        -1.0 / latent_dim, 1.0 / latent_dim)
    return qtz_codebook

class Intensifier(nn.Module):
    def __init__(self, intensification_method):
        super(Intensifier, self).__init__()
        self.init_norm_layer(intensification_method)

    def init_norm_layer(self, intensification_method):
        if intensification_method == 'identity':
            self.intensifier = nn.Identity()
        elif intensification_method == 'arcsinh':
            self.intensifier = Fn(torch.arcsinh)
        elif intensification_method == 'sinh':
            self.intensifier = Fn(torch.sinh)
        else:
            raise ValueError('Unsupported intensification method.')

    def forward(self, input):
        return self.intensifier(input)

class Fn(nn.Module):
    def __init__(self, fn):
        super(Fn, self).__init__()
        self.fn = fn

    def forward(self, input):
        return self.fn(input)

class TrainableEltwiseLayer(nn.Module):
    def __init__(self, ndim, mean=0, std=.1):
        super(TrainableEltwiseLayer, self).__init__()

        #weights = torch.exp(torch.normal(mean=mean, std=std, size=(ndim,)).abs())
        weights = torch.ones(ndim)
        self.weights = nn.Parameter(weights)

    # input, [bsz,ndim]
    def forward(self, input):
        return input * self.weights

class Quantization(nn.Module):
    """ Quantization layer for latent variables.
        @Param
          calculate_loss: whether calculate codebook loss or not
    """
    def __init__(self, calculate_loss, **kwargs):
        super(Quantization, self).__init__()
        self.kwargs = kwargs

        self.calculate_loss = calculate_loss
        self.quantization_strategy = kwargs["quantization_strategy"]

        self.beta = kwargs["qtz_beta"]
        self.num_embed = kwargs["qtz_num_embed"]
        self.latent_dim = kwargs["qtz_latent_dim"]
        # todo, if we toggle between batched and unbatched hps model, we need to remove this
        self.classify_redshift = get_bool_classify_redshift(**kwargs)

    def partial_loss(self, z, z_q):
        codebook_loss = torch.mean((z_q.detach() - z)**2) + \
            torch.mean((z_q - z.detach())**2) * self.beta
        return codebook_loss

    def hard_quantize(self, z, codebook, ret, qtz_args):
        """ Hard quantization only applied when quantize latent.
            @Param
              z: latent variable [bsz,1,embed_dim]
              codebook: [num_embeds,embed_dim]
        """
        assert(self.kwargs["quantize_latent"])

        weights, z_shape = None, z.shape
        z_f = z.view(-1,self.latent_dim) # flatten
        assert(z_f.shape[-1] == z.shape[-1])

        min_embed_ids = find_closest_tensor(z_f, codebook) # [bsz]

        # replace each z with closest embedding
        encodings = one_hot(min_embed_ids, num_classes=self.num_embed) # [bsz,num_embed]
        encodings = encodings.type(z.dtype)

        # codebook here is the original codebook
        z_q = torch.matmul(encodings, codebook).view(z_shape)
        # straight-through estimator
        z_q = z + (z_q - z).detach()

        ret["min_embed_ids"] = min_embed_ids
        if self.calculate_loss:
            ret["codebook_loss"] = self.partial_loss(z, z_q)
        return z_q

    def soft_quantize(self, z, codebook, ret, qtz_args):
        """ Soft quantization can be applied at both quantize latent and spectra.
            @Param
              z: coefficients [(...,)bsz,1,num_embed]
              codebook: codebook spectra [...,num_embed,bsz,nsmpl]
        """
        if qtz_args["find_embed_id"]:
            min_embed_ids = torch.argmax(z, dim=-1)
        else: min_embed_ids = None

        weights = z
        if self.kwargs["temped_qtz"]:
            weights = weights * qtz_args["temperature"] * self.kwargs["qtz_temperature_scale"]
        # print(z.shape, torch.sum(z, dim=-1)[:,0])
        # print(qtz_args["temperature"], self.kwargs["qtz_temperature_scale"])

        # choice 1: use softmax
        # weights = nn.functional.softmax(weights, dim=-1) # [bsz,1,num_embeds]
        # choice 2: regularize s.t. l2 norm of weights sum to 1
        # regu = torch.pow(torch.sum(weights**2, dim=-1, keepdim=True), 0.5)
        # weights = weights / (regu + 1e-10)

        # print(weights.shape, codebook.shape)

        # codebook here is codebook spectra
        if self.kwargs["quantize_spectra"]:
            if codebook.ndim == 3:
                codebook = codebook.permute(1,0,2) # [bsz,num_embeds,nsmpl]
            elif codebook.ndim == 4:
                num_bins = codebook.shape[0]
                if weights.ndim == 3:
                    weights = weights[None,...].tile(num_bins,1,1,1) # [...,bsz,1,num_embds]
                else:
                    assert weights.ndim == 4 # [bsz,1,num_bins,num_embeds]
                    weights = weights.permute(2,0,1,3)
                codebook = codebook.permute(0,2,1,3) # [...,bsz,num_embds,nsmpl]
            else: raise ValueError()

        # [...,bsz,1,nsmpl]
        # print(weights.shape, codebook.shape)
        z_q = torch.matmul(weights, codebook)

        if qtz_args["find_embed_id"]:
            ret["min_embed_ids"] = min_embed_ids
        if qtz_args["save_qtz_weights"]:
            # todo: currently, our hps model is batched and if we classify redshift
            # we end up with an extra dimension for each redshift bin which
            # however is identical in terms of qtz weights, thus we keep only one
            if self.classify_redshift:
                ret["qtz_weights"] = weights[0]
            else: ret["qtz_weights"] = weights
        # if qtz_args["save_codebook_spectra"]:
        #     ret["codebook_spectra"] = codebook # [bsz,num_embeds,full_nsmpl]
        return z_q

    def forward(self, z, codebook, ret, qtz_args):
        """ @Param
               z:        logits  or latent variables
                         [bsz,1,embed_dim/num_embeds] (hard/soft qtz)
               codebook: codebook for qtz [num_embeds,(bsz,)embed_dim] (soft)
               ret:      collection of results to return
               qtz_args (deafultdict: False):
                 temperature:   soft qtz temp (current num of steps)
                 find_embed_id: whether find embed id, only used for soft quantization
                                hard qtz always requires find embed id
                                soft qtz requires only when plotting embed map
                 save_codebook_spectra: save codebook spectra locally (only for soft qtz)
                 save_soft_qtz_weights: save weights for each code (only for soft qtz)
        """
        if self.quantization_strategy == "hard":
            z_q = self.hard_quantize(z, codebook, ret, qtz_args)
        elif self.quantization_strategy == "soft":
            z_q = self.soft_quantize(z, codebook, ret, qtz_args)
        else:
            raise ValueError("Unsupported quantization strategy")
        return z, z_q

class ArgMax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits):
        """ @Param
              logits: [bsz,num_bins]
              spectra: [num_bins,bsz,nsmpl]
            @Return
              spectra: indexed argmax spectra according to logits [bsz,nsmpl]
        """
        num_bins = logits.shape[1]
        ids = torch.argmax(logits, dim=-1) # [bsz]
        ctx.save_for_backward(ids, torch.tensor([num_bins]))

        encodings = F.one_hot(ids, num_classes=num_bins) # [bsz,num_bins]
        encodings = encodings.type(logits.dtype)
        # spectra = torch.matmul(encodings[:,None], spectra.permute(1,0,2))[:,0]
        # return spectra
        return encodings

    @staticmethod
    def backward(ctx, grad_output):
        """ Expand returned grad to fit forward spectra size.
            Fill entries corresponds to argmax ids (according to logits) with
              grad_output and leave the rest gradients as zero.

            @Context
              num_bins: [1]
              argmax_ids: [bsz]
            @Return
              grad_updated: [num_bins,bsz,nsmpl]
        """
        argmax_ids, num_bins = ctx.saved_tensors
        bsz = argmax_ids.shape[0]
        device = grad_output.device

        sp = list(num_bins) + list(grad_output.shape)
        grad_updated = torch.zeros(sp).to(device)
        batch_ids = torch.arange(bsz).to(device)
        ids = torch.cat((argmax_ids[None,:], batch_ids[None,:]), dim=0)

        grad_updated[ids[0,:],ids[1,:]] = grad_output
        # print(grad_updated[13,0])
        # print(grad_updated[9,3])
        return grad_updated

def calculate_bayesian_redshift_logits(
    loss, mask, gt_spectra, recon_fluxes, redshift_logits, **kwargs
):
    """ Calculate bayesian logits for redshfit classification.
        @Param
          mask:       [bsz,num_smpls]
          gt_spectra: [bsz,4+2*nbanbds,num_smpls]
                      (wave/flux/ivar/weight/trans_mask/trans(nbands)/band_mask(nbands))
          recon_fluxes: [num_bins,bsz,num_smpls]
          redshift_logits: p(z | spectra_recon)  [bsz,num_bins]
        @Return
          logits: p(z | spectra_gt) [bsz,num_bins]
    """
    num_bins = len(recon_fluxes)
    gt_fluxes = gt_spectra[:,1]*mask[None,...].tile(num_bins,1,1)
    spectra_loss_bin_wise = loss(gt_fluxes, recon_fluxes*mask)
    spectra_loss_bin_wise = torch.mean(spectra_loss_bin_wise, dim=-1) # [num_bins,bsz]
    spectra_logits = torch.exp(-spectra_loss_bin_wise) # p(spectra_recon | spectra_gt)

    logits = redshift_logits * spectra_logits.T # [bsz,num_bins]
    logits = logits / torch.sum(logits, dim=-1)[:,None]
    return logits

def calculate_spectra_loss(
        loss_func, mask, gt_spectra, recon_fluxes, ret,
        save_lambdawise_loss=False, train_with_lambdawise_weights=False,
        suffix="", **kwargs
):
    """
    Calculate spectra loss.
    @Param
      mask:       [bsz,num_smpls]
      gt_spectra: [bsz,4+2*nbanbds,num_smpls]
                  (wave/flux/ivar/weight/trans_mask/trans(nbands)/band_mask(nbands))
      recon_fluxes: [(num_bins,)bsz,num_smpls]
    @AddToDict
      binwise_loss:    [bsz,n_bins]
      lambdawise_loss: [bsz,n_bins,nsmpls]
    """
    n_bins = recon_fluxes.shape[0]
    brute_force = recon_fluxes.ndim == 3
    apply_gt_redshift = recon_fluxes.ndim == 2

    if apply_gt_redshift:
        lambdawise_loss = loss_func(gt_spectra, recon_fluxes, mask) # [bsz,nsmpl]
    elif brute_force:
        lambdawise_loss = loss_func(
            gt_spectra[None,:].tile(n_bins,1,1,1), recon_fluxes,
            mask[None,:].tile(n_bins,1,1)
        ) # [n_bins,bsz,nsmpls]
    else: raise ValueError()

    assert recon_fluxes.shape == lambdawise_loss.shape

    if save_lambdawise_loss:
        """
        We save lambdawise loss when
          plot spectra with lambdawise loss or
          plot and save global restframe loss as training weights
        Here `lambdawise_loss` may contain `nan` which will be masked out
          in the following plotting functions.
        Also, we save before multiplying weights as we need the original loss
          for visualization purposes.
        """
        nm = f"spectra_lambdawise_loss{suffix}"
        if apply_gt_redshift:
            ret[nm] = lambdawise_loss
        elif brute_force:
            ret[nm] = lambdawise_loss.permute(1,0,2)

    if apply_gt_redshift:
        lambdawise_loss[mask == 0] = 0
        assert not torch.isnan(lambdawise_loss).any() # [bsz,nsmpl]
        if kwargs["spectra_loss_reduction"] == "sum":
            spectrawise_loss = torch.sum(lambdawise_loss, dim=-1)
        elif kwargs["spectra_loss_reduction"] == "mean":
            mask = torch.sum(mask, dim=-1) # [bsz,]
            # assert not (mask == 0).any()
            valid = mask != 0
            mask = mask[valid]
            lambdawise_loss = lambdawise_loss[valid]
            spectrawise_loss = (torch.sum(lambdawise_loss, dim=-1) / mask)
        else: raise ValueError()
        ret[f"spectrawise_loss{suffix}"] = spectrawise_loss

    elif brute_force:
        if train_with_lambdawise_weights:
            weights = ret["lambdawise_weights"] # [nbins,bsz,nsmpl]
            lambdawise_loss = lambdawise_loss * weights
        """
        From now on, `lambdawise_loss` is used only for binwise loss calculation.
        As we still need to keep the batched shape of the tensor, we cannot directly mask
          out `nan` which will give a flattened tensor. Thus we replace `nan` with 0.
        """
        mask = mask[None,...].tile(n_bins,1,1)
        lambdawise_loss[mask == 0] = 0
        assert not torch.isnan(lambdawise_loss).any() # [nbins,bsz,nsmpl]
        if kwargs["spectra_lambdawise_loss_reduction"] == "sum":
            binwise_loss = torch.sum(lambdawise_loss, dim=-1).T # [nbins,bsz]
        elif kwargs["spectra_lambdawise_loss_reduction"] == "mean":
            mask = torch.sum(mask, dim=-1) # [nbins,bsz] duplicate in dim 0
            mask[mask == 0] = 1 # avoid division by 0 todo: need to mask out
            binwise_loss = (torch.sum(lambdawise_loss, dim=-1) / mask).T
        else: raise ValueError()
        ret[f"spectra_binwise_loss{suffix}"] = binwise_loss

def calculate_redshift_logits(beta, ret, suffix=""):
    """
    Calculate logits for redshift bins as softmax of negative loss.
    """
    logits = -ret[f"spectra_binwise_loss{suffix}"] * beta
    ret[f"redshift_logits{suffix}"] = F.softmax(logits, dim=-1)
