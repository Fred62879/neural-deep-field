# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.functional import one_hot
from wisp.utils.common import set_seed
from wisp.utils.numerical import find_closest_tensor


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
    """Normalizes the matrix according to the L1 norm.

    Args:
        x (torch.FloatTensor): A matrix.

    Returns:
        (torch.FloatTensor): A normalized matrix.
    """
    abscolsum = torch.sum(torch.abs(x), dim=0)
    abscolsum = torch.min(torch.stack([1.0/abscolsum, torch.ones_like(abscolsum)], dim=0), dim=0)[0]
    return x * abscolsum[None,:]

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
    set_seed()
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
        self.temperature = kwargs["qtz_soft_temperature"]

        self.beta = kwargs["qtz_beta"]
        self.num_embed = kwargs["qtz_num_embed"]
        self.latent_dim = kwargs["qtz_latent_dim"]

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
        encodings = one_hot(min_embed_ids, self.num_embed) # [n,num_embed]
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
        """ Hard quantization can be applied at both quantize latent and spectra.
            @Param
              z: logits [bsz,1,num_embed]
              codebook: codebook spectra [...,num_embed,bsz,embed_dim]
        """
        if qtz_args["find_embed_id"]:
            min_embed_ids = torch.argmax(z, dim=-1)
        else: min_embed_ids = None

        weights = z * qtz_args["temperature"] * self.kwargs["qtz_temperature_scale"]
        # choice 1: use softmax
        # weights = nn.functional.softmax(weights, dim=-1) # [bsz,1,num_embeds]
        # choice 2: regularize s.t. l2 norm of weights sum to 1
        regu = torch.pow(torch.sum(weights**2, dim=-1, keepdim=True), 0.5)
        weights = weights / (regu + 1e-10)

        # codebook here is codebook spectra
        if self.kwargs["quantize_spectra"]:
            if codebook.ndim == 3:
                codebook = codebook.permute(1,0,2) # [bsz,num_embeds,nsmpl]
            elif codebook.ndim == 4:
                num_bins = codebook.shape[0]
                weights = weights[None,...].tile(num_bins,1,1,1) # [...,bsz,1,num_embds]
                codebook = codebook.permute(0,2,1,3) # [...,bsz,num_embds,nsmpl]
            else: raise ValueError()

        # [...,bsz,1,nsmpl]
        z_q = torch.matmul(weights, codebook)

        if qtz_args["find_embed_id"]:
            ret["min_embed_ids"] = min_embed_ids
        if qtz_args["save_qtz_weights"]:
            ret["qtz_weights"] = weights
        if qtz_args["save_codebook_spectra"]:
            ret["codebook_spectra"] = codebook # [bsz,num_embeds,full_nsmpl]
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
