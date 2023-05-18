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
    torch.manual_seed(seed)
    qtz_codebook = nn.Embedding(num_embed, latent_dim)
    qtz_codebook.weight.data.uniform_(
        -1.0 / latent_dim, 1.0 / latent_dim)
    return qtz_codebook

class Normalization(nn.Module):
    def __init__(self, norm_method):
        super(Normalization, self).__init__()
        self.init_norm_layer(norm_method)

    def init_norm_layer(self, norm_method):
        if norm_method == 'identity':
            self.norm = nn.Identity()
        elif norm_method == 'arcsinh':
            self.norm = Fn(torch.arcsinh)
        elif norm_method == 'sinh':
            self.norm = Fn(torch.sinh)
        else:
            raise ValueError('Unsupported normalization method.')

    def forward(self, input):
        return self.norm(input)

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

    def quantize(self, z, codebook, temperature, find_embed_id, save_codebook):
        """ @Param
              codebook [num_embeds,embed_dim]
        """
        if self.quantization_strategy == "soft":
            if find_embed_id:
                min_embed_ids = torch.argmax(z, dim=-1)
            else: min_embed_ids = None

            # weights = nn.functional.softmax(
            #     z * temperature * self.kwargs["qtz_temperature_scale"], dim=-1
            # ) # [bsz,1,num_embeds]
            weights = z * temperature * self.kwargs["qtz_temperature_scale"]
            # regularize s.t. l2 norm of weights sum to 1
            regu = torch.pow(torch.sum(weights**2, dim=-1, keepdim=True), 0.5) + 1e-10
            weights = weights / regu

            if self.kwargs["quantize_spectra"]:
                codebook = codebook.permute(1,0,2) # [bsz,num_embeds,full_nsmpl]

            z_q = torch.matmul(weights, codebook)

            # import numpy as np
            # np.save('codebook.npy', codebook.detach().cpu().numpy())
            # np.save('zq.npy',z_q.detach().cpu().numpy())

        elif self.quantization_strategy == "hard":
            assert(self.kwargs["quantize_latent"])

            weights, z_shape = None, z.shape
            z_f = z.view(-1,self.latent_dim) # flatten
            assert(z_f.shape[-1] == z.shape[-1])

            min_embed_ids = find_closest_tensor(z_f, codebook) # [bsz]

            # replace each z with closest embedding
            encodings = one_hot(min_embed_ids, self.num_embed) # [n,num_embed]
            encodings = encodings.type(z.dtype)

            z_q = torch.matmul(encodings, codebook).view(z_shape)

        else:
            raise ValueError("Unsupported quantization strategy")

        # at this point, codebook is either still in the original form
        # or became codebook spectra (which we need when recon codebook spectra
        #  individually for each spectra)
        return z_q, min_embed_ids, codebook, weights

    def partial_loss(self, z, z_q):
        codebook_loss = torch.mean((z_q.detach() - z)**2) + \
            torch.mean((z_q - z.detach())**2) * self.beta
        return codebook_loss

    def forward(self, z, codebook, ret, qtz_args):
        """ @Param
               z:        logits  or latent variables
                         [bsz,1,embed_dim/num_embeds] (hard/soft qtz)
               codebook: codebook for qtz [num_embeds,embed_dim]
               ret:      collection of results to return
               qtz_args (deafultdict: False):
                 temperature:   soft qtz temp (current num of steps)
                 find_embed_id: whether find embed id, only used for soft quantization
                                hard qtz always requires find embed id
                                soft qtz requires only when plotting embed map
                 save_codebook: save codebook weights value to local
                 save_soft_qtz_weights: save weights for each code (when doing soft qtz)
        """
        z_q, min_embed_ids, codebook, codebook_weights = self.quantize(
            z, codebook, qtz_args["temperature"], qtz_args["find_embed_id"],
            qtz_args["save_codebook"])

        if self.quantization_strategy == "hard":
            ret["min_embed_ids"] = min_embed_ids

            if self.calculate_loss:
                ret["codebook_loss"] = self.partial_loss(z, z_q)

            # straight-through estimator
            z_q = z + (z_q - z).detach()

        elif self.quantization_strategy == "soft":
            if qtz_args["save_codebook"]:
                ret["codebook"] = codebook # [bsz,num_embeds,full_nsmpl]

            if qtz_args["find_embed_id"]:
                ret["min_embed_ids"] = min_embed_ids

            if qtz_args["save_soft_qtz_weights"]:
                ret["soft_qtz_weights"] = codebook_weights

        else: raise ValueError("Unsupported quantization strategy")
        return z, z_q
