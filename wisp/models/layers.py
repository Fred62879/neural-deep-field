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

        self.init_codebook(kwargs["qtz_seed"])

    def init_codebook(self, seed):
        torch.manual_seed(seed)
        self.qtz_codebook = nn.Embedding(self.num_embed, self.latent_dim)
        self.qtz_codebook.weight.data.uniform_(
            -1.0 / self.latent_dim, 1.0 / self.latent_dim)

    def quantize(self, z, temperature, find_embed_id):
        if self.quantization_strategy == "soft":
            if find_embed_id:
                min_embed_ids = torch.argmax(z, dim=-1)
            else: min_embed_ids = None

            weights = nn.functional.softmax( # [bsz,1,num_embeds]
                z * temperature * self.kwargs["qtz_temperature_scale"], dim=-1)
            z_q = torch.matmul(weights, self.qtz_codebook.weight)

        elif self.quantization_strategy == "hard":
            weights, z_shape = None, z.shape
            z_f = z.view(-1,self.latent_dim) # flatten
            min_embed_ids = find_closest_tensor(z_f, self.qtz_codebook.weight) # [bsz]

            # replace each z with closest embedding
            encodings = one_hot(min_embed_ids, self.num_embed) # [n,num_embed]
            encodings = encodings.type(z.dtype)
            z_q = torch.matmul(encodings, self.qtz_codebook.weight).view(z_shape)

        else: raise ValueError("Unsupported quantization strategy")
        return z_q, min_embed_ids, weights

    def partial_loss(self, z, z_q):
        codebook_loss = torch.mean((z_q.detach() - z)**2) + \
            torch.mean((z_q - z.detach())**2) * self.beta
        return codebook_loss

    def forward(self, z, ret, temperature=1, find_embed_id=False,
                save_codebook=False, save_soft_qtz_weights=False):

        """ @Param
               z: latent variables
               ret: collection of results to return
               temperature: used for softmax quantization
               find_embed_id: whether find embed id, only used for soft quantization
                              hard qtz always requires find embed id
                              soft qtz requires only when plotting embed map
               save_codebook: save codebook weights value to local
               save_soft_qtz_weights: save weights for each code (when doing soft qtz)
        """
        z_q, min_embed_ids, codebook_weights = self.quantize(z, temperature, find_embed_id)

        if self.quantization_strategy == "hard":
            ret["min_embed_ids"] = min_embed_ids

            if self.calculate_loss:
                ret["codebook_loss"] = self.partial_loss(z, z_q)

            # straight-through estimator
            z_q = z + (z_q - z).detach()

        elif self.quantization_strategy == "soft":
            if find_embed_id:
                ret["min_embed_ids"] = min_embed_ids

            if save_soft_qtz_weights:
                ret["soft_qtz_weights"] = codebook_weights

        else: raise ValueError("Unsupported quantization strategy")

        if save_codebook:
            ret["codebook"] = self.qtz_codebook.weight.detach().cpu().numpy()

        return z, z_q
