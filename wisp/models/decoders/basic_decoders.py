# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import ortho_group
from wisp.utils.common import get_gpu_info, query_GPU_mem, set_seed

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=1, hidden_dim=128, seed=0):
        super(MLP, self).__init__()

        def block(dim_in, dim_out, seed=0, activate=True):
            # set_seed()
            block = [ nn.Linear(dim_in, dim_out) ]
            if activate:
                block.append(nn.ReLU(inplace=True))
            return block

        net = []
        net.extend(block(input_dim, hidden_dim, seed))
        for i in range(num_layers):
            net.extend(block(hidden_dim, hidden_dim, seed + i + 1))
        net.extend(block(hidden_dim, output_dim, seed + num_layers + 1, False))
        self.model = nn.Sequential(*net)

    def forward(self, x):
        return self.model(x)

class BasicDecoder(nn.Module):
    """Super basic but super useful MLP class.
    """
    def __init__(self, input_dim, output_dim, activation,
                 bias, layer = nn.Linear,
                 num_layers = 1,
                 hidden_dim = 128,
                 batch_norm = False,
                 skip       = [],
                 skip_all_layers=False,
                 activation_before_skip=False,
                 skip_with_same_dim=False,
                 skip_with_same_dim_sep_layers=False
    ):
        """Initialize the BasicDecoder.

        Args:
            input_dim (int): Input dimension of the MLP.
            output_dim (int): Output dimension of the MLP.
            activation (function): The activation function to use.
            bias (bool): If True, use bias.
            layer (nn.Module): The MLP layer module to use.
            num_layers (int): The number of hidden layers in the MLP.
            hidden_dim (int): The hidden dimension of the MLP.
            skip (List[int]): List of layer indices where the input dimension is concatenated.
            skip_same_dim:    Convert skip input to be of same dim as hidden layer latents.

        Returns:
            (void): Initializes the class.
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.bias = bias
        self.layer = layer
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.batch_norm = batch_norm
        self.activation_before_skip = activation_before_skip
        assert self.activation_before_skip

        self.skip = skip
        if skip_all_layers:
            self.skip = np.arange(num_layers)

        self.skip_with_same_dim = skip_with_same_dim and len(self.skip) != 0
        self.skip_with_same_dim_sep_layers = \
            skip_with_same_dim_sep_layers and self.skip_with_same_dim

        self.init()

    def get_first_layer_input_dim(self):
        if self.skip_with_same_dim and not self.skip_with_same_dim_sep_layers:
            # if we perform skip with single conversion layer, we convert input to
            #  same dim as hidden latents and then pass it to following layers
            in_dim = self.hidden_dim
        else: in_dim = self.input_dim
        return in_dim

    def get_input_dim(self, i):
        if i == 0:
            in_dim = self.get_first_layer_input_dim()
        elif i in self.skip:
            if self.skip_with_same_dim: in_dim = self.hidden_dim
            else: in_dim = self.hidden_dim + self.input_dim
        else:
            in_dim = self.hidden_dim
        return in_dim

    def init(self):
        """ Builds the actual MLP.
        """
        # initializes mlp layers
        layers = []
        for i in range(self.num_layers):
            in_dim = self.get_input_dim(i)
            layer = self.layer(in_dim, self.hidden_dim, bias=self.bias)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
        self.lout = self.layer(self.hidden_dim, self.output_dim, bias=self.bias)

        # initializes batch normalization layers
        if self.batch_norm:
            self.bns = []
            for i in range(self.num_layers):
                self.bns.append(nn.BatchNorm1d(self.hidden_dim))
            self.bns = nn.ModuleList(self.bns)

        # initializes conversion layers for skip
        if self.skip_with_same_dim:
            if self.skip_with_same_dim_sep_layers:
                convert_layers = []
                for i in range(self.num_layers):
                    if i in self.skip:
                        layer = self.layer(self.input_dim, self.hidden_dim, bias=self.bias)
                        convert_layers.append(layer)
                self.convert_layers = nn.ModuleList(convert_layers)
            else:
                self.convert_layer = self.layer(self.input_dim, self.hidden_dim, bias=self.bias)

    def forward(self, x, return_h=False):
        """
        @Params
          x (torch.FloatTensor): Some tensor of shape [batch, ..., input_dim]
          return_h (bool): If True, also returns the last hidden layer.
        @Return
          (torch.FloatTensor, (optional) torch.FloatTensor):
            - The output tensor of shape [batch, ..., output_dim]
            - The last hidden layer of shape [batch, ..., hidden_dim]
        """
        if self.skip_with_same_dim and not self.skip_with_same_dim_sep_layers:
            x = self.convert_layer(x)

        h = self.forward_one_layer(0, x, x)
        for i in range(1, self.num_layers):
            h = self.forward_one_layer(i, x, h)
        out = self.lout(h)

        if return_h: return out, h
        return out

    def forward_one_layer(self, i, x, h):
        """
        Forward through one layer considering batch norm and skip.
        @Param
          i: i-th layer
          x: original input (x0)
          h: output from previous layer (x_{i-1})
        """
        if i in self.skip:
            if self.skip_with_same_dim:
                if self.skip_with_same_dim_sep_layers:
                    assert i == 0 or h.shape != x.shape
                    x_skip = self.convert_layers[i](x)
                else:
                    assert h.shape == x.shape
                    x_skip = x

                # if self.activation_before_skip:
                h = x_skip + self.forward_mlp(i, h)
                # else:
                #     h = x_skip + h
                #     h = self.forward_mlp(i, h)
            else:
                if i != 0:
                    h = torch.cat([x, h], dim=-1)
                h = self.forward_mlp(i, h)
        else:
            h = self.forward_mlp(i, h)
        return h

    def forward_mlp(self, i, h):
        l = self.layers[i]
        print(i, h.shape)
        h = l(h)
        if self.batch_norm:
            if h.ndim == 3: # [bsz,nsmpl,dim]
                h = h.permute(0,2,1)
                h = self.bns[i](h)
                h = h.permute(0,2,1)
            elif h.ndim == 4: # [nbins,bsz,nsmpl,dim]
                nbins,bsz,nsmpl,dim = h.shape
                h = h.permute(0,1,3,2).view(nbins*bsz,dim,nsmpl)
                h = self.bns[i](h)
                h = h.view(nbins,bsz,dim,nsmpl).permute(0,1,3,2)
            else:
                raise ValueError()
        h = self.activation(h)
        return h

    def initialize_last_layer_equal(self):
        """ Initializes the last layer (w and bias) such that outputs are the same.
        """
        w = self.lout.weight
        w_sp = list(w.shape)
        nsmpl = w.shape[0]
        w_ = torch.mean(w, dim=0)
        w_new = w_[None:,].tile(nsmpl,1)
        self.lout.weight = nn.Parameter( torch.full(w_sp, torch.mean(w).item()) )
        if self.bias:
            bias = self.lout.bias
            b_sp = list(bias.shape)
            self.lout.bias = nn.Parameter( torch.full(b_sp, torch.mean(bias).item()) )

    def initialize(self, get_weight):
        """ Initializes the MLP layers with some initialization functions.
            @Params
              get_weight (function): A function which returns a matrix given a matrix.
            @Return:
              (void): Initializes the layer weights.
        """
        ms = []
        for i, w in enumerate(self.layers):
            m = get_weight(w.weight)
            ms.append(m)
        for i in range(len(self.layers)):
            self.layers[i].weight = nn.Parameter(ms[i])
        m = get_weight(self.lout.weight)
        self.lout.weight = nn.Parameter(m)

def orthonormal(weight):
    """ Initialize the layer as a random orthonormal matrix.

    Args:
        weight (torch.FloatTensor): Matrix of shape [M, N]. Only used for the shape.

    Returns:
        (torch.FloatTensor): Matrix of shape [M, N].
    """
    m = ortho_group.rvs(dim=max(weight.shape))
    #m = np.dot(m.T, m)
    m = m[:weight.shape[0],:weight.shape[1]]
    return torch.from_numpy(m).float()

def svd(weight):
    """ Initialize the layer with the U,V of SVD.

    Args:
        weight (torch.FloatTensor): Matrix of shape [M, N].

    Returns:
        (torch.FloatTensor): Matrix of shape [M, N].
    """
    U,S,V = torch.svd(weight)
    return torch.matmul(U, V.T)

def spectral_normalization(weight):
    """ Initialize the layer with spectral normalization.

    Args:
        weight (torch.FloatTensor): Matrix of shape [M, N].

    Returns:
        (torch.FloatTensor): Matrix of shape [M, N].
    """
    U,S,V = torch.svd(weight)
    return weight / S.max()

def identity(weight):
    """ Initialize the layer with identity matrix.

    Args:
        weight (torch.FloatTensor): Matrix of shape [M, N].

    Returns:
        (torch.FloatTensor): Matrix of shape [M, N].
    """
    return torch.diag(torch.ones(weight.shape[0]))

def average(weight):
    """ Initialize the layer by normalizing the weights.

    Args:
        weight (torch.FloatTensor): Matrix of shape [M, N].

    Returns:
        (torch.FloatTensor): Matrix of shape [M, N].
    """
    return weight / weight.sum()
