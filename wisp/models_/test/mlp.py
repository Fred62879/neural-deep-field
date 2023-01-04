
import sys
import torch
import numpy as np
import torch.nn as nn

from wisp.models.test.pe import PE_All
from wisp.models.test.common import MLP_Relu
from wisp.models.test.mfn import FourierNet, GaborNet
from wisp.models.test.siren import Sine_Layer, Linr_Layer


class PEMLP(nn.Module):
    ''' ReLU MLP with positional encoding '''
    def __init__(self, pe_cho, pe_args, mlp_args):
        super(PEMLP, self).__init__()
        self.init_net(pe_cho, pe_args, mlp_args)

    def init_net(self, pe_cho, pe_args, mlp_args):
        self.encd = PE_All(pe_cho, pe_args)
        self.mlp = MLP_Relu(mlp_args)

    ''' @Param
          coords [bsz,(nsmpl,)dim]
        @Return
          output:[bsz,(nsmpl,)dim_out]
    '''
    def forward(self, coords):
        encd_coords = self.encd(coords)
        return self.mlp(encd_coords)

class SIREN(nn.Module):
    # Siren mlp
    def __init__(self, mlp_args):
        super(SIREN, self).__init__()
        self.init_siren(mlp_args)

    def init_siren(self, mlp_args):
        (dim_in, dim_hidden, dim_out, num_hidden_layers, last_linr, \
         first_w0, hidden_w0, coords_scaler, seed, float_tensor) = mlp_args

        self.coords_scaler = coords_scaler

        first_layer = Sine_Layer(dim_in, dim_hidden, is_first=True,
                                 w0=first_w0, seed=seed)
        net= [first_layer]

        for i in range(num_hidden_layers):
            cur_layer = Sine_Layer(dim_hidden, dim_hidden, is_first=False,
                                   w0=hidden_w0, seed=i+1+seed)
            net.append(cur_layer)

        last_seed = num_hidden_layers + 1 + seed
        if last_linr:
           final_layer = Linr_Layer(dim_hidden, dim_out, hidden_w0,
                                     last_seed, float_tensor)
        else:
            final_layer = Sine_Layer(dim_hidden, dim_out, is_first=False,
                                     w0=hidden_w0, seed=last_seed)
        net.append(final_layer)
        self.net = nn.Sequential(*net)

        ''' @Param
              coords [bsz,...,dim_in]
              covar  (None)
        @Return
              output [bsz,...,dim_out]
        '''
    def forward(self, coords):
        return self.net(coords * self.coords_scaler)

class MLP_All(nn.Module):
    ''' Wrapper of mlp supporting all possible mlp choices '''
    def __init__(self, mlp_cho, mlp_args, pe_cho=None, pe_args=None):
        super(MLP_All, self).__init__()
        self.init_mlp(mlp_cho, mlp_args, pe_cho, pe_args)

    def init_mlp(self, mlp_cho, mlp_args, pe_cho, pe_args):

        if mlp_cho == 'relumlp':
            self.mlp = MLP_Relu(mlp_args)

        elif mlp_cho == 'pemlp':
            assert(pe_cho is not None and pe_args is not None)
            self.mlp = PEMLP(pe_cho, pe_args, mlp_args)

        elif mlp_cho == 'siren':
            self.mlp = SIREN(mlp_args)

        elif mlp_cho == 'fourier_mfn':
            self.mlp = FourierNet(mlp_args)

        elif mlp_cho == 'gabor_mfn':
            self.mlp = GaborNet(mlp_args)
        else:
            raise Exception('Unsupported mlp choice')

    def forward(self, input):
        return self.mlp(input)

'''
class MLP_All(nn.Module):
    def __init__(self, mlp_cho, mlp_args):
        super(MLP_All, self).__init__()
        self.init_mlp(mlp_args)

    def init_mlp(self, mlp_args):
        coord_multiplier = args.coord_multiplier

        if mlp_cho == 'pemlp':
            assert(coord_multiplier == 1)
            self.mlp = PEMLP(mlp_args)

        elif mlp_cho == 'siren':
            self.mlp = SIREN(mlp_args)

        elif mlp_cho == 'fourier_mfn':
            assert(coord_multiplier == 1)
            self.mlp = FourierNet(mlp_args)
                #(args.mlp_dim_in, args.mlp_dim_hidden, 1, args.mfn_omega, args.mfn_num_layers,
                # args.img_sz, args.mfn_w_scale, args.mfn_bias, args.mfn_output_act,
                # args.verbose)

        elif mlp_cho == 'gabor_mfn':
            assert(coord_multiplier == 1)
            self.mlp = GaborNet(mlp_args)
                #(args.mlp_dim_in, args.mlp_dim_hidden, 1, args.mfn_omega, args.mfn_num_layers,
                # args.img_sz, args.mfn_w_scale, args.mfn_alpha, args.mfn_beta,
                # args.mfn_bias, args.mfn_output_act, args.verbose)
        else:
            raise Exception('Unsupported mlp choice')

    def forward(self, input):
        return self.mlp(input)
'''
