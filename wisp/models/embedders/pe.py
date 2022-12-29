
import torch
import torch.nn as nn
import logging as log


class PE(nn.Module): # pe0
    ''' conventional positional encoding
        P: |1 0 0 2 0 0 3 0 0 ...|
           |0 1 0 0 2 0 0 3 0 ...|
           |0 0 1 0 0 2 0 0 3 ...|
        min_deg, min_deg+1, ..., max_deg

        Input:  [bsz,(nsmpl,)dim]
        Output: [bsz,(nsmpl,)2*pe_dim]
    '''
    def __init__(self, pe_args):
        super(PE, self).__init__()

        (min_deg, max_deg, float_tensor, verbose) = pe_args

        scales = torch.arange(min_deg, max_deg).type(float_tensor)
        scales *= 2 * torch.pi
        self.scales = scales[...,None] # [pe_dim,1]
        if verbose:
            print('    NeRF PE with with deg {}~{}'.format(min_deg, max_deg))

    def forward(self, input):
        (coords, _) = input
        shape = list(coords.shape[:-1])+[-1]
        coords_lifted = (coords[...,None,:] * self.scales).view(shape)
        return torch.cat((torch.sin(coords_lifted), torch.cos(coords_lifted)), dim=-1)

class IntePE(nn.Module): # pe1
    ''' integrated PE as in mip-nerf
        P: |1 0 0 2 0 0 3 0 0 ...|
           |0 1 0 0 2 0 0 3 0 ...|
           |0 0 1 0 0 2 0 0 3 ...|
        input:  coords [bsz,(nsmpl,)dim]
                covar  [bsz,(nsmpl,)3]
        output: [bsz,(nsmpl,)pe_dim]
        pe_dim  2*dim*3*(max_deg - min_deg)
    '''
    def __init__(self, min_deg, max_deg, float_tensor, verbose):
        super(IntePE, self).__init__()

        scales = torch.arange(min_deg, max_deg).type(float_tensor)
        self.scales = scales[...,None] # [pe_dim,1]
        self.var_scales = self.scales**2
        if verbose:
            print('    Integrated PE with with deg {}~{}'.format(min_deg, max_deg))

    def encd_covar(self, covar):
        covar_lifted = covar[...,None,:]*self.var_scales # [bsz,(nsmpl,)~,dim]
        covar_lifted = (covar_lifted).flatten(-2,-1)
        return torch.exp(-.5 * covar_lifted) # [bsz,(nsmpl,)pe_dim]

    def forward(self, input):
        (coords, covar) = input
        exp_var = self.encd_covar(covar)
        # coords lifted to pe basis [...,pe_dim]
        shape = list(coords.shape[:-1])+[-1]
        coords_lifted = (coords[...,None,:] * self.scales).view(shape)
        return torch.cat( (torch.sin(coords_lifted) * exp_var,
                           torch.cos(coords_lifted) * exp_var ), dim=-1)

class PE_MixDim(nn.Module): # pe2
    ''' positional encoding where encoding of dims are added up
        P: |1 2 3 ...|
           |1 2 3 ...|
           |1 2 3 ...|
           min_deg, .min_deg+1, ... max_deg

        input:  [bsz,(nsmpl,)dim]
        output: [bsz,(nsmpl,)2*pe_dim]
    '''
    def __init__(self, pe_args):
        super(PE_MixDim, self).__init__()

        (dim, min_deg, max_deg, float_tensor, verbose) = pe_args

        self.dim = dim
        scales = torch.arange(min_deg, max_deg).type(float_tensor)
        self.scales = scales[None,...] # [1,pe_dim]
        if verbose:
            print('    mix dim PE with deg {}~{}'.format(min_deg, max_deg))

    def forward(self, input):
        (coords, _) = input
        encd_coords = coords[...,0:1]@self.scales
        for i in range(1,self.dim):
            encd_coords += coords[...,i:i+1]@self.scales
        return torch.cat((torch.sin(encd_coords),
                          torch.cos(encd_coords)), dim=-1)

class IntePE_MixDim(nn.Module): # pe3
    ''' integrated PE where encoding of dims are added up
        P: |1 2 3 ...|
           |1 2 3 ...|
           |1 2 3 ...|

        omegas: [dim], diagonal of covariance matrix of each coord
                set as hyperparameters and being same for all coords
        input:  [bsz,dim]/[bsz,nsmpl,dim]
        output: [bsz,pe_dim]/[bsz,nsmpl,pe_dim]
        pe_dim  dim*(max_deg - min_deg)
    '''
    def __init__(self, dim, min_deg, max_deg, covar, float_tensor, verbose):
        super(IntePE_MixDim, self).__init__()

        self.dim = dim
        scales = torch.arange(min_deg, max_deg)\
                      .type(float_tensor)

        self.exp_var = self.encd_covar(scales, covar)
        self.scales = scales[None,...]
        if verbose:
            print('    mix deg Integrated PE with with deg {}~{}'.format(min_deg, max_deg))

    def encd_covar(self, scales, covar):
        covar_lifted = scales[None,:].tile(len(covar),1)
        for i in range(self.dim):
            covar_lifted[i] *= covar[i]
        return torch.exp(-.5*torch.sum(covar_lifted, dim=0))

        #var_lifted = (covar * scales**2).flatten()
        #print('var lifted', var_lifted)
        #return torch.exp(-.5 * var_lifted)

    def forward(self, input):
        (coords, covar) = input
        # coord lifted to pe basis [...,pe_dim]
        encd_coords = coords[...,0:1]@self.scales
        for i in range(1,self.dim):
            encd_coords += coords[...,i:i+1]@self.scales
        exp_covar = self.encd_covar(covar)
        return torch.cat((torch.sin(encd_coords) * exp_var,
                          torch.cos(encd_coords) * exp_var), dim=-1)

class PE_Rand(nn.Module):
    ''' randomized positional encoding
        now this is a duplicate of rand_gaus
        P = |0.31 0.23 0.11 ... |
            |0.12 0.28 0.29 ... |
            |0.88 0.42 0.98 ... |
        P = 2*pi*factor * P
    '''
    def __init__(self, pe_args):
        super(PE_Rand, self).__init__()

        (dim, pe_dim, pe_omega, pe_bias, float_tensor, verbose, seed) = pe_args

        self.dim = dim
        self.bias = pe_bias
        self.pe_dim = pe_dim
        self.omega = pe_omega
        self.float_tensor = float_tensor

        # here mapping is initialized as an linr layer while inte pe rand has a tensor
        self.mappings = [self.init_mapping(i + seed) for i in range(self.dim)]

        if verbose: print(f'randomized PE with {pe_dim} dim and omega {pe_omega}')

    def init_mapping(self, seed):
        mapping = nn.Linear(1, self.pe_dim, bias=self.bias)
        mapping.weight = nn.Parameter(self.randmz_weights(seed), requires_grad=False)
        return mapping

    def randmz_weights(self, seed):
        torch.manual_seed(seed)
        weight = torch.randn(self.pe_dim) # ~N(0,1)
        weight = 2 * torch.pi * self.omega * weight
        return weight.reshape((self.pe_dim, 1)).type(self.float_tensor)

    ''' input:  [bsz,(nsmpl,)dim]
        output: [bsz,(nsmpl,)pe_dim]
    '''
    def forward(self, input):
        (coords, _) = input
        encd_coords = self.mappings[0](coords[...,0:1])
        for i in range(1,self.dim):
            encd_coords += self.mappings[i](coords[...,i:i+1])
        return torch.cos(encd_coords)

class IntePERand(nn.Module):
    ''' integrated randomized positional encoding
        assuming each dim is independent from each other
        P |0.31 0.23 0.11 ... |
          |0.12 0.28 0.29 ... |
          |0.88 0.42 0.98 ... |

        input:  [bsz,(nsmpl,)dim]
        output: [bsz,(nsmpl,)pe_dim]
    '''
    def __init__(self, dim, pe_dim, omega, float_tensor, verbose, seed):
        super(IntePERand, self).__init__()

        self.dim = dim
        self.pe_dim = pe_dim
        self.omega = omega
        self.float_tensor = float_tensor

        # here mapping is initialized as an tensor while pe rand has a linr layer
        self.mappings = torch.stack([self.randmz_weights(i, i + seed)
                                     for i in range(self.dim)]) # [dim,pe_dim]
        if verbose:
            print('Integrated Randomized PE with {} dim'.format(self.pe_dim))

    def encd_covar(self, covar):
        covar_lifted = self.mappings**2
        for i in range(self.dim):
            covar_lifted[i] *= covar[i]
        return torch.exp(-.5*torch.sum(covar_lifted, dim=0))

    def randmz_weights(self, seed=0):
        torch.manual_seed(seed)
        weight = torch.randn(self.pe_dim)
        weight = 2 * torch.pi * self.omega * weight
        return weight.type(self.float_tensor)

    def forward(self, input):
        (coords, covar) = input
        encd_coords = coords[...,0:1]@self.mappings[0:1]
        for i in range(1,self.dim):
            encd_coords += coords[...,i:i+1]@self.mappings[i:i+1]
        exp_var = self.encd_covar(covar)
        return torch.cos(encd_coords) * exp_var

class RandGaus(nn.Module):
    ''' randomized gaussian as in fourier features
        P = |0.31 0.23 0.11 ... |
            |0.12 0.28 0.29 ... |
            |0.88 0.42 0.98 ... |
        P = 2*pi*factor * P
    '''
    def __init__(self, pe_args):
        super(RandGaus, self).__init__()
        (dim, pe_dim, omega, sigma, pe_bias, seed, verbose) = pe_args

        self.dim = dim
        self.omega = omega
        self.sigma = sigma
        self.bias = pe_bias
        self.pe_dim = pe_dim//2

        self.mappings = nn.ModuleList([self.init_mapping(i + seed)
                                       for i in range(dim)])
        if verbose:
            print(f'= randmized Gaussian with {pe_dim} dim and sigma {sigma}')

    def init_mapping(self, seed):
        mapping = nn.Linear(1, self.pe_dim, bias=self.bias)
        mapping.weight = nn.Parameter(self.randmz_weights(seed),requires_grad=False)
        #print('weight', mapping.weight.isnan().any())
        #print(mapping.weight)
        return mapping

    def randmz_weights(self, seed):
        torch.manual_seed(seed)
        weight = torch.empty(self.pe_dim).normal_(mean=0.,std=self.sigma**2)
        weight = 2 * torch.pi * self.omega * weight
        weight = torch.FloatTensor(weight[:,None])
        return weight

    ''' input:  [bsz,(nsmpl,)dim]
        output: [bsz,(nsmpl,)pe_dim]
    '''
    def forward(self, coords):
        encd_coords = self.mappings[0](coords[...,0:1])
        for i in range(1,self.dim):
            encd_coords += self.mappings[i](coords[...,i:i+1])
        return torch.cat((torch.cos(encd_coords),
                          torch.sin(encd_coords)), dim=-1)

class InteRandGaus(nn.Module):
    ''' integrated randomized Gaussian
        assuming each dim is independent from each other

        input:  coords [bsz,(nsmpl,)dim]
                covar  [dim], variance for each dim
        output: [bsz,(nsmpl,)pe_dim]
    '''
    def __init__(self, dim, pe_dim, gaus_sigma, pe_factor, float_tensor, covar, verbose):
        super(InteRandGaus, self).__init__()

        self.dim = dim
        self.pe_dim = pe_dim//2
        self.sigma = gaus_sigma
        self.factor = pe_factor
        self.float_tensor = float_tensor

        self.mappings = torch.stack([self.randmz_weights(i)
                                     for i in range(dim)]) # [dim,pe_dim]
        self.exp_var = self.encd_covar(covar)
        if verbose:
            print('    Integrated Randomized Gaus with {} dim'.format(self.pe_dim))

    def encd_covar(self, covar):
        covar_lifted = self.mappings**2
        for i in range(self.dim):
            covar_lifted[i] *= covar[i]
        return torch.exp(-.5*torch.sum(covar_lifted, dim=0)).type(self.float_tensor)

    def randmz_weights(self, seed):
        torch.manual_seed(seed)
        weight = torch.empty(self.pe_dim).normal_(mean=0.,std=self.sigma**2)
        return 2 * torch.pi * weight.type(self.float_tensor)

    def forward(self, input):
        encd_coords = input[...,0:1]@self.mappings[0:1]
        for i in range(1,self.dim):
            encd_coords += input[...,i:i+1]@self.mappings[i:i+1]
        return torch.cat((torch.cos(encd_coords)*self.exp_var,
                          torch.sin(encd_coords)*self.exp_var), dim=-1)

class RandGausLinr(nn.Module): # pe8
    ''' randomized gaussian as in fourier features where P is multiplied with (1,...n)
        P = |0.31 0.23 0.11 ... |   | 1 2 3 ... |
            |0.12 0.28 0.29 ... | * | 1 2 3 ... |
            |0.88 0.42 0.98 ... |   | 1 2 3 ... |

        input:  [bsz,(nsmpl,)3]
        output: [bsz,(nsmpl,)pe_dim]
    '''
    def __init__(self, pe_args):
        super(RandGausLinr, self).__init__()

        #(dim, pe_dim, sigma, omega, pe_bias, float_tensor, verbose) = pe_args
        (dim, pe_dim, sigma, omega, pe_bias, verbose) = pe_args

        self.dim = dim
        self.sigma = sigma
        self.omega = omega
        self.bias = pe_bias
        self.pe_dim = pe_dim//2
        #self.float_tensor = float_tensor

        #self.scale = torch.arange(start=0,end=self.pe_dim)
        self.mappings = nn.ModuleList([self.init_mapping(i) for i in range(dim)])
        if verbose:
            log.info(f"linear transformed Randmized Gaussian with {pe_dim} \
            dim and sigma {sigma} omega {omega}")

    def init_mapping(self, seed):
        mapping = nn.Linear(1, self.pe_dim, bias=self.bias)
        mapping.weight = nn.Parameter(self.randmz_weights(), requires_grad=False)
        #print('weight', mapping.weight.isnan().any())
        #print(mapping.weight)
        return mapping

    def randmz_weights(self):
        weight = torch.empty(self.pe_dim).normal_(mean=0.,std=self.sigma**2)
        weight = 2 * torch.pi * weight * self.omega #self.scale
        weight = torch.FloatTensor(weight.reshape((self.pe_dim, 1)))
        return weight

    def forward(self, input): # [bsz,...,input_dim]
        encd_input = self.mappings[0](input[...,0:1])
        for i in range(1,self.dim):
            encd_input += self.mappings[i](input[...,i:i+1])
        #print('encd_input', encd_input.isnan().any())
        return torch.cat((torch.cos(encd_input),
                          torch.sin(encd_input)), dim=-1)

class RandIGausLinr(nn.Module): # pe9
    ''' integrated RandGausLinr, assuming each dim is indep from each other
        P = |0.31 0.23 0.11 ... |   | 1 2 3 ... |
            |0.12 0.28 0.29 ... | * | 1 2 3 ... |
            |0.88 0.42 0.98 ... |   | 1 2 3 ... |

        input:  coords [bsz,(nsmpl,)dim]
                covar  [bsz,(nsmpl,)dim]
        output: [bsz,(nsmpl,)pe_dim]
    '''
    def __init__(self, dim, pe_dim, sigma, omega, pe_bias, float_tensor, verbose):
        super(RandIGausLinr, self).__init__()

        self.dim = dim
        self.sigma = sigma
        self.omega = omega
        self.bias = pe_bias
        self.pe_dim = pe_dim//2
        self.float_tensor = float_tensor

        self.scale = torch.arange(start=0,end=self.pe_dim) #.type(float_tensor)
        self.mappings = torch.stack([self.randmz_weights(i)
                                     for i in range(dim)]) # [dim,pe_dim]
        self.var_mappings = self.mappings**2
        if verbose:
            print('    Linr transformed Integrated Randmized Gaussian with {} dim and sigma {} omega {}'\
                  .format(pe_dim, sigma, omega))

    def randmz_weights(self, seed):
        torch.manual_seed(seed)
        weight = torch.empty(self.pe_dim).normal_(mean=0.,std=self.sigma**2)
        #print('pre', weight)
        weight = 2 * torch.pi * weight * self.scale
        #print('post', weight)
        return weight.type(self.float_tensor)

    def encd_covar(self, covar):
        #covar_lifted = self.mappings**2
        #for i in range(self.dim):
        #    covar_lifted[i] *= covar[i]
        #return torch.exp(-.5*torch.sum(covar_lifted, dim=0)).type(self.float_tensor)
        covar_lifted = covar@self.var_mappings # [bsz,(nsmpl,)pe_dim]
        return torch.exp(-.5 * covar_lifted)

    def forward(self, input):
        (coords, covar) = input
        encd_coords = coords[...,0:1]@self.mappings[0:1]
        for i in range(1,self.dim):
            encd_coords += coords[...,i:i+1]@self.mappings[i:i+1]
        exp_var = self.encd_covar(covar)
        return torch.cat((torch.cos(encd_coords) * exp_var,
                          torch.sin(encd_coords) * exp_var), dim=-1)

class PE_All(nn.Module):
    ''' Positional encoding, supports multiple different versions
        Input:  coords [bsz,(nsmpl,)dim]
                covar [bsz,(nsmpl,)3] / None
        Output: [bsz,(nsmpl,)dim_out]
    '''
    def __init__(self, pe_cho, pe_args):
        super(PE_All, self).__init__()

        if pe_cho == 'pe':
            self.encd = PE(pe_args)
        elif pe_cho == 'pe_mix_dim':
            self.encd = PE_Mix_Dim(pe_args)
        elif pe_cho == 'pe_rand':
            self.encd = PE_Rand(pe_args)
        elif pe_cho == 'rand_gaus':
            self.encd = Rand_Gaus(pe_args)
        elif pe_cho == 'rand_gaus_linr':
            self.encd = Rand_GausLinr(pe_args)
        else:
            raise Exception('Unsupported positional encoding pe choice')

    def forward(self, input):
        encd_coords = self.encd(input) # [bsz,(nsmpl,)pe_dim]
        return encd_coords

'''
class PE_All(nn.Module):
     Positional encoding, supports multiple different versions
        Input:  coords [bsz,(nsmpl,)dim]
                covar [bsz,(nsmpl,)3] / None
        Output: [bsz,(nsmpl,)dim_out]

    def __init__(self, dim_in, pe_cho, float_tensor, pe_dim=None, pe_min_deg=None,
                 pe_max_deg=None, pe_bias=False, pe_omega=None, pe_factor=None,
                 gaus_sigma=None, gaus_omega=None, ipe_covar=None, ipe=False, verbose=False):

        super(PE_All, self).__init__()

        if pe_cho == 'pe':
            (pe_min_deg, pe_max_deg, float_tensor, verbose) = pe_args
            if ipe:
                self.encd = IntePE\
                    (pe_min_deg, pe_max_deg, float_tensor, verbose)
            else:
                self.encd = PE(pe_min_deg, pe_max_deg, float_tensor, verbose)

        elif pe_cho == 'pe_mix_dim':
            (dim_in, pe_min_deg, pe_max_deg, ipe_covar, float_tensor, verbose) = pe_args
            if ipe:
                self.encd = IntePE_MixDim\
                    (dim_in, pe_min_deg, pe_max_deg, ipe_covar, float_tensor, verbose)
            else:
                self.encd = PE_Mix_Dim(dim, pe_min_deg, pe_max_deg, float_tensor, verbose)

        elif pe_cho == 'pe_rand': # duplicate of rand_gaus
            (dim_in, pe_dim, pe_omega, pe_factor, pe_bias, float_tensor, verbose) = pe_args
            if ipe:
                self.encd = IntePERand\
                    (dim_in, pe_dim, pe_omega, float_tensor, ipe_covar, verbose)
            else:
                self.encd = PERand\
                    (dim_in, pe_dim, pe_omega, pe_factor, pe_bias, float_tensor, verbose)

        elif pe_cho == 'rand_gaus':
            (dim_in, pe_dim, gaus_sigma, pe_factor, pe_bias, float_tensor, verbose) = pe_args
            if ipe:
                self.encd = InteRandGaus\
                    (dim_in, pe_dim, gaus_sigma, pe_factor,
                     float_tensor, ipe_covar, verbose)
            else:
                self.encd = RandGaus\
                    (dim_in, pe_dim, gaus_sigma, pe_factor,
                     pe_bias, float_tensor, verbose)
        elif pe_cho == 'rand_gaus_linr':
            (dim_in, pe_dim, gaus_sigma, gaus_omega, pe_bias, float_tensor, verbose) = pe_args
            if ipe:
                self.encd = RandIGausLinr\
                    (dim_in, pe_dim, gaus_sigma, gaus_omega,
                     pe_bias, float_tensor, verbose)
            else:
                self.encd = RandGausLinr\
                    (dim_in, pe_dim, gaus_sigma, gaus_omega,
                     pe_bias, float_tensor, verbose)
        else:
            raise Exception('Unsupported positional encoding pe choice')

    def forward(self, input):
        encd_coords = self.encd(input) # [bsz,(nsmpl,)pe_dim]
        return encd_coords
'''
