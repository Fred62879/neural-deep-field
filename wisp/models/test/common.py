
import torch
import torch.nn as nn

class MLP_Relu(nn.Module):
    def __init__(self, mlp_args):
        super(MLP_Relu, self).__init__()

        (dim_in, dim_hidden, dim_out, num_hidden_layers, seed) = mlp_args

        def block(dim_in, dim_out, seed, activate=True):
            torch.manual_seed(seed)
            block = [ nn.Linear(dim_in, dim_out) ]
            if activate:
                block.append(nn.ReLU(inplace=True))
            return block

        net = []
        net.extend(block(dim_in, dim_hidden, seed))
        for i in range(num_hidden_layers):
            net.extend(block(dim_hidden, dim_hidden, seed + i + 1))
        net.extend(block(dim_hidden, dim_out, seed + num_hidden_layers + 1, False))
        self.model = nn.Sequential(*net)

    # input:  [bsz,(nsmpl,)pe_dim]
    # output: [bsz,(nsmpl,)num_bands]
    def forward(self, coords):
        return self.model(coords)

class Normalization(nn.Module):
    def __init__(self, norm_cho):
        super(Normalization, self).__init__()
        self.init_norm_layer(norm_cho)

    def init_norm_layer(self, norm_cho):
        if norm_cho == 'identity':
            self.norm = nn.Identity()
        elif norm_cho == 'arcsinh':
            self.norm = Fn(torch.arcsinh)
        elif norm_cho == 'sinh':
            self.norm = Fn(torch.sinh)
        else:
            raise Exception('Unsupported normalization choice')

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
