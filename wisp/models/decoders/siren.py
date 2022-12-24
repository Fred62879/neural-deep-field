
import torch
import numpy as np
import torch.nn as nn


class Siren(nn.Module):

    def __init__(self, dim_in, dim_out, num_layers, dim_hidden, first_w0,
                 hidden_w0, seed, coords_scaler, last_linear=False):

        super(Siren, self).__init__()

        self.coords_scaler = coords_scaler

        # init siren
        first_layer = Sine_Layer(dim_in, dim_hidden, is_first=True, w0=first_w0, seed=seed)
        net= [first_layer]

        for i in range(num_layers):
            cur_layer = Sine_Layer(dim_hidden, dim_hidden, is_first=False,
                                   w0=hidden_w0, seed=i + 1 + seed)
            net.append(cur_layer)

        last_seed = num_layers + 1 + seed
        if last_linear:
            final_layer = Linr_Layer(dim_hidden, dim_out, hidden_w0, last_seed, float_tensor)
        else:
            final_layer = Sine_Layer(dim_hidden, dim_out, is_first=False,
                                     w0=hidden_w0, seed=last_seed)
        net.append(final_layer)
        self.net = nn.Sequential(*net)

        """ @Param
              coords [bsz,...,dim_in]
            @Return
              output [bsz,...,dim_out]
        """
    def forward(self, coords):
        print(coords.isnan().any())
        ret = self.net(coords * self.coords_scaler)
        print(ret.isnan().any())
        return ret

class Sine_Layer(nn.Module):
    """ One mlp layer of siren
        Input:  [bsz,nsmpl,inchl],  torch.(cuda.)double
        Output: [bsz,nsmpl,outchl], torch.(cuda.)float
    """

    def __init__(self, dim_in, dim_out, bias=True, is_first=False, w0=30, seed=0):
        super(Sine_Layer, self).__init__()

        self.w0 = w0
        self.dim_in = dim_in
        self.dim_out= dim_out
        self.is_first = is_first

        self.linear = nn.Linear(dim_in, dim_out, bias=bias)
        self.init_weights(seed)

    def init_weights(self, seed):
        std = 1/self.dim_in if self.is_first \
            else np.sqrt(6 / self.dim_in) / self.w0
        w = torch.empty((self.dim_out, self.dim_in), dtype=torch.float)\
                 .uniform_(-std, std)
        self.linear.weight = nn.Parameter(w)

    def forward(self, input):
        return torch.sin(self.w0 * self.linear(input))

class Linr_Layer(nn.Module):

    def __init__(self, dim_hidden, dim_out, hidden_w0, seed, float_tensor):
        super(Linr_Layer, self).__init__()

        self.layer = nn.Linear(dim_hidden, dim_out)
        std = np.sqrt(6 / dim_hidden) / hidden_w0
        torch.manual_seed(seed)
        w = torch.empty((dim_out, dim_hidden)).\
            uniform_(-std, std).type(float_tensor)
        self.layer.weight = nn.Parameter(w)

    def forward(self, input):
        return self.layer(input)

"""
# feature map plotting
class SIREN3D(nn.Module):
    def __init__(self, inchl, mlp_dim_hidden, outchl, num_hidden_layers, nsmpl, lambdas, transms,
                 last_linear=False, first_w0=30., hidden_w0=30., has_dot_layer=True):

        super(SIREN3D, self).__init__()

        self.nsmpl = nsmpl
        self.inchl = inchl
        self.mlp_dim_hidden = mlp_dim_hidden
        self.outchl = outchl
        self.num_hidden_layers = num_hidden_layers

        self.lambdas = lambdas # [nsmpl,bsz,1]
        self.transms = transms # [nsmpl,nchl]

        net = []

        # siren layers
        net.append(SineLayer(inchl, mlp_dim_hidden, is_first=True, w0=first_w0)) # add first layer
        for i in range(num_hidden_layers): # add hidden layers
            net.append(SineLayer(mlp_dim_hidden, mlp_dim_hidden, is_first=False, w0=hidden_w0))
        if last_linear: # add last layer
            final_linear = nn.Linear(mlp_dim_hidden, 1)
            std = np.sqrt(6 / mlp_dim_hidden) / hidden_w0
            w = torch.empty((1, mlp_dim_hidden), dtype=torch.float).uniform_(-std, std)
            final_linear.weight = nn.Parameter(w)
            net.append(final_linear)
        else:
            net.append(SineLayer(mlp_dim_hidden, 1, is_first=False, w0=hidden_w0))

        # dot product layer
        if has_dot_layer:
            net.append(DotProductEin(transms))

        self.net = nn.Sequential(*net)

    # input:  bsz x inchl, (inchl is 2), torch.(cuda.)double
    # output: bsz x outchl, torch.(cuda.)float
    def forward(self, input):
        input3d = torch.tile(input, (self.nsmpl, 1))
        input3d = input3d.view((self.nsmpl,-1, 2))      # [nsmpl,bsz,2]
        input3d = torch.cat((input3d, self.lambdas[:,:input3d.shape[1]]), 2) # [nsmpl,bsz,3]
        return self.net(input3d)

    # plot distribution of output of each layer for selected neurons
    # input n x 2, float tensor
    def plotDistriAllLayers(self, input, fn_prefx, neuron_chos):
        input = torch.tile(input, (self.nsmpl, 1))
        input = input.view((self.nsmpl,-1, 2))      # [nsmpl,bsz,2]
        input = torch.cat((input, self.lambdas), 2) # [nsmpl,bsz,3]

        for i in range(self.num_hidden_layers+2):

            subnet = self.net[:i+1]
            with torch.no_grad():
                out = subnet(input) # n x 512
            x = out.detach().cpu().numpy()

            if i == self.num_hidden_layers + 1:
                neuron_chos = [0] #,1,2,3,4]

            for neuron in neuron_chos:
                cx = x[:,:,neuron]
                #np.save(fn_prefx + str(neuron) + ".npy", cx)
                plotImageHist(cx.flatten(), fn=fn_prefx + "/L" + str(i) + "_" + str(neuron) + "_activ.png")

    # plot distribution of output (before activation) of each layer for given neurons
    # input: n x 2
    def plotDistriAllLayersInactv(self, input, fn_prefx, neuron_chos):
        input = torch.tile(input, (self.nsmpl, 1))
        input = input.view((self.nsmpl,-1, 2))      # [nsmpl,bsz,2]
        input = torch.cat((input, self.lambdas), 2) # [nsmpl,bsz,3]

        for i in range(self.num_hidden_layers+2):
            cur_input = input
            print(cur_input.shape)

            # get activated output from i-1\th layer
            if i: # not fisrt layer
                subnet_1 = self.net[:i]
                if torch.cuda.is_available():
                    subnet_1.cuda()

                with torch.no_grad():
                    cur_input = subnet_1(cur_input) # [nsmpl,bsz,512]

            # get weight for ith linear layer (assuming we have bias)
            weight = None
            for j, w in enumerate(self.net.parameters()):
                if j%2 == 0 and j/2 == i:
                    weight = w
                    break

            inchl = self.inchl if i == 0 else self.mlp_dim_hidden
            outchl = self.mlp_dim_hidden if i != self.num_hidden_layers+1 else 1
            #outchl = self.mlp_dim_hidden if i < self.num_hidden_layers else 5 if i == self.num_hidden_layers else 1
            subnet = nn.Linear(inchl, outchl)
            subnet.weight = weight
            if torch.cuda.is_available():
                subnet.cuda()

            with torch.no_grad():
                out = subnet(cur_input) # [nsmpl,bsz,outchl]
            x = out.detach().cpu().numpy()

            if i == self.num_hidden_layers + 1:
                neuron_chos = [0] #,1,2,3,4]

            for neuron in neuron_chos:
                cx = x[:,:,neuron]
                plotImageHist(cx.flatten(), fn=fn_prefx + "/L" + str(i) + "_" + str(neuron) + "_inactiv.png")

"""
