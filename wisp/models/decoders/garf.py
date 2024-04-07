
import torch
import torch.nn as nn
import logging as log

from wisp.utils.common import set_seed


class Garf(nn.Module):
    def __init__(
            self, input_dim, output_dim, num_hidden_layers,
            hidden_dim, gaussian_sigma, skip_all_layers
    ):
        super(Garf, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_hidden_layers
        self.gaussian_sigma = gaussian_sigma

        # self.skip = kwargs["wave_encoder_skip_layers"]
        # self.output_dim = kwargs["wave_embed_dim"]
        # self.hidden_dim = kwargs["wave_encoder_hidden_dim"]
        # self.num_layers = kwargs["wave_encoder_num_hidden_layers"]
        # self.gaussian_sigma = kwargs["wave_encoder_gaussian_sigma"]
        # if kwargs["wave_encoder_skip_all_layers"]:
        #     self.skip = np.arange(self.num_layers)

        if skip_all_layers:
            self.skip = np.arange(self.num_layers)
        self.perform_skip = len(self.skip) > 0
        self.init_net()

    def init_net(self):
        if self.num_layers == 0:
            self.layer = torch.nn.Linear(self.input_dim, self.output_dim)
        else:
            self.first_layer = torch.nn.Linear(self.input_dim, self.hidden_dim)
            self.layers = []
            for i in range(self.num_layers):
                if i not in self.skip:
                    cur_layer = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
                else: cur_layer = torch.nn.Linear(self.hidden_dim + self.hidden_dim, self.hidden_dim)
                self.layers.append(cur_layer)
            self.layers = torch.nn.ModuleList(self.layers)
            self.last_layer = torch.nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, coords):
        if self.num_layers == 0:
            # feat = self.layer(coords)
            feat = self.gaussian(self.layer(coords))
        else:
            feat = self.first_forward(coords)
            if self.perform_skip: points_enc = feat
            for i, l in enumerate(self.layers):
                feat = l(feat)
                feat = self.gaussian(feat)
                if i in self.skip:
                    feat = torch.cat([points_enc, feat], -1)
            feat = self.last_layer(feat)
        return feat

    def first_forward(self, x):
        x_ = self.first_layer(x)
        mu = torch.mean(x_, axis = -1).unsqueeze(-1)
        out = (-0.5*(mu - x_)**2 / self.gaussian_sigma**2).exp()
        return out

    def gaussian(self, x):
        out = (-0.5*(x)**2 / self.gaussian_sigma**2).exp()
        return out
