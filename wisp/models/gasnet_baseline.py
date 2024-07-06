import torch

from torch import nn
from wisp.utils.common import init_redshift_bins

class GasNet(nn.Module):
    def __init__(self, **kwargs):
        super(GasNet, self).__init__()

        self.kwargs = kwargs
        self._print = False
        self.redshift_model_method = kwargs["redshift_model_method"]
        self.regress_redshift = self.redshift_model_method == "regression"
        self.classify_redshift = self.redshift_model_method == "classification"

        self.init_net()

    def init_net(self):
        reduced_num_samples = 1
        if self.regress_redshift:
            out_channels = 1
        elif self.classify_redshift:
            num_redshift_bins = len(init_redshift_bins(**self.kwargs))
            out_channels = num_redshift_bins
        else: raise ValueError()

        self.net = nn.Sequential(
            Block(1, 16, 5, 5, 2),
            Block(16, 32, 5, 5, 2),
            Block(32, 64, 5, 5, 2),
            Block(64, 128, 5, 5, 2),
            Block(128, 256, 5, 5, 2),
            Block(256, 512, 5, 5, 2),
            Dense(reduced_num_samples, 512, out_channels))

    def forward(
            self, channels, spectra_mask, spectra_source_data,
            selected_ids=None, idx=None
    ):
        """
        @Params
        @Return
           redshift: [bsz]
           redshift_logits: [bsz,nbins]
        """
        ret = {}
        flux = spectra_source_data[:,1]
        flux = (flux * spectra_mask)[:,None] # [bsz,1,nsmpls]
        output = self._forward(flux)
        if self.redshift_model_method == "regression":
            ret["redshift"] = output[...,0]
        elif self.redshift_model_method == "classification":
            ret["redshift_logits"] = output
        else: raise ValueError()
        return ret

    def _forward(self, x):
        x = self.net(x)
        return x

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size, stride):
        super(Block, self).__init__()

        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(pool_size, stride=1, padding=2))

    def forward(self, x):
        x = self.net(x) + self.skip(x)
        return x

class Dense(nn.Module):
    def __init__(self, num_samples, in_channels, out_channels):
        super(Dense, self).__init__()
        self.dense = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_channels * num_samples, 256),
            nn.Linear(256, out_channels))

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.dense(x)
        return x
