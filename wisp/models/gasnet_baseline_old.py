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
        self.block1 = Block(1, 16, 5, 5, 2)
        self.block2 = Block(16, 32, 5, 5, 2)
        self.block3 = Block(32, 64, 5, 5, 2)
        self.block4 = Block(64, 128, 5, 5, 2)
        self.block5 = Block(128, 256, 5, 5, 2)
        self.block6 = Block(256, 512, 5, 5, 2)

        reduced_num_samples = 1
        if self.regress_redshift:
            out_channels = 1
        elif self.classify_redshift:
            num_redshift_bins = len(init_redshift_bins(**self.kwargs))
            out_channels = num_redshift_bins
        else: raise ValueError()
        self.dense = Dense(reduced_num_samples, 512, out_channels)

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
        if self._print: print(x.shape)
        x = self.block1(x)
        if self._print: print('1', x.shape)
        x = self.block2(x)
        if self._print: print('2', x.shape)
        x = self.block3(x)
        if self._print: print('3', x.shape)
        x = self.block4(x)
        if self._print: print('4', x.shape)
        x = self.block5(x)
        if self._print: print('5', x.shape)
        x = self.block6(x)
        if self._print: print('6', x.shape)
        x = self.dense(x)
        if self._print: print('dense', x.shape)
        return x

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size, stride):
        super(Block, self).__init__()
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride)
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, stride=stride)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(pool_size, stride=1, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_skip = self.skip(x)
        # print('skip', x_skip.shape)
        x = self.conv1(x)
        # print('conv1', x.shape)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        # print('conv2', x.shape)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        # print('pool', x.shape)
        x += x_skip
        return x

class Dense(nn.Module):
    def __init__(self, num_samples, in_channels, out_channels):
        super(Dense, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(in_channels * num_samples, 256)
        self.dense2 = nn.Linear(256, out_channels)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        # print('flatten', x.shape)
        x = self.dropout(x)
        x = self.dense1(x)
        # print('dense1', x.shape)
        x = self.dense2(x)
        # print('dense2', x.shape)
        return x
