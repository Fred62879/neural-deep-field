
import torch
import torch.nn as nn
import torch.nn.functional as F

from wisp.utils import PerfTimer
from wisp.utils.common import init_redshift_bins

from wisp.models.decoders import BasicDecoder
from wisp.models.embedders.encoder import Encoder

class SpectraBaseline(nn.Module):
    def __init__(self, **kwargs):
        super(SpectraBaseline, self).__init__()
        assert kwargs["model_redshift"], "we must model redshift during pretrain"
        self.kwargs = kwargs
        self.redshift_model_method = kwargs["redshift_model_method"]

        self.init_model()

    def init_model(self):
        input_dim = self.kwargs["baseline_decoder_input_dim"]
        if self.redshift_model_method == "regression":
            output_dim = 1
        elif self.redshift_model_method == "classification":
            output_dim = len(init_redshift_bins(**self.kwargs))
        else: raise ValueError()

        self.decoder = BasicDecoder(
            input_dim, output_dim, True,
            num_layers=self.kwargs["baseline_decoder_num_hidden_layers"] + 1,
            hidden_dim=self.kwargs["baseline_decoder_hidden_dim"],
            batch_norm=self.kwargs["baseline_decoder_batch_norm"],
            skip=[],
            skip_method=self.kwargs["baseline_decoder_skip_method"],
            skip_all_layers=self.kwargs["baseline_decoder_skip_all_layers"],
            activate_before_skip=self.kwargs["baseline_decoder_activate_before_skip"],
            skip_add_conversion_method= \
                self.kwargs["baseline_decoder_skip_add_conversion_method"])

    def index_latents(self, data, selected_ids, idx):
        ret = data
        if selected_ids is not None:
            ret = ret[selected_ids]
        if idx is not None:
            ret = ret[idx]
        return ret

    def forward(self, channels, wave_range, spectra_masks, spectra_source_data,
                idx=None, selected_ids=None):
        """
        @Params
          spectra_masks: [bsz,nsmpl]
          spectra_source_data: [bsz,3,nsmpl]
        @Return
          logits: [bsz*nsmpl]
        """
        ret = {}
        wave = spectra_source_data[:,0]
        spectra = spectra_source_data[:,1]
        # print(torch.sum(spectra_masks, dim=-1))
        # print(wave.shape, spectra.shape, spectra_masks.shape)
        # todo: incorporate spectra mask into forward?
        # print(spectra.shape, spectra_masks.shape)
        output = self.decoder(spectra * spectra_masks)
        if self.redshift_model_method == "regression":
            ret["redshift"] = output.flatten()
        elif self.redshift_model_method == "classification":
            ret["redshift_logits"] = F.softmax(output, dim=-1)
        else: raise ValueError()
        return ret
