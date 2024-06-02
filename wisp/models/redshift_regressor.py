
import torch
import torch.nn as nn
import torch.nn.functional as F

from wisp.utils import PerfTimer

from wisp.models.decoders import BasicDecoder
from wisp.models.embedders.encoder import Encoder

class RedshiftRegressor(nn.Module):
    def __init__(self, **kwargs):
        super(RedshiftRegressor, self).__init__()
        assert kwargs["model_redshift"], "we must model redshift during pretrain"
        self.kwargs = kwargs

        self.init_model()

    def init_model(self):
        # embedder_args = (
        #     1, self.kwargs["wave_embed_dim"],
        #     self.kwargs["wave_embed_omega"],
        #     self.kwargs["wave_embed_sigma"],
        #     self.kwargs["wave_embed_bias"],
        #     self.kwargs["wave_embed_seed"])
        # self.encoder = Encoder(
        #     input_dim=1,
        #     encode_method=self.kwargs["wave_encode_method"],
        #     embedder_args=embedder_args,
        #     **self.kwargs)

        # input_dim = 2 * self.kwargs["wave_embed_dim"]
        input_dim = self.kwargs["regressor_decoder_input_dim"]
        output_dim = 1

        self.decoder = BasicDecoder(
            input_dim, output_dim, True,
            num_layers=self.kwargs["classifier_decoder_num_hidden_layers"] + 1,
            hidden_dim=self.kwargs["classifier_decoder_hidden_dim"],
            batch_norm=self.kwargs["classifier_decoder_batch_norm"])

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
        print(spectra.shape)
        print(wave[0])
        print(spectra[0])
        assert 0
        # print(wave.shape, spectra.shape, spectra_masks.shape)
        # todo: incorporate spectra mask into forward
        # print(spectra_masks.shape, spectra_lambdawise_losses.shape, wave.shape)
        # print(wave[0])
        # pe_wave = self.encoder(wave)
        # pe_losses = self.encoder(spetra_lambdawise_losses)
        # print(pe_wave.shape, pe_losses.shape)
        redshift = self.decoder(spectra).flatten()
        ret["redshift"] = redshift
        return ret
