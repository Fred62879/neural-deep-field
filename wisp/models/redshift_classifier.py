
import torch
import torch.nn as nn
import torch.nn.functional as F

from wisp.utils import PerfTimer

from wisp.models.decoders import BasicDecoder
from wisp.models.embedders.encoder import Encoder

class RedshiftClassifier(nn.Module):
    def __init__(self, **kwargs):
        super(RedshiftClassifier, self).__init__()
        assert kwargs["model_redshift"], "we must model redshift during pretrain"
        self.kwargs = kwargs
        self.wave_multiplier = kwargs["wave_multiplier"]

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
        if self.kwargs["classify_based_on_loss"]:
            input_dim = self.kwargs["classifier_decoder_input_dim"]
        elif self.kwargs["classify_based_on_concat_spectra"] or \
             self.kwargs["classify_based_on_concat_wave_loss"]:
            input_dim = 2 * self.kwargs["classifier_decoder_input_dim"]
        elif self.kwargs["classify_based_on_concat_wave_spectra"]:
            input_dim = 3 * self.kwargs["classifier_decoder_input_dim"]
        else: raise ValueError()

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

    def shift_wave(self, wave, redshift):
        wave = wave / (1 + redshift[:,None])
        return wave

    def linear_norm_wave(self, wave, wave_range):
        (lo, hi) = wave_range
        # return self.wave_multiplier * (wave - lo) / (hi - lo)
        return wave / hi

    def forward(
            self, channels, wave, wave_range, spectra_mask, spectra_redshift,
            gt_spectra, recon_spectra, spectra_lambdawise_losses, idx=None, selected_ids=None
    ):
        """
        @Params
          wave: [bsz,nsmpl]
          gt_spectra: [bsz,nsmpl]
          recon_spectra: [bsz,nbins,nsmpl]
          spectra_mask: [bsz,nsmpl]
          spectra_lambdawise_losses: [bsz,nbins,nsmpl]
        @Return
          logits: [bsz*nsmpl]
        """
        ret = {}
        if self.kwargs["classify_based_on_loss"]:
            # print('*', spectra_mask.shape, spectra_lambdawise_losses.shape, wave.shape)
            # print(wave[0])
            # pe_wave = self.encoder(wave)
            # pe_losses = self.encoder(spetra_lambdawise_losses)
            # print(pe_wave.shape, pe_losses.shape)
            # todo: incorporate spectra mask into forward
            input = spectra_lambdawise_losses * spectra_mask[:,None]

        elif self.kwargs["classify_based_on_concat_spectra"]:
            nbins = recon_spectra.shape[1]
            input = torch.cat((gt_spectra[:,None].tile(1,nbins,1) * spectra_mask[:,None],
                               recon_spectra * spectra_mask[:,None]), dim=-1)
            # print(input.shape, input.dtype, input.device)

        elif self.kwargs["classify_based_on_concat_wave_loss"]:
            nbins = spectra_lambdawise_losses.shape[1]
            # print(torch.min(wave), torch.max(wave))
            wave = self.shift_wave(wave, spectra_redshift)
            # print(torch.min(wave), torch.max(wave))
            wave = self.linear_norm_wave(wave, wave_range)
            # print(torch.min(wave), torch.max(wave))
            # print(wave.shape, wave[0])
            # print(spectra_lambdawise_losses[0])
            input = torch.cat((wave[:,None].tile(1,nbins,1) * spectra_mask[:,None],
                               spectra_lambdawise_losses * spectra_mask[:,None]), dim=-1)

        elif self.kwargs["classify_based_on_concat_wave_spectra"]:
            nbins = recon_spectra.shape[1]
            # print(wave.shape, gt_spectra.shape, recon_spectra.shape)
            # print(torch.min(wave), torch.max(wave))
            wave = self.shift_wave(wave, spectra_redshift)
            # print(torch.min(wave), torch.max(wave))
            wave = self.linear_norm_wave(wave, wave_range)
            # print(torch.min(wave), torch.max(wave))
            # print(wave[0], gt_spectra[0], recon_spectra[0])
            input = torch.cat((recon_spectra * spectra_mask[:,None],
                               gt_spectra[:,None].tile(1,nbins,1) * spectra_mask[:,None],
                               wave[:,None].tile(1,nbins,1) * spectra_mask[:,None]), dim=-1)
        else:
            raise ValueError()

        logits = self.decoder(input)
        logits = logits.flatten() # [bsz,nbins,1] -> [n,]
        assert not torch.isnan(logits).any()
        ret["redshift_logits"] = logits
        return ret
