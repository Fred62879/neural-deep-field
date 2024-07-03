
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
            batch_norm=self.kwargs["classifier_decoder_batch_norm"],
            skip=[],
            skip_method=self.kwargs["classifier_decoder_skip_method"],
            skip_all_layers=self.kwargs["classifier_decoder_skip_all_layers"],
            activate_before_skip=self.kwargs["classifier_decoder_activate_before_skip"],
            skip_add_conversion_method= \
                self.kwargs["classifier_decoder_skip_add_conversion_method"])

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


    def mask_invalid(self, data, mask):
        data[~mask] = 0
        return data

    def forward(
            self, channels, spectra_mask,
            wave=None, wave_range=None, spectra_redshift=None,
            gt_spectra=None, recon_spectra=None,
            spectra_lambdawise_losses=None,
            idx=None, selected_ids=None
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
            input = self.mask_invalid(
                spectra_lambdawise_losses, spectra_mask[:,None].tile(1,nbins,1))

        elif self.kwargs["classify_based_on_concat_spectra"]:
            nbins = recon_spectra.shape[1]
            input = torch.cat((
                self.mask_invalid(gt_spectra, spectra_mask)[:,None].tile(1,nbins,1),
                self.mask_invalid(recon_spectra, spectra_mask[:,None].tile(1,nbins,1))), dim=-1)

        elif self.kwargs["classify_based_on_concat_wave_loss"]:
            nbins = spectra_lambdawise_losses.shape[1]
            wave = self.shift_wave(wave, spectra_redshift)
            wave = self.linear_norm_wave(wave, wave_range)
            input = torch.cat((
                self.mask_invalid(wave, spectra_mask)[:,None].tile(1,nbins,1),
                self.mask_invalid(
                    spectra_lambdawise_losses, spectra_mask[:,None].tile(1,nbins,1))), dim=-1)

        elif self.kwargs["classify_based_on_concat_wave_spectra"]:
            nbins = recon_spectra.shape[1]
            # print(torch.min(wave), torch.max(wave))
            wave = self.shift_wave(wave, spectra_redshift)
            # print(torch.min(wave), torch.max(wave))
            wave = self.linear_norm_wave(wave, wave_range)
            # print(torch.min(wave), torch.max(wave))
            # print(wave[0], gt_spectra[0], recon_spectra[0])
            input = torch.cat((
                self.mask_invalid(wave, spectra_mask)[:,None].tile(1,nbins,1),
                self.mask_invalid(gt_spectra, spectra_mask)[:,None].tile(1,nbins,1),
                self.mask_invalid(recon_spectra, spectra_mask[:,None].tile(1,nbins,1))), dim=-1)
        else:
            raise ValueError()

        logits = self.decoder(input)
        # logits = logits.flatten() # [bsz,nbins,1] -> [n,]
        assert not torch.isnan(logits).any()
        # todo, calculate logits use sigmoid
        logits = torch.sigmoid(logits)
        ret["redshift_logits"] = logits
        return ret
