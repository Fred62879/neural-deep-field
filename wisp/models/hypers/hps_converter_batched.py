
import torch
import warnings
import torch.nn as nn

from wisp.utils.common import get_bool_classify_redshift

from wisp.models.embedders import Encoder
from wisp.models.decoders import BasicDecoder, Siren


class HyperSpectralConverter(nn.Module):
    """ Processing module, no weights to update here.
    """
    def __init__(self, _qtz_spectra=True, _model_redshift=True, **kwargs):
        """ @Param:
              wave_encoder: encoder network for lambda.
        """
        super(HyperSpectralConverter, self).__init__()

        self.kwargs = kwargs
        self.sample_bins = False
        self._qtz_spectra = _qtz_spectra
        self._model_redshift = _model_redshift
        self.wave_multiplier = kwargs["wave_multiplier"]
        self.optimize_one_latent_for_all_redshift_bins = \
            kwargs["brute_force_redshift"] and \
            not kwargs["optimize_latents_for_each_redshift_bin"]

        self.encode_wave = kwargs["encode_wave"]
        self.quantize_spectra = kwargs["quantize_spectra"]
        self.combine_method = kwargs["hps_combine_method"]
        self.linear_norm_wave = kwargs["linear_norm_wave"]

        self.classify_redshift = get_bool_classify_redshift(**kwargs)
        self.use_global_spectra_loss_as_lambdawise_weights = \
            kwargs["use_global_spectra_loss_as_lambdawise_weights"] or \
            kwargs["infer_use_global_loss_as_lambdawise_weights"]

        if self.encode_wave:
            self.init_encoder()

    def init_encoder(self):
        if self.kwargs["wave_encode_method"] == "positional_encoding":
            embedder_args = (
                1, self.kwargs["wave_embed_dim"],
                self.kwargs["wave_embed_omega"],
                self.kwargs["wave_embed_sigma"],
                self.kwargs["wave_embed_bias"],
                self.kwargs["wave_embed_seed"])

            self.wave_encoder = Encoder(
                input_dim=1,
                encode_method=self.kwargs["wave_encode_method"],
                embedder_args=embedder_args,
                **self.kwargs)
        else:
            assert not self.kwargs["encode_wave"]

    def toggle_sample_bins(self, sample: bool):
        self.sample_bins = sample

    ####################
    # forward operations
    ####################

    def _linear_norm_wave(self, wave, wave_bound):
        (lo, hi) = wave_bound
        # return self.wave_multiplier * (wave - lo) / (hi - lo)
        return self.wave_multiplier * wave / hi

    def shift_wave(self, wave, redshift, selected_bins_mask, ret):
        """ Convert observed lambda to emitted lambda.
            @Param
              wave: [bsz,nsmpl,1]
              redshift: [bsz] or [n_bins]
              selected_bins_mask: 1 for chose bins, 0 for others [bsz,n_bins]
            @Return
              emitted_wave: [(num_bins,)bsz,nsmpl,1]
        """
        bsz, nsmpl, _ = wave.shape

        wave = wave.permute(1,2,0)
        if self.classify_redshift:
            num_bins = len(redshift)
            wave = wave[...,None].tile(1,1,1,num_bins)
        wave = wave / (1 + redshift) # dont use `/=` this will change wave object

        if wave.ndim == 3:
            wave = wave.permute(2,0,1)
        elif wave.ndim == 4:
            wave = wave.permute(3,2,0,1)
        else: raise ValueError("Wrong wave dimension when redshifting.")

        if self.sample_bins:
            """
            We need to apply the mask with the batch dimension at dim 0
              e.g. `wave[selected_bins_mask.T]` will mess up the batch dim,
              spectra 0 may end up as spectra 1.
            """
            assert selected_bins_mask is not None
            if wave.ndim == 3:
                pass
            elif wave.ndim == 4:
                # wave = wave[selected_bins_mask.T[...,None,None].tile(1,1,nsmpl,1)]
                # wave = wave.view(-1,bsz,nsmpl,1)
                wave = wave.permute(1,0,2,3)[selected_bins_mask].view(
                    bsz,-1,nsmpl,1).permute(1,0,2,3)
            else: raise ValueError()

        if self.use_global_spectra_loss_as_lambdawise_weights:
            ret["emitted_wave"] = wave[...,0] # [(nbins,)bsz,nsmpl]
        return wave

    def combine_spatial_spectral(self, spatial, spectral):
        """ Combine spatial with spectral latent variables.
            @Param
              spatial:   [(num_codes,)bsz,1,2/embed_dim] if qtz_spectra
                         [bsz,1,(num_bins),embed_dim]    o.w.
              spectral:  [(num_bins,)bsz,nsmpl,1/embed_dim]
            @Return
              if add:    [(num_codes,num_bins,)bsz,nsmpls,embed_dim]
              if concat: [(num_codes,num_bins,)bsz,nsmpls,spa_dim+spe_dim]
        """
        print(spatial.shape, spectral.shape)
        nsmpls = spectral.shape[-2]
        if spatial.ndim == 4 and self._qtz_spectra:
            num_codes = spatial.shape[0]
        if spectral.ndim == 4: num_bins = spectral.shape[0]

        if spatial.ndim == 3:
            if spectral.ndim == 3:
                spatial = spatial.tile(1,nsmpls,1)
            elif spectral.ndim == 4:
                assert self._qtz_spectra or self.optimize_one_latent_for_all_redshift_bins
                spatial = spatial[None,...].tile(num_bins,1,nsmpls,1)
            else: raise ValueError()

        elif spatial.ndim == 4:
            if spectral.ndim == 3:
                assert self._qtz_spectra
                spatial = spatial.tile(1,1,nsmpls,1)
                spectral = spectral[None,...].tile(num_codes,1,1,1)
            elif spectral.ndim == 4:
                if self._qtz_spectra:
                    spatial = spatial[None,...].tile(num_bins,1,1,nsmpls,1)
                    spectral = spectral[:,None,...].tile(1,num_codes,1,1,1)
                else:
                    spatial = spatial.permute(2,0,1,3).tile(1,1,nsmpls,1)
        else:
            raise ValueError("Wrong wave dimension when combining.")

        # spatial and spectral now are both [...,bsz,nsmpl,?]
        if self.combine_method == "add":
            assert(spatial.shape == spectral.shape)
            latents = spatial + spectral # [...,embed_dim]
        elif self.combine_method == "concat":
            latents = torch.cat((spatial, spectral), dim=-1)
        else:
            raise ValueError("Unsupported spatial-spectral combination method.")
        return latents

    def forward(self, wave, latents, redshift, wave_bound, selected_bins_mask, ret):
        """ Process wave (shift, encode, if required) and
              combine with RA/DEC (original state or encoded) to hyperspectral latents.
            @Param
              wave:     lambda values used for casting. [bsz,nsmpl,1]
              latents:  2D coords / encoded coords    / codebook.
                        [bsz,1,2] / [bsz,1,embed_dim] / [num_code,bsz,1,embed_dim]
              redshift: redshift value, unique for each pixel.
                        if classification modeling: [num_bins]
                        else: [bsz]
            @Return
              latents:  hyperspectral latents (i.e. ra/dec/wave)
        """
        embed_dim = latents.shape[-1]

        if redshift is None:
            if self._model_redshift:
                warnings.warn("model redshift without providing redshift values!")
        else: wave = self.shift_wave(wave, redshift, selected_bins_mask, ret)
        if self.linear_norm_wave:
            wave = self._linear_norm_wave(wave, wave_bound)

        if self.encode_wave:
            wave = self.wave_encoder(wave) # [...,bsz,num_samples,wave_embed_dim]
        # else:
        #     # assert coords are not encoded as well, should only use siren in this case
        #     assert not self.kwargs["encode_coords"] and embed_dim == 2

        latents = self.combine_spatial_spectral(latents, wave)
        return latents
