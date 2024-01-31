
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
        self._qtz_spectra = _qtz_spectra
        self._model_redshift = _model_redshift
        self.wave_multiplier = kwargs["wave_multiplier"]

        self.encode_wave = kwargs["encode_wave"]
        self.quantize_spectra = kwargs["quantize_spectra"]
        self.combine_method = kwargs["hps_combine_method"]

        self.classify_redshift = get_bool_classify_redshift(**kwargs)

        self.init_encoder()

    def init_encoder(self):
        if self.kwargs["wave_encode_method"] == "positional_encoding":
            embedder_args = (
                1,
                self.kwargs["wave_embed_dim"],
                self.kwargs["wave_embed_omega"],
                self.kwargs["wave_embed_sigma"],
                self.kwargs["wave_embed_bias"],
                self.kwargs["wave_embed_seed"]
            )
            self.wave_encoder = Encoder(
                input_dim=1,
                encode_method=self.kwargs["wave_encode_method"],
                embedder_args=embedder_args,
                **self.kwargs
            )

        elif self.kwargs["wave_encode_method"] == "relumlp":
            # we abuse basic decoder and use it as an encoder here
            self.wave_encoder = BasicDecoder(
                1, self.kwargs["wave_embed_dim"],
                torch.relu, bias=True, layer=nn.Linear,
                num_layers=self.kwargs["wave_encoder_num_hidden_layers"] + 1,
                hidden_dim=self.kwargs["wave_encoder_hidden_dim"], skip=[]
            )

        elif self.kwargs["wave_encode_method"] == "siren":
            self.wave_encoder = Siren(
                1, self.kwargs["wave_embed_dim"],
                num_layers=self.kwargs["wave_encoder_num_hidden_layers"] + 1,
                dim_hidden=self.kwargs["wave_encoder_hidden_dim"],
                first_w0=self.kwargs["wave_encoder_siren_first_w0"],
                hidden_w0=self.kwargs["wave_encoder_siren_hidden_w0"],
                seed=self.kwargs["wave_encoder_siren_seed"],
                coords_scaler=self.kwargs["wave_encoder_siren_coords_scaler"],
                last_linear=self.kwargs["wave_encoder_siren_last_linear"],
            )

        else:
            assert not self.kwargs["encode_wave"]

    ####################
    # forward operations
    ####################

    def linear_norm_wave(self, wave, wave_bound):
        (lo, hi) = wave_bound # 3940, 10870
        return self.wave_multiplier * (wave - lo) / (hi - lo)

    def shift_wave(self, wave, redshift):
        """ Convert observed lambda to emitted lambda.
            @Param
              wave: [bsz,nsmpl,1]
              redshift: [bsz] or [num_bins]
            @Return
              emitted_wave: [(num_bins,)bsz,nsmpl,1]
        """
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
        nsmpls = spectral.shape[-2]
        if spatial.ndim == 4 and self._qtz_spectra:
            num_codes = spatial.shape[0]
        if spectral.ndim == 4: num_bins = spectral.shape[0]
        # print(spatial.shape, spectral.shape)

        if spatial.ndim == 3:
            if spectral.ndim == 3:
                spatial = spatial.tile(1,nsmpls,1)
            elif spectral.ndim == 4:
                assert self._qtz_spectra
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

    def forward(self, wave, latents, redshift, wave_bound):
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
        else: wave = self.shift_wave(wave, redshift)
        wave = self.linear_norm_wave(wave, wave_bound)

        if self.encode_wave:
            wave = self.wave_encoder(wave) # [...,bsz,num_samples,wave_embed_dim]
        else:
            # assert coords are not encoded as well, should only use siren in this case
            assert not self.kwargs["encode_coords"] and embed_dim == 2

        latents = self.combine_spatial_spectral(latents, wave)
        return latents
