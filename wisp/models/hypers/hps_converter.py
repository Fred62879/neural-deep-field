
import torch
import torch.nn as nn

from wisp.models.embedders import Encoder
from wisp.models.decoders import BasicDecoder, Siren


class HyperSpectralConverter(nn.Module):
    """ Processing module, no weights to update here.
    """
    def __init__(self, **kwargs):
        """ @Param:
              wave_encoder: encoder network for lambda.
        """
        super(HyperSpectralConverter, self).__init__()

        self.kwargs = kwargs
        self.encode_wave = kwargs["encode_wave"]
        self.quantize_spectra = kwargs["quantize_spectra"]
        self.combine_method = kwargs["hps_combine_method"]

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

    def linear_norm_wave(self, wave, wave_bound):
        (lo, hi) = wave_bound # 3940, 10870
        return (wave - lo) / (hi - lo)
        # return 2*(wave - lo) / (hi - lo)-1

    def shift_wave(self, wave, redshift):
        wave = wave.permute(1,2,0)

        if wave.ndim == 3: # [nsmpl,1,bsz]
            wave = wave / (1 + redshift) # dont use `/=` this will change wave object
        elif wave.ndim == 4: # [nbands,nsmpl,1,bsz]
            wave = wave / (1 + redshift)
        else:
            raise Exception("Wrong wave dimension when doing wave shifting.")

        del redshift
        wave = wave.permute(2,0,1)
        return wave

    def combine_spatial_spectral(self, spatial, spectral):
        """ Combine spatial with spectral latent variables.
            @Param
              spatial:   [bsz,num_samples,2 or embed_dim]
              spectral:  [bsz,num_samples,1 or embed_dim]
            @Return
              if add:    [bsz,num_samples,embed_dim]
              if concat: [bsz,num_samples,3 or spa_embed_dim+spe_embed_dim]
        """
        #print('****')
        #print(spatial.shape, spatial)
        #assert 0
        if self.combine_method == "add":
            assert(spatial.shape == spectral.shape)
            latents = spatial + spectral # [...,embed_dim]
        elif self.combine_method == "concat":
            latents = torch.cat((spatial, spectral), dim=-1)
        else:
            raise ValueError("Unrecognized spatial-spectral combination method.")

        del spatial, spectral
        return latents

    def forward(self, wave, latents, redshift, wave_bound):
        """ Process wave (shift, encode, if required) and
              combine with RA/DEC (original state or encoded) to hyperspectral latents.
            @Param
              wave:     lambda values used for casting.   [bsz,nsmpl,1]
              latents:  (original or encoded) 2D coords.  [bsz,1,2 or coords_embed_dim]
              redshift: redshift value, unique for each pixel. [bsz]
            @Return
              latents:  hyperspectral latents (i.e. ra/dec/wave)
        """
        num_samples = wave.shape[-2]
        coords_encode_dim = latents.shape[-1]

        if redshift is not None:
            wave = self.shift_wave(wave, redshift)

        # normalize lambda values to [0,1]
        #print(wave[0,:,0])
        wave = self.linear_norm_wave(wave, wave_bound)
        #import numpy as np
        #np.save('tmp.npy',wave.detach().cpu().numpy())
        #print(wave[0,:,0])
        #assert 0

        if self.encode_wave:
            assert(coords_encode_dim != self.kwargs["space_dim"])
            wave = self.wave_encoder(wave) # [bsz,num_samples,wave_embed_dim]
            #print(wave.shape, wave)

        else: # assert coords are not encoded as well, should only use siren in this case
            if self.kwargs["coords_encode_method"] == "grid" and self.kwargs["grid_dim"] == 3:
                assert(coords_encode_dim == self.kwargs["space_dim"])
            else:
                assert(coords_encode_dim == 2)
                latents = latents[...,:2] # remove dummy 3rd dim

        latents = latents.tile(1,num_samples,1) # [bsz,nsamples,encode_dim or 2]
        latents = self.combine_spatial_spectral(latents, wave)
        return latents
