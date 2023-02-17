
import torch
import torch.nn as nn


class HyperSpectralConverter(nn.Module):
    """ Processing module, no weights to update here.
    """
    def __init__(self, wave_encoder, **kwargs):
        """ @Param:
              wave_encoder: encoder network for lambda.
        """
        super(HyperSpectralConverter, self).__init__()

        self.kwargs = kwargs
        self.combine_method = kwargs["hps_combine_method"]
        self.wave_encode_method = kwargs["wave_encode_method"]

        self.wave_encoder = wave_encoder

    def linear_norm_wave(self, wave, wave_bound):
        (lo, hi) = wave_bound # 3940, 10870
        return (wave - lo) / (hi - lo)

    def shift_wave(self, wave, redshift):
        if self.kwargs["print_shape"]: print('hps_converter, shift wave', wave.shape)
        if self.kwargs["print_shape"]: print('hps_converter, shift wave', redshift.shape)
        wave = wave.permute(1,2,0)
        #print(torch.min(redshift), torch.max(redshift))

        if wave.ndim == 3:
            nsmpl = wave.shape[1] # [bsz,nsmpl,1]
            # wave += redshift #- 0.1
            wave /= (1 + redshift)
        elif wave.ndim == 4:
            nsmpl = wave.shape[2] # [bsz,nbands,nsmpl,1]
            # wave += redshift #- 0.1
            wave /= (1 + redshift)
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
        """ Process wave (refshift, encode, if required) and
              combine with RA/DEC (original state or encoded) to hyperspectral latents.
            @Param
              wave:     lambda values used for casting.   [bsz,num_samples,1]
              latents:  (original or encoded) 2D coords.  [bsz,1,2 or coords_embed_dim]
              redshift: redshift value, unique for each pixel. [bsz]
            @Return
              latents:  hyperspectral latents (i.e. ra/dec/wave)
        """
        if self.kwargs["print_shape"]: print('hps_converter',latents.shape)

        # print(wave.shape, latents.shape)
        num_samples = wave.shape[-2]
        coords_encode_dim = latents.shape[-1]

        #print(latents.shape, latents)
        #if len(latents) > 1:
        #    print(latents[2080])
        if redshift is not None:
            wave = self.shift_wave(wave, redshift)

        # normalize lambda values to [0,1]
        wave = self.linear_norm_wave(wave, wave_bound)

        if self.wave_encode_method == "positional":
            assert(coords_encode_dim != 2)
            wave = self.wave_encoder(wave) # [bsz,num_samples,wave_embed_dim]
        else:
            # assert 2D coords are not encoded as well, should use siren in this case
            assert(coords_encode_dim == 2)

        if self.kwargs["print_shape"]: print('hps_converter, embedded wave', wave.shape)

        latents = latents.tile(1,num_samples,1) # [bsz,nsamples,encode_dim or 2]
        if self.kwargs["print_shape"]: print('hps_converter, latents',latents.shape)
        latents = self.combine_spatial_spectral(latents, wave)

        #del wave, redshift
        return latents
