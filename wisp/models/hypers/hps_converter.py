
import torch
import torch.nn as nn


class HyperSpectralConverter(nn.Module):
    """ Processing module, no weights to update here.
    """

    def __init__(self, wave_embedder, **kwargs):
        """ @Param:
              wave_embedder: embedder network for lambda.
        """
        super(HyperSpectralConverter, self).__init__()

        self.kwargs = kwargs
        self.verbose = kwargs["verbose"]
        self.combine_method = kwargs["hps_combine_method"]

        self.wave_embed_method = kwargs["wave_embed_method"]
        self.wave_embedder = wave_embedder

    def shift_wave(self, wave, redshift):
        if wave.ndim == 3:
            nsmpl = wave.shape[1] # [bsz,nsmpl,1]
            redshift = torch.exp(redshift) - 0.1
            wave += redshift[:,:,None].tile(1,nsmpl,1)
            #wave /= (1 + redshift[:,:,None].tile(1,nsmpl,1))
        elif wave.ndim == 4:
            nsmpl = wave.shape[2] # [bsz,nbands,nsmpl,1]
            redshift = torch.exp(redshift) - 0.1
            wave += redshift[:,:,:,None].tile(1,self.nbands,nsmpl,1)
            #wave /= (1 + redshift[:,:,:,None].tile(1,self.nbands,nsmpl,1))
        else:
            raise Exception("Wrong wave dimension when doing wave shifting.")
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
            hps_latents = spatial + spectral # [...,embed_dim]
        elif self.combine_method == "concat":
            #print(spatial.shape, spectral.shape)
            hps_latents = torch.cat((spatial, spectral), dim=-1)
        else:
            raise ValueError("Unrecognized spatial-spectral combination method.")
        return hps_latents

    def forward(self, wave, latents, redshift=None):
        """ Process wave (refshifting, embedding, if required) and
              combine with RA/DEC (original state or embedded) to hyperspectral latents.
            @Param
              wave:    lambda values used for casting.   [1,bsz,num_samples,1]
              latents: (original or embedded) 2D coords. [bsz,1,2 or coords_embed_dim]
            @Return
              hps_latents: ra/dec/wave coords
        """
        wave = wave[0] # [bsz,num_samples,1]  # replace ***********
        num_samples = wave.shape[-2]
        coords_embed_dim = latents.shape[-1]

        if redshift is not None:
            wave = self.shift_wave(wave, redshift)

        if self.wave_embed_method == "positional":
            assert(coords_embed_dim != 2)
            wave = self.wave_embedder(wave) # [bsz,num_samples,wave_embed_dim]
        else:
            # assert 2D coords are not embedded as well, should use siren in this case
            assert(coords_embed_dim == 2)

        latents = latents.tile(1,num_samples,1) # [bsz,nsamples,embed_dim or 2]
        hps_latents = self.combine_spatial_spectral(latents, wave)
        return hps_latents
