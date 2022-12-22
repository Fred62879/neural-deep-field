
import torch
import torch.nn as nn

from wisp.models.embedders.pe import RandGausLinr


class HyperSpectralConverter(nn.Module):

    def __init__(self, **kwargs):
        super(HyperSpectralConverter, self).__init__()

        self.kwargs = kwargs
        self.verbose = kwargs["verbose"]
        self.convert_method = kwargs["hps_convert_method"]
        self.wave_embed_method = kwargs["wave_embed_method"]

        if self.wave_embed_method is not None:
            self.init_embedder()

    def init_embedder(self):
        pe_dim = self.kwargs["wave_embed_dim"]
        sigma = 1
        omega = 1
        pe_bias = True
        self.embedder = RandGausLinr((1, pe_dim, sigma, omega, pe_bias, self.verbose))
        self.decoder_input_dim = pe_dim

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

    def forward(self, wave, radec, redshift=None):
        """ Process wave (refshifting, embedding, if required) and
              convert RA/DEC (original state or embedded) to hyperspectral coordinates.

            @Param
              wave:   lambda values used for casting [1,bsz,num_samples,1]
              latent: embedded 2D coords [bsz,2 or coords_embed_dim]
            @Return
              hps_coords: ra/dec/wave coords
                          if convert method is add, then [bsz,num_samples,embed_dim]
                          if concat, then [bsz,num_samples, coords_embed_dim+wave_embed_dim]
        """
        wave = wave[0,:,:] # [bsz,num_samples,1]  # replace ***********
        num_samples = wave.shape[-2]
        coords_embed_dim = radec.shape[-1]

        if redshift is not None:
            wave = self.shift_wave(wave, redshift)

        if self.wave_embed_method == "positional":
            assert(coords_embed_dim != 2)
            wave = self.embedder(wave) # [bsz,num_samples,wave_embed_dim]
        else:
            # assert 2D coords are not embedded as well, should use siren in this case
            assert(coords_embed_dim == 2)

        radec = radec[:,None].tile(1,num_samples,1) # [bsz,num_samples,coords_embed_dim or 2]

        if self.convert_method == "add":
            assert(radec.shape == wave.shape)
            hyperspectral_coords = radec + wave # [...,embed_dim]
        elif self.convert_method == "concat":
            hyperspectral_coords = torch.cat((radec, wave), dim=-1) # [...,c_embed_dim+w_embed_dim]
        else:
            raise ValueError("Unrecognized converting method.")
        return hyperspectral_coords
