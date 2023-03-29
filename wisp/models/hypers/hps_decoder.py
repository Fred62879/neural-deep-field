
import torch
import torch.nn as nn

from wisp.utils import PerfTimer
from wisp.models.decoders import Decoder
from wisp.models.embedders import Encoder
from wisp.models.layers import Normalization
from wisp.models.hypers.hps_converter import HyperSpectralConverter
from wisp.models.hypers.hps_integrator import HyperSpectralIntegrator


class HyperSpectralDecoder(nn.Module):

    def __init__(self, integrate=True, scale=True, **kwargs):

        super(HyperSpectralDecoder, self).__init__()

        self.kwargs = kwargs
        self.scale = scale

        self.convert = HyperSpectralConverter(**kwargs)
        self.spectra_decoder = Decoder(**kwargs)
        self.scale = Normalization(kwargs["mlp_output_norm_method"])
        self.inte = HyperSpectralIntegrator(integrate=integrate, **kwargs)

    def reconstruct_spectra(self, wave, latents, scaler, redshift, wave_bound, scale=True):
        latents = self.convert(wave, latents, redshift, wave_bound)

        # import numpy as np
        # np.save('/scratch/projects/vision/code/implicit-universe-wisp/latents.npy',
        #         latents.detach().cpu().numpy())

        spectra = self.spectra_decoder(latents)[...,0]

        # np.save('/scratch/projects/vision/code/implicit-universe-wisp/spectra.npy',
        #         spectra.detach().cpu().numpy())
        # assert 0

        if self.scale and scaler is not None:
            # spectra = (torch.exp(scaler) * spectra.T).T
            spectra = (scaler * spectra.T).T

        spectra = self.scale(spectra)
        return spectra

    def train_with_full_wave(self, latents, full_wave, full_wave_bound, num_spectra_coords, ret):
        """ During training, some latents will be decoded, combining with full wave.
            Currently only supports spectra coords (incl. gt, dummy that requires
              spectrum plotting during training time).
            Latents that require full wave are placed at the end of the tensor.
        """
        latents = latents[-num_spectra_coords:]
        scaler = None if ret["scaler"] is None else ret["scaler"][-num_spectra_coords:]
        redshift = None if ret["redshift"] is None else ret["redshift"][-num_spectra_coords:]
        full_wave = full_wave[None,:,None].tile(num_spectra_coords,1,1)

        if self.kwargs["print_shape"]: print('hps_decoder', latents.shape)
        if self.kwargs["print_shape"]: print('hps_decoder, full wave', full_wave.shape)

        ret["spectra"] = self.reconstruct_spectra(full_wave, latents, scaler, redshift, full_wave_bound)

    def forward(self, latents, wave, trans, nsmpl, ret, full_wave=None, full_wave_bound=None, num_spectra_coords=-1):
        """ @Param
              latents:   (encoded or original) coords. [bsz,num_samples,coords_encode_dim or 2 or 3]
              wave:      lambda values, used to convert ra/dec to hyperspectral latents. [bsz,num_samples]
              trans:     corresponding transmission values of lambda. [bsz,num_samples]
              nsmpl:     average number of lambda samples falling within each band. [num_bands]
              full_wave: not None if do spectra supervision. [num_spectra_coords,full_num_samples]
              full_wave_bound: min and max value of lambda
              num_spectra_coords: > 0 if spectra supervision.

              ret (output from nerf and/or quantization): {
                "scaler":        unique scaler value for each coord. [bsz,1]
                "redshift":      unique redshift value for each coord. [bsz,1]
                "embed_ids":     ids of embedding each pixel's latent is quantized to.
                "codebook_loss": loss for codebook optimization.
              }

            @Return (add new fields to input data)
              intensity: reconstructed pixel values
              spectra:   reconstructed spectra
        """
        timer = PerfTimer(activate=self.kwargs["activate_timer"], show_memory=False)

        if self.kwargs["print_shape"]: print('hps_decoder',latents.shape)

        # spectra supervision, train with all lambda values instead of sampled lambda
        if num_spectra_coords > 0:
            self.train_with_full_wave(latents, full_wave, full_wave_bound, num_spectra_coords, ret)
            latents = latents[:-num_spectra_coords]
            if ret["scaler"] is not None: ret["scaler"] = ret["scaler"][:-num_spectra_coords]
            if ret["redshift"] is not None: ret["redshift"] = ret["redshift"][:-num_spectra_coords]
            if self.kwargs["print_shape"]: print('hps_decoder', latents.shape)

        if latents.shape[0] > 0:
            spectra = self.reconstruct_spectra(wave, latents, ret["scaler"], ret["redshift"], full_wave_bound)
            if "spectra" not in ret: ret["spectra"] = spectra
            if self.kwargs["print_shape"]: print('hps_decoder', spectra.shape)

            #timer.check("hps decoder, integration")
            intensity = self.inte(spectra, trans, nsmpl)
            if self.kwargs["print_shape"]: print('hps_decoder', intensity.shape)

            ret["intensity"] = intensity
