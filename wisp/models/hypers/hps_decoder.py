
import torch
import torch.nn as nn

from wisp.utils import PerfTimer
from wisp.models.decoders import Decoder
from wisp.models.hypers.hps_converter import HyperSpectralConverter
from wisp.models.hypers.hps_integrator import HyperSpectralIntegrator
from wisp.models.layers import Normalization, Quantization


class HyperSpectralDecoder(nn.Module):

    def __init__(self, integrate=True, scale=True, **kwargs):

        super(HyperSpectralDecoder, self).__init__()

        self.kwargs = kwargs
        self.scale = scale

        self.convert = HyperSpectralConverter(**kwargs)
        self.spectra_decoder = Decoder(**kwargs)
        self.norm = Normalization(kwargs["mlp_output_norm_method"])
        self.inte = HyperSpectralIntegrator(integrate=integrate, **kwargs)
        if kwargs["quantize_spectra"]:
            self.qtz = Quantization(False, **kwargs)

    def reconstruct_spectra(self, input, wave, scaler, redshift, wave_bound, ret,
                            codebook, qtz_args, quantize_spectra):
        if quantize_spectra:
            bsz = wave.shape[0]
            # each input coord has #num_code spectra generated
            latents = torch.stack([
                self.convert(wave, code.tile(bsz,1,1), redshift, wave_bound)
                for code in codebook.weight
            ], dim=0) # [num_code,bsz,nsmpl,dim]
        else:
            latents = self.convert(wave, input, redshift, wave_bound)

        spectra = self.spectra_decoder(latents)[...,0]

        if quantize_spectra:
            _, spectra = self.qtz(input, spectra, ret, qtz_args)
            spectra = spectra[:,0] # [bsz,nsmpl]

        if self.scale and scaler is not None:
            spectra = (scaler * spectra.T).T

        spectra = self.norm(spectra)
        return spectra

    def forward_with_full_wave(self, latents, full_wave, full_wave_bound,
                               num_spectra_coords, ret, codebook, qtz_args, quantize_spectra):
        """ During training, some latents will be decoded, combining with full wave.
            Currently only supports spectra coords (incl. gt, dummy that requires
              spectrum plotting during training time).
            Latents that require full wave are placed at the end of the tensor.
        """
        latents = latents[-num_spectra_coords:]
        scaler = None if ret["scaler"] is None else ret["scaler"][-num_spectra_coords:]
        redshift = None if ret["redshift"] is None else ret["redshift"][-num_spectra_coords:]
        full_wave = full_wave[None,:,None].tile(num_spectra_coords,1,1)

        ret["spectra"] = self.reconstruct_spectra(
            latents, full_wave, scaler, redshift, full_wave_bound, ret,
            codebook, qtz_args, quantize_spectra)

        if ret["scaler"] is not None:
            ret["scaler"] = ret["scaler"][:-num_spectra_coords]
        if ret["redshift"] is not None:
            ret["redshift"] = ret["redshift"][:-num_spectra_coords]

    def forward(self, latents,
                wave, trans, nsmpl, full_wave_bound,
                full_wave=None, num_spectra_coords=-1,
                codebook=None, qtz_args=None, quantize_spectra=False, ret=None):
        """ @Param
              latents:   (encoded or original) coords or logits for quantization.
                         [bsz,1,space_dim or coords_encode_dim]

            - hyperspectral
              wave:      lambda values, used to convert ra/dec to hyperspectral latents.
                         [bsz,num_samples,1]
              trans:     corresponding transmission values of lambda. [(bsz,)nbands,num_samples]
              nsmpl:     average number of lambda samples falling within each band. [num_bands]
              full_wave_bound: min and max value of lambda

            - spectra supervision
              full_wave: not None if do spectra supervision.
                           [num_spectra_coords,full_num_samples]
              num_spectra_coords: > 0 if spectra supervision.

            - spectra qtz
              codebook
              qtz_args

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

        if num_spectra_coords > 0:
            # forward the last #num_spectra_coords latents with all lambda
            self.forward_with_full_wave(
                latents, full_wave, full_wave_bound, num_spectra_coords,
                ret, codebook, qtz_args, quantize_spectra)

            latents = latents[:-num_spectra_coords]
            if latents.shape[0] == 0: return

        spectra = self.reconstruct_spectra(
            latents, wave, ret["scaler"], ret["redshift"], full_wave_bound, ret,
            codebook, qtz_args, quantize_spectra)

        if "spectra" not in ret:
            ret["spectra"] = spectra

        intensity = self.inte(spectra, trans, nsmpl)
        ret["intensity"] = intensity
