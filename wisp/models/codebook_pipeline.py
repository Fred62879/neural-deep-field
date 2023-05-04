
import torch
import torch.nn as nn

from wisp.models.hypers import HyperSpectralDecoder
from wisp.models.decoders import BasicDecoder, SpatialDecoder
from wisp.models.layers import get_layer_class, init_codebook, Quantization


class CodebookPipeline(nn.Module):
    def __init__(self, **kwargs):
        super(CodebookPipeline, self).__init__()

        self.kwargs = kwargs
        self.init_model()

    def init_model(self):
        self.codebook = init_codebook(
            self.kwargs["qtz_seed"], self.kwargs["qtz_num_embed"], self.kwargs["qtz_latent_dim"])

        self.spatial_decoder = SpatialDecoder(
            output_scaler=False,
            output_redshift=self.kwargs["generate_redshift"],
            qtz_calculate_loss=False,
            **self.kwargs)

        self.hps_decoder = HyperSpectralDecoder(
            integrate=False, scale=False, **self.kwargs)

    def forward(self, latents, wave, trans, nsmpl, full_wave_bound, codebook, qtz_args, ret):
        latents = self.spatial_decoder(latents, self.codebook, qtz_args, ret)

        self.hps_decoder(latents, wave, trans, nsmpl, full_wave_bound,
                         codebook=self.codebook, qtz_args=qtz_args,
                         quantize_spectra=True, ret=ret)
        return ret["spectra"]
