
import torch
import torch.nn as nn

from collections import defaultdict
from wisp.models.nefs import BaseNeuralField
from wisp.models.hypers import HyperSpectralDecoder
from wisp.models.decoders import BasicDecoder, SpatialDecoder
from wisp.models.layers import get_layer_class, init_codebook, Quantization


class CodebookPretrainNerf(BaseNeuralField):
    def __init__(self, **kwargs):
        super(CodebookPretrainNerf, self).__init__()

        self.kwargs = kwargs
        self.init_model()

    def get_nef_type(self):
        return 'codebook_pretrain'

    def register_forward_functions(self):
        """ Register forward functions with the channels that they output.
        """
        channels = ["intensity","spectra","redshift","soft_qtz_weights"]
        self._register_forward_function(self.pretrain, channels)

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

    def pretrain(self, coords, full_wave, full_wave_bound, qtz_args):
        ret = defaultdict(lambda: None)
        # print(coords.shape)
        # print(full_wave.shape)
        # print(self.codebook.weight.shape)

        latents = self.spatial_decoder(coords[:,None,:], self.codebook, qtz_args, ret)
        self.hps_decoder(latents, full_wave[None,:,None], None, None, full_wave_bound,
                         codebook=self.codebook, qtz_args=qtz_args,
                         quantize_spectra=True, ret=ret)
        return ret
