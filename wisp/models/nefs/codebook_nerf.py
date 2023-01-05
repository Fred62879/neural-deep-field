
import torch

from collections import defaultdict
from wisp.models.nefs import BaseNeuralField
from wisp.models.hypers import HyperSpectralDecoder


class CodebookNef(BaseNeuralField):
    """ Model for returning hardcoded value.
    """
    def __init__(self, integrate, **kwargs):
        super(CodebookNef, self).__init__(**kwargs)

        self.hps_decoder = HyperSpectralDecoder(
            integrate=integrate, scale=False, **kwargs)

        torch.cuda.empty_cache()
        self._forward_functions = {}
        self.register_forward_functions()
        self.supported_channels = set([
            channel for channels in self._forward_functions.values()
            for channel in channels])

    def get_nef_type(self):
        return 'codebook'

    def register_forward_functions(self):
        channels = ["intensity"]
        self._register_forward_function( self.get_codebook_spectra, channels )

    def get_codebook_spectra(self, coords, wave, trans, nsmpl):
        """ Output given latents without any modifications.
            @Params:
              coords (torch.FloatTensor): tensor of shape [1, batch, num_samples, 2/3]
            @Return
              {"indensity": torch.FloatTensor }:
                - Output intensity tensor of shape [batch, num_samples, 3]
        """
        ret = defaultdict(lambda: None)
        self.hps_decoder(coords, wave, trans, nsmpl, ret)
        return ret
