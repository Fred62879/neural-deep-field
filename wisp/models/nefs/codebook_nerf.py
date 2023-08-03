
import torch

from collections import defaultdict
from wisp.models.nefs import BaseNeuralField
from wisp.models.hypers import HyperSpectralDecoder


class CodebookNef(BaseNeuralField):
    """ Model for codebook spectra inferrence without considering redshift
         (applicable for both pretrain and main train).
        Generate spectra only, w.o. redshift, scaler, or integration.
    """
    def __init__(self, **kwargs):
        super(CodebookNef, self).__init__()

        self.hps_decoder = HyperSpectralDecoder(
            scale=False,
            add_bias=False,
            integrate=False,
            intensify=False,
            qtz_spectra=False,
            _model_redshift=False,
            **kwargs
        )
        torch.cuda.empty_cache()

    def get_nef_type(self):
        return "codebook"

    def register_forward_functions(self):
        channels = ["spectra","intensity"]
        self._register_forward_function( self.recon_codebook_spectra, channels )

    def recon_codebook_spectra(self, coords, wave, wave_range):
        """ @Params:
              coords: codebook latents [(1,)bsz,num_samples,latents_dim]
              wave:   full range wave [bsz,num_samples,1]
            @Return
              ret: {
                "intensity": [bsz,num_samples,3]
              }
        """
        ret = defaultdict(lambda: None)
        self.hps_decoder(coords, wave, None, None, wave_range, ret=ret)
        return ret
