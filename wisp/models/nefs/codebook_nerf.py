
import torch

from collections import defaultdict
from wisp.models.nefs import BaseNeuralField
from wisp.models.hypers import HyperSpectralDecoder


class CodebookNef(BaseNeuralField):
    """ Model for codebook inferrence. Generate spectra only
          w.o. redshift, scaler, or integration.
    """
    def __init__(self, integrate, **kwargs):
        super(CodebookNef, self).__init__()

        self.hps_decoder = HyperSpectralDecoder(
            integrate=integrate, scale=False,
            _model_redshift=False, **kwargs)

        torch.cuda.empty_cache()

    def get_nef_type(self):
        return 'codebook'

    def register_forward_functions(self):
        channels = ["intensity"]
        self._register_forward_function( self.recon_codebook_spectra, channels )

    def recon_codebook_spectra(self, coords, wave, full_wave_bound, qtz_args=None):
        """ Output given latents without any modifications.
            @Params:
              coords: [(1,)bsz,num_samples,latents_dim]
              wave:   full wave [bsz,num_samples,1]
            @Return
              ret: {
                "intensity": [bsz,num_samples,3]
              }
        """
        ret = defaultdict(lambda: None)
        self.hps_decoder(coords, wave, None, None, full_wave_bound,
                         qtz_args=qtz_args, ret=ret)
        return ret
