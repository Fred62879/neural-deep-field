
import torch
import torch.nn as nn

from collections import defaultdict
from wisp.models.nefs import BaseNeuralField
from wisp.models.hypers import HyperSpectralDecoder
from wisp.models.decoders import BasicDecoder, SpatialDecoder
from wisp.models.layers import get_layer_class, init_codebook, Quantization


class CodebookPretrainNerf(BaseNeuralField):
    def __init__(self, pretrain_pixel_supervision, _model_redshift=True, **kwargs):
        super(CodebookPretrainNerf, self).__init__()

        self.kwargs = kwargs
        self.model_redshift = _model_redshift
        self.pixel_supervision = pretrain_pixel_supervision
        self.init_model()

    def get_nef_type(self):
        return 'codebook_pretrain'

    def register_forward_functions(self):
        """ Register forward functions with the channels that they output.
        """
        channels = ["intensity","spectra","redshift","soft_qtz_weights","codebook"]
        self._register_forward_function(self.pretrain, channels)

    def init_model(self):
        self.codebook = init_codebook(
            self.kwargs["qtz_seed"], self.kwargs["qtz_num_embed"], self.kwargs["qtz_latent_dim"])

        self.spatial_decoder = SpatialDecoder(
            output_scaler=False,
            output_redshift=False,  #self.kwargs["generate_redshift"],
            apply_redshift=True, #self.kwargs["apply_gt_redshift"],
            qtz_calculate_loss=False,
            **self.kwargs)

        self.hps_decoder = HyperSpectralDecoder(
            integrate=self.pixel_supervision,
            scale=self.pixel_supervision,
            _model_redshift=self.model_redshift,
            **self.kwargs)

    def pretrain(self, coords, wave, full_wave_bound, trans=None, nsmpl=None, qtz_args=None, specz=None):
        """ Pretrain codebook.
            @Param
              coords: [num_supervision_spectra,latent_dim]
              wave:   full wave [bsz,nsmpl,1]
        """
        ret = defaultdict(lambda: None)
        bsz = coords.shape[0]
        coords = coords[:,None]

        latents = self.spatial_decoder(coords, self.codebook, qtz_args, ret, specz=specz)

        self.hps_decoder(
            latents, wave, trans, nsmpl, full_wave_bound,
            codebook=self.codebook, qtz_args=qtz_args,
            quantize_spectra=True, ret=ret
        )

        return ret
