
import torch
import torch.nn as nn
import torch.nn.functional as F

from wisp.utils import PerfTimer
from wisp.utils.common import get_input_latent_dim
from wisp.models.decoders.basic_decoders import BasicDecoder


class ScalerDecoder(nn.Module):
    """ Accept as input latent variables and quantize based on
          a codebook which is optimizaed simultaneously during training
    """
    def __init__(self, output_bias, output_scaler, qtz, **kwargs):
        super(ScalerDecoder, self).__init__()
        self.kwargs = kwargs

        self.output_bias = qtz and output_bias
        self.output_scaler = qtz and output_scaler

        if self.output_scaler or self.output_bias:
            self.init_model()

    def init_model(self):
        if self.kwargs["split_latent"]:
            self.input_dim = self.kwargs["scaler_latent_dim"]
        else: self.input_dim = get_input_latent_dim(**self.kwargs)

        output_dim = 0
        if self.output_bias: output_dim += 1
        if self.output_scaler: output_dim += 1

        self.scaler_decoder = BasicDecoder(
            self.input_dim, output_dim, True,
            layer_type=self.kwargs["scaler_decod_layer_type"],
            activation_type=self.kwargs["scaler_decod_activation_type"],
            num_layers=self.kwargs["scaler_decod_num_hidden_layers"] + 1,
            hidden_dim=self.kwargs["scaler_decod_hidden_dim"],
            skip=self.kwargs["scaler_decod_skip_layers"]
        )

    def forward(self, z, ret, scaler_latent_mask=None):
        """ Decode latent variables to various spatial information we need.
            @Param
              z: raw 2D coordinate or embedding of 2D coordinate [batch_size,1,dim]
              codebook: codebook used for quantization
              qtz_args: arguments for quantization operations
              specz: spectroscopic (gt) redshift
              sup_id: id of pixels to supervise with gt redshift (OBSOLETE)
        """
        timer = PerfTimer(activate=self.kwargs["activate_model_timer"],
                          show_memory=self.kwargs["show_memory"])
        timer.reset()

        if scaler_latent_mask is not None:
            latent = z[:,0] * scaler_latent_mask
        else: latent = z[:,0]

        if self.output_scaler or self.output_bias:
            out = self.scaler_decoder(latent)
            if self.output_scaler:
                ret["scaler"] = out[...,0]
                if self.output_bias:
                    ret["bias"] = out[...,1]

        timer.check("spatial_decod::scaler done")
