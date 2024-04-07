
import torch
import torch.nn as nn
import torch.nn.functional as F

from wisp.utils import PerfTimer
from wisp.utils.common import get_input_latent_dim

from wisp.models.layers import Quantization
from wisp.models.decoders.basic_decoders import BasicDecoder
from wisp.models.decoders.scaler_decoder import ScalerDecoder
from wisp.models.decoders.redshift_decoder import RedshiftDecoder


class SpatialDecoder(nn.Module):
    """ Accept as input latent variables and quantize based on
          a codebook which is optimizaed simultaneously during training
    """
    def __init__(self, output_bias, output_scaler, qtz_calculate_loss, **kwargs):
        super(SpatialDecoder, self).__init__()

        self.kwargs = kwargs

        self.output_bias = output_bias
        self.output_scaler = output_scaler
        self.model_redshift = kwargs["model_redshift"]

        # we either quantize latents or spectra or none
        self.quantize_latent = kwargs["quantize_latent"]
        self.quantize_spectra = kwargs["quantize_spectra"]
        self.qtz = self.quantize_latent or self.quantize_spectra
        assert not (self.quantize_latent and self.quantize_spectra)

        self.qtz_calculate_loss = qtz_calculate_loss
        self.quantization_strategy = kwargs["quantization_strategy"]
        self.decode_spatial_embedding = kwargs["decode_spatial_embedding"]

        self.init_model()

    def init_model(self):
        self.input_dim = get_input_latent_dim(**self.kwargs)

        if self.quantize_spectra and self.kwargs["optimize_spectra_latents_as_logits"]:
            pass
        elif self.decode_spatial_embedding or self.qtz:
            self.init_decoder()

        if self.quantize_latent:
            self.qtz = Quantization(self.qtz_calculate_loss, **self.kwargs)

        if self.output_scaler or self.output_bias:
            self.scaler_decoder = ScalerDecoder(
                self.output_bias, self.output_scaler, qtz=self.qtz, **self.kwargs)

        if self.model_redshift:
            self.redshift_decoder = RedshiftDecoder(**self.kwargs)

    def init_decoder(self):
        if self.quantize_latent:
            if self.quantization_strategy == "soft":
                # decode into score corresponding to each code
                output_dim = self.kwargs["qtz_num_embed"]
            elif self.quantization_strategy == "hard":
                output_dim = self.kwargs["qtz_latent_dim"]
            else:
                raise ValueError("Unsupporteed quantization strategy.")
        elif self.quantize_spectra:
            output_dim = self.kwargs["qtz_num_embed"]
        elif self.decode_spatial_embedding:
            output_dim = self.kwargs["spatial_decod_output_dim"]
        else: return

        skip_layers = self.kwargs["spatial_decod_skip_layers"]

        self.decode = BasicDecoder(
            self.input_dim, output_dim, True,
            layer_type=self.kwargs["spatial_decod_layer_type"],
            activation_type=self.kwargs["spatial_decod_activation_type"],
            num_layers=self.kwargs["spatial_decod_num_hidden_layers"] + 1,
            hidden_dim=self.kwargs["spatial_decod_hidden_dim"],
            skip=skip_layers
        )

    def forward(self, z, codebook, qtz_args, ret,
                specz=None,
                scaler_latents=None,
                redshift_latents=None,
                sup_id=None, # DELETE
                init_redshift_prob=None # debug
    ):
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

        if self.output_scaler or self.output_bias:
            self.scaler_decoder(latent, ret, scaler_latents)

        if self.model_redshift:
            self.redshift_decoder(
                z, ret, specz, redshift_latents, init_redshift_prob)

        # quantize/decode/nothing
        if self.quantize_spectra:
            coeff = z if self.kwargs["optimize_spectra_latents_as_logits"] else self.decode(z)
            ret["codebook_logits"] = coeff[:,0]
        elif self.quantize_latent:
            z, q_z = self.qtz(z, codebook.weight, ret, qtz_args)
        elif self.decode_spatial_embedding:
            z = self.decode(z)
        timer.check("spatial_decod::qtz done")

        ret["latents"] = z
        if self.quantize_spectra: return coeff
        elif self.quantize_latent: return q_z
        return z
