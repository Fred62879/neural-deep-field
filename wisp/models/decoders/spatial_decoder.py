
import torch
import torch.nn as nn
import torch.nn.functional as F

from wisp.utils import PerfTimer
from wisp.models.decoders import BasicDecoder, MLP
from wisp.utils.common import get_input_latent_dim
from wisp.models.activations import get_activation_class
from wisp.models.layers import get_layer_class, Quantization
from wisp.models.decoders.scaler_decoder import ScalerDecoder
from wisp.models.decoders.redshift_decoder import RedshiftDecoder


class SpatialDecoder(nn.Module):
    """ Accept as input latent variables and quantize based on
          a codebook which is optimizaed simultaneously during training
    """
    def __init__(self, output_bias, output_scaler, decode_redshift, qtz_calculate_loss, **kwargs):
        super(SpatialDecoder, self).__init__()

        self.kwargs = kwargs

        self.output_bias = output_bias
        self.output_scaler = output_scaler
        self.decode_redshift = decode_redshift
        self.model_redshift = kwargs["model_redshift"]
        self.apply_gt_redshift = kwargs["apply_gt_redshift"]

        # we either quantize latents or spectra or none
        self.quantize_z = kwargs["quantize_latent"]
        self.quantize_spectra = kwargs["quantize_spectra"]
        self.qtz = self.quantize_z or self.quantize_spectra
        assert not (self.quantize_z and self.quantize_spectra)

        self.qtz_calculate_loss = qtz_calculate_loss
        self.quantization_strategy = kwargs["quantization_strategy"]
        self.decode_spatial_embedding = kwargs["decode_spatial_embedding"]

        self.init_model()

    def init_model(self):
        self.input_dim = get_input_latent_dim(**self.kwargs)

        if self.quantize_spectra and self.kwargs["optimize_codebook_latents_as_logits"]:
            pass
        elif self.decode_spatial_embedding or self.qtz:
            self.init_decoder()

        if self.quantize_z:
            self.qtz = Quantization(self.qtz_calculate_loss, **self.kwargs)

        if self.output_scaler or self.output_bias:
            self.scaler_decoder = ScalerDecoder(
                self.output_bias, self.output_scaler, qtz=self.qtz, **self.kwargs)

        if self.model_redshift and self.decode_redshift:
            self.redshift_decoder = RedshiftDecoder(**self.kwargs)

    def init_decoder(self):
        if self.quantize_z:
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

        self.decode = BasicDecoder(
            self.input_dim, output_dim,
            get_activation_class(self.kwargs["spatial_decod_activation_type"]),
            bias=True, layer=get_layer_class(self.kwargs["spatial_decod_layer_type"]),
            num_layers=self.kwargs["spatial_decod_num_hidden_layers"] + 1,
            hidden_dim=self.kwargs["spatial_decod_hidden_dim"],
            skip=self.kwargs["spatial_decod_skip_layers"]
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
            if self.decode_redshift:
                self.redshift_decoder(
                    z, ret, specz, redshift_latents, init_redshift_prob)
            elif self.apply_gt_redshift:
                assert specz is not None
                ret["redshift"] = specz

        # decode/quantize
        if self.quantize_spectra:
            if self.kwargs["optimize_codebook_latents_as_logits"]:
                logits = z
                ret["codebook_logits"] = logits[:,0]
            else:
                logits = self.decode(z)
        elif self.quantize_z:
            z, q_z = self.qtz(z, codebook.weight, ret, qtz_args)
        elif self.decode_spatial_embedding:
            z = self.decode(z)
        timer.check("spatial_decod::qtz done")

        ret["latents"] = z
        if self.quantize_spectra: return logits
        if self.quantize_z:       return q_z
        return z
