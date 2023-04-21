
import torch
import torch.nn as nn

from wisp.models.decoders import BasicDecoder
from wisp.utils.common import get_input_latents_dim
from wisp.models.activations import get_activation_class
from wisp.models.layers import get_layer_class, Quantization


class SpatialDecoder(nn.Module):
    """ Accept as input latent variables and quantize based on
          a codebook which is optimizaed simultaneously during training
    """
    def __init__(self, qtz_calculate_loss, **kwargs):
        super(SpatialDecoder, self).__init__()

        self.kwargs = kwargs

        # we either quantize latents or spectra or none
        self.quantize_z = kwargs["quantize_latent"]
        self.quantize_spectra = kwargs["quantize_spectra"]
        assert not (self.quantize_z and self.quantize_spectra)

        self.qtz_calculate_loss = qtz_calculate_loss
        self.quantization_strategy = kwargs["quantization_strategy"]

        self.output_scaler = self.quantize_z and kwargs["generate_scaler"]
        self.output_redshift = self.quantize_z and kwargs["generate_redshift"]
        self.decode_spatial_embedding = kwargs["decode_spatial_embedding"]

        self.input_dim = get_input_latents_dim(**kwargs)
        self.init_model()

    def init_model(self):
        if self.decode_spatial_embedding: # or self.quantize_z:
            self.init_decoder()

        if self.quantize_z:
            self.qtz = Quantization(self.qtz_calculate_loss, **self.kwargs)

        if self.output_scaler:
            self.init_scaler_decoder()

        if self.output_redshift:
            self.init_redshift_decoder()

    def init_scaler_decoder(self):
        self.scaler_decoder = BasicDecoder(
            self.input_dim, 1,
            get_activation_class(self.kwargs["scaler_decod_activation_type"]),
            bias=True, layer=get_layer_class(self.kwargs["scaler_decod_layer_type"]),
            num_layers=self.kwargs["scaler_decod_num_hidden_layers"] + 1,
            hidden_dim=self.kwargs["scaler_decod_hidden_dim"], skip=[])

    def init_redshift_decoder(self):
        self.redshift_decoder = BasicDecoder(
            self.input_dim, 1,
            get_activation_class(self.kwargs["redshift_decod_activation_type"]),
            bias=True, layer=get_layer_class(self.kwargs["redshift_decod_layer_type"]),
            num_layers=self.kwargs["redshift_decod_num_hidden_layers"] + 1,
            hidden_dim=self.kwargs["redshift_decod_hidden_dim"], skip=[])
        self.redshift_adjust = nn.ReLU(inplace=True)

    def init_decoder(self):
        if self.quantize_z:
            if self.quantization_strategy == "soft":
                # decode into score corresponding to each code
                output_dim = self.kwargs["qtz_num_embed"]
            elif self.quantization_strategy == "hard":
                output_dim = self.kwargs["qtz_latent_dim"]
            else:
                raise ValueError("Unsupporteed quantization strategt.")
        elif self.decode_spatial_embedding:
            output_dim = self.kwargs["spatial_decod_output_dim"]

        self.decoder = BasicDecoder(
            self.input_dim, output_dim,
            get_activation_class(self.kwargs["spatial_decod_activation_type"]),
            bias=True, layer=get_layer_class(self.kwargs["spatial_decod_layer_type"]),
            num_layers=self.kwargs["spatial_decod_num_hidden_layers"] + 1,
            hidden_dim=self.kwargs["spatial_decod_hidden_dim"], skip=[])

    def forward(self, z, ret, qtz_args):
                # temperature=1, find_embed_id=False,
                # save_codebook=False, save_soft_qtz_weights=False):
        """ Decode latent variables
            @Param
              z: raw 2D coordinate or embedding of 2D coordinate [batch_size,1,dim]
        """
        if self.output_scaler:
            scaler = self.scaler_decoder(z[:,0])[...,0]
        else: scaler = None

        if self.output_redshift:
            redshift = self.redshift_decoder(z[:,0])[...,0]
            redshift = self.redshift_adjust(redshift)
        else: redshift = None

        if self.decode_spatial_embedding: # or self.quantize_z:
            z = self.decoder(z)

        if self.quantize_z:
            # z, z_q = self.qtz(z, ret, temperature=temperature,
            #                   find_embed_id=find_embed_id,
            #                   save_codebook=save_codebook,
            #                   save_soft_qtz_weights=save_soft_qtz_weights)
            z, z_q = self.qtz(z, ret, **qtz_args)

        elif self.quantize_spectra: pass

        ret["latents"] = z
        ret["scaler"] = scaler
        ret["redshift"] = redshift

        if self.quantize_z: return z_q
        return z
