
import torch
import torch.nn as nn

from wisp.utils import PerfTimer
from wisp.models.decoders import BasicDecoder, MLP
from wisp.utils.common import get_input_latents_dim
from wisp.models.activations import get_activation_class
from wisp.models.layers import get_layer_class, Quantization


class SpatialDecoder(nn.Module):
    """ Accept as input latent variables and quantize based on
          a codebook which is optimizaed simultaneously during training
    """
    def __init__(self, output_scaler, qtz_calculate_loss, **kwargs):
        super(SpatialDecoder, self).__init__()

        self.kwargs = kwargs

        # we either quantize latents or spectra or none
        self.quantize_z = kwargs["quantize_latent"]
        self.quantize_spectra = kwargs["quantize_spectra"]
        self.qtz = self.quantize_z or self.quantize_spectra
        assert not (self.quantize_z and self.quantize_spectra)

        self.qtz_calculate_loss = qtz_calculate_loss
        self.quantization_strategy = kwargs["quantization_strategy"]

        self.output_scaler = self.qtz and output_scaler

        # we either pred redshift and supervise or apply gt redshift directly or semi-sup
        assert sum([kwargs["apply_gt_redshift"],
                    kwargs["redshift_unsupervision"],
                    kwargs["redshift_semi_supervision"]]) <= 1

        self.model_redshift = kwargs["model_redshift"] and self.qtz
        self.apply_gt_redshift = self.model_redshift and \
            kwargs["apply_gt_redshift"] and self.qtz

        self.decode_spatial_embedding = kwargs["decode_spatial_embedding"]

        self.input_dim = get_input_latents_dim(**kwargs)
        self.init_model()

    def init_model(self):
        if self.decode_spatial_embedding or self.qtz:
            self.init_decoder()

        if self.quantize_z:
            self.qtz = Quantization(self.qtz_calculate_loss, **self.kwargs)

        if self.output_scaler:
            self.init_scaler_decoder()

        if self.model_redshift:
            self.init_redshift_decoder()

    def init_scaler_decoder(self):
        self.scaler_decoder = BasicDecoder(
            self.input_dim, 1,
            get_activation_class(self.kwargs["scaler_decod_activation_type"]),
            bias=True, layer=get_layer_class(self.kwargs["scaler_decod_layer_type"]),
            num_layers=self.kwargs["scaler_decod_num_hidden_layers"] + 1,
            hidden_dim=self.kwargs["scaler_decod_hidden_dim"], skip=[])
        # self.scaler_decoder = MLP(
        #     self.input_dim, 1,
        #     self.kwargs["scaler_decod_num_hidden_layers"],
        #     self.kwargs["scaler_decod_hidden_dim"])

    def init_redshift_decoder(self):
        self.redshift_decoder = BasicDecoder(
            self.input_dim, 1,
            get_activation_class(self.kwargs["redshift_decod_activation_type"]),
            bias=True, layer=get_layer_class(self.kwargs["redshift_decod_layer_type"]),
            num_layers=self.kwargs["redshift_decod_num_hidden_layers"] + 1,
            hidden_dim=self.kwargs["redshift_decod_hidden_dim"], skip=[])
        # self.redshift_decoder = MLP(
        #     self.input_dim, 1,
        #     self.kwargs["redshift_decod_num_hidden_layers"],
        #     self.kwargs["redshift_decod_hidden_dim"])
        self.redshift_adjust = nn.ReLU(inplace=True)

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
            hidden_dim=self.kwargs["spatial_decod_hidden_dim"], skip=[])
        # self.decode = MLP(
        #     self.input_dim, output_dim,
        #     self.kwargs["spatial_decod_num_hidden_layers"],
        #     self.kwargs["spatial_decod_hidden_dim"])

    def forward(self, z, codebook, qtz_args, ret, specz=None, sup_id=None):
        """ Decode latent variables
            @Param
              z: raw 2D coordinate or embedding of 2D coordinate [batch_size,1,dim]
              specz: spectroscopic (gt) redshift
              sup_id: id of pixels to supervise with gt redshift
        """
        timer = PerfTimer(activate=self.kwargs["activate_model_timer"], show_memory=False)
        timer.reset()

        if self.output_scaler:
            scaler = self.scaler_decoder(z[:,0])[...,0]
        else: scaler = None
        timer.check("spatial_decod::scaler done")

        if self.apply_gt_redshift:   # dont generate redshift
            assert specz is not None
            redshift = specz
            ret["redshift"] = redshift
        elif self.model_redshift:
            redshift = self.redshift_decoder(z[:,0])[...,0]
            redshift = self.redshift_adjust(redshift + 0.5)
        else:
            redshift = None
        timer.check("spatial_decod::redshift done")

        if self.quantize_spectra:
            logits = self.decode(z)
        else:
            if self.decode_spatial_embedding:
                z = self.decode(z)
            if self.quantize_z:
                z, z_q = self.qtz(z, codebook.weight, ret, qtz_args)
        timer.check("spatial_decod::qtz done")

        ret["latents"] = z
        ret["scaler"] = scaler
        ret["redshift"] = redshift

        if self.quantize_spectra: return logits
        if self.quantize_z:       return z_q
        return z
