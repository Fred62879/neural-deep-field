
import torch
import torch.nn as nn
import torch.nn.functional as F

from wisp.utils import PerfTimer
from wisp.models.decoders import BasicDecoder, MLP
from wisp.utils.common import get_input_latents_dim
from wisp.models.activations import get_activation_class
from wisp.models.layers import get_layer_class, Quantization


class SpatialDecoder(nn.Module):
    """ Accept as input latent variables and quantize based on
          a codebook which is optimizaed simultaneously during training
    """
    def __init__(self, output_bias, output_scaler, qtz_calculate_loss, **kwargs):
        super(SpatialDecoder, self).__init__()

        self.kwargs = kwargs

        # we either quantize latents or spectra or none
        self.quantize_z = kwargs["quantize_latent"]
        self.quantize_spectra = kwargs["quantize_spectra"]
        self.qtz = self.quantize_z or self.quantize_spectra
        assert not (self.quantize_z and self.quantize_spectra)

        self.qtz_calculate_loss = qtz_calculate_loss
        self.quantization_strategy = kwargs["quantization_strategy"]
        self.decode_spatial_embedding = kwargs["decode_spatial_embedding"]

        self.output_bias = self.qtz and output_bias
        self.output_scaler = self.qtz and output_scaler

        # we either pred redshift and supervise or apply gt redshift directly or semi-sup
        assert sum([kwargs["apply_gt_redshift"],
                    kwargs["redshift_unsupervision"],
                    kwargs["redshift_semi_supervision"]]) <= 1

        self.model_redshift = kwargs["model_redshift"]
        self.redshift_model_method = kwargs["redshift_model_method"]
        self.apply_gt_redshift = self.model_redshift and kwargs["apply_gt_redshift"]

        self.init_model()

    def init_model(self):
        self.input_dim = get_input_latents_dim(**self.kwargs)

        if self.decode_spatial_embedding or self.qtz:
            self.init_decoder()

        if self.quantize_z:
            self.qtz = Quantization(self.qtz_calculate_loss, **self.kwargs)

        if self.output_scaler or self.output_bias:
            self.init_scaler_decoder()

        if not self.apply_gt_redshift:
            self.init_redshift_decoder()

    def init_scaler_decoder(self):
        output_dim = 0
        if self.output_bias: output_dim += 1
        if self.output_scaler: output_dim += 1

        self.scaler_decoder = BasicDecoder(
            self.input_dim, output_dim,
            get_activation_class(self.kwargs["scaler_decod_activation_type"]),
            bias=True, layer=get_layer_class(self.kwargs["scaler_decod_layer_type"]),
            num_layers=self.kwargs["scaler_decod_num_hidden_layers"] + 1,
            hidden_dim=self.kwargs["scaler_decod_hidden_dim"],
            skip=self.kwargs["scaler_decod_skip_layers"]
        )

    def init_redshift_decoder(self):
        if self.redshift_model_method == "regression":
            output_dim = 1
        elif self.redshift_model_method == "classification":
            self.init_redshift_bins()
            output_dim = self.num_redshift_bins
        else: raise ValueError()

        self.redshift_decoder = BasicDecoder(
            self.input_dim, output_dim,
            get_activation_class(self.kwargs["redshift_decod_activation_type"]),
            bias=True, layer=get_layer_class(self.kwargs["redshift_decod_layer_type"]),
            num_layers=self.kwargs["redshift_decod_num_hidden_layers"] + 1,
            hidden_dim=self.kwargs["redshift_decod_hidden_dim"],
            skip=self.kwargs["redshift_decod_skip_layers"]
        )
        self.redshift_adjust = nn.ReLU(inplace=True)

    def init_redshift_bins(self):
        if self.kwargs["use_gpu"]:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else: device = "cpu"
        self.redshift_bin_center = torch.arange(
            self.kwargs["redshift_lo"],
            self.kwargs["redshift_hi"],
            self.kwargs["redshift_bin_width"]
        ).to(device)
        self.num_redshift_bins = len(self.redshift_bin_center)
        offset = self.kwargs["redshift_bin_width"] / 2
        self.redshift_bin_center += offset

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

    ##########
    # Forward
    ##########

    def generate_redshift(self, z, ret, specz, timer):
        redshift, redshift_logits = None, None
        if self.apply_gt_redshift:
            assert specz is not None
            ret["redshift"] = specz
        elif self.model_redshift:
            if self.redshift_model_method == "regression":
                redshift = self.redshift_decoder(z[:,0])[...,0]
                ret["redshift"] = self.redshift_adjust(redshift + 0.5)
            elif self.redshift_model_method == "classification":
                bsz = z.shape[0]
                ret["redshift"]= self.redshift_bin_center[None,:].tile(bsz,1) # [bsz,num_bins]
                ret["redshift_bin_prob"] = F.softmax(
                    self.redshift_decoder(z[:,0]), dim=-1) # [bsz,num_bins]
                ret["redshift_one_hot"] = torch.argmax(
                    ret["redshift_bin_prob"], dim=-1)
            else:
                raise ValueError("Unsupported redshift model method!")
        timer.check("spatial_decod::redshift done")

    def generate_scaler(self, z, ret, timer):
        if self.output_scaler or self.output_bias:
            out = self.scaler_decoder(z[:,0])
            if self.output_scaler:
                ret["scaler"] = out[...,0]
                if self.output_bias:
                    ret["bias"] = out[...,1]
        timer.check("spatial_decod::scaler done")

    def forward(self, z, codebook, qtz_args, ret, specz=None, sup_id=None):
        """ Decode latent variables to various spatial information we need.
            @Param
              z: raw 2D coordinate or embedding of 2D coordinate [batch_size,1,dim]
              codebook: codebook used for quantization
              qtz_args: arguments for quantization operations
              specz: spectroscopic (gt) redshift
              sup_id: id of pixels to supervise with gt redshift (OBSOLETE)
        """
        timer = PerfTimer(activate=self.kwargs["activate_model_timer"], show_memory=False)
        timer.reset()

        self.generate_scaler(z, ret, timer)
        self.generate_redshift(z, ret, specz, timer)

        # decode/quantize
        if self.quantize_spectra:
            logits = self.decode(z)
        elif self.quantize_z:
            z, z_q = self.qtz(z, codebook.weight, ret, qtz_args)
        elif self.decode_spatial_embedding:
            z = self.decode(z)
        timer.check("spatial_decod::qtz done")

        ret["latents"] = z
        if self.quantize_spectra: return logits
        if self.quantize_z:       return z_q
        return z
