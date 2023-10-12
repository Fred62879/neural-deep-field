
import torch
import torch.nn as nn
import torch.nn.functional as F

from wisp.utils import PerfTimer
from wisp.utils.common import get_input_latent_dim, init_redshift_bins

from wisp.models.layers import get_layer_class
from wisp.models.decoders import BasicDecoder
from wisp.models.decoders.basic_decoders import average
from wisp.models.activations import get_activation_class


class RedshiftDecoder(nn.Module):
    """ Decoder input latent variables to redshift values.
    """
    def __init__(self, **kwargs):
        super(RedshiftDecoder, self).__init__()

        self.kwargs = kwargs
        self.init_model()

    def init_model(self):
        if self.kwargs["split_latent"]:
            self.input_dim = self.kwargs["redshift_logit_latent_dim"]
        else: self.input_dim = get_input_latent_dim(**self.kwargs)

        self.redshift_model_method = self.kwargs["redshift_model_method"]

        if self.redshift_model_method == "regression":
            output_dim = 1
            self.redshift_adjust = nn.ReLU(inplace=True)
        elif self.redshift_model_method == "classification":
            self.init_redshift_bins()
            output_dim = self.num_redshift_bins
        else: raise ValueError("Unsupported redshift modeling method!")

        if not self.kwargs["optimize_redshift_latents_as_logits"]:
            self.redshift_decoder = BasicDecoder(
                self.input_dim, output_dim,
                get_activation_class(self.kwargs["redshift_decod_activation_type"]),
                bias=True, layer=get_layer_class(self.kwargs["redshift_decod_layer_type"]),
                num_layers=self.kwargs["redshift_decod_num_hidden_layers"] + 1,
                hidden_dim=self.kwargs["redshift_decod_hidden_dim"],
                skip=self.kwargs["redshift_decod_skip_layers"]
            )
            # self.redshift_decoder.initialize_last_layer_equal()
            # for n,p in self.redshift_decoder.lout.named_parameters(): print(n, p)
        else: assert self.num_redshift_bins == self.kwargs["redshift_logit_latent_dim"]

    def init_redshift_bins(self):
        if self.kwargs["use_gpu"]:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else: device = "cpu"
        self.redshift_bin_center = init_redshift_bins(
            self.kwargs["redshift_lo"], self.kwargs["redshift_hi"],
            self.kwargs["redshift_bin_width"]
        )
        self.redshift_bin_center = self.redshift_bin_center.to(device)
        self.num_redshift_bins = len(self.redshift_bin_center)

    def forward(self, z, ret, specz=None, redshift_latents=None, init_redshift_prob=None):
        """ Decode latent variables to various spatial information we need.
            @Param
              z: raw 2D coordinate or embedding of 2D coordinate [batch_size,1,dim]
              specz: spectroscopic (gt) redshift
        """
        timer = PerfTimer(activate=self.kwargs["activate_model_timer"],
                          show_memory=self.kwargs["show_memory"])
        timer.reset()

        if self.kwargs["split_latent"]:
            assert redshift_latents is not None
            latents = redshift_latents # [bsz,dim]
            assert latents.shape[-1] == self.kwargs["redshift_logit_latent_dim"]
        else: latents = z[:,0]

        if self.redshift_model_method == "regression":
            redshift = self.redshift_decoder(latents)[...,0]
            ret["redshift"] = self.redshift_adjust(redshift + 0.5)

        elif self.redshift_model_method == "classification":
            ret["redshift"]= self.redshift_bin_center # [num_bins]

            if self.kwargs["optimize_redshift_latents_as_logits"]:
                ret["redshift_logits"] = F.softmax(redshift_latents, dim=-1)
            else:
                logits = self.redshift_decoder(latents)
                if init_redshift_prob is not None:
                    logits = logits + init_redshift_prob
                ret["redshift_logits"] = F.softmax(logits, dim=-1) # [bsz,num_bins]
        else:
            raise ValueError("Unsupported redshift model method!")

        timer.check("spatial_decod::redshift done")
