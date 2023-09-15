
import torch
import torch.nn as nn
import torch.nn.functional as F

from wisp.utils import PerfTimer
from wisp.utils.common import get_input_latents_dim

from wisp.models.decoders import BasicDecoder
from wisp.models.layers import get_layer_class
from wisp.models.activations import get_activation_class


class RedshiftDecoder(nn.Module):
    """ Decoder input latent variables to redshift values.
    """
    def __init__(self, _apply_gt_redshift, **kwargs):
        super(RedshiftDecoder, self).__init__()

        self.kwargs = kwargs

        # at most one can be true
        assert sum([kwargs["apply_gt_redshift"],
                    kwargs["redshift_unsupervision"],
                    kwargs["redshift_semi_supervision"]]) <= 1

        self.apply_gt_redshift = _apply_gt_redshift

        if not self.apply_gt_redshift:
            self.init_model()

    def init_model(self):
        self.input_dim = get_input_latents_dim(**self.kwargs)
        self.redshift_model_method = self.kwargs["redshift_model_method"]

        if self.redshift_model_method == "regression":
            output_dim = 1
            self.redshift_adjust = nn.ReLU(inplace=True)
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

    def init_redshift_bins(self):
        if self.kwargs["use_gpu"]:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else: device = "cpu"
        self.redshift_bin_center = torch.arange(
            self.kwargs["redshift_lo"],
            self.kwargs["redshift_hi"],
            self.kwargs["redshift_bin_width"]).to(device)
        self.num_redshift_bins = len(self.redshift_bin_center)
        offset = self.kwargs["redshift_bin_width"] / 2
        self.redshift_bin_center += offset

    def forward(self, z, ret, specz=None):
        """ Decode latent variables to various spatial information we need.
            @Param
              z: raw 2D coordinate or embedding of 2D coordinate [batch_size,1,dim]
              specz: spectroscopic (gt) redshift
        """
        timer = PerfTimer(activate=self.kwargs["activate_model_timer"], show_memory=False)
        timer.reset()

        if self.apply_gt_redshift:
            assert specz is not None
            ret["redshift"] = specz
        else:
            if self.redshift_model_method == "regression":
                redshift = self.redshift_decoder(z[:,0])[...,0]
                ret["redshift"] = self.redshift_adjust(redshift + 0.5)
            elif self.redshift_model_method == "classification":
                # bsz = z.shape[0]
                # ret["redshift"]= self.redshift_bin_center[None,:].tile(bsz,1) # [bsz,num_bins]
                ret["redshift"]= self.redshift_bin_center # [num_bins]
                ret["redshift_logits"] = F.softmax(
                    self.redshift_decoder(z[:,0]), dim=-1) # [bsz,num_bins]
            else:
                raise ValueError("Unsupported redshift model method!")

        timer.check("spatial_decod::redshift done")
