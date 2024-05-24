
import torch
import torch.nn as nn

from wisp.utils import PerfTimer
from wisp.utils.common import get_bool_classify_redshift, \
    get_bool_has_redshift_latents, get_bool_weight_spectra_loss_with_global_restframe_loss, \
    get_bool_save_lambdawise_spectra_loss

from wisp.models.nefs import BaseNeuralField
from wisp.models.embedders.encoder import Encoder
from wisp.models.decoders import BasicDecoder, SpatialDecoder
from wisp.models.hypers import HyperSpectralDecoder, HyperSpectralDecoderB
from wisp.models.layers import get_layer_class, init_codebook, Quantization


class RedshiftClassifier():
    def __init__(self, **kwargs):
        super(RedshiftClassifier, self).__init__()
        assert kwargs["model_redshift"], "we must model redshift during pretrain"
        self.kwargs = kwargs

        self.init_model()

    def init_model(self):
        embedder_args = (
            1, self.kwargs["wave_embed_dim"],
            self.kwargs["wave_embed_omega"],
            self.kwargs["wave_embed_sigma"],
            self.kwargs["wave_embed_bias"],
            self.kwargs["wave_embed_seed"])

        self.encoder = Encoder(
            input_dim=1,
            encode_method=self.kwargs["wave_encode_method"],
            embedder_args=embedder_args,
            **self.kwargs)

        input_dim = 2 * self.kwargs["wave_embed_dim"]
        output_dim = 1
        self.decoder = BasicDecoder(
            input_dim, output_dim, True,
            num_layers=self.kwargs["decoder_num_hidden_layers"] + 1,
            hidden_dim=self.kwargs["decoder_hidden_dim"],
            batch_norm=self.kwargs["decoder_batch_norm"],
            layer_type=self.kwargs["decoder_layer_type"],
            activation_type=self.kwargs["decoder_activation_type"],
            skip=self.kwargs["decoder_skip_layers"],
            skip_method=self.kwargs["decoder_latents_skip_method"],
            skip_all_layers=self.kwargs["decoder_latents_skip_all_layers"],
            activate_before_skip=self.kwargs["decoder_activate_before_latents_skip"],
            skip_add_conversion_method=\
                self.kwargs["decoder_latents_skip_add_conversion_method"]
        )

    def index_latents(self, data, selected_ids, idx):
        ret = data
        if selected_ids is not None:
            ret = ret[selected_ids]
        if idx is not None:
            ret = ret[idx]
        return ret

    def forward(self, lambdawise_losses, wave, idx=None, selected_ids=None):
        print(lambdawise_losses.shape, wave.shape)

        assert 0
