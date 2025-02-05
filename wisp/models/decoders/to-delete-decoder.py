
import torch.nn as nn

from wisp.models.grids import *
from wisp.models.layers import get_layer_class
from wisp.models.activations import get_activation_class
from wisp.utils.common import get_input_latent_dim, query_GPU_mem

from wisp.models.decoders.basic_decoders import BasicDecoder, MLP
#import sys
#sys.path.insert(0, "./wisp/models/decoders")
#from wisp.models.decoders.garf import Garf
#from wisp.models.decoders.siren import Siren
#from wisp.models.decoders.basic_decoders import BasicDecoder


class Decoder(nn.Module):
    """ Wrapper class for different implementations of the MLP decoder.
    """
    def __init__(self, **kwargs):
        super(Decoder, self).__init__()

        self.kwargs = kwargs
        self.init_decoder()

    def get_input_dim(self):
        if self.kwargs["space_dim"] == 2:

            if self.kwargs["coords_encode_method"] == "positional_encoding":
                assert(self.kwargs["decoder_activation_type"] == "relu")
                input_dim = self.kwargs["coords_embed_dim"]

            elif self.kwargs["coords_encode_method"] == "grid":
                assert(self.kwargs["decoder_activation_type"] == "relu")
                input_dim = get_input_latent_dim(**self.kwargs)
            else:
                assert(self.kwargs["decoder_activation_type"] == "sin")
                input_dim = self.kwargs["space_dim"]

        else: #if kwargs["space_dim"] == 3:
            if self.kwargs["quantize_latent"] or self.kwargs["quantize_spectra"]:
                latents_dim = self.kwargs["qtz_latent_dim"]
            elif self.kwargs["decode_spatial_embedding"]:
                latents_dim = self.kwargs["spatial_decod_output_dim"]
            else:
                latents_dim = get_input_latent_dim(**self.kwargs)

            if self.kwargs["encode_wave"]:
                if self.kwargs["hps_combine_method"] == "add":
                    assert(self.kwargs["wave_embed_dim"] == latents_dim)
                    input_dim = self.kwargs["wave_embed_dim"]
                elif self.kwargs["hps_combine_method"] == "concat":
                    input_dim = self.kwargs["wave_embed_dim"] + latents_dim

            else: # coords and wave are not encoded
                input_dim = 3

        return input_dim

    def get_output_dim(self):
        if self.kwargs["space_dim"] == 2:
            return self.kwargs["num_bands"]
        return 1

    def init_decoder(self):
        """ Initializes MLP.
        """
        input_dim = self.get_input_dim()
        output_dim = self.get_output_dim()

        if self.kwargs["decoder_activation_type"] == "relu":
            # self.decoder = BasicDecoder(
            #     input_dim, output_dim,
            #     bias=True,
            #     activation=get_activation_class(self.kwargs["decoder_activation_type"]),
            #     layer=get_layer_class(self.kwargs["decoder_layer_type"]),
            #     num_layers=self.kwargs["decoder_num_layers"] + 1,
            #     hidden_dim=self.kwargs["decoder_hidden_dim"],
            #     skip=[])
            self.decoder = MLP(input_dim, output_dim,
                               self.kwargs["decoder_num_hidden_layers"],
                               hidden_dim=self.kwargs["decoder_hidden_dim"])

        elif self.kwargs["decoder_activation_type"] == "sin":
            self.decoder = Siren(
                input_dim, output_dim,
                self.kwargs["decoder_num_layers"],
                self.kwargs["decoder_hidden_dim"],
                self.kwargs["siren_first_w0"],
                self.kwargs["siren_hidden_w0"],
                self.kwargs["siren_seed"],
                self.kwargs["siren_coords_scaler"],
                self.kwargs["siren_last_linear"])

        # elif self.kwargs["decoder_activation_type"] == "gaussian":
        #     self.decoder = Garf(input_dim, **self.kwargs)

        else: raise ValueError("Unrecognized decoder activation type.")

    def forward(self, input):
        return self.decoder(input)
