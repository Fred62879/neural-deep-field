
import torch

from wisp.models.grids import *
from wisp.utils import PerfTimer
from wisp.models.nefs import BaseNeuralField
from wisp.models.decoders import BasicDecoder, Siren
#from wisp.models.siren import Siren
from wisp.models.activations import get_activation_class
from wisp.models.layers import get_layer_class, Normalization
from wisp.models.embedders import get_positional_embedder, RandGaus


class NeuralHyperSpectral(BaseNeuralField):
    """ Model for encoding RA/DEC coordinates.
        In hyperspectral setup, output is embedded latent variables.
        Otherwise, output is decoded pixel intensities.

        Different from how the base_nef works,
          here we use either the positional encoding or the grid for embedding.

        Usage:
          2D: embedding (positional/grid) -- relu -- sinh
              no embedding -- siren
          3D: embedding (positional/grid)
              no embedding
    """

    def init_embedder(self):
        if self.kwargs["coords_embed_method"] != "positional": return

        pe_dim = self.kwargs["coords_embed_dim"]
        sigma = 1
        omega = 1
        pe_bias = True
        verbose = False
        seed = 0
        input_dim = 2
        self.embedder = RandGaus((input_dim, pe_dim, omega, sigma, pe_bias, seed, verbose))

    def init_grid(self):
        """ Initialize the grid object. """
        if self.kwargs["coords_embed_method"] != "grid": return

        if self.grid_type == "OctreeGrid":
            grid_class = OctreeGrid
        elif self.grid_type == "CodebookOctreeGrid":
            grid_class = CodebookOctreeGrid
        elif self.grid_type == "TriplanarGrid":
            grid_class = TriplanarGrid
        elif self.grid_type == "HashGrid":
            grid_class = HashGrid
        else:
            raise NotImplementedError

        self.grid = grid_class(
            self.feature_dim, grid_dim=self.grid_dim,
            base_lod=self.base_lod, num_lods=self.num_lods,
            interpolation_type=self.interpolation_type,
            multiscale_type=self.multiscale_type, **self.kwargs)

        self.grid.init_from_geometric(
            self.kwargs["min_grid_res"], self.kwargs["max_grid_res"], self.num_lods)

        if self.multiscale_type == 'cat':
            self.effective_feature_dim = self.grid.feature_dim * self.num_lods
        else:
            self.effective_feature_dim = self.grid.feature_dim

    def init_decoder(self):
        """ Initializes the decoder object.
        """
        # hyperspectral setup w.o/ quantization doesn't need decoder
        if self.space_dim == 3 and not self.kwargs["quantize_latent"]:
            return

        if self.kwargs["coords_embed_method"] == "positional":
            assert(self.activation_type == "relu")
            input_dim = self.kwargs["coords_embed_dim"]
        elif self.kwargs["coords_embed_method"] == "grid":
            assert(self.activation_type == "relu")
            input_dim = self.effective_feature_dim
        else:
            assert(self.activation_type == "sin")
            input_dim = self.space_dim

        # set decoder output dimension
        if self.kwargs["quantize_latent"]:
            output_dim = self.kwargs["qtz_latent_dim"] + \
                self.kwargs["generate_scaler"] + self.kwargs["generate_redshift"]
        else:
            output_dim = self.output_dim

        # intialize decoder
        if self.space_dim == 3 and self.kwargs["quantize_latent"]:
            self.decoder_latent = BasicDecoder(
                input_dim, output_dim,
                get_activation_class(self.activation_type),
                True, layer=get_layer_class(self.layer_type),
                num_layers=self.num_layers+1,
                hidden_dim=self.hidden_dim, skip=[])

        elif self.activation_type == "relu":
            self.decoder_intensity = BasicDecoder(
                input_dim, output_dim,
                get_activation_class(self.activation_type),
                True, layer=get_layer_class(self.layer_type),
                num_layers=self.num_layers+1,
                hidden_dim=self.hidden_dim, skip=[])

        elif self.activation_type == "sin":
            # self.decoder_intensity = Siren(
            #     input_dim, output_dim, self.num_layers, self.hidden_dim,
            #     self.kwargs["siren_first_w0"], self.kwargs["siren_hidden_w0"],
            #     self.kwargs["siren_seed"], self.kwargs["siren_coords_scaler"],
            #     self.kwargs["siren_last_linear"])
            #dim_in, dim_hidden, dim_out, num_hidden_layers, last_linr, \
            #first_w0, hidden_w0, coords_scaler, seed, float_tensor
            self.decoder_intensity = Siren((2,256,5,3,False,24,6,8,0,torch.FloatTensor.cuda))

        else: raise ValueError("Unrecognized decoder activation type.")

        self.norm = Normalization(self.kwargs["mlp_output_norm_method"])

    def get_nef_type(self):
        return 'hyperspectral'

    def register_forward_functions(self):
        """ Register forward functions with the channels that they output.
            For hyperspectral setup, we need latent embedding as output.
            Otherwise, we need pixel intensity values as output.
        """
        if self.space_dim == 2:
            channels = ["intensity"]
        else:
            channels = ["latents"]
            if self.kwargs["quantize_latent"]:
                channels.append("scaler")
                channels.append("redshift")

        self._register_forward_function( self.hyperspectral, channels )

    def hyperspectral(self, coords, pidx=None, lod_idx=None):
        """ Compute hyperspectral intensity for the provided coordinates.
            @Params:
              coords (torch.FloatTensor): tensor of shape [1, batch, num_samples, 2/3]
              pidx (torch.LongTensor): SPC point_hierarchy indices of shape [batch].
                                       Unused in the current implementation.
              lod_idx (int): index into active_lods. If None, will use the maximum LOD.
                             Currently interpolation doesn't use this.
            @Return
              {"indensity": torch.FloatTensor }:
                - Output intensity tensor of shape [batch, num_samples, 3]
        """
        timer = PerfTimer(activate=False, show_memory=True)

        if self.kwargs["print_shape"]: print('hyper nef',coords.shape)

        if coords.ndim == 2:
            batch, _ = coords.shape
        elif coords.ndim == 3:
            batch, num_samples, _ = coords.shape
        elif coords.ndim == 4: # **** replace
            _, _, num_samples, dim = coords.shape
            coords = coords.view(-1,num_samples,dim)
            batch = coords.shape[0]
        else:
            raise Exception("Wrong coordinate dimension.")

        if self.kwargs["print_shape"]: print('hyper nef', coords.shape)
        timer.check("rf_hyperspectral_preprocess")

        # embed 2D coords into high-dimensional vectors with PE or the grid
        if self.kwargs["coords_embed_method"] == "positional":
            feats = self.embedder(coords) # [bsz,num_samples,coords_embed_dim]
            timer.check("rf_hyperspectral_pe")

        elif self.kwargs["coords_embed_method"] == "grid":
            if lod_idx is None:
                lod_idx = len(self.grid.active_lods) - 1

            feats = self.grid.interpolate(coords, lod_idx)
            feats = feats.reshape(-1, self.effective_feature_dim)
            timer.check("rf_hyperspectra_interpolate")
        else:
            feats = coords

        timer.check("rf_hyperspectral_embedding")

        if self.space_dim == 3:
            feats = feats.reshape(batch, num_samples, -1)
            if not self.kwargs["quantize_latent"]:
                return dict(latents=feats)

            latents = self.decoder_latent(feats)
            if self.kwargs["generate_scaler"]:
                if self.kwargs["generate_redshift"]:
                    scaler = latents[...,-2:-1]
                    redshift = latents[...,-1:]
                    latents = latents[...,:-2]
                else:
                    redshift = None
                    scaler = latents[...,-1:]
                    latents = latents[...,:-1]
            else: redshift, scaler = None, None
            return dict(latents=latents, scaler=scaler, redshift=redshift)

        intensity = self.decoder_intensity(feats)
        intensity = self.norm(intensity)
        timer.check("rf_hyperspectral_decode")
        return dict(intensity=intensity)
