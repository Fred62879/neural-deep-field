
import torch
import torch.nn as nn

from wisp.utils import PerfTimer
from wisp.utils.common import print_shape, get_input_latents_dim

from wisp.models.decoders import Decoder, BasicDecoder
from wisp.models.activations import get_activation_class
from wisp.models.hypers.hps_converter import HyperSpectralConverter
from wisp.models.hypers.hps_integrator import HyperSpectralIntegrator
from wisp.models.layers import Intensifier, Quantization, get_layer_class


class HyperSpectralDecoder(nn.Module):

    def __init__(self, scale=True,
                 add_bias=True,
                 integrate=True,
                 intensify=True,
                 qtz_spectra=True,
                 _model_redshift=True,
                 **kwargs
    ):
        super(HyperSpectralDecoder, self).__init__()

        self.kwargs = kwargs
        self.scale = scale
        self.add_bias = add_bias
        self.intensify = intensify
        self.qtz_spectra = qtz_spectra

        self.convert = HyperSpectralConverter(
            _model_redshift=_model_redshift, **kwargs
        )
        self.redshift_model_method = self.kwargs["redshift_model_method"]

        self.init_decoder()
        if self.qtz_spectra:
            self.qtz = Quantization(False, **kwargs)
        if self.intensify:
            self.intensifier = Intensifier(kwargs["intensification_method"])
        self.inte = HyperSpectralIntegrator(integrate=integrate, **kwargs)

    def get_input_dim(self):
        if self.kwargs["space_dim"] == 2:

            if self.kwargs["coords_encode_method"] == "positional_encoding":
                assert(self.kwargs["decoder_activation_type"] == "relu")
                input_dim = self.kwargs["coords_embed_dim"]

            elif self.kwargs["coords_encode_method"] == "grid":
                assert(self.kwargs["decoder_activation_type"] == "relu")
                input_dim = get_input_latents_dim(**self.kwargs)
            else:
                assert(self.kwargs["decoder_activation_type"] == "sin")
                input_dim = self.kwargs["space_dim"]

        else: #if kwargs["space_dim"] == 3:
            if self.kwargs["quantize_latent"] or self.kwargs["quantize_spectra"]:
                latents_dim = self.kwargs["qtz_latent_dim"]
            elif self.kwargs["decode_spatial_embedding"]:
                latents_dim = self.kwargs["spatial_decod_output_dim"]
            else:
                latents_dim = get_input_latents_dim(**self.kwargs)

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
        # self.spectra_decoder = Decoder(**kwargs)
        input_dim = self.get_input_dim()
        output_dim = self.get_output_dim()
        self.spectra_decoder = BasicDecoder(
            input_dim, output_dim,
            get_activation_class(self.kwargs["decoder_activation_type"]),
            bias=True, layer=get_layer_class(self.kwargs["decoder_layer_type"]),
            num_layers=self.kwargs["decoder_num_hidden_layers"] + 1,
            hidden_dim=self.kwargs["decoder_hidden_dim"],
            skip=self.kwargs["decoder_skip_layers"]
        )

    ##################
    # model forwarding
    ##################

    def reconstruct_spectra(self, input, wave, scaler, bias, redshift,
                            wave_bound, ret, codebook, qtz_args
    ):
        """ Reconstruct emitted (under possibly multiple redshift values) spectra
              using given input and wave.
            Scale spectra intensity using scaler, bias, and sinh func, if needed.
            @Param:
               input: 2D coords or embedded latents or logits [bsz,1,2/embed_dim]
               wave:  [bsz,num_samples,1]
               redshift: [num_bins] if model redshift using classification model
                         [bsz] o.w.
               wave_bound: lo and hi lambda values used for linear normalization [2]
               codebook: nn.Parameter([num_embed,embed_dim])
            @Return
               spectra: reconstructed emitted spectra [bsz,num_nsmpl]
        """
        # assert (self.redshift_model_method == "regression" and redshift.ndim == 1) or \
        #     (self.redshift_model_method == "classification" and redshift.ndim != 1)
        bsz = wave.shape[0]

        if self.qtz_spectra:
            codes = codebook.weight[:,None,None].tile(1,bsz,1,1)
            latents = self.convert(
                wave, codes, redshift, wave_bound) # [...,num_embed,bsz,nsmpl,dim]
            codebook_spectra = self.spectra_decoder(latents)[...,0] # [...,num_embed,bsz,nsmpl]

            # input here is logits
            _, spectra = self.qtz(input, codebook_spectra, ret, qtz_args)
            spectra = torch.squeeze(spectra, dim=-2) # [bsz,nsmpl]
        else:
            latents = self.convert(wave, input, redshift, wave_bound) # [...,bsz,nsmpl,dim]
            spectra = self.spectra_decoder(latents)[...,0] # [...,bsz,nsmpl]

        if self.redshift_model_method == "classification":
            # spectra [num_redshift_bins,bsz,nsmpl]; logits [bsz,num_redshift_bins]
            spectra = torch.matmul(ret["redshift_logits"][:,None], spectra.permute(1,0,2))[:,0]

        if self.scale:
            assert scaler is not None
            spectra = (scaler * spectra.T).T
        if self.add_bias:
            assert bias is not None
            spectra = spectra + bias[:,None]
        if self.intensify:
            spectra = self.intensifier(spectra)

        return spectra

    def forward_sup_spectra(self, latents, wave, full_wave_bound,
                            num_spectra_coords, ret, codebook, qtz_args):
        """ During training, some latents will be decoded, combining with full wave.
            Currently only supports spectra coords (incl. gt, dummy that requires
              spectrum plotting during training time).
            Latents that require full wave are placed at the end of the tensor.
        """
        latents = latents[-num_spectra_coords:]
        bias = None if ret["bias"] is None else ret["bias"][-num_spectra_coords:]
        scaler = None if ret["scaler"] is None else ret["scaler"][-num_spectra_coords:]
        redshift = None if ret["redshift"] is None else ret["redshift"][-num_spectra_coords:]

        ret["sup_spectra"] = self.reconstruct_spectra(
            latents, wave, scaler, bias, redshift,
            full_wave_bound, ret, codebook, qtz_args
        )

        if ret["bias"] is not None:
            ret["bias"] = ret["bias"][:-num_spectra_coords]
        if ret["scaler"] is not None:
            ret["scaler"] = ret["scaler"][:-num_spectra_coords]
        if ret["redshift"] is not None:
            ret["redshift"] = ret["redshift"][:-num_spectra_coords]

    def forward(self, latents,
                wave, trans, nsmpl, full_wave_bound,
                trans_mask=None,
                num_sup_spectra=-1, sup_spectra_wave=None,
                codebook=None, qtz_args=None, ret=None):
        """ @Param
            latents:   (encoded or original) coords or logits for quantization.
                         [bsz,1,space_dim or coords_encode_dim]

            - hyperspectral
              wave:      lambda values, used to convert ra/dec to hyperspectral latents.
                         [bsz,num_samples,1]
              trans:     corresponding transmission values of lambda. [(bsz,)nbands,num_samples]
              nsmpl:     average number of lambda samples falling within each band. [num_bands]
              full_wave_bound: min and max value of lambda
                               used for linear normalization of lambda
            - spectra supervision
              num_spectra_coords: > 0 if spectra supervision.
                                  The last #num_spectra_coords entries in latents are for sup.
              sup_spectra_wave: not None if do spectra supervision.
                                [num_spectra_coords,full_num_samples]
            - spectra qtz
              codebook    nn.Parameter [num_embed,embed_dim]
              qtz_args

            ret (output from nerf and/or quantization): {
                "scaler":        unique scaler value for each coord. [bsz,1]
                "redshift":      unique redshift value for each coord. [bsz,1]
                "embed_ids":     ids of embedding each pixel's latent is quantized to.
                "codebook_loss": loss for codebook optimization.
              }

            @Return (added to `ret`)
              spectra:   reconstructed spectra
              intensity: reconstructed pixel values
              sup_spectra: reconstructed spectra used for spectra supervision
        """
        timer = PerfTimer(activate=self.kwargs["activate_model_timer"], show_memory=False)
        timer.reset()

        perform_spectra_supervision = num_sup_spectra > 0

        if perform_spectra_supervision:
            self.forward_sup_spectra(
                latents, sup_spectra_wave, full_wave_bound,
                num_sup_spectra, ret, codebook, qtz_args
            )
            latents = latents[:-num_sup_spectra]
            if latents.shape[0] == 0: return

        ret["spectra"] = self.reconstruct_spectra(
            latents, wave, ret["scaler"], ret["bias"], ret["redshift"],
            full_wave_bound, ret, codebook, qtz_args
        )
        timer.check("hps_decoder::spectra reconstruced")

        intensity = self.inte(ret["spectra"], trans, trans_mask, nsmpl)
        ret["intensity"] = intensity
        timer.check("hps_decoder::integration done")
