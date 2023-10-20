
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from wisp.utils import PerfTimer
from wisp.utils.common import get_input_latent_dim, get_bool_classify_redshift

from wisp.models.decoders import Decoder, BasicDecoder
from wisp.models.activations import get_activation_class
from wisp.models.hypers.hps_integrator import HyperSpectralIntegrator
from wisp.models.hypers.hps_converter_batched import HyperSpectralConverter
from wisp.models.layers import Intensifier, Quantization, get_layer_class, ArgMax, \
    calculate_bayesian_redshift_logits

class HyperSpectralDecoderB(nn.Module):

    def __init__(self, scale=True,
                 add_bias=True,
                 integrate=True,
                 intensify=True,
                 qtz_spectra=True,
                 _model_redshift=True,
                 **kwargs
    ):
        super(HyperSpectralDecoderB, self).__init__()

        self.kwargs = kwargs
        self.scale = scale
        self.add_bias = add_bias
        self.intensify = intensify
        self.qtz_spectra = qtz_spectra

        self.reduction_order = "qtz_first"
        self.classify_redshift = _model_redshift and get_bool_classify_redshift(**kwargs)

        self.recon_codebook_spectra = kwargs["regu_codebook_spectra"] and \
            self.kwargs["space_dim"] == 3 and self.kwargs["quantize_spectra"]

        self.convert = HyperSpectralConverter(
            _model_redshift=_model_redshift, **kwargs
        )

        self.init_decoder()
        if self.qtz_spectra:
            self.qtz = Quantization(False, **kwargs)
        if self.intensify:
            self.intensifier = Intensifier(kwargs["intensification_method"])
        self.inte = HyperSpectralIntegrator(integrate=integrate, **kwargs)

    ##################
    # Initialization
    ##################

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
        else:
            assert self.kwargs["space_dim"] == 3
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
                else:
                    raise ValueError()
            else:
                # coords and wave are not encoded
                input_dim = 3

        return input_dim

    def get_output_dim(self):
        if self.kwargs["space_dim"] == 2:
            return self.kwargs["num_bands"]
        return 1

    def init_decoder(self):
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
    # Setters
    ##################

    def set_batch_reduction_order(self, order="qtz_first"):
        """ When we do spectra quantization over multiple redshift bins,
            our forwared spectra has two batch dim.
            If we want to save spectra under each bin, we need to perform qtz first.
            If we want to plot codebook spectra, we need to average over all redshift bins first.
            @Param
              spectra: [num_redshift_bins,num_embed,bsz,num_smpl]
              order: `qtz_first` or `bin_avg_first`
        """
        self.reduction_order = order

    def set_bayesian_redshift_logits_calculation(self, loss, mask, gt_spectra):
        """ Set function to calculate bayesian logits for redshift classification.
        """
        self.calculate_bayesian_redshift_logits = partial(
            calculate_bayesian_redshift_logits, loss, mask, gt_spectra)

    ##################
    # Model forwarding
    ##################

    def classify_redshift4D(self, spectra, ret):
        """ @Param
              spectra [num_redshift_bins,num_embed,bsz,nsmpl]
              ret["redshift_logits"] [bsz,num_redshift_bins]
            @Return
              spectra [num_embed,bsz,nsmpl]
        """
        assert self.kwargs["redshift_classification_method"] == "weighted_avg"
        num_embed = spectra.shape[1]
        spectra = torch.einsum("ij,jkim->kim", ret["redshift_logits"], spectra)
        # spectra = torch.matmul(
        #     ret["redshift_logits"][:,None,:,None].tile(1,num_embed,1,1), # [bsz,n_embed,1,.]
        #     spectra.permute(2,1,0,3) # [bsz,n_embed,n_bins,nsmpl]
        # )[:,:,0]
        return spectra

    def classify_redshift3D(self, spectra, ret):
        """ @Param
              spectra [num_redshift_bins,bsz,nsmpl]
              ret["redshift_logits"] [bsz,num_redshift_bins]
            @Return
              spectra [bsz,nsmpl]
        """
        assert self.kwargs["redshift_classification_method"] == "weighted_avg"
        spectra = torch.matmul(
            ret["redshift_logits"][:,None], spectra.permute(1,0,2))[:,0]
        # elif self.kwargs["redshift_classification_method"] == "argmax":
        #     # spectra = ArgMax.apply(ret["redshift_logits"], spectra)
        #     ids = ArgMax.apply(ret["redshift_logits"])
        #     spectra = torch.matmul(ids[:,None], spectra.permute(1,0,2))[:,0]
        # elif self.kwargs["redshift_classification_method"] == "bayesian_weighted_avg":
        #     logits = self.calculate_bayesian_redshift_logits(
        #         spectra, ret["redshift_logits"], **self.kwargs) # [bsz,num_bins]
        #     spectra = torch.matmul(logits, spectra.permute(1,0,2))[:,0]
        # else: raise ValueError()
        return spectra

    def quantize_spectra(self, logits, codebook_spectra, ret, qtz_args):
        _, spectra = self.qtz(logits, codebook_spectra, ret, qtz_args)
        spectra = torch.squeeze(spectra, dim=-2) # [...,bsz,nsmpl]
        return spectra

    def spectra_dim_reduction(self, input, spectra, ret, qtz_args):
        if self.qtz_spectra:
            if self.reduction_order == "qtz_first":
                spectra = self.quantize_spectra(input, spectra, ret, qtz_args)
                if self.classify_redshift:
                    ret["spectra_all_bins"] = spectra
                    spectra = self.classify_redshift3D(spectra, ret)

            elif self.reduction_order == "bin_avg_first":
                if self.classify_redshift:
                    spectra = self.classify_redshift4D(spectra, ret)
                ret["codebook_spectra"] = spectra.permute(1,0,2)
                spectra = self.quantize_spectra(input, spectra, ret, qtz_args)
            else:
                raise ValueError()
        else:
            if self.classify_redshift:
                spectra = self.classify_redshift3D(spectra, ret)
        return spectra

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
        bsz = wave.shape[0]

        if self.qtz_spectra:
            codes = codebook.weight[:,None,None].tile(1,bsz,1,1)
            latents = self.convert(wave, codes, redshift, wave_bound)
                                                           # [...,num_embed,bsz,nsmpl,dim]
            spectra = self.spectra_decoder(latents)[...,0] # [...,num_embed,bsz,nsmpl]
        else:
            latents = self.convert(wave, input, redshift, wave_bound) # [...,bsz,nsmpl,dim]
            spectra = self.spectra_decoder(latents)[...,0] # [...,bsz,nsmpl]

        spectra = self.spectra_dim_reduction(input, spectra, ret, qtz_args) # [bsz,nsmpl]

        if self.scale:
            assert scaler is not None
            spectra = (scaler * spectra.T).T
        if self.add_bias:
            assert bias is not None
            spectra = spectra + bias[:,None]
        if self.intensify:
            spectra = self.intensifier(spectra)

        return spectra

    def forward_codebook_spectra(self, codebook, full_wave, full_wave_bound, ret):
        """
            @Params
              codebook: [num_embed,dim]
              full_wave: [nsmpl]
              full_wave_masks: [nsmpl]
            @Return
              ret["codebook_spectra"]: [num_embed,nsmpl]
        """
        n = codebook.weight.shape[0]
        latents = self.convert(
            full_wave[None,:,None].tile(n,1,1),
            codebook.weight[:,None], None, full_wave_bound
        ) # [num_embed,nsmpl,dim]
        ret["full_range_codebook_spectra"] = self.spectra_decoder(latents)[...,0]

    def forward(self, latents,
                wave, trans, nsmpl, full_wave_bound,
                trans_mask=None,
                num_sup_spectra=0, sup_spectra_wave=None,
                codebook=None, qtz_args=None, ret=None,
                full_wave=None
    ):
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

            - full_wave
              full_wave_massk

            @Return (added to `ret`)
              spectra:   reconstructed spectra
              intensity: reconstructed pixel values
              sup_spectra: reconstructed spectra used for spectra supervision
        """
        assert num_sup_spectra >= 0
        bsz = latents.shape[0]
        timer = PerfTimer(activate=self.kwargs["activate_model_timer"],
                          show_memory=self.kwargs["show_memory"])
        timer.reset()

        if self.kwargs["regu_codebook_spectra"]:
            self.forward_codebook_spectra(codebook, full_wave, full_wave_bound, ret)

        redshift = None if ret["redshift"] is None else \
            ret["redshift"] if self.classify_redshift else ret["redshift"][:bsz]

        ret["spectra"] = self.reconstruct_spectra(
            latents, wave,
            None if ret["scaler"] is None else ret["scaler"][:bsz],
            None if ret["bias"] is None else ret["bias"][:bsz],
            redshift,
            full_wave_bound, ret, codebook, qtz_args
        )
        timer.check("hps_decoder::spectra reconstruced")

        intensity = self.inte(ret["spectra"], trans, trans_mask, nsmpl)
        ret["intensity"] = intensity
        timer.check("hps_decoder::integration done")
