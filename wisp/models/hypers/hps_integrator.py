
import torch
import torch.nn as nn

class HyperSpectralIntegrator(nn.Module):
    """ Approximate the integration of the spectra over the transmission function.

        @Param
          covr_rnge: [nbands]
            only used for uniform sampling, records # of wave samples
            (for full wave data, not sampled wave data) within each band
          nsmpl_within_each_band: [nbands]
            - during training, all entries are the same, being nsmpl_per_band
            - during infer, each entry is the # of wave samples within each band

        Note: nsmpl_within_bands and covr_rnge are needed for bandwise integration only.
              mixture integration has nsmpl_within_bands as an forward parameter.
    """

    def __init__(self, integrate=True, **kwargs):
        super(HyperSpectralIntegrator, self).__init__()

        self.num_bands = kwargs["num_bands"]
        self.uniform_sample = kwargs["uniform_sample_trans"]
        self.trans_sample_method = kwargs["trans_sample_method"]

        if not integrate:
            self.integration_method = "identity"
        else:
            self.integration_method = kwargs["integration_method"]

        if self.trans_sample_method == "hardcode":
            self.nsmpl = kwargs["hardcode_num_trans_samples"]

        self.init_integration_funcs()
        self.register_function()

    def init_integration_funcs(self):
        self.funcs = {
            "hardcode": {
                "identity": self.identity,
                "dot_prod": self.dot_prod_hdcd,
                "trapezoid": self.trapezoid_hdcd,
                "simpson": self.simpson_hdcd
            },
            "bandwise": {
                "identity": self.identity,
                "dot_prod": self.dot_prod_bdws,
                "trapezoid": self.trapezoid_bdws,
                "simpson": self.simpson_bdws
            },
            "mixture": {
                "identity": self.identity,
                "dot_prod": self.dot_prod_mix,
                "trapezoid": self.trapezoid_mix,
                "simpson": self.simpson_mix
            }
        }

    def register_function(self):
        self.integrate = self.funcs[self.trans_sample_method][self.integration_method]

    def forward(self, spectra, **kwargs):
        """ @Param
              spectra: [bsz,nsmpl]
              trans:   train [bsz,nbands,nsmpl]
                       infer [nbands,nsmpl]
              nsmpl:   train [bsz,nbands]/[1,]
                       infer [nbands]/[1,]
            @Return
              pixl_val [bsz,nbands]
        """
        return self.integrate(spectra, **kwargs)

    #############
    # Helper methods
    #############

    def identity(self, spectra, **kwargs):
        return spectra

    # @dot product
    def dot_prod_hdcd(self, spectra, **kwargs):
        return torch.einsum("ij,lj->il", spectra, kwargs["trans"]) / self.nsmpl

    def dot_prod_nonuniform_bdws(self, spectra, **kwargs):
        """ Wave is sampled according to the transmission function
              thus only need to sum up spectra values.
        """
        trans = kwargs["trans"]
        if trans is None or trans.ndim == 2:   # infer
            cumsum = torch.zeros((self.num_bands+1))
            cumsum[1:] = torch.cumsum(kwargs["nsmpl_within_each_band"], dim=0)
            cumsum = cumsum.type(torch.IntTensor)

            res =torch.stack([
                torch.sum(spectra[:,cumsum[i]:cumsum[i+1]], dim=1) \
                / kwargs["nsmpl_within_each_band"][i]
                for i in range(self.num_bands) ]).permute(1,0)

        elif trans.ndim == 3: # train
            res = torch.stack([
                torch.sum(spectra[:,i,:], dim=1) \
                / kwargs["nsmpl_within_each_band"][i]
                for i in range(self.num_bands) ]).permute(1,0)
        else:
            raise ValueError("Wrong transmission dimension when doing integration.")
        return res

    def dot_prod_uniform_bdws(self, spectra, trans):
        if trans.ndim == 2:   # infer (img & spectra)
            res = torch.einsum("ij,kj->ik", spectra, trans)
        elif trans.ndim == 3: # train
            res = torch.einsum("ijk,ijk->ij", spectra, trans)
        else:
            raise Exception("! wrong dimension of transmission when doing integration")
        # todo: do we still need to multiply covr_rnge during infer
        res *= kwargs["covr_rnge"] # equivalent to divided by inverse prob
        res /= kwargs["nsmpl_within_each_band"]  # average pixel value
        return res

    def dot_prod_bdws(self, spectra, **kwargs):
        if self.unismpl: res = self.dot_prod_uniform_bdws(spectra, kwargs)
        else: res = self.dot_prod_nonuniform_bdws(spectra, kwargs)
        return res

    def dot_prod_mix(self, spectra, **kwargs):
        trans = kwargs["trans"][0] # **** replace ***********
        if trans.ndim == 2:   # infer
            dp = torch.einsum("ij,kj->ik", spectra, trans)
        elif trans.ndim == 3: # train
            dp = torch.einsum("ij,ilj->il", spectra, trans)
        else:
            raise Exception("wrong dimension of transmission when doing integration")
        dp /= kwargs["nsmpl"][0] # *** replace ***********
        return dp

    # @trapezoid
    def trapezoid_hdcd(self, spectra, **kwargs):
        return torch.stack([torch.trapezoid(spectra * trans)
                            for trans in transms]).T  / self.nsmpl

    def trapezoid_bdws(self, spectra, **kwargs):
        return torch.stack([torch.trapezoid(spectra[:,band] * trans[:,band])
                            for band in range(self.num_bands)]).T / kwargs["nsmpl_within_each_band"]

    def trapezoid_mix(self, spectra, **kwargs):
        return torch.stack([torch.trapezoid(spectra * kwargs["trans"][:,band])
                            for band in range(self.num_bands)]).T / kwargs["nsmpl_within_each_band"]
    # @simpsons
    def simpson_hdcd(self, spectra, **kwargs):
        # integration across all channels
        return torch.stack([self.simpson(spectra * trans)
                            for trans in transms]).T  / self.nsmpl

    def simpson_bdws(self, spectra, **kwargs):
        # integration across all channels
        return torch.stack([self.simpson(spectra[:,band] * trans[:,band])
                            for band in range(self.num_bands)]).T / kwargs["nsmpl_within_each_band"]

    def simpson_mix(self, spectra, **kwargs):
        # integration across all channels
        return torch.stack([self.simpson(spectra * kwargs["trans"][:,band])
                            for band in range(self.num_bands)]).T / kwargs["nsmpl_within_each_band"]
