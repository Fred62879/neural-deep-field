
import torch
import torch.nn as nn

class Integrator(nn.Module):
    ''' Approximate the integration of two functions
    '''

    def __init__(self, inte_cho, args, nsmpl_within_bands=None, covr_rnge=None):
        # nsmpl_within_bands and covr_rnge are needed for bandwise integration only
        # mixture integration has nsmpl_within_bands as an forward arg
        super(Integrator, self).__init__()

        self.mc_cho = args.mc_cho
        self.inte_cho = inte_cho
        self.init_integration(args, nsmpl_within_bands, covr_rnge)

    def init_integration(self, args, nsmpl_within_bands, covr_rnge):
        if self.mc_cho == 'mc_hardcode':
            self.integrate = Integrator_hardcode(self.inte_cho, args.num_trans_smpl, args.plot_eltws_prod)
        elif self.mc_cho == 'mc_bandwise':
            self.integrate = Integrator_bandwise \
                (self.inte_cho, covr_rnge, nsmpl_within_bands, args.uniform_smpl, args.plot_eltws_prod)
        elif self.mc_cho == 'mc_mixture':
            self.integrate = Integrator_mixture(self.inte_cho, args.plot_eltws_prod)

    def forward(self, input):
        return self.integrate(input)


class Integrator_Base(nn.Module):
    def __init__(self, get_eltws_prod):
        super(Integrator_Base, self).__init__()

        self.get_eltws_prod = get_eltws_prod

    ''' For spectrum plotting only
        Input:  spectra [bsz,nsmpl]/[bsz,nsmpl,1]
                transms [nbands,nsmpl]
        Output: pixel_val  []
                eltws_prod [bsz,nbands,nsmpl]
    '''
    def identity(self, spectra, transms, nsmpl_within_bands=None):
        '''
        if self.get_eltws_prod:
            #print(spectra.shape, transms.shape)
            #print(spectra[0,305:466]*transms[2,305:466])
            nbands = transms.shape[0]
            if transms.ndim == 3: transms = transms[...,0]
            eltws_prod = torch.stack([spectra*transms[i]
                                      for i in range(nbands)]).permute(1,0,2)
            return spectra, eltws_prod
        '''
        return spectra

    # y [bsz,nsmpl]
    def simpson(self, y): # batch simpson integration
        return (y[:,0] + 2*torch.sum(y[:,1:-1],dim=1) +
                2*torch.sum(y[:,1:-1:2], dim=1)+ y[:,-1]) /3

    def manual_dot_prod(self, spectra, transms):
        bsz, nbands, nsmpl = transms.shape
        res = torch.zeros((bsz, nbands))
        for i in range(nbands):
            res[:,i] = torch.sum(spectra*transms[:,i],dim=1)
        return res

    def init_layer(self, cho):
        if cho == 'identity':
            self.integrate = self.identity
        elif cho == 'dot_product':
            self.integrate = self.dot_prod
        elif cho == 'trapezoid':
            self.integrate = self.trapezoid
        elif cho == 'simpson':
            self.integrate = self.simpson_inte
        else:
            raise Exception('Unsupported integration choice')

class Integrator_hardcode(Integrator_Base):
    ''' Hardcode Integrator
        Input:  spectra [bsz,nsmpl]
                transms: [nbands,nsmpl]
        Output: pixl_val [bsz,nbands]
    '''
    def __init__(self, cho, nsmpl_train_smpl, get_eltws_prod):
        Integrator_Base.__init__(self, get_eltws_prod)
        self.nsmpl = nsmpl_train_smpl
        self.init_layer(cho)

    def simpson_inte(self, spectra, transms):
        # integration across all channels
        return torch.stack([self.simpson(spectra * trans)
                            for trans in transms]).T  / self.nsmpl

    def trapezoid(self, spectra, transms):
        return torch.stack([torch.trapezoid(spectra * trans)
                            for trans in transms]).T  / self.nsmpl

    def dot_prod(self, spectra, transms):
        return torch.einsum('ij,lj->il', spectra, transms) / self.nsmpl

    def forward(self, input):
        (spectra, transms) = input
        assert(spectra.shape[1] == self.nsmpl)
        return self.integrate(spectra, transms)


class Integrator_bandwise(Integrator_Base):
    ''' Bandwise integration
        @Param
          covr_rnge: [nbands]
            only used for uniform sampling, records # of wave samples
            (for full wave data, not sampled wave data) within each band
          nsmpl_within_each_band: [nbands]
            - during training, all entries are the same, being nsmpl_per_band
            - during infer, each entry is the # of wave samples within each band
          get_eltws_prod: only has effect when identity is used (i.e. plot spectrum)

        Note: non-uniform inference only supports infer using all lambda
    '''
    def __init__(self, cho, covr_rnge, nsmpl_within_each_band, unismpl, get_eltws_prod):
        Integrator_Base.__init__(self, get_eltws_prod)
        self.init_layer(cho)

        self.unismpl = unismpl
        self.covr_rnge = covr_rnge # inverse unifm sampling prob [nbands,]
        self.nbands = len(nsmpl_within_each_band)
        self.nsmpl_within_each_band = nsmpl_within_each_band

    def simpson_inte(self, spectra, trans):
        # integration across all channels
        nbands = trans.shape[1]
        return torch.stack([self.simpson(spectra[:,band] * trans[:,band])
                            for band in range(nbands)]).T / self.nsmpl_within_each_band

    def trapezoid(self, spectra, trans):
        nbands = trans.shape[1]
        return torch.stack([torch.trapezoid(spectra[:,band] * trans[:,band])
                            for band in range(nbands)]).T / self.nsmpl_within_each_band

    def manual_dot_prod(self, spectra, trans):
        bsz, nbands, nsmpl = spectra.shape
        res = torch.zeros((bsz, nbands))
        for i in range(nbands):
            res[:,i] = torch.sum(spectra[:,i]*trans[:,i],dim=1)
        return res/self.nsmpl_within_each_band

    ''' wave is sampled according to the transmission function
        thus only need to sum up spectra values
    '''
    def dot_prod_nonuniform(self, spectra, trans):
        #print(self.nsmpl_within_each_band)
        if trans is None or trans.ndim == 2:   # infer
            cumsum = torch.zeros((self.nbands+1))
            cumsum[1:] = torch.cumsum(self.nsmpl_within_each_band,dim=0)
            cumsum = cumsum.type(torch.IntTensor)

            res =torch.stack([
                torch.sum(spectra[:,cumsum[i]:cumsum[i+1]], dim=1) \
                / self.nsmpl_within_each_band[i]
                for i in range(self.nbands) ]).permute(1,0)

        elif trans.ndim == 3: # train
            res = torch.stack([
                torch.sum(spectra[:,i,:], dim=1) \
                / self.nsmpl_within_each_band[i]
                for i in range(self.nbands) ]).permute(1,0)
        else:
            raise Exception('! wrong dimension of transmission when doing integration')
        return res

    def dot_prod_uniform(self, spectra, trans):
        assert(self.covr_rnge is not None and self.nsmpl_within_each_band is not None)
        if trans.ndim == 2:   # infer (img & spectra)
            res = torch.einsum('ij,kj->ik', spectra, trans)
        elif trans.ndim == 3: # train
            res = torch.einsum('ijk,ijk->ij', spectra, trans)
        else:
            raise Exception('! wrong dimension of transmission when doing integration')
        # todo: do we still need to multiply covr_rnge during infer
        res *= self.covr_rnge # equivalent to divided by inverse prob
        res /= self.nsmpl_within_each_band  # average pixel value
        return res

    def dot_prod(self, spectra, trans):
        if self.unismpl: res = self.dot_prod_uniform(spectra, trans)
        else: res = self.dot_prod_nonuniform(spectra, trans)
        return res

    ''' @Param
          spectra: train [bsz,nbands,nsmpl]
                   infer [bsz,nsmpl]
          trans:   train [bsz,nbands,nsmpl]
                   infer [nbands,nsmpl]
        @Return
          pixl_val [bsz,nbands]
        Note: the extra dimension (nbands) is only needed during training.
    '''
    def forward(self, input):
        (spectra, trans) = input
        return self.integrate(spectra, trans)


class Integrator_mixture(Integrator_Base):
    ''' Mixture integration '''
    def __init__(self, cho, get_eltws_prod):
        Integrator_Base.__init__(self, get_eltws_prod)
        self.init_layer(cho)

    def simpson_inte(self, spectra, transms, nsmpl_within_each_band):
        # integration across all channels
        nbands = transms.shape[1]
        return torch.stack([self.simpson(spectra * transms[:,band])
                            for band in range(nbands)]).T / nsmpl_within_each_band

    def trapezoid(self, spectra, transms, nsmpl_within_each_band):
        nbands = transms.shape[1]
        return torch.stack([torch.trapezoid(spectra * transms[:,band])
                            for band in range(nbands)]).T / nsmpl_within_each_band

    def dot_prod(self, spectra, trans, nsmpl_within_each_band):
        #print(spectra.shape, trans.shape, nsmpl_within_each_band.shape)
        if trans.ndim == 2:   # infer
            dp = torch.einsum('ij,kj->ik', spectra, trans)
        elif trans.ndim == 3: # train
            dp = torch.einsum('ij,ilj->il', spectra, trans)
        else:
            raise Exception('! wrong dimension of transmission when doing integration')
        return dp / nsmpl_within_each_band

    ''' @Param
          spectra: [bsz,nsmpl]
          trans:   train [bsz,nbands,nsmpl]
                   infer [nbands,nsmpl]
          nsmpl:   train [bsz,nbands]/[1,]
                   infer [nbands]/[1,]
        @Return
          pixl_val [bsz,nbands]
    '''
    def forward(self, input):
        (spectra, transms, nsmpl_within_each_band) = input
        return self.integrate(spectra, transms, nsmpl_within_each_band)
