
import sys
import math
import time
import torch
import numpy as np
import torch.nn as nn

sys.path.insert(0, './models')
from mlp import MLP_All
from pe import PE_All
from common import Normalization
from vae import Quantized_Encoder
from integration import Integration


class Monte_Carlo(nn.Module):
    ''' Monte Carlo wrapper that initializes different monte carlo
          models based on choices of wave/lambda sampling methods
    '''
    def __init__(self, norm_cho, inte_cho, pe_coord, pe_wave, encode,
                 encoder_output_scaler, quantizer_calculate_loss,
                 args, covr_rnge=None, nsmpl_within_each_band=None,
                 full_wave=None, full_trans=None, num_coords_with_full_wave=-1):

        super(Monte_Carlo, self).__init__()

        mc_cho = args.mc_cho
        if mc_cho == 'mc_hardcode':
            self.mc = MC_Hardcode(norm_cho, inte_cho, pe_coord, pe_wave, encode,
                                  encoder_output_scaler, quantizer_calculate_loss, args,
                                  full_wave, full_trans, num_coords_with_full_wave)

        elif mc_cho == 'mc_bandwise':
            assert(covr_rnge is not None or not args.uniform_smpl)
            self.mc = MC_Bandwise(norm_cho, inte_cho, nsmpl_within_each_band, covr_rnge,
                                  pe_coord, pe_wave, encode, encoder_output_scaler,
                                  quantizer_calculate_loss, args,
                                  full_wave, full_trans, num_coords_with_full_wave)

        elif mc_cho == 'mc_mixture':
            self.mc = MC_Mixture(norm_cho, inte_cho, pe_coord, pe_wave, encode,
                                 encoder_output_scaler, quantizer_calculate_loss, args,
                                 full_wave, full_trans, num_coords_with_full_wave)

        else: raise('Unsupporeted monte carlo choice')

    def forward(self, input):
        return self.mc(input)

class MC_Base(nn.Module):
    ''' Base class for three different Monte carlo models (each support one sampling method)
        @Param
          nsmpl_within_each_band & covr_rnge: used for bandwise integration only
            mixture integration has nsmpl_within_each_band as a forward arg
    '''
    def __init__(self, norm_cho, inte_cho, pe_coord, pe_wave, encode,
                 encoder_output_scaler, quantizer_calculate_loss, args,
                 covr_rnge=None, nsmpl_within_each_band=None,
                 full_wave=None, full_trans=None, num_coords_with_full_wave=-1):

        super(MC_Base, self).__init__()

        self.nbands = args.num_bands

        self.norm_cho = norm_cho
        self.inte_cho = inte_cho
        self.mc_cho = args.mc_cho
        self.mlp_cho = args.mlp_cho

        self.encode = encode
        self.pe_wave = pe_wave
        self.pe_coord = pe_coord
        self.redshift_wave = args.encoder_output_redshift
        self.encoder_output_scaler = encoder_output_scaler
        self.quantizer_calculate_loss = quantizer_calculate_loss

        self.num_bands = args.num_bands
        # these two are only used for bandwise mc, otherwise they are None
        self.covr_rnge = covr_rnge
        self.nsmpl_within_each_band = nsmpl_within_each_band

        # during training, we can train some coords with full wave while
        # the majority others with sampled waves (for bandwise & mixture only)
        self.partial_train_full_wave = full_wave is not None and \
            full_trans is not None and num_coords_with_full_wave > 0
        self.full_wave = full_wave
        self.full_trans = full_trans
        self.num_coords_with_full_wave = num_coords_with_full_wave

    def print(self):
        encoder_str = 'encoding ' if self.encode else ''
        encoder_str += ('scaler-generating ' if self.encoder_output_scaler else '')
        print(f'= {self.mc_cho} monte carlo model: {encoder_str}{self.mlp_cho} with {self.norm_cho} normalization using {self.inte_cho} integration')

    def init_net(self, args):
        if self.pe_coord:
            self.coord_pe = PE_All(args.coord_pe_cho, args.coord_pe_args)

        if self.encode:
            self.encoder = Quantized_Encoder \
                (args.latent_dim, args.num_embd, args.encoder_cho,
                 args.encoder_mlp_args, args.encoder_output_scaler,
                 args.encoder_output_redshift, self.quantizer_calculate_loss,
                 args.encoder_quantize, args.vae_beta, args.cdbk_seed)

        if self.pe_wave:
            self.wave_pe = PE_All(args.wave_pe_cho, args.wave_pe_args)

        self.mlp = MLP_All(self.mlp_cho, args.mlp_args, args.pe_cho, args.pe_args)
        self.norm = Normalization(self.norm_cho)
        self.integration = Integration \
            (self.inte_cho, args, self.nsmpl_within_each_band, self.covr_rnge)

        if self.partial_train_full_wave:
            self.init_full_wave()

        if args.verbose: self.print()

    # tile coords before concatenation with wave
    def tile_coords(self, coords, nsmpl, wave_dim):
        if wave_dim == 3:   # hardcode/mixture
            coords = coords[:,None,:].tile(1,nsmpl,1)
        elif wave_dim == 4: # bandwise
            coords = coords[:,None,None,:].tile(1,self.num_bands,nsmpl,1)
        else:
            raise Exception('! incorrect lambda dimension when tiling coords')
        return coords

    ''' Tile and positional encode full wave if required
        @Param
          full_wave:  [full_nsmpl]
          full_trans: [nbands,full_nsmpl]
    '''
    def init_full_wave(self):
        self.full_nsmpl = self.full_wave.shape[0]
        self.full_wave = self.full_wave[None,:,None].tile(self.num_coords_with_full_wave,1,1)
        # do not PE full wave during init as we may need to do redshift first
        #if self.pe_wave: self.full_wave = self.wave_pe([self.full_wave,None]) # [bsz,nbands,nsmpl]

    # set batch size for pixls with sampled wave and full wave (spectra supervision)
    def set_bsz(self, coords, wave):
        self.nsmpl = wave.shape[-2]
        self.bsz_full_pixl = self.num_coords_with_full_wave \
            if self.partial_train_full_wave else 0
        self.bsz_sampled_pixl = len(coords) -self.bsz_full_pixl

    # generate latent from input coords
    def forward_coords(self, coords):
        if self.pe_coord:
            coords = self.coord_pe([coords, None])
        if self.encode:
            (coords, latents, scaler, redshift, cdbk_err, embd_ids) = self.encoder([coords, None])
            if self.partial_train_full_wave:
                # the last #bsz_full_pixl pixels are spectra supervision pixels
                # exclude these from the below variables
                latents = latents[:-self.bsz_full_pixl]
                if embd_ids is not None:
                    embd_ids = embd_ids[:-self.bsz_full_pixl]
            return (coords, latents, scaler, redshift, cdbk_err, embd_ids)
        return (coords, None, None, None, None, None)

    ''' Convert observed lambda to emitted lambda
        Hardcode redshift value to have minimum -0.1
        @Param
          wave [bsz,(nbands,)nsmpl,1]
          redshift [bsz,1]/None
    '''
    def shift_wave(self, wave, redshift):
        if redshift is None:
            pass
        elif wave.ndim == 3:
            nsmpl = wave.shape[1] # [bsz,nsmpl,1]
            redshift = torch.exp(redshift) - 0.1
            wave += redshift[:,:,None].tile(1,nsmpl,1)
            #wave /= (1 + redshift[:,:,None].tile(1,nsmpl,1))
        elif wave.ndim == 4:
            nsmpl = wave.shape[2] # [bsz,nbands,nsmpl,1]
            redshift = torch.exp(redshift) - 0.1
            wave += redshift[:,:,:,None].tile(1,self.nbands,nsmpl,1)
            #wave /= (1 + redshift[:,:,:,None].tile(1,self.nbands,nsmpl,1))
        else:
            raise Exception('! wrong wave dimension when doing wave shifting')
        return wave

    # redshift and pe wave if required
    def forward_wave(self, wave, redshift, lo, len):
        if self.redshift_wave and redshift is not None:
            wave = self.shift_wave(wave, redshift[lo:lo+len])
        if self.pe_wave:
            wave = self.wave_pe([wave, None]) # [bsz,(nbands,)nsmpl,pe_dim]
        return wave

    ''' Concatenate coord with wave before passing to mlp
        @Param
          coords [bsz,2/coord_pe_dim/latent_dim]
                 is the concatenation of coords with sampled wave and coords with full wave, if any
          wave   [bsz,(nbands,)nsmpl_per_band,1/wave_pe_dim]/[bsz,(nbands,)nsmpl_full,1/wave_pe_dim]
        @Return
          mlp_input: all coords concated with sampled lambda
                     [bsz,nbands,nsmpl_per_band,1]/[bsz,nsmpl,1]
          mlp_input_full_wave: only specified coords concatenated with all lambda
                               [bsz,full_nsmpl,1] no bandwise
    '''
    def concat_coord_wave(self, coords, wave, nsmpl):
        coords = self.tile_coords(coords, nsmpl, wave.ndim)
        mlp_input = torch.cat((coords, wave), -1)
        return mlp_input

    ''' Pass input thru MLP to get spectra
        @Param
          mlp_input [bsz,nsmpl,input_dim]
          scaler    [bsz,1]
          covar     OUTDATED
          full_wave: whether use all lambda
    '''
    def forward_mlp(self, mlp_input, scaler, covar, use_full_wave=False):
        spectra = self.mlp([mlp_input, covar]) # [bsz,nsmpl,1]
        spectra = self.norm(spectra[...,0])
        if scaler is not None: # cdbk spectrum plot doesnt have scaler
            if use_full_wave: scaler = scaler[-self.bsz_full_pixl:]
            else: scaler = scaler[:self.bsz_sampled_pixl].tile(1,self.nsmpl)
`            spectra *= torch.exp(scaler)
        return spectra

    ''' Integrate spectra over transmission to get pixel value
        @Param
          spectra [bsz,nsmpl,1]/[bsz,nbands,nsmpl,1]
          trans   train: [bsz,nbands,nsmpl]
                  infer: [nbands,nsmpl]
    '''
    def forward_spectra(self, spectra, trans, nsmpl_within_each_band):
        if self.mc_cho == 'mc_mixture':
            output = self.integration([spectra, trans, nsmpl_within_each_band])
        else: output = self.integration([spectra, trans])
        return output

    ''' @Param
          coords [bsz(+?),2] (incl. coord for spectra supervision pixl at the end)
          wave   train: [bsz(,nbands),nsmpl,1] (excl. spectra supervision pixl)
                 infer: [bsz,nsmpl,1]
          trans  train: [bsz,nbands,nsmpl] (excl. spectra supervision pixl)
                 infer: [nbands,nsmpl]
        @Return
          output [bsz,nbands] (pixl values)
    '''
    def _forward(self, coords, covar, wave, trans, nsmpl_within_each_band=None):
        self.set_bsz(coords, wave)
        # during inference, # wave samples may be > last batch size
        wave = wave[:self.bsz_sampled_pixl]
        if trans is not None: trans = trans[:self.bsz_sampled_pixl]
        (coords, latents, scaler, redshift, cdbk_err, embd_ids) = self.forward_coords(coords)

        # forward pass for coords with partially sampled lambda
        wave = self.forward_wave(wave, redshift, 0, self.bsz_sampled_pixl)
        mlp_input = self.concat_coord_wave(coords[:self.bsz_sampled_pixl], wave[:self.bsz_sampled_pixl], self.nsmpl)
        spectra = self.forward_mlp(mlp_input, scaler, covar)
        output = self.forward_spectra(spectra, trans, nsmpl_within_each_band)

        # forward pass for coords with fully sampled lambda
        # currently only use this for spectra for supervision (don't go further to integration)
        if self.partial_train_full_wave:
            full_wave = self.forward_wave \
                (self.full_wave.clone(), redshift, self.bsz_sampled_pixl, self.bsz_full_pixl)
            mlp_input_full_wave = self.concat_coord_wave \
                (coords[-self.bsz_full_pixl:], full_wave, self.full_nsmpl)
            full_spectra = self.forward_mlp(mlp_input_full_wave, scaler, covar, use_full_wave=True)
        else: full_spectra = None

        return (output, full_spectra, cdbk_err, embd_ids, latents)


##########################
# 3 exact implementations

class MC_Hardcode(MC_Base):
    ''' Monte carlo model with hardcoded samples of lambda '''
    def __init__(self, norm_cho, inte_cho, pe_coord, pe_wave, encode,
                 encoder_output_scaler, quantizer_calculate_loss, args,
                 full_wave, full_trans, num_coords_with_full_wave):

        MC_Base.__init__(self, norm_cho, inte_cho, pe_coord, pe_wave,
                         encode, encoder_output_scaler, quantizer_calculate_loss,
                         args, full_wave=full_wave, full_trans=full_trans,
                         num_coords_with_full_wave=num_coords_with_full_wave)
        self.init_net(args)

    ''' @Param
          coords[bsz,(nsmpl,)dim] (pre-concat with lambda when loading data)
          covar [bsz,(nsmpl,)dim]
        @Return
          pixl_val [bsz,dim_out]
    '''
    def forward(self, input):
        (coords, covar, trans) = input
        # tmp, coords is concat with lambda, need to slice
        wave, coords = coords[...,-1:], coords[...,:-1]
        return self._forward(coords, covar, wave, trans)


class MC_Bandwise(MC_Base):
    ''' Monte carlo NeRF that supports bandwise sampling of lambda '''
    def __init__(self, norm_cho, inte_cho, nsmpl_within_each_band,
                 covr_rnge, pe_coord, pe_wave, encode, encoder_output_scaler,
                 quantizer_calculate_loss, args,
                 full_wave, full_trans, num_coords_with_full_wave):

        MC_Base.__init__(self, norm_cho, inte_cho, pe_coord, pe_wave, encode,
                         encoder_output_scaler, quantizer_calculate_loss, args,
                         covr_rnge=covr_rnge, nsmpl_within_each_band=nsmpl_within_each_band,
                         full_wave=full_wave, full_trans=full_trans,
                         num_coords_with_full_wave=num_coords_with_full_wave)
        self.init_net(args)

    ''' @Param
          coords: [bsz,2],
          wave:   [bsz,nbands,nsmpl,1]
          trans:  train [bsz,nbands,nsmpl]
                  infer [nbands,nsmpl]
          nsmpl_within_each_band: [nbands]
    '''
    def forward(self, input):
        (coords, covar, wave, trans) = input
        return self._forward(coords, covar, wave, trans)


class MC_Mixture(MC_Base):
    ''' Monte carlo NeRF that supports mixture sampling of lambda '''
    def __init__(self, norm_cho, inte_cho, pe_coord, pe_wave, encode,
                 encoder_output_scaler, quantizer_calculate_loss, args,
                 full_wave, full_trans, num_coords_with_full_wave):

        MC_Base.__init__(self, norm_cho, inte_cho, pe_coord, pe_wave, encode,
                         encoder_output_scaler, quantizer_calculate_loss, args,
                         full_wave=full_wave, full_trans=full_trans,
                         num_coords_with_full_wave=num_coords_with_full_wave)
        self.init_net(args)

    ''' @Param
          coords: [bsz,2],
          wave:   [bsz,nsmpl,1]
          trans:  train [bsz,nbands,nsmpl]
                  infer [nbands,nsmpl]
          nsmpl_within_each_band: [nbands]
        @Return
          full_spectra: spectra of full wave for spectra supervision
          output:       pixel value (training & img recon) OR spectra (spectra recon) [bsz,num_bands]
          cdbk_err:  codebook error of embedding quantization if performed else None
    '''
    def forward(self, input):
        (coords, covar, wave, trans, nsmpl_within_each_band) = input
        return self._forward(coords, covar, wave, trans,
                             nsmpl_within_each_band=nsmpl_within_each_band)
