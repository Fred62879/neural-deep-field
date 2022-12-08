
import torch
import numpy as np
import logging as log

from astropy.io import fits
from astropy.wcs import WCS
from os.path import join, exists
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from wisp.utils.numerical import normalize_coords, normalize


class FITSData:
    ''' Data class for FITS files. '''

    def __init__(self, dataset_path, **kwargs):
        self.kwargs = kwargs
        self.tile_ids = kwargs["fits_tile_ids"]
        self.subtile_ids = kwargs["fits_subtile_ids"]
        self.footprints = kwargs["fits_footprints"]

        self.num_bands = kwargs["num_bands"]
        self.sensors_full_name = kwargs['sensors_full_name']
        self.u_band_scale = kwargs["u_band_scale"]
        self.load_weights = kwargs["weight_train"]
        self.cutout_based_train = kwargs["cutout_based_train"]
        self.load_fits_data_cache = kwargs["load_fits_data_cache"]

        self.start_r = kwargs["start_r"]
        self.start_c = kwargs["start_c"]
        self.fits_cutout_sz = kwargs["fits_cutout_sz"]
        self.use_full_fits = kwargs["use_full_fits"]

        self.num_rows = []
        self.num_cols = []

        self.set_path(dataset_path)
        self.compile_fits_fnames()
        self.load_all_fits()
        self.get_world_coords_all_fits()

    def set_path(self, dataset_path):
        self.dataset_path = dataset_path
        self.input_path = join(self.dataset_path, 'input')
        self.output_path = join(self.dataset_path, 'output')

        # input data
        self.input_fits_path = join(self.input_path, 'input_fits')

        img_data_path = join(self.input_path, self.kwargs['sensor_collection_name'], 'img_data')
        suffix = self.kwargs['fits_choice_id']
        if not self.use_full_fits:
            suffix += '_' + str(self.fits_cutout_sz) + '_' + str(self.start_r) + '_' + str(self.start_c)
        self.coords_fname = join(img_data_path, 'coords_' + suffix + '.npy')
        self.weights_fname = join(img_data_path, 'weights_' + suffix + '.npy')
        self.coords_range_fname = join(img_data_path, 'coords_range_' + suffix + '.npy')
        self.pixels_fname = join(img_data_path, 'pixels_' + self.kwargs['train_pixels_norm'] + '_' + suffix + '.npy')

        trans_path = join(self.input_path, self.kwargs['sensor_collection_name'], 'transmission')

        mask_path = join(self.input_path, 'mask')
        if self.kwargs['mask_config'] == 'region':
            mask_str = '_' + str(self.kwargs['m_start_r']) + '_' \
                + str(self.kwargs['m_start_c']) + '_' \
                + str(self.kwargs['mask_sz'])
        else:
            mask_str = '_' + str(float(100 * self.kwargs['sample_ratio']))

        if self.kwargs['inpaint_cho'] == 'spectral_inpaint':
            self.mask_fn = join(self.mask_path, str(self.kwargs['fits_cutout_sz']) + mask_str + '.npy')
            self.masked_pixl_id_fn = join(mask_path, str(self.kwargs['fits_cutout_sz']) + mask_str + '_masked_id.npy')
        else:
            self.mask_fn, self.masked_pixl_id_fn = None, None

    def compile_fits_fnames(self):
        """ Get fnames of all given fits input for all bands. """
        self.fits_ids, self.fits_groups, self.fits_wgroups = [], {}, {}
        for footprint, tile_id, subtile_id in zip(
                self.footprints, self.tile_ids, self.subtile_ids):

            mtile= tile_id + 'c' + subtile_id
            tile = tile_id + '%2C' + subtile_id
            fits_id = footprint + tile_id + subtile_id

            hsc_fits_fname = np.array(
                ['calexp-' + band + '-' + footprint + '-' + tile + '.fits'
                for band in self.sensors_full_name if 'HSC' in band])
            nb_fits_fname = np.array(
                ['calexp-' + band + '-' + footprint + '-' + tile + '.fits'
                for band in self.sensors_full_name if 'NB' in band])
            megau_fits_fname = np.array(
                ['Mega-' + band + '_' + footprint + '_' + mtile + '.fits'
                for band in self.sensors_full_name if 'u' in band])
            megau_weights_fname = np.array(
                ['Mega-' + band + '_' + footprint + '_' + mtile + '.weight.fits'
                for band in self.sensors_full_name if 'u' in band])

            self.fits_ids.append(fits_id)
            self.fits_groups[fits_id] = np.concatenate((hsc_fits_fname, nb_fits_fname, megau_fits_fname))
            self.fits_wgroups[fits_id] = np.concatenate((hsc_fits_fname, nb_fits_fname, megau_weights_fname))

    ###############
    # Load FITS data
    ###############

    def load_header(self, fits_id, full_fits):
        ''' Load header for both full tile and current cutout. '''
        fits_fname = self.fits_groups[fits_id][0]
        id = 0 if 'Mega-u' in fits_fname else 1
        hdu = fits.open(join(self.input_fits_path, fits_fname))[id]
        header = hdu.header

        if full_fits:
            num_rows, num_cols = header['NAXIS2'], header['NAXIS1']
        else:
            pos = (self.start_c + self.fits_cutout_sz//2,
                   self.start_r + self.fits_cutout_sz//2)
            wcs = WCS(header)
            cutout = Cutout2D(hdu.data, position=pos, size=self.fits_cutout_sz, wcs=wcs)
            header = cutout.wcs.to_header()
            num_rows, num_cols = self.fits_cutout_sz, self.fits_cutout_sz

        self.num_rows.append(num_rows)
        self.num_cols.append(num_cols)
        return header, num_rows, num_cols

    def load_one_fits(self, fits_id, load_pixels=True):
        ''' Load pixel values or variance from one FITS file (tile_id/subtile_id).
            Load pixel and weights separately to avoid using up mem.
        '''
        cur_data = []

        for i in range(self.num_bands):
            if load_pixels:
                fits_fname = self.fits_groups[fits_id][i]

                # u band pixel vals in first hdu, others in 2nd hdu
                is_u = 'Mega-u' in fits_fname
                id = 0 if is_u else 1

                pixels = fits.open(join(self.input_fits_path, fits_fname))[id].data
                if is_u: # scale u and u* band pixel values
                    pixels /= self.u_bands_scale

                if not self.use_full_fits:
                    pixels = pixels[self.start_r:self.start_r + self.fits_cutout_sz,
                                    self.start_c:self.start_c + self.fits_cutout_sz]

                if not self.kwargs['train_pixels_norm'] == 'linear':
                    pixels = normalize(pixels, self.kwargs["train_pixels_norm"], gt=pixels)
                cur_data.append(pixels.flatten())

            else: # load weights
                fits_wfname = self.fits_wgroups[fits_id][i]
                # u band weights in first hdu, others in 4th hdu
                id = 0 if 'Mega-u' in fits_wfname else 3
                var = fits.open(join(self.input_fits_path, fits_wfname))[id].data

                # u band weights stored as inverse variance, others as variance
                if id == 3: weight = var
                else:       weight = 1 / (var + 1e-6) # avoid division by 0
                if self.use_full_fits:
                    cur_data.append(weight.flatten())
                else:
                    var = var[self.start_r:self.start_r + self.fits_cutout_sz,
                              self.start_c:self.start_c + self.fits_cutout_sz].flatten()
                    cur_data.append(var)

        if load_pixels:
            return np.array(cur_data).T      # [npixels,nbands]
        return np.sqrt(np.array(cur_data).T) # [npixels,nbands]

    def load_all_fits(self, to_tensor=True, save_cutout=False):
        ''' Load all images (and weights) and flatten into one array.
            @Return
              pixels:  [npixels,nbands]
              weights: [npixels,nbands]
        '''
        if self.cutout_based_train:
            raise Exception('Cutout based train only works on one fits file.')

        cached = self.load_fits_data_cache and exists(self.pixels_fname) \
            (not self.load_weights or exists(self.weights_fname))

        if cached:
            log.info('Load cached FITS data.')
            pixls = np.load(self.pixels_fname)
            if self.load_weights:
                weights = np.load(self.weights_fname)
            else: weights = None
        else:
            log.info('Loading FITS data.')
            if self.load_weights:
                log.info('Loading weights.')
                weights = np.concatenate([ self.load_one_fits(fits_id, load_pixels=False)
                                           for fits_id in self.fits_ids ])
                np.save(self.weights_fname, weights)
            else: weights = None

            log.info('Loading pixels.')
            pixels = np.concatenate([ self.load_one_fits(fits_id)
                                      for fits_id in self.fits_ids ])
            if self.kwargs['train_pixels_norm'] == 'linear':
                pixels = normalize(pixels, 'linear')
            np.save(self.pixels_fname, pixels)

        print('train pixels max ', np.round(np.max(pixels, axis=0), 3))
        print('train pixels min ', np.round(np.min(pixels, axis=0), 3))


    ##############
    # Load coords
    ##############

    def get_mgrid_tensor(self, sidelen, lo=0, hi=1, dim=2, flat=True):
        ''' Generates a flattened grid of (x,y,...) coords in [-1,1] (Tensor version).'''
        tensors = tuple(dim * [torch.linspace(lo, hi, steps=sidelen)])
        mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
        if flat: mgrid = mgrid.reshape(-1, dim)
        return mgrid

    def get_mgrid_np(self, sidelen, lo=0, hi=1, dim=2, indexing='ij', flat=True):
        ''' Generates a flattened grid of (x,y,...) coords in [-1,1] (numpy version).'''
        arrays = tuple(dim * [np.linspace(lo, hi, num=sidelen)])
        mgrid = np.stack(np.meshgrid(*arrays, indexing=indexing), axis=-1)
        if flat: mgrid = mgrid.reshape(-1,dim) # [sidelen**2,dim]
        return mgrid

    def get_world_coords_one_fits(self, fits_id):
        ''' Get ra/dec coords from one fits file and normalize.
            pix2world calculate coords in x-y order
              coords can be indexed using r-c
            @Return
              coords: 2D coordinates [npixels,2]
        '''
        header, num_rows, num_cols = self.load_header(fits_id, True)
        xids = np.tile(np.arange(0, num_cols), num_rows)
        yids = np.repeat(np.arange(0, num_rows), num_cols)

        wcs = WCS(header)
        ras, decs = wcs.all_pix2world(xids, yids, 0) # x-y pixel coord
        if self.use_full_fits:
            coords = np.array([ras, decs]).T
        else:
            coords = np.concatenate(( ras.reshape((num_rows, num_cols, 1)),
                                      decs.reshape((num_rows, num_cols, 1)) ), axis=2)
            coords = coords[self.start_r:self.start_r + self.fits_cutout_sz,
                            self.start_c:self.start_c + self.fits_cutout_sz]
            coords = coords.reshape(-1,2)
        return coords

    def get_world_coords_all_fits(self):
        ''' Get ra/dec coord from all fits files and normalize. '''
        if exists(self.coords_fname):
            log.info('Loading coords from cache.')
            coords = np.load(self.coords_fname)
        else:
            log.info('Generating coords.')
            coords = np.concatenate([ self.get_world_coords_one_fits(fits_id)
                                      for fits_id in self.fits_ids ])
            coords, coords_range = normalize_coords(coords)
            np.save(self.coords_fname, coords)
            np.save(self.coords_range_fname, np.array(coords_range))

    """
    def reshape_coords(coords, args, infer=False, spectrum=False, coord_wave=None):
        nsmpl = get_num_trans_smpl(infer, args)
        coords = coords.unsqueeze(1).tile(1,nsmpl,1) # [npixls,nsmpl,2]
        if args.mc_cho == 'mc_hardcode': # [npixls,nsmpl,2]
            assert(coord_wave is not None)
            bsz = coords.shape[0]
            coords = torch.cat((coords, coord_wave[:bsz]), dim=-1)
        elif args.mc_cho == 'mc_bandwise': # [npixls,num_bands,nsmpl,2]
            if not spectrum:
                coords = coords.unsqueeze(1).tile(1,args.num_bands,1,1)
        elif args.mc_cho == 'mc_mixture': pass
        else: raise Exception('Unsupported monte carlo choice')
        return coords

    def get_coords(nr, nc, header, args, coord_wave=None, pos=None,
                   infer=True, coords=None, spectrum=False):
        ''' Generate coordinates dep on monte carlo choice. '''
        if coords is None:
            # xy coord, origin top left [npixls,2]
            #coords = get_mgrid_np(args.img_sz, indexing='xy')
            coords = get_radec(nr, nc, header, args)
            if pos is not None: # get coordinates for selected pixl only
                coords = coords[pos]

        coords = torch.from_numpy(coords).type(args.float_tensor)
        if args.dim == 2:
            return coords
        #return reshape_coords(coords, args, infer, spectrum, coord_wave=coord_wave)
        return coords
    """

    ################
    # mask creation
    ################

    def create_mask_one_band(n, ratio, seed):
        # generate mask, first 100*ratio% pixls has mask value 1 (unmasked)
        ids = np.arange(n)
        random.seed(seed)
        random.shuffle(ids)
        offset = int(ratio*n)
        mask = np.zeros(n)
        mask[ids[:offset]] = 1
        return mask, ids[-offset:]

    def create_mask(mask_seed, npixls, num_bands, inpaint_bands, mask_config, mask_args, verbose):
        if mask_config == 'rand_diff':
            if verbose: print('= mask diff pixels in diff bands')
            ratio = mask_args[0]
            mask, masked_ids = np.ones((npixls, num_bands)), []
            for i in inpaint_bands:
                mask[:,i], cur_band_masked_ids = create_mask_one_band(npixls, ratio, i + mask_seed)
                masked_ids.append(cur_band_masked_ids)
            # [npixls,nbands], [nbands,num_masked_pixls]
            masked_ids = np.array(masked_ids)

        elif mask_config == 'rand_same':
            if verbose: print('= mask same pixels in diff bands')
            ratio = mask_args[0]
            mask, masked_ids = create_mask_one_band(npixls, ratio, mask_seed)
            mask = np.tile(mask[:,None], (1,num_bands))
            # [npixls, nbands], [num_masked_pixls]
            print(masked_ids.shape)

        elif mask_config == 'region': # NOT TESTED
            assert(False)
            if verbose: print('= mask region')
            (m_start_r, m_start_c, msz) = mask_args
            rs = np.arange(m_start_r, m_start_r+msz)
            cs = np.arange(m_start_c, m_start_c+msz)
            grid = np.stack(np.meshgrid(*tuple([rs,cs]),indexing='ij'), \
                            axis=-1).reshape((-1,2))
            m_ids = np.array(list(map(lambda p: p[0]*nr+p[1], grid)))
            nm_ids = np.array(list(set(ids)-set(m_ids))) # id of pixels used for training
        else:
            raise Exception('Unsupported mask config')
        return mask, masked_ids

    def load_mask(args, flat=True, to_bool=True, to_tensor=True):
        ''' Load (or generate) mask dependeing on config for spectral inpainting.
            If train bands and inpaint bands form a smaller set of
              the band of the current mask file, then we load only and
              slice the corresponding dimension from the larger mask.
        '''
        npixls = args.img_sz**2
        mask_fname = args.mask_fname
        masked_id_fname = args.masked_pixl_id_fname

        if exists(mask_fname) and exists(masked_id_fname):
            if args.verbose: print(f'= loading spectral mask from {mask_fname}')
            mask = np.load(mask_fname)
            masked_ids = np.load(masked_id_fname)
        else:
            assert(len(args.filters) == len(args.train_bands) + len(args.inpaint_bands))
            if args.mask_config == 'region':
                maks_args = [args.m_start_r, args.m_start_c, args.msz]
            else: mask_args = [args.sample_ratio]
            mask, masked_ids = create_mask(args.mask_seed, npixls, args.num_bands, args.inpaint_bands,
                                           args.mask_config, mask_args, args.verbose)
            np.save(mask_fname, mask)
            np.save(masked_id_fname, masked_ids)

        num_smpl_pixls = [np.count_nonzero(mask[:,i]) for i in range(mask.shape[1])]
        if args.verbose: print('= sampled pixls for each band of spectral mask', num_smpl_pixls)

        # slice mask, leave inpaint bands only
        mask = mask[:,args.inpaint_bands] # [npixls,num_inpaint_bands]
        if to_bool:   mask = (mask == 1) # conver to boolean array
        if not flat:  mask = mask.reshape((args.img_sz, args.img_sz, -1))
        if to_tensor: mask = torch.tensor(mask, device=args.device)
        return mask, masked_ids

    def spatial_masking(pixls, coords, args, weights=None):
        # mask data spatially
        mask = load_mask(args)[...,0]
        pixls, coords = pixls[mask], coords[mask]
        if self.load_weights:
            assert(weights is not None)
            weights = weights[mask]
        print("    spatial mask: total num train pixls: {}, with {} per epoch".
              format(len(mask), int(len(mask)*args.train_ratio)))
        return pixls, coords, weights

    ##############
    # Load test data
    ##############

    def load_fake_gt_spectrum(full_wave, float_tensor, fake_spectra_cho, gt_spectra_fname):
        if exists(gt_spectra_fname):
            gt_spectrum = np.load(gt_spectra_fname)
            gt_spectrum = torch.tensor(gt_spectrum).type(float_tensor)
            return gt_spectrum

        gt_spectrum = np.ones((len(full_wave), 2))
        gt_spectrum[:,0] = full_wave
        if fake_spectra_cho == 0:
            gt_spectrum[:,1] = np.ones(len(full_wave))
        elif fake_spectra_cho == 1:
            gt_spectrum [:,1]= 2* np.arange(0, len(full_wave))/1000
        elif fake_spectra_cho == 2:
            val = np.arange(0, len(full_wave))
            gt_spectrum[:,1] = (np.sin(val/2**7)+1)/2
        elif fake_spectra_cho == 3:
            val = np.arange(0, len(full_wave))
            gt_spectrum[:,1] = (np.sin(val/2**6)+1)/2
        elif fake_spectra_cho == 4:
            val = np.arange(0, len(full_wave))
            gt_spectrum[:,1] = (np.sin(val/2**5)+1)/2
        elif fake_spectra_cho == 5:
            val = np.arange(0, len(full_wave))
            gt_spectrum[:,1] = (np.sin(val/2**5)+1)/2
        elif fake_spectra_cho == 6:
            val = np.arange(0, len(full_wave))
            gt_spectrum[:,1] = (np.sin(val/2**3)+1)/2
        else:
            raise Exception('unsupported fake spectrum choice')

        np.save(gt_spectra_fname, gt_spectrum)
        gt_spectrum = torch.tensor(gt_spectrum).type(float_tensor)
        return gt_spectrum

    def load_fake_data(recon, args, spectrum=False):
        ''' Load hardcoded pixl and coords for unit test. '''
        float_tensor = args.float_tensor

        # get wave, trans
        full_wave = np.load(args.full_wave_fname)
        full_trans = torch.tensor(np.load(args.full_trans_fname)).type(float_tensor)
        full_nsmpl = len(full_wave)

        #val = np.load(args.avg_nsmpl_fname)
        #avg_nsmpl = torch.tensor(val).type(float_tensor)

        gt_spectrum = load_fake_gt_spectrum\
            (full_wave, float_tensor, args.fake_spectra_cho, args.gt_spectra_fnames[0])[:,1]

        # get pixl values
        pixls = (full_trans@gt_spectrum / full_nsmpl).unsqueeze(0) # [1,nbands]

        # get coords
        nsmpl = get_num_trans_smpl(recon, args)
        coords = torch.tensor([args.fake_coord]).type(float_tensor)
        coords = coords.unsqueeze(1).tile(1,nsmpl,1) # [1,nsmpl,2]
        if args.mc_cho == 'mc_bandwise' and not spectrum:
            coords = coords.unsqueeze(1).tile(1,args.num_bands,1,1)

        return pixls, coords

    #########
    # getter
    #########

    def get_img_sz(self):
        return self.num_rows, self.num_cols

    def get_pixels(self, to_tensor=True):
        assert(exists(self.pixels_fname))
        pixels = np.load(self.pixels_fname)
        if to_tensor:
            pixels = torch.FloatTensor(pixels) #.to(self.device)
        return pixels

    def get_weights(self, to_tensor=True):
        if not self.load_weights:
            return None

        assert(exists(self.weights_fname))
        weights = np.load(self.weights_fname)
        if to_tensor:
            weights = torch.FloatTensor(weights) #.to(self.device)
        return weights

    def get_coords(self, to_tensor=True):
        assert(exists(self.coords_fname))
        coords = np.load(self.coords_fname)
        if to_tensor:
            coords = torch.FloatTensor(coords) #.to(self.device)
        return coords

    def get_mask(self):
        if self.kwargs["inpaint_cho"] == 'spatial_inpaint':
            self.pixls, self.coords, self.weights = utils.spatial_masking\
                (self.pixls, self.coords, self.kwargs, weights=self.weights)

        elif self.kwargs["inpaint_cho"] == 'spectral_inpaint':
            self.relative_train_bands = self.kwargs["relative_train_bands"]
            self.relative_inpaint_bands = self.kwargs["relative_inpaint_bands"]
            self.mask, self.masked_pixl_ids = utils.load_mask(self.args)
            self.num_masked_pixls = self.masked_pixl_ids.shape[0]

        # iv) get ids of cutout pixels
        if self.save_cutout:
            self.cutout_pixl_ids = utils.generate_cutout_pixl_ids\
                (self.cutout_pos, self.fits_cutout_sz, self.img_sz)
        else: self.cutout_pixl_ids = None
        return

    def get_spectra(self):
        """ Load spectra data if do spectra supervision """
        trusted_spectra_wave_range = [self.kwargs["trusted_wave_lo"],
                                      self.kwargs["trusted_wave_hi"]]
        self.gt_spectra = trans_utils.load_supervision_gt_spectra_all \
            (self.kwargs["spectra_supervision_gt_spectra_fns"],
             trusted_spectra_wave_range,
             self.kwargs["trans_smpl_interval"], self.float_tensor)

        self.spectra_supervision_gt_spectra_pixl_ids = \
            self.kwargs["spectra_supervision_gt_spectra_pixl_ids"]

        # get id bound of trusted wave range
        wave_hi = int(min(self.kwargs["trusted_wave_hi"], int(np.max(self.wave))))
        self.trusted_spectra_wave_id_hi = np.argmin(self.wave < wave_hi)
        self.trusted_spectra_wave_id_lo = np.argmax(self.wave >= self.kwargs["trusted_wave_lo"])

# FITS class ends
#################

def generate_recon_cutout_pixel_ids(pos, cutout_sz, num_cols):
    """
    """
    (r, c) = pos
    rlo, rhi = r, r + cutout_sz
    clo, chi = c, c + cutout_sz
    rs = np.arange(rlo, rhi)
    id_inits = rs * num_cols + clo
    ids = reduce(lambda acc, id_init:
                 acc + list(np.arange(id_init, id_init + cutout_sz)),
                 id_inits, [])
    return np.array(ids)
