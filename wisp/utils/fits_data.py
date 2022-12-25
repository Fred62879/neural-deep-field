
import torch
import numpy as np
import logging as log

from astropy.io import fits
from astropy.wcs import WCS
from functools import reduce
from os.path import join, exists
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from wisp.utils.common import generate_hdu
from wisp.utils.plot import plot_horizontally
from wisp.utils.numerical import normalize_coords, normalize, \
    calculate_metrics, calculate_zscale_ranges_multiple_FITS


class FITSData:
    ''' Data class for FITS files. '''

    def __init__(self, dataset_path, device, **kwargs):
        self.kwargs = kwargs
        self.device = device
        self.verbose = kwargs["verbose"]
        self.footprints = kwargs["fits_footprints"]
        self.tile_ids = kwargs["fits_tile_ids"]
        self.subtile_ids = kwargs["fits_subtile_ids"]

        self.num_bands = kwargs["num_bands"]
        self.u_band_scale = kwargs["u_band_scale"]
        self.load_weights = kwargs["weight_train"]
        self.sensors_full_name = kwargs['sensors_full_name']
        #self.cutout_based_train = kwargs["cutout_based_train"]
        self.load_fits_data_cache = kwargs["load_fits_data_cache"]

        self.use_full_fits = kwargs["use_full_fits"]
        self.fits_cutout_sizes = kwargs["fits_cutout_sizes"]
        self.fits_cutout_start_pos = kwargs["fits_cutout_start_pos"]

        self.headers = {}
        self.num_rows = {}
        self.num_cols = {}

        self.compile_fits_fnames()
        self.set_path(dataset_path)

        self.load_headers()
        self.load_all_fits()
        self.get_world_coords_all_fits()

    ################
    # Generate filenames
    ################

    def compile_fits_fnames(self):
        """ Get fnames of all given fits input for all bands. """
        self.fits_ids, self.fits_groups, self.fits_wgroups = [], {}, {}
        for footprint, tile_id, subtile_id in zip(
                self.footprints, self.tile_ids, self.subtile_ids):

            utile= tile_id + 'c' + subtile_id
            tile = tile_id + '%2C' + subtile_id
            fits_id = footprint + tile_id + subtile_id

            hsc_fits_fname = np.array(
                ['calexp-' + band + '-' + footprint + '-' + tile + '.fits'
                for band in self.sensors_full_name if 'HSC' in band])
            nb_fits_fname = np.array(
                ['calexp-' + band + '-' + footprint + '-' + tile + '.fits'
                for band in self.sensors_full_name if 'NB' in band])
            megau_fits_fname = np.array(
                ['Mega-' + band + '_' + footprint + '_' + utile + '.fits'
                for band in self.sensors_full_name if 'u' in band])
            megau_weights_fname = np.array(
                ['Mega-' + band + '_' + footprint + '_' + utile + '.weight.fits'
                for band in self.sensors_full_name if 'u' in band])

            self.fits_ids.append(fits_id)
            self.fits_groups[fits_id] = np.concatenate((hsc_fits_fname, nb_fits_fname, megau_fits_fname))
            self.fits_wgroups[fits_id] = np.concatenate((hsc_fits_fname, nb_fits_fname, megau_weights_fname))

    def set_path(self, dataset_path):
        input_path = join(dataset_path, 'input')
        img_data_path = join(input_path, self.kwargs['sensor_collection_name'], 'img_data')

        self.input_fits_path = join(input_path, 'input_fits')

        # suffix that uniquely identifies the currently selected group of
        #   tiles with the corresponding cropping parameters, if any
        suffix, self.gt_img_fnames = "", {}
        if self.use_full_fits:
            for fits_id in self.fits_ids:
                suffix += f"_{fits_id}"
                self.gt_img_fnames[fits_id] = join(img_data_path, f'gt_img_{fits_id}')
        else:
            for (fits_id, size, (r,c)) in zip(
                    self.fits_ids, self.fits_cutout_sizes, self.fits_cutout_start_pos):
                suffix += f"_{fits_id}_{size}_{r}_{c}"
                self.gt_img_fnames[fits_id] = join(
                    img_data_path, f'gt_img_{fits_id}_{size}_{r}_{c}')

        norm_str = self.kwargs['train_pixels_norm']

        # image data path creation
        self.coords_fname = join(img_data_path, f'coords{suffix}.npy')
        self.weights_fname = join(img_data_path, f'weights{suffix}.npy')
        self.pixels_fname = join(img_data_path, f'pixels_{norm_str}{suffix}.npy')
        self.coords_range_fname = join(img_data_path, f'coords_range{suffix}.npy')
        self.zscale_ranges_fname = join(img_data_path, f"zscale_ranges{suffix}.npy")

        # mask path creation
        mask_path = join(input_path, 'mask')
        if self.kwargs['mask_config'] == 'region':
            mask_str = '_' + str(self.kwargs['m_start_r']) + '_' \
                + str(self.kwargs['m_start_c']) + '_' \
                + str(self.kwargs['mask_size'])
        else: mask_str = '_' + str(float(100 * self.kwargs['sample_ratio']))

        if self.kwargs['inpaint_cho'] == 'spectral_inpaint':
            self.mask_fn = join(self.mask_path, str(self.kwargs['fits_cutout_size']) + mask_str + '.npy')
            self.masked_pixl_id_fn = join(mask_path, str(self.kwargs['fits_cutout_size']) + mask_str + '_masked_id.npy')
        else:
            self.mask_fn, self.masked_pixl_id_fn = None, None

    ###############
    # Load FITS data
    ###############

    def load_header(self, index, fits_id, full_fits):
        ''' Load header for both full tile and current cutout. '''
        fits_fname = self.fits_groups[fits_id][0]
        id = 0 if 'Mega-u' in fits_fname else 1
        hdu = fits.open(join(self.input_fits_path, fits_fname))[id]
        header = hdu.header

        if full_fits:
            num_rows, num_cols = header['NAXIS2'], header['NAXIS1']
        else:
            size = self.fits_cutout_sizes[index]
            (r, c) = self.fits_cutout_start_pos[index] # start position (r/c)
            pos = (c + size//2, r + size//2)           # center position (x/y)
            wcs = WCS(header)
            cutout = Cutout2D(hdu.data, position=pos, size=self.fits_cutout_size, wcs=wcs)
            header = cutout.wcs.to_header()
            num_rows, num_cols = self.fits_cutout_size, self.fits_cutout_size

        self.headers[fits_id] = header
        self.num_rows[fits_id] = num_rows
        self.num_cols[fits_id] = num_cols

    def load_headers(self):
        for index, fits_id in enumerate(self.fits_ids):
            self.load_header(index, fits_id, True)

    def load_one_fits(self, index, fits_id, load_pixels=True):
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
                    pixels /= self.u_band_scale

                if not self.use_full_fits:
                    size = self.fits_cutout_sizes[index]
                    (r, c) = self.fits_cutout_start_pos[index] # start position (r/c)
                    pixels = pixels[r:r+size, c:c+size]

                if not self.kwargs['train_pixels_norm'] == 'linear':
                    pixels = normalize(pixels, self.kwargs["train_pixels_norm"], gt=pixels)
                cur_data.append(pixels)

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
                    size = self.fits_cutout_sizes[index]
                    (r, c) = self.fits_cutout_start_pos[index] # start position (r/c)
                    var = var[r:r+size, c:c+size].flatten()
                    cur_data.append(var)

        if load_pixels:
            # save gt np img individually for each fits file
            # since different fits may differ in size
            cur_data = np.array(cur_data)    # [nbands,sz,sz]
            np.save(self.gt_img_fnames[fits_id], cur_data)
            plot_horizontally(cur_data, self.gt_img_fnames[fits_id])

            if self.kwargs["to_HDU"]:
                generate_hdu(self.headers[fits_id], cur_data,
                             self.gt_img_fnames[fits_id] + ".fits")

            # flatten into pixels for ease of training
            cur_data = cur_data.reshape(self.num_bands, -1).T
            return cur_data # [npixels,nbands]

        # load weights
        return np.sqrt(np.array(cur_data).T) # [npixels,nbands]

    def load_all_fits(self, to_tensor=True, save_cutout=False):
        ''' Load all images (and weights) and flatten into one array.
            @Return
              pixels:  [npixels,nbands]
              weights: [npixels,nbands]
        '''
        #if self.cutout_based_train:
        #    raise Exception('Cutout based train only works on one fits file.')

        cached = self.load_fits_data_cache and exists(self.pixels_fname) and \
            ([exists(fname) for fname in self.gt_img_fnames]) and \
            (not self.load_weights or exists(self.weights_fname)) and \
            exists(self.zscale_ranges_fname)

        if cached:
            if self.verbose: log.info('FITS data cached.')
        else:
            if self.verbose: log.info('Loading FITS data.')
            if self.load_weights:
                if self.verbose: log.info('Loading weights.')
                weights = np.concatenate([ self.load_one_fits(index, fits_id, load_pixels=False)
                                           for index, fits_id in enumerate(self.fits_ids) ])
                np.save(self.weights_fname, weights)
            else: weights = None

            if self.verbose: log.info('Loading pixels.')
            pixels = [ self.load_one_fits(index, fits_id) # nfits*[npixels,nbands]
                       for index, fits_id in enumerate(self.fits_ids) ]

            # calcualte zscale range for pixel normalization
            zscale_ranges = calculate_zscale_ranges_multiple_FITS(pixels)
            np.save(self.zscale_ranges_fname, zscale_ranges)

            pixels = np.concatenate(pixels) # [total_npixels,nbands]

            # apply normalization to pixels as specified
            if self.kwargs['train_pixels_norm'] == 'linear':
                pixels = normalize(pixels, 'linear')

            np.save(self.pixels_fname, pixels)
            pixel_max = np.round(np.max(pixels, axis=0), 3)
            pixel_min = np.round(np.min(pixels, axis=0), 3)
            log.info(f'train pixels max {pixel_max}')
            log.info(f'train pixels min {pixel_min}')

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

    def get_world_coords_one_fits(self, id, fits_id):
        ''' Get ra/dec coords from one fits file and normalize.
            pix2world calculate coords in x-y order
              coords can be indexed using r-c
            @Return
              coords: 2D coordinates [npixels,2]
        '''
        num_rows, num_cols = self.num_rows[fits_id], self.num_cols[fits_id]
        xids = np.tile(np.arange(0, num_cols), num_rows)
        yids = np.repeat(np.arange(0, num_rows), num_cols)

        wcs = WCS(self.headers[fits_id])
        ras, decs = wcs.all_pix2world(xids, yids, 0) # x-y pixel coord
        if self.use_full_fits:
            coords = np.array([ras, decs]).T
        else:
            coords = np.concatenate(( ras.reshape((num_rows, num_cols, 1)),
                                      decs.reshape((num_rows, num_cols, 1)) ), axis=2)
            size = self.fits_cutout_sizes[id]
            (r, c) = self.fits_cutout_start_pos[id] # start position (r/c)
            coords = coords[r:r+size,c:c+size].reshape(-1,2)
        return coords

    def get_world_coords_all_fits(self):
        ''' Get ra/dec coord from all fits files and normalize. '''
        if exists(self.coords_fname):
            log.info('Loading coords from cache.')
            #coords = np.load(self.coords_fname)
        else:
            log.info('Generating coords.')
            coords = np.concatenate([ self.get_world_coords_one_fits(id, fits_id)
                                      for id, fits_id in enumerate(self.fits_ids) ])
            coords, coords_range = normalize_coords(coords)
            np.save(self.coords_fname, coords)
            np.save(self.coords_range_fname, np.array(coords_range))

    """
    def reshape_coords(coords, args, infer=False, spectrum=False, coord_wave=None):
        nsmpl = get_num_trans_smpl(infer, args)
        coords = coords.unsqueeze(1).tile(1,nsmpl,1) # [npixls,nsmpl,2]
        if args.mc_cho == 'mc_hardcode': # [npixls,nsmpl,2]
            assert(coord_wave is not None)
            bsize = coords.shape[0]
            coords = torch.cat((coords, coord_wave[:bsize]), dim=-1)
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
            #coords = get_mgrid_np(args.img_size, indexing='xy')
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
            (m_start_r, m_start_c, msize) = mask_args
            rs = np.arange(m_start_r, m_start_r+msize)
            cs = np.arange(m_start_c, m_start_c+msize)
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
        npixls = args.img_size**2
        mask_fname = args.mask_fname
        masked_id_fname = args.masked_pixl_id_fname

        if exists(mask_fname) and exists(masked_id_fname):
            if args.verbose: print(f'= loading spectral mask from {mask_fname}')
            mask = np.load(mask_fname)
            masked_ids = np.load(masked_id_fname)
        else:
            assert(len(args.filters) == len(args.train_bands) + len(args.inpaint_bands))
            if args.mask_config == 'region':
                maks_args = [args.m_start_r, args.m_start_c, args.msize]
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
        if not flat:  mask = mask.reshape((args.img_size, args.img_size, -1))
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

    def get_zscale_ranges(self, fits_id=None):
        zscale_ranges = np.load(self.zscale_ranges_fname)
        if fits_id is not None:
            id = self.fits_ids.index(fits_id)
            zscale_ranges = zscale_ranges[id]
        return zscale_ranges

    def get_fits_ids(self):
        return self.fits_ids

    def get_img_sizes(self):
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
                (self.cutout_pos, self.fits_cutout_size, self.img_size)
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

    ############
    # Utilities
    ############

    def restore_evaluate_one_tile(self, index, fits_id, num_pixels_acc, pixels, func, kwargs):
        if self.use_full_fits:
            num_rows, num_cols = self.num_rows[fits_id], self.num_cols[fits_id]
        else: num_rows, num_cols = self.fits_cutout_sizes[index], self.fits_cutout_sizes[index]
        cur_num_pixels = num_rows * num_cols

        cur_tile = np.array(pixels[num_pixels_acc : num_pixels_acc + cur_num_pixels]).T. \
            reshape((self.num_bands, num_rows, num_cols))

        cur_metrics, cur_metrics_zscale = func(self, fits_id, cur_tile, **kwargs)

        num_pixels_acc += cur_num_pixels
        return num_pixels_acc, cur_metrics, cur_metrics_zscale

    def restore_evaluate_tiles(self, pixels, func, kwargs):
        """ Restore original FITS/cutouts given flattened pixels.
            Perform given func (e.g. metric calculation) on each restored FITS/cutout image.
            @Param
               pixels: flattened pixels, [npixels, nbands]
            @Return
               pixels: list of np array of size [nbands,nrows,ncols]
        """
        elem_type = type(pixels)

        if type(pixels) is list:
            pixels = torch.stack(pixels)

        if type(pixels).__module__ == "torch":
            if pixels.device != "cpu":
                pixels = pixels.detach().cpu()
            pixels = pixels.numpy()

        if kwargs["calculate_metrics"]:
            metric_options = kwargs["metric_options"]
            metrics = np.zeros((len(metric_options), 0, self.num_bands))
            metrics_zscale = np.zeros((len(metric_options), 0, self.num_bands))
        else: metrics, metrics_zscale = None, None

        num_pixels_acc = 0
        for index, fits_id in enumerate(self.fits_ids):
            num_pixels_acc, cur_metrics, cur_metrics_zscale = self.restore_evaluate_one_tile(
                index, fits_id, num_pixels_acc, pixels, func, kwargs)

            if kwargs["calculate_metrics"]:
                metrics = np.concatenate((metrics, cur_metrics), axis=1)
                metrics_zscale = np.concatenate((metrics_zscale, cur_metrics_zscale), axis=1)

        return metrics, metrics_zscale

# FITS class ends
#################

def recon_img_and_evaluate(recon_pixels, dataset, **kwargs):
    """ Reconstruct multiband image save locally and calculate metrics.
        @Return:
           metrics(_z): metrics of current model [n_metrics,1,ntiles,nbands]
    """
    recon_fname = join(kwargs["dir"], kwargs["fname"])
    if kwargs["recon_norm"]: recon_fname += "_norm"
    if kwargs["recon_flat_trans"]: recon_fname += "_flat"

    # get metrics for all recon tiles using current model
    metrics, metrics_zscale = dataset.restore_evaluate_tiles(
        recon_pixels, func=evaluate_func, kwargs=kwargs)
    if metrics is not None and metrics_zscale is not None:
        return metrics[:,None], metrics_zscale[:,None]
    return None, None

def evaluate_func(class_obj, fits_id, recon_tile, **kwargs):
    """ Image evaluation function (e.g. saving, metric calculation),
          passed to astro_dataset class and performed after
          images being restored to the original shape.
        @Param:
          class_obj:   class object, FITSData class
          fits_id:     id of current fits tile to recon and evaluate
          recon_tile:  restored recon tile [nbands,sz,sz]
        @Return:
          metrics(_z): metrics of current model for current fits tile, [n_metrics,1,nbands]
    """
    #if denorm_args is not None: recon *= denorm_args
    verbose = kwargs["verbose"]
    fname = kwargs["fname"]
    dir = kwargs["dir"]

    # restore flattened pixels to original shape
    # a group of tiles are saved locally as soon as restore finished
    # don't wait until all tiles are restored to save RAM
    np_fname = join(dir, f"{fits_id}_{fname}.npy")
    np.save(np_fname, recon_tile)

    if kwargs["to_HDU"]: # generate fits image
        fits_fname = join(dir, f"{fits_id}_{fname}.fits")
        generate_hdu(class_obj.headers[fits_id], recon_tile, fits_fname)

    # if mask is not None: # inpaint: fill unmasked pixels with gt value
    #     recon = restore_unmasked(recon, np.copy(gt), mask)
    #     if fn is not None:
    #         np.save(fn + "_restored.npy", recon)

    # plot recon tile
    png_fname = join(dir, f"{fits_id}_{fname}.png")
    zscale_ranges = class_obj.get_zscale_ranges(fits_id)
    plot_horizontally(recon_tile, png_fname, zscale_ranges=zscale_ranges)
    recon_max = np.round(np.max(recon_tile, axis=(1,2)), 1)
    if verbose: log.info(f"recon. pixel max {recon_max}")

    # calculate metrics
    if kwargs["calculate_metrics"]:
        gt_fname = class_obj.gt_img_fnames[fits_id] + '.npy'
        gt_tile = np.load(gt_fname)
        gt_max = np.round(np.max(gt_tile, axis=(1,2)), 1)
        if verbose: log.info(f"GT. pixel max {gt_max}")

        metrics = calculate_metrics(
            recon_tile, gt_tile, kwargs["metric_options"])[:,None]
        metrics_zscale = calculate_metrics(
            recon_tile, gt_tile, kwargs["metric_options"], zscale=True)[:,None]
        return metrics, metrics_zscale
    return None, None


## Abandoned
# def get_recon_cutout_pixel_ids(pos, fits_cutout_size, recon_cutout_size,
#                                num_rows, num_cols, cutout_tile_id, use_full_fits):

#     """ Get id of pixels within cutout to reconstruct.
#         Cutout could be within the original image or the same as original.
#         If train over multiple tiles, we reconstruct cutout only for one of the tile.
#         @Param
#           pos: starting r/c position of cutout
#                (local r/c relative to original image start r/c position)
#           fits_cutout_size: size of fits cutout (if not use full fits)
#                           this is the size of the image we train over.
#           recon_cutout_size: size of cutout to reconstruct.
#                            this is <= fits_cutout_size
#           num_rows/cols: number of rows/columns of original image
#                          (map from tile_id to num_rows/cols)
#         @Return
#           ids: pixels ids (with 0 being the first pixel in the original image)
#     """
#     # count #pixels before the selected tile
#     offset = 0
#     for (tile_id, num_row), (_, num_col) in zip(num_rows.items(), num_cols.items()):
#         if tile_id == cutout_tile_id: break
#         if use_full_fits: offset += num_row * num_col
#         else: offset += cutout_size**2

#     (r, c) = pos
#     rlo, rhi = r, r + recon_cutout_size
#     clo, chi = c, c + recon_cutout_size
#     rs = np.arange(rlo, rhi)

#     num_col = num_cols[cutout_tile_id] if use_full_fits else fits_cutout_size
#     id_inits = rs * num_col + clo
#     ids = reduce(lambda acc, id_init:
#                  acc + list(np.arange(id_init, id_init + recon_cutout_size)),
#                  id_inits, [])

#     ids = np.array(ids) + offset
#     return ids
