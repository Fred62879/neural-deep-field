
import torch
import numpy as np
import logging as log

from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from os.path import join, exists
from astropy.nddata import Cutout2D
from wisp.utils.common import worldToPix
from astropy.coordinates import SkyCoord

from wisp.utils.common import generate_hdu
from wisp.utils.plot import plot_horizontally, mark_on_img
from wisp.datasets.data_utils import add_dummy_dim, create_uid
from wisp.utils.numerical import normalize_coords, normalize, \
    calculate_metrics, calculate_zscale_ranges_multiple_FITS


class FITSData:
    """ Data class for FITS files. """

    def __init__(self, dataset_path, device, **kwargs):
        self.kwargs = kwargs
        self.load_weights = kwargs["weight_train"]
        self.qtz = kwargs["quantize_latent"] or kwargs["quantize_spectra"]

        if not self.require_any_data(kwargs["tasks"]): return

        self.device = device
        self.verbose = kwargs["verbose"]
        self.num_bands = kwargs["num_bands"]
        self.u_band_scale = kwargs["u_band_scale"]
        self.sensors_full_name = kwargs["sensors_full_name"]
        self.load_fits_data_cache = kwargs["load_fits_data_cache"]

        self.use_full_fits = kwargs["use_full_fits"]
        self.fits_cutout_num_rows = kwargs["fits_cutout_num_rows"]
        self.fits_cutout_num_cols = kwargs["fits_cutout_num_cols"]
        self.fits_cutout_sizes = kwargs["fits_cutout_sizes"]
        self.fits_cutout_start_pos = kwargs["fits_cutout_start_pos"]

        self.data = {}

        self.compile_fits_fnames()
        self.set_path(dataset_path)
        self.init()

    #############
    # Initializations
    #############

    def require_any_data(self, tasks):
        """ Find all required data based on given self.tasks. """
        tasks = set(tasks)

        self.require_weights = "train" in tasks and self.load_weights
        self.require_pixels = len(tasks.intersection({
            "train","recon_img","log_pixel_value"})) != 0

        ## TODO: REMOVE require_scaler
        self.require_scaler = self.kwargs["space_dim"] == 3 and self.qtz \
            and self.kwargs["generate_scaler"]
        self.require_redshift = self.kwargs["space_dim"] == 3 and self.qtz \
            and self.kwargs["generate_redshift"] and self.kwargs["redshift_supervision"]

        self.require_coords = self.kwargs["spectra_supervision"] or \
            self.require_scaler or self.require_redshift or \
            len(tasks.intersection({
                "train","recon_img","recon_synthetic_band","recon_gt_spectra"})) != 0

        return self.require_coords or self.require_pixels or \
            self.require_weights or self.require_redshift or \
            self.require_scaler or "recon_codebook_spectra" in tasks

    def init(self):
        """ Load all needed data. """
        self.load_headers()

        if self.require_coords:
            #self.get_world_coords_all_fits()
            self.get_pixel_coords_all_fits()

        if self.require_pixels:
            self.load_all_fits()

        # if self.require_redshift:
        #     self.get_redshift_all_fits()

    def set_path(self, dataset_path):
        input_path = join(dataset_path, "input")
        self.input_fits_path = join(input_path, "input_fits")
        img_data_path = join(input_path, self.kwargs["sensor_collection_name"], "img_data")
        paths = [input_path, img_data_path, self.input_fits_path]

        norm = self.kwargs["gt_img_norm_cho"]
        norm_str = self.kwargs["train_pixels_norm"]

        self.gt_paths, self.gt_img_fnames, self.gt_img_distrib_fnames = [], {}, {}

        if self.use_full_fits:
            for fits_uid in self.fits_uids:
                gt_path = join(img_data_path, f"{norm}_{fits_uid}")

                self.gt_paths.append(gt_path)
                self.gt_img_fnames[fits_uid] = join(gt_path, "gt_img")
                self.gt_img_distrib_fnames[fits_uid] = join(gt_path, "gt_img_distrib")
        else:
            for (fits_uid, num_rows, num_cols, (r,c)) in zip(
                    self.fits_uids, self.fits_cutout_num_rows,
                    self.fits_cutout_num_cols, self.fits_cutout_start_pos):
                gt_path = join(img_data_path, f"{norm}_{fits_uid}_{num_rows}_{num_cols}_{r}_{c}")

                self.gt_paths.append(gt_path)
                self.gt_img_fnames[fits_uid] = join(gt_path, "gt_img")
                self.gt_img_distrib_fnames[fits_uid] = join(gt_path, "gt_img_distrid")

        # generate folder for each gt image
        paths.extend(self.gt_paths)

        # image data path creation
        suffix = create_uid(self, **self.kwargs)
        self.coords_fname = join(img_data_path, f"coords{suffix}.npy")
        self.weights_fname = join(img_data_path, f"weights{suffix}.npy")
        self.pixels_fname = join(img_data_path, f"pixels_{norm_str}{suffix}.npy")
        self.coords_range_fname = join(img_data_path, f"coords_range{suffix}.npy")
        self.zscale_ranges_fname = join(img_data_path, f"zscale_ranges{suffix}.npy")

        # create path
        for path in paths:
            Path(path).mkdir(parents=True, exist_ok=True)

    def compile_fits_fnames(self):
        """ Get fnames of all given fits input for all bands.
        """
        # sort fits fnames (so we can locate a random pixel from different fits)
        footprints = self.kwargs["fits_footprints"]
        tile_ids = self.kwargs["fits_tile_ids"]
        subtile_ids = self.kwargs["fits_subtile_ids"]
        footprints.sort();tile_ids.sort();subtile_ids.sort()

        self.fits_uids, self.fits_groups, self.fits_wgroups = [], {}, {}

        for footprint, tile_id, subtile_id in zip(footprints, tile_ids, subtile_ids):
            utile= tile_id + "c" + subtile_id
            tile = tile_id + "%2C" + subtile_id

            # fits ids can be duplicated only if we crop cutout from full fits
            # where we may have multiple cutouts from the same fits (thus same ids)
            fits_uid = footprint + tile_id + subtile_id
            self.fits_uids.append(fits_uid)

            # avoid duplication
            if fits_uid in self.fits_groups and fits_uid in self.fits_wgroups:
                continue

            hsc_fits_fname = np.array(
                ["calexp-" + band + "-" + footprint + "-" + tile + ".fits"
                for band in self.kwargs["sensors_full_name"] if "HSC" in band])
            nb_fits_fname = np.array(
                ["calexp-" + band + "-" + footprint + "-" + tile + ".fits"
                for band in self.kwargs["sensors_full_name"] if "NB" in band])
            megau_fits_fname = np.array(
                ["Mega-" + band + "_" + footprint + "_" + utile + ".fits"
                for band in self.kwargs["sensors_full_name"] if "u" in band])
            megau_weights_fname = np.array(
                ["Mega-" + band + "_" + footprint + "_" + utile + ".weight.fits"
                for band in self.kwargs["sensors_full_name"] if "u" in band])

            self.fits_groups[fits_uid] = np.concatenate(
                (hsc_fits_fname, nb_fits_fname, megau_fits_fname))

            self.fits_wgroups[fits_uid] = np.concatenate(
                (hsc_fits_fname, nb_fits_fname, megau_weights_fname))

        self.num_fits = len(self.fits_uids)

        # make sure no duplicate fits ids exist if use full tile
        if self.use_full_fits:
            assert( len(self.fits_uids) == len(set(self.fits_uids)))

    ###############
    # Load FITS data
    ###############

    def load_header(self, index, fits_uid):
        """ Load header for both full tile and current cutout. """
        fits_fname = self.fits_groups[fits_uid][0]
        id = 0 if "Mega-u" in fits_fname else 1
        hdu = fits.open(join(self.input_fits_path, fits_fname))[id]
        header = hdu.header

        if self.use_full_fits:
            num_rows, num_cols = header["NAXIS2"], header["NAXIS1"]
        else:
            (r, c) = self.fits_cutout_start_pos[index] # start position (r/c)
            num_rows = self.fits_cutout_num_rows[index]
            num_cols = self.fits_cutout_num_cols[index]
            pos = (c + num_cols//2, r + num_rows//2)

            wcs = WCS(header)
            cutout = Cutout2D(hdu.data, position=pos, size=(num_rows,num_cols), wcs=wcs)
            header = cutout.wcs.to_header()

        self.headers[fits_uid] = header
        self.num_rows[fits_uid] = num_rows
        self.num_cols[fits_uid] = num_cols

    def load_headers(self):
        self.headers, self.num_rows, self.num_cols = {}, {}, {}
        for index, fits_uid in enumerate(self.fits_uids):
            self.load_header(index, fits_uid)

    def load_one_fits(self, index, fits_uid, load_pixels=True):
        """ Load pixel values or variance from one FITS file (tile_id/subtile_id).
            Load pixel and weights separately to avoid using up mem.
        """
        cur_data = []

        for i in range(self.num_bands):
            if load_pixels:
                fits_fname = self.fits_groups[fits_uid][i]

                # u band pixel vals in first hdu, others in 2nd hdu
                is_u = "Mega-u" in fits_fname
                id = 0 if is_u else 1

                pixels = fits.open(join(self.input_fits_path, fits_fname))[id].data
                if is_u: # scale u and u* band pixel values
                    pixels /= self.u_band_scale

                if not self.use_full_fits:
                    (r, c) = self.fits_cutout_start_pos[index] # start position (r/c)
                    num_rows = self.num_rows[fits_uid]
                    num_cols = self.num_cols[fits_uid]
                    pixels = pixels[r:r+num_rows, c:c+num_cols]

                if not self.kwargs["train_pixels_norm"] == "linear":
                    pixels = normalize(pixels, self.kwargs["train_pixels_norm"], gt=pixels)
                cur_data.append(pixels)

            else: # load weights
                fits_wfname = self.fits_wgroups[fits_uid][i]
                # u band weights in first hdu, others in 4th hdu
                id = 0 if "Mega-u" in fits_wfname else 3
                var = fits.open(join(self.input_fits_path, fits_wfname))[id].data

                # u band weights stored as inverse variance, others as variance
                if id == 3: weight = var
                else:       weight = 1 / (var + 1e-6) # avoid division by 0
                if self.use_full_fits:
                    cur_data.append(weight.flatten())
                else:
                    (r, c) = self.fits_cutout_start_pos[index] # start position (r/c)
                    num_rows = self.fits_cutout_num_rows[index]
                    num_cols = self.fits_cutout_num_cols[index]
                    var = var[r:r+num_rows, c:c+num_cols].flatten()
                    cur_data.append(var)

        if load_pixels:
            # save gt np img individually for each fits file
            # since different fits may differ in size
            cur_data = np.array(cur_data) # [nbands,sz,sz]
            np.save(self.gt_img_fnames[fits_uid], cur_data)
            plot_horizontally(cur_data, self.gt_img_fnames[fits_uid], "plot_img")

            if self.kwargs["to_HDU"]:
                generate_hdu(self.headers[fits_uid], cur_data,
                             self.gt_img_fnames[fits_uid] + ".fits")

            # flatten into pixels for ease of training
            cur_data = cur_data.reshape(self.num_bands, -1).T
            return cur_data # [npixels,nbands]

        # load weights
        return np.sqrt(np.array(cur_data).T) # [npixels,nbands]

    def load_all_fits(self, to_tensor=True, save_cutout=False):
        """ Load all images (and weights) and flatten into one array.
            @Return
              pixels:  [npixels,nbands]
              weights: [npixels,nbands]
        """
        #if self.cutout_based_train:
        #    raise Exception("Cutout based train only works on one fits file.")

        cached = self.load_fits_data_cache and exists(self.pixels_fname) and \
            all([exists(fname) for fname in self.gt_img_fnames]) and \
            (not self.load_weights or exists(self.weights_fname)) and \
            exists(self.zscale_ranges_fname)

        if cached:
            if self.verbose: log.info("FITS data cached.")
            pixels = np.load(self.pixels_fname)
            if self.load_weights:
                weights = np.load(self.weights_fname)
        else:
            if self.verbose: log.info("Loading FITS data.")
            if self.load_weights:
                if self.verbose: log.info("Loading weights.")
                weights = np.concatenate([ self.load_one_fits(index, fits_uid, load_pixels=False)
                                           for index, fits_uid in enumerate(self.fits_uids) ])
                np.save(self.weights_fname, weights)
            else: weights = None

            if self.verbose: log.info("Loading pixels.")
            pixels = [ self.load_one_fits(index, fits_uid) # nfits*[npixels,nbands]
                       for index, fits_uid in enumerate(self.fits_uids) ]

            # calcualte zscale range for pixel normalization
            zscale_ranges = calculate_zscale_ranges_multiple_FITS(pixels)
            np.save(self.zscale_ranges_fname, zscale_ranges)

            pixels = np.concatenate(pixels) # [total_npixels,nbands]

            # apply normalization to pixels as specified
            if self.kwargs["train_pixels_norm"] == "linear":
                pixels = normalize(pixels, "linear")
            elif self.kwargs["train_pixels_norm"] == "zscale":
                pixels = normalize(pixels, "zscale", gt=pixels)

            np.save(self.pixels_fname, pixels)

        if self.kwargs["plot_img_distrib"]:
            for fits_uid in self.fits_uids:
                cur_data = np.load(self.gt_img_fnames[fits_uid] + ".npy")
                plot_horizontally(cur_data, self.gt_img_distrib_fnames[fits_uid], "plot_distrib")

        pixel_max = np.round(np.max(pixels, axis=0), 3)
        pixel_min = np.round(np.min(pixels, axis=0), 3)
        log.info(f"train pixels max {pixel_max}")
        log.info(f"train pixels min {pixel_min}")

        self.data["pixels"] = torch.FloatTensor(pixels)
        if self.load_weights:
            self.data["weights"] = torch.FloatTensor(weights)

    ##############
    # Load redshifts
    ##############

    def get_redshift_one_fits(self, id, fits_uid):
        if self.use_full_fits:
            num_rows, num_cols = self.num_rows[fits_uid], self.num_cols[fits_uid]
            redshifts = -1 * np.ones((num_rows, num_cols))
        else:
            num_rows = self.fits_cutout_num_rows[index]
            num_cols = self.fits_cutout_num_cols[index]
            redshifts = -1 * np.ones((num_rows, num_cols))

        return redshifts

    def get_redshift_all_fits(self):
        """ Load dummy redshift values for now.
        """
        redshift = [ self.get_redshift_one_fits(id, fits_uid)
                     for id, fits_uid in enumerate(self.fits_uids) ]
        redshift = np.array(redshift).flatten()
        self.data["redshift"] = torch.FloatTensor(redshift)

    ##############
    # Load coords
    ##############

    # def get_world_coords_one_fits(self, id, fits_uid):
    #     """ Get ra/dec coords from one fits file and normalize.
    #         pix2world calculate coords in x-y order
    #           coords can be indexed using r-c
    #         @Return
    #           coords: 2D coordinates [npixels,2]
    #     """
    #     num_rows, num_cols = self.num_rows[fits_uid], self.num_cols[fits_uid]
    #     xids = np.tile(np.arange(0, num_cols), num_rows)
    #     yids = np.repeat(np.arange(0, num_rows), num_cols)

    #     wcs = WCS(self.headers[fits_uid])
    #     ras, decs = wcs.all_pix2world(xids, yids, 0) # x-y pixel coord
    #     if self.use_full_fits:
    #         coords = np.array([ras, decs]).T
    #     else:
    #         coords = np.concatenate(( ras.reshape((num_rows, num_cols, 1)),
    #                                   decs.reshape((num_rows, num_cols, 1)) ), axis=2)
    #         # size = self.fits_cutout_sizes[id]
    #         num_rows = self.num_rows[fits_uid]
    #         num_cols = self.num_cols[fits_uid]

    #         (r, c) = self.fits_cutout_start_pos[id] # start position (r/c)
    #         coords = coords[r:r+num_rows,c:c+num_cols].reshape(-1,2)
    #     return coords

    # def get_world_coords_all_fits(self):
    #     """ Get ra/dec coord from all fits files and normalize.
    #         @Return
    #           coords [num_pixels,1,2]
    #     """
    #     if exists(self.coords_fname):
    #         log.info("Loading coords from cache.")
    #         coords = np.load(self.coords_fname)
    #     else:
    #         log.info("Generating coords.")
    #         coords = np.concatenate([ self.get_world_coords_one_fits(id, fits_uid)
    #                                   for id, fits_uid in enumerate(self.fits_uids) ])
    #         coords, coords_range = normalize_coords(coords)
    #         np.save(self.coords_fname, coords)
    #         np.save(self.coords_range_fname, np.array(coords_range))

    #     self.data["coords"] = add_dummy_dim(coords, **self.kwargs)

    def get_pixel_coords_all_fits(self):
        # assert(not self.use_full_fits)
        # assert(len(self.fits_uids) == 1)
        for id, fits_uid in enumerate(self.fits_uids):
            num_rows, num_cols = self.num_rows[fits_uid], self.num_cols[fits_uid]
            self.get_mgrid_np(num_rows, num_cols)
            # assert(num_rows == num_cols)
            # size = self.fits_cutout_sizes[id]
            # self.get_mgrid_np(size)

    def get_mgrid_np(self, num_rows, num_cols, lo=-1, hi=1, dim=2, indexing='ij', flat=True):
    #def get_mgrid_np(self, sidelen, lo=-1, hi=1, dim=2, indexing='ij', flat=True):
        """ Generates a flattened grid of (x,y,...) coords in [-1,1] (numpy version).
        """
        # arrays = tuple(dim * [np.linspace(lo, hi, num=sidelen)])
        # mgrid = np.stack(np.meshgrid(*arrays, indexing=indexing), axis=-1)

        x = np.linspace(lo, hi, num=num_cols)
        y = np.linspace(lo, hi, num=num_rows)
        mgrid = np.stack(np.meshgrid(x, y, indexing=indexing), axis=-1)

        if flat: mgrid = mgrid.reshape(-1,dim) # [sidelen**2,dim]
        self.data["coords"] = add_dummy_dim(mgrid, **self.kwargs)

    def get_mgrid_tensor(self, sidelen, lo=-1, hi=1, dim=2, flat=True):
        """ Generates a flattened grid of (x,y,...) coords in [-1,1] (Tensor version).
        """
        tensors = tuple(dim * [torch.linspace(lo, hi, steps=sidelen)])
        mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
        if flat: mgrid = mgrid.reshape(-1, dim)
        self.data["coords"] = add_dummy_dim(mgrid, **self.kwargs)

    #############
    # Getters
    #############

    def get_zscale_ranges(self, fits_uid=None):
        zscale_ranges = np.load(self.zscale_ranges_fname)
        if fits_uid is not None:
            id = self.fits_uids.index(fits_uid)
            zscale_ranges = zscale_ranges[id]
        return zscale_ranges

    def get_fits_uids(self):
        return self.fits_uids

    def get_num_rows(self):
        return self.num_rows

    def get_num_cols(self):
        return self.num_cols

    def get_num_coords(self):
        return self.data["coords"].shape[0]

    def get_fits_cutout_sizes(self):
        return self.fits_cutout_sizes

    def get_fits_cutout_start_pos(self):
        return self.fits_cutout_start_pos

    def get_pixels(self, idx=None):
        if idx is not None:
            return self.data["pixels"][idx]
        return self.data["pixels"]

    def get_weights(self, idx=None):
        if idx is not None:
            return self.data["weights"][idx]
        return self.data["weights"]

    def get_redshifts(self, idx=None):
        if idx is not None:
            return self.data["redshift"][idx]
        return self.data["redshift"]

    def get_coord(self, idx):
        if type(idx) == list:
            for id in idx:
                assert(id >= 0 and id < len(self.data["coords"]))
        else:
            assert(id >= 0 and id < len(self.data["coords"]))
        return self.data["coords"][idx]

    def get_coords(self, idx=None):
        """ Get all coords [n,1,2] """
        if idx is not None:
            return self.data["coords"][idx]
        return self.data["coords"]

    def get_gt_img_fnames(self):
        return self.gt_img_fnames

    def get_gt_paths(self):
        return self.gt_paths

    ############
    # Utilities
    ############

    def mark_on_img(self, coords, markers, fits_id):
        """ Mark spectra pixels on gt image
            @Param
              coords: r, c
        """
        fits_uid = self.fits_uids[fits_id]
        gt_img_fname = self.gt_img_fnames[fits_uid]
        gt_img = np.load(gt_img_fname + ".npy")
        png_fname = gt_img_fname + "_marked.png"
        mark_on_img(png_fname, gt_img, coords, markers)

    def convert_from_world_coords(self, ra, dec, neighbour_size, footprint, tile_id, subtile_id):
        """ Get coordinate of pixel with given ra/dec
            @Param:
               fits_id:        index of fits where the selected spectra comes from
               spectra_id:     index of the selected spectra
               neighbour_size: size of neighbouring area to average for spectra
               spectra_data

            We can either
             i) directly normalize the given ra/dec with coordinates range or
            ii) get pixel id based on ra/dec and index from loaded coordinates.

            Since ra/dec values from spectra data may not be as precise as
              real coordinates calculated from wcs package, method i) tends to
              be inaccurate and method ii) is more reliable.
            However, if given ra/dec is not within loaded coordinates, we
              can only use method i)
        """
        fits_uid = footprint + tile_id + subtile_id
        fits_id = self.fits_uids.index(fits_uid)

        # ra/dec values from spectra data may not be exactly the same as real coords
        # this normalized ra/dec may thus be slightly different from real coords
        # (ra_lo, ra_hi, dec_lo, dec_hi) = np.load(self.fits_obj.coords_range_fname)
        # coord_loose = ((ra - ra_lo) / (ra_hi - ra_lo),
        #                (dec - dec_lo) / (dec_hi - dec_lo))

        # get a random header
        random_name = f"calexp-HSC-G-{footprint}-{tile_id}%2C{subtile_id}.fits"
        fits_fname = join(self.input_fits_path, random_name)
        header = fits.open(fits_fname)[1].header

        # index coord from original coords array to get accurate coord
        # this only works if spectra coords is included in the loaded coords
        (r, c) = worldToPix(header, ra, dec) # image coord, r and c coord within full tile

        if self.use_full_fits:
            img_coords = np.array([r, c, fits_id])
        else:
            start_pos = self.fits_cutout_start_pos[fits_id]
            img_coords = np.array([r - start_pos[0], c - start_pos[1], fits_id])

        pixel_ids = self.get_pixel_ids(fits_uid, r, c, neighbour_size)
        grid_coords = self.get_coord(pixel_ids)
        # print(r, c, pixel_ids, coords_accurate, self.kwargs["fits_cutout_start_pos"])
        return img_coords, grid_coords, pixel_ids

    def calculate_local_id(self, r, c, index, fits_uid):
        """ Count number of pixels before given position in given tile.
        """
        if self.use_full_fits:
            r_lo, c_lo = 0, 0
            total_cols = self.num_cols[fits_uid]
        else:
            (r_lo, c_lo) = self.fits_cutout_start_pos[index]
            # total_cols = self.fits_cutout_sizes[index]
            total_cols = self.num_cols[fits_uid]

        local_id = total_cols * (r - r_lo) + c - c_lo
        return local_id

    def calculate_global_offset(self, fits_uid):
        """ Count total number of pixels before the given tile.
            Assume given fits_uid is included in loaded fits ids which
              is sorted in alphanumerical order.
            @Return
               id: index of given fits id inside all loaded tiles
               base_count: total # pixels before current tile
        """
        id, base_count, found = 0, 0, False

        # count total number of pixels before the given tile
        for cur_fits_uid in self.fits_uids:
            if cur_fits_uid == fits_uid: found = True; break
            if self.use_full_fits:
                base_count += self.num_rows[cur_fits_uid] * self.num_cols[cur_fits_uid]
            else:
                base_count += self.num_rows[cur_fits_uid] * self.num_cols[cur_fits_uid]
                # base_count += self.fits_cutout_sizes[id]**2
            id += 1

        assert(found)
        return id, base_count

    def calculate_neighbour_ids(self, base_count, r, c, neighbour_size, index, fits_uid):
        """ Get global id of coords within neighbour_size of given coord (specified by r/c).
            For neighbour_size being: 2, 3, 4, the collected ids:
            . .   . . .   . . . .
            . *   . * .   . . . .
                  . . .   . . * .
                          . . . .
        """
        ids = []
        offset = neighbour_size // 2
        for i in range(r - offset, r + (neighbour_size - offset)):
            for j in range(c - offset, c + (neighbour_size - offset)):
                local_id = self.calculate_local_id(i, j, index, fits_uid)
                ids.append(base_count + local_id)
        return ids

    def get_pixel_ids(self, fits_uid, r, c, neighbour_size):
        """ Get global id of given position based on its
              local r/c position in given fits tile.
            If neighbour_size is > 1, also find id of neighbour pixels within neighbour_size.
        """
        index, base_count = self.calculate_global_offset(fits_uid)
        if neighbour_size <= 1:
            local_id = self.calculate_local_id(r, c, index, fits_uid)
            ids = [local_id + base_count]
        else:
            ids = self.calculate_neighbour_ids(base_count, r, c, neighbour_size, index, fits_uid)
        return ids

    def evaluate(self, index, fits_uid, recon_tile, **re_args):
        """ Image evaluation function (e.g. saving, metric calculation).
            @Param:
              fits_uid:     id of current fits tile to evaluate
              recon_tile:  restored fits tile [nbands,sz,sz]
            @Return:
              metrics(_z): metrics of current model for current fits tile, [n_metrics,1,nbands]
        """
        dir = re_args["dir"]
        fname = re_args["fname"]
        verbose = re_args["verbose"]

        #if denorm_args is not None: recon_tile *= denorm_args
        # if mask is not None: # inpaint: fill unmasked pixels with gt value
        #     recon = restore_unmasked(recon, np.copy(gt), mask)
        #     if fn is not None:
        #         np.save(fn + "_restored.npy", recon)

        if re_args["log_max"]:
            #recon_min = np.round(np.min(recon_tile, axis=(1,2)), 1)
            #recon_mean = np.round(np.mean(recon_tile, axis=(1,2)), 1)
            recon_max = np.round(np.max(recon_tile, axis=(1,2)), 1)
            # log.info(f"recon. pixel min {recon_min}")
            # log.info(f"recon. pixel mean {recon_mean}")
            log.info(f"recon. pixel max {recon_max}")

        if re_args["save_locally"]:
            np_fname = join(dir, f"{fits_uid}_{fname}.npy")
            #if restore_args["recon_norm"]: np_fname += "_norm"
            if "recon_synthetic_band" in re_args and re_args["recon_synthetic_band"]:
                np_fname += "_synthetic"
            np.save(np_fname, recon_tile)

        if re_args["to_HDU"]:
            fits_fname = join(dir, f"{fits_uid}_{fname}.fits")
            generate_hdu(class_obj.headers[fits_uid], recon_tile, fits_fname)

        if "plot_func" in re_args:
            png_fname = join(dir, f"{fits_uid}_{fname}.png")
            if re_args["zscale"]:
                zscale_ranges = self.get_zscale_ranges(fits_uid)
                re_args["plot_func"](recon_tile, png_fname, zscale_ranges=zscale_ranges)
            elif re_args["match_fits"]:
                re_args["plot_func"](recon_tile, png_fname, index)
            else:
                re_args["plot_func"](recon_tile, png_fname)

        if re_args["calculate_metrics"]:
            gt_fname = self.gt_img_fnames[fits_uid] + ".npy"
            gt_tile = np.load(gt_fname)
            gt_max = np.round(np.max(gt_tile, axis=(1,2)), 1)
            log.info(f"GT. pixel max {gt_max}")

            metrics = calculate_metrics(
                recon_tile, gt_tile, re_args["metric_options"])[:,None]
            metrics_zscale = calculate_metrics(
                recon_tile, gt_tile, re_args["metric_options"], zscale=True)[:,None]
            return metrics, metrics_zscale
        return None, None

    def restore_evaluate_zoomed_tile(self, recon_tile, fits_uid, **re_args):
        """ Crop smaller cutouts from reconstructed image.
            Helpful to evaluate local reconstruction quality when recon is large.
        """
        id = re_args["cutout_fits_uids"].index(fits_uid)
        zscale_ranges = self.get_zscale_ranges(fits_uid)

        for i, (size, (r,c)) in enumerate(
                zip(re_args["cutout_sizes"][id], re_args["cutout_start_pos"][id])
        ):
            zoomed_gt = np.load(self.gt_img_fnames[fits_uid] + ".npy")[:,r:r+size,c:c+size]
            zoomed_gt_fname = str(self.gt_img_fnames[fits_uid]) + f"_zoomed_{size}_{r}_{c}"
            plot_horizontally(zoomed_gt, zoomed_gt_fname, "plot_img")

            zoomed_recon = recon_tile[:,r:r+size,c:c+size]
            zoomed_recon_fname = join(re_args["zoomed_recon_dir"],
                                      str(re_args["zoomed_recon_fname"]) + f"_{fits_uid}_{i}")
            plot_horizontally(zoomed_recon, zoomed_recon_fname,
                              "plot_img", zscale_ranges=zscale_ranges)

    def restore_evaluate_one_tile(self, index, fits_uid, num_pixels_acc, pixels, **re_args):
        if self.use_full_fits:
            num_rows, num_cols = self.num_rows[fits_uid], self.num_cols[fits_uid]
        else:
            num_rows, num_cols = self.num_rows[fits_uid], self.num_cols[fits_uid]
            # num_rows, num_cols = self.fits_cutout_sizes[index], self.fits_cutout_sizes[index]
        cur_num_pixels = num_rows * num_cols

        cur_tile = np.array(pixels[num_pixels_acc : num_pixels_acc + cur_num_pixels]).T. \
            reshape((re_args["num_bands"], num_rows, num_cols))

        if "zoom" in re_args and re_args["zoom"] and fits_uid in re_args["cutout_fits_uids"]:
            self.restore_evaluate_zoomed_tile(cur_tile, fits_uid, **re_args)

        cur_metrics, cur_metrics_zscale = self.evaluate(index, fits_uid, cur_tile, **re_args)
        num_pixels_acc += cur_num_pixels
        return num_pixels_acc, cur_metrics, cur_metrics_zscale

    def restore_evaluate_tiles(self, pixels, **re_args):
        """ Restore original FITS/cutouts from given flattened pixels.
            Then evaluate (metric calculation) each restored FITS/cutout image.
            @Param
               pixels: flattened pixels, [npixels, nbands]
            @Return
               pixels: list of np array of size [nbands,nrows,ncols] (saved locally)
               metrics(_z): metrics for all tiles of current model [n_metrics,1,ntiles,nbands]
        """
        elem_type = type(pixels)
        if type(pixels) is list:
            pixels = torch.stack(pixels)
        if type(pixels).__module__ == "torch":
            if pixels.device != "cpu":
                pixels = pixels.detach().cpu()
            pixels = pixels.numpy()

        if re_args["calculate_metrics"]:
            metric_options = re_args["metric_options"]
            metrics = np.zeros((len(metric_options), 0, self.num_bands))
            metrics_zscale = np.zeros((len(metric_options), 0, self.num_bands))
        else: metrics, metrics_zscale = None, None

        num_pixels_acc = 0
        for index, fits_uid in enumerate(self.fits_uids):
            num_pixels_acc, cur_metrics, cur_metrics_zscale = self.restore_evaluate_one_tile(
                index, fits_uid, num_pixels_acc, pixels, **re_args)

            if re_args["calculate_metrics"]:
                metrics = np.concatenate((metrics, cur_metrics), axis=1)
                metrics_zscale = np.concatenate((metrics_zscale, cur_metrics_zscale), axis=1)

        return metrics, metrics_zscale

# FITS class ends
#################
