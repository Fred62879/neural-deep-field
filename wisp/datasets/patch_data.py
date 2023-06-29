
import torch
import numpy as np
import logging as log

from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from os.path import join, exists
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord

from wisp.utils.common import generate_hdu
from wisp.utils.plot import plot_horizontally
from wisp.utils.numerical import normalize, calculate_zscale_ranges
from wisp.datasets.data_utils import set_input_path, create_patch_uid, \
    create_patch_fname, create_selected_patches_uid, get_mgrid_np, add_dummy_dim


class PatchData:
    """ Data class for each patch. """

    def __init__(self, dataset_path, tract, patch,
                 load_pixels=False, load_coords=False, load_weights=False, load_spectra=False,
                 cutout_num_rows=None, cutout_num_cols=None, cutout_start_pos=None,
                 pixel_norm_cho=None, full_patch=True, spectra_obj=None, **kwargs
    ):
        """ @Param
               dataset_path: path of local data directory.
               tract: tract id of current patch (e.g. 9812)
               patch: patch id of current patch (e.g. 1,5)
        """
        self.kwargs = kwargs

        self.tract = tract
        self.patch = patch
        self.load_pixels = load_pixels
        self.load_coords = load_coords
        self.load_weights = load_weights
        self.load_spectra = load_spectra
        self.use_full_patch = full_patch
        self.cutout_num_rows = cutout_num_rows
        self.cutout_num_cols = cutout_num_cols
        self.cutout_start_pos = cutout_start_pos
        self.pixel_norm_cho = pixel_norm_cho
        self.spectra_obj = spectra_obj

        self.verbose = kwargs["verbose"]
        self.num_bands = kwargs["num_bands"]
        self.u_band_scale = kwargs["u_band_scale"]
        self.sensors_full_name = kwargs["sensors_full_name"]
        self.load_patch_data_cache = kwargs["load_patch_data_cache"]
        self.qtz = kwargs["quantize_latent"] or kwargs["quantize_spectra"]

        self.patch_uid = create_patch_uid(tract, patch)
        self.set_path(dataset_path)

        self.verify_patch_exists(tract, patch)
        if not self.patch_exists(): return

        self.compile_patch_fnames()
        self.load_data()

    #############
    # Initializations
    #############

    def verify_patch_exists(self, tract, patch):
        fname = create_patch_fname(tract, patch, "HSC-G")
        fname = join(self.input_patch_path, fname)
        self.patch_file_exists = exists(fname)

    def compile_patch_fnames(self):
        """ Get fnames of all given patch input for all bands.
        """
        hsc_patch_fname = np.array(
            [create_patch_fname(self.tract, self.patch, band)
            for band in self.kwargs["sensors_full_name"] if "HSC" in band])
        nb_patch_fname = np.array(
            [create_patch_fname(self.tract, self.patch, band)
            for band in self.kwargs["sensors_full_name"] if "NB" in band])
        megau_patch_fname = np.array(
            [create_patch_fname(self.tract, self.patch, band, megau=True)
            for band in self.kwargs["sensors_full_name"] if "u" in band])
        megau_weights_fname = np.array(
            [create_patch_fname(self.tract, self.patch, band, megau=True, weights=True)
            for band in self.kwargs["sensors_full_name"] if "u" in band])

        self.patch_group = np.concatenate(
            (hsc_patch_fname, nb_patch_fname, megau_patch_fname))

        self.patch_wgroup = np.concatenate(
            (hsc_patch_fname, nb_patch_fname, megau_weights_fname))

    def set_path(self, dataset_path):
        self.input_patch_path, img_data_path = set_input_path(
            dataset_path, self.kwargs["sensor_collection_name"])

        norm = self.kwargs["gt_img_norm_cho"]
        suffix = f"{norm}_{self.patch_uid}"
        if not self.use_full_patch:
            (r, c) = self.cutout_start_pos
            suffix = suffix + f"_{self.cutout_num_rows}_{self.cutout_num_cols}_{r}_{c}"
        self.gt_path = join(img_data_path, suffix)
        self.gt_img_fname = join(self.gt_path, "gt_img")
        self.gt_img_distrib_fname = join(self.gt_path, "gt_img_distrib")

        # create path
        for path in [img_data_path, self.gt_path]:
            Path(path).mkdir(parents=True, exist_ok=True)

    def load_data(self):
        self.data = {}

        self.load_header()

        if self.load_coords:
            self.get_world_coords()
            # self.get_grid_coords()

        if self.load_pixels or self.load_weights:
            self.load_patch()

        if self.load_spectra:
            self.load_spectra_data()

    #############
    # Getters
    #############

    def patch_exists(self):
        return self.patch_file_exists

    def get_patch_uid(self):
        return self.patch_uid

    def get_header(self):
        return self.header

    def get_num_rows(self):
        """ Return num of rows of current patch (full or cutout). """
        return self.cur_num_rows

    def get_num_cols(self):
        return self.cur_num_cols

    def get_gt_path(self):
        return self.gt_path

    def get_gt_img_fname(self):
        return self.gt_img_fname

    def get_zscale_ranges(self):
        return data["zscale_ranges"]

    def get_pixels(self, idx=None):
        if idx is not None:
            return self.data["pixels"][idx]
        return self.data["pixels"]

    def get_weights(self, idx=None):
        if idx is not None:
            return self.data["weights"][idx]
        return self.data["weights"]

    def get_num_coords(self):
        return self.data["coords"].shape[0]

    def get_coords(self, idx=None):
        """ Get indexed (world) coords [n,1,2] """
        if idx is not None:
            return self.data["coords"][idx]
        return self.data["coords"]

    def get_spectra_id_map(self, idx=None):
        if idx is not None:
            return self.data["spectra_id_map"][idx]
        return self.data["spectra_id_map"]

    def get_spectra_bin_map(self, idx=None):
        if idx is not None:
            return self.data["spectra_bin_map"][idx]
        return self.data["spectra_bin_map"]

    def get_spectra_pixel_ids(self):
        """ Get id of pixels with gt spectra data. """
        return self.spectra_pixel_ids

    def get_spectra_pixel_fluxes(self, idx=None):
        if idx is not None:
            return self.data["spectra_pixel_fluxes"][idx]
        return self.data["spectra_pixel_fluxes"]

    def get_spectra_pixel_redshift(self, idx=None):
        if idx is not None:
            return self.data["spectra_pixel_redshift"][idx]
        return self.data["spectra_pixel_redshift"]

    ############
    # Utilities
    ############

    def calculate_local_id(self, r, c):
        """ Count number of pixels before given position in current patch.
        """
        # return self.cur_num_cols * r + c
        if not self.use_full_patch:
            start_r, start_c = self.cutout_start_pos
            r = r - start_r
            c = c - start_c
        return self.cur_num_cols * r + c

    def calculate_neighbour_ids(self, r, c, neighbour_size):
        """ Get id of pixels within neighbour_size of given position in current patch.
            e.g. For neighbour_size being: 2, 3, 4, the collected ids:
            . .   . . .   . . . .
            . *   . * .   . . . .
                  . . .   . . * .
                          . . . .
            @Return: ids [n,]
        """
        ids = []
        offset = neighbour_size // 2
        for i in range(r - offset, r + (neighbour_size - offset)):
            for j in range(c - offset, c + (neighbour_size - offset)):
                local_id = self.calculate_local_id(i, j)
                ids.append(local_id)
        return ids

    def get_pixel_ids(self, r, c, neighbour_size=1):
        """ Get id of given position in current patch.
            If neighbour_size is > 1, also find id of neighbour pixels within neighbour_size.
            @Param: r/c, img coord in terms of the full patch
            @Return: ids [n,]
        """
        if neighbour_size <= 1:
            local_id = self.calculate_local_id(r, c)
            ids = [local_id]
        else:
            ids = self.calculate_neighbour_ids(r, c, neighbour_size)
        ids = np.array(ids)
        return ids

    ############
    # Helpers
    ############

    def load_header(self):
        """ Load header for both full patch and current cutout. """
        patch_fname = self.patch_group[0]
        id = 0 if "Mega-u" in patch_fname else 1
        hdu = fits.open(join(self.input_patch_path, patch_fname))[id]
        header = hdu.header
        self.full_num_rows, self.full_num_cols = header["NAXIS2"], header["NAXIS1"]

        if self.use_full_patch:
            self.cur_num_rows = self.full_num_rows
            self.cur_num_cols = self.full_num_cols
        else:
            (r, c) = self.cutout_start_pos
            num_rows = self.cutout_num_rows
            num_cols = self.cutout_num_cols
            pos = (c + num_cols//2, r + num_rows//2)

            wcs = WCS(header)
            cutout = Cutout2D(hdu.data, position=pos, size=(num_rows,num_cols), wcs=wcs)
            header = cutout.wcs.to_header()

            self.cur_num_rows = num_rows
            self.cur_num_cols = num_cols

        self.header = header

    def load_patch(self):
        """ Load all images (and weights) and flatten into one array.
            @Return
              pixels:  [npixels,nbands]
              weights: [npixels,nbands]
        """
        if self.verbose: log.info("Loading PATCH data.")
        pixels, weights = self.read_fits_file()

        if self.load_pixels:
            self.data["pixels"] = pixels
            self.data["zscale_ranges"] = calculate_zscale_ranges(pixels)

        if self.load_weights:
            self.data["weights"] = weights

    def filter_spectra(self, coords):
        """ Filter out coords not present in current patch (out of range).
            Useful if we only use part of the current patch.
            @Param
              coords: [n,2] img coords of all spectra pixels in current patch.
        """
        ids = np.arange(len(coords))
        if not self.use_full_patch:
            r, c = self.cutout_start_pos
            valid = (coords[:,0] >= r) & (coords[:,0] < r + self.cur_num_rows) & \
                (coords[:,1] >= c) & (coords[:,1] < c + self.cur_num_cols)
            ids = ids[valid]
        return ids

    def load_spectra_data(self):
        """ Load spectra fluxes and redshift values for all pixels with gt spectra.
        """
        path = self.spectra_obj.get_processed_spectra_path()
        cur_patch_coords_fname = join(path, f"{self.patch_uid}_coords.npy")
        cur_patch_spectra_fname = join(path, f"{self.patch_uid}_spectra.npy")
        cur_patch_redshift_fname = join(path, f"{self.patch_uid}_redshift.npy")
        coords = np.load(cur_patch_coords_fname)
        # num_spectra_full_patch = len(coords)
        # print(coords)
        spectra = np.load(cur_patch_spectra_fname) # [n,2] [wave,flux]
        redshift = np.load(cur_patch_redshift_fname)

        valid_spectra_ids = self.filter_spectra(coords)
        coords = coords[valid_spectra_ids]
        spectra = spectra[valid_spectra_ids]
        redshift = redshift[valid_spectra_ids]

        self.num_spectra = len(coords)

        if not self.use_full_patch:
            r, c = self.cutout_start_pos
        else: r, c = 0, 0

        spectra_bin_map = np.zeros((self.cur_num_rows, self.cur_num_cols)).astype(bool)
        spectra_bin_map[coords[:,0]-r,coords[:,1]-c] = 1
        spectra_bin_map = spectra_bin_map.flatten()

        ids = np.arange(self.num_spectra)
        spectra_id_map = np.full((self.cur_num_rows, self.cur_num_cols), -1).astype(int)
        spectra_id_map[coords[:,0]-r,coords[:,1]-c] = ids
        spectra_id_map = spectra_id_map.flatten()

        self.spectra_pixel_ids = self.get_pixel_ids(coords[:,0], coords[:,1])
        self.data["spectra_id_map"] = spectra_id_map
        self.data["spectra_bin_map"] = spectra_bin_map
        self.data["spectra_pixel_wave"] = spectra[:,0]
        self.data["spectra_pixel_fluxes"] = spectra[:,1]
        self.data["spectra_pixel_redshift"] = redshift

    def get_world_coords(self):
        """ Get ra/dec coords from current patch and normalize.
            pix2world calculate coords in x-y order
              coords can be indexed using r-c
            @Return
              coords: 2D coordinates [npixels,2]
        """
        xids = np.tile(np.arange(0, self.full_num_cols), self.full_num_rows)
        yids = np.repeat(np.arange(0, self.full_num_rows), self.full_num_cols)

        wcs = WCS(self.header)
        ras, decs = wcs.all_pix2world(xids, yids, 0) # x-y pixel coord
        if self.use_full_patch:
            coords = np.array([ras, decs]).T
        else:
            coords = np.concatenate((
                ras.reshape((self.full_num_rows, self.full_num_cols, 1)),
                decs.reshape((self.full_num_rows, self.full_num_cols, 1)) ), axis=2)
            (r, c) = self.cutout_start_pos # start position (r/c)
            coords = coords[r : r+self.cutout_num_rows,
                            c : c+self.cutout_num_cols].reshape(-1,2)
        self.data["coords"] = coords

    # def get_grid_coords(self):
    #     coords = get_mgrid_np(self.cur_num_rows, self.cur_num_cols)
    #     self.data["coords"] = coords

    def read_fits_file(self):
        """ Load pixel values or variance from one PATCH file (patch_id/subpatch_id).
            Load pixel and weights separately to avoid using up mem.
        """
        cur_pixels, cur_weights = [], []

        for i in range(self.num_bands):
            if self.load_pixels:
                patch_fname = self.patch_group[i]

                # u band pixel vals in first hdu, others in 2nd hdu
                is_u = "Mega-u" in patch_fname
                id = 0 if is_u else 1

                pixels = fits.open(join(self.input_patch_path, patch_fname))[id].data
                if is_u: # scale u and u* band pixel values
                    pixels /= self.u_band_scale

                if not self.use_full_patch:
                    (r, c) = self.cutout_start_pos
                    pixels = pixels[r:r+self.cutout_num_rows, c:c+self.cutout_num_cols]

                if not self.pixel_norm_cho == "linear":
                    pixels = normalize(pixels, self.pixel_norm_cho, gt=pixels)
                cur_pixels.append(pixels)

            if self.load_weights: # load weights
                patch_wfname = self.patch_wgroup[i]
                # u band weights in first hdu, others in 4th hdu
                id = 0 if "Mega-u" in patch_wfname else 3
                var = fits.open(join(self.input_patch_path, patch_wfname))[id].data

                # u band weights stored as inverse variance, others as variance
                if id == 3: weight = var
                else:       weight = 1 / (var + 1e-6) # avoid division by 0
                if self.use_full_patch:
                    cur_data.append(weight.flatten())
                else:
                    (r, c) = self.cutout_start_pos
                    var = var[r:r+self.cutout_num_rows, c:c+self.cutout_num_cols].flatten()
                    cur_weights.append(var)

        if self.load_pixels:
            cur_pixels = np.array(cur_pixels) # [nbands,sz,sz]
            np.save(self.gt_img_fname, cur_pixels)

            plot_horizontally(cur_pixels, self.gt_img_fname, "plot_img")

            if self.kwargs["to_HDU"]:
                generate_hdu(self.header, cur_pixels, self.gt_img_fname + ".fits")

            if self.kwargs["plot_img_distrib"]:
                if np.isnan(cur_pixels).any():
                    print(self.tract, self.patch, "contains nan")
                plot_horizontally(cur_pixels, self.gt_img_distrib_fname, "plot_distrib")

            # flatten pixels for ease of training [npixels,nbands]
            cur_pixels = cur_pixels.reshape(self.num_bands, -1).T

        if self.load_weights:
            cur_weights = np.sqrt(np.array(cur_weights).T) # [npixels,nbands]

        return cur_pixels, cur_weights

# PATCH class ends
#################
