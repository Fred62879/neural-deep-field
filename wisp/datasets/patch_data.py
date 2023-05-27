
import torch
import numpy as np
import logging as log

from pathlib import Path
from astropy.io import patch
from astropy.wcs import WCS
from os.path import join, exists
from astropy.nddata import Cutout2D
from wisp.utils.common import worldToPix
from astropy.coordinates import SkyCoord

from wisp.utils.common import generate_hdu
from wisp.utils.plot import plot_horizontally, mark_on_img
from wisp.datasets.data_utils import add_dummy_dim, create_uid
from wisp.utils.numerical import normalize_coords,  \
    calculate_metrics, calculate_zscale_ranges


class PatchData:
    """ Data class for each patch. """

    def __init__(self, dataset_path, tract, patch,
                 load_pixels=False, load_coords=False, load_weights=False,
                 cutout_num_rows=None, cutout_num_cols=None, cutout_start_pos=None,
                 **kwargs
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
        self.cutout_num_row = patch_cutout_num_rows
        self.cutout_num_col = patch_cutout_num_cols
        self.cutout_start_pos = patch_cutout_start_pos

        self.verbose = kwargs["verbose"]
        self.num_bands = kwargs["num_bands"]
        self.u_band_scale = kwargs["u_band_scale"]
        self.use_full_patch = kwargs["use_full_patch"]
        self.sensors_full_name = kwargs["sensors_full_name"]
        self.load_patch_data_cache = kwargs["load_patch_data_cache"]
        self.qtz = kwargs["quantize_latent"] or kwargs["quantize_spectra"]

        self.compile_patch_fnames()
        self.set_path(dataset_path)
        self.init()

    #############
    # Initializations
    #############

    def compile_patch_fnames(self):
        """ Get fnames of all given patch input for all bands.
        """
        self.patch_uid = self.tract + self.patch

        upatch = self.patch.replace(",","c")
        patch = self.patch.replace(",","2%C")

        hsc_patch_fname = np.array(
            ["calexp-" + band + "-" + tract + "-" + patch + ".patch"
            for band in self.kwargs["sensors_full_name"] if "HSC" in band])
        nb_patch_fname = np.array(
            ["calexp-" + band + "-" + tract + "-" + patch + ".patch"
            for band in self.kwargs["sensors_full_name"] if "NB" in band])
        megau_patch_fname = np.array(
            ["Mega-" + band + "_" + tract + "_" + upatch + ".patch"
            for band in self.kwargs["sensors_full_name"] if "u" in band])
        megau_weights_fname = np.array(
            ["Mega-" + band + "_" + tract + "_" + upatch + ".weight.patch"
            for band in self.kwargs["sensors_full_name"] if "u" in band])

        self.patch_group = np.concatenate(
            (hsc_patch_fname, nb_patch_fname, megau_patch_fname))

        self.patch_wgroup = np.concatenate(
            (hsc_patch_fname, nb_patch_fname, megau_weights_fname))

    def set_path(self, dataset_path):
        input_path = join(dataset_path, "input")
        self.input_patch_path = join(input_path, "input_patch")
        img_data_path = join(input_path, self.kwargs["sensor_collection_name"], "img_data")

        norm = self.kwargs["gt_img_norm_cho"]
        norm_str = self.kwargs["train_pixels_norm"]

        if self.use_full_patch:
            self.gt_path = join(img_data_path, f"{norm}_{self.patch_uid}")
        else:
            (r, c) = self.cutout_start_pos
            self.gt_path = join(
                img_data_path,
                f"{norm}_{self.patch_uid}_{self.cutout_num_rows}_{self.cutout_num_cols}_{r}_{c}"
            )

        self.gt_img_fname = join(self.gt_path, "gt_img")
        self.gt_img_distrib_fname = join(self.gt_path, "gt_img_distrib")

        # create path
        for path in [img_data_path, self.gt_path]:
            Path(path).mkdir(parents=True, exist_ok=True)

    def init(self):
        """ Load all needed data. """
        self.data = {}

        self.load_header()

        if self.load_coords:
            self.get_pixel_coords_all_patch()

        if self.load_pixels or self.load_weights:
            self.load_patch(to_tensor=self.to_tensor)

    ###############
    # Load PATCH data
    ###############

    def load_header(self):
        """ Load header for both full patch and current cutout. """
        patch_fname = self.patch_group[0]
        id = 0 if "Mega-u" in patch_fname else 1
        hdu = patch.open(join(self.input_patch_path, patch_fname))[id]
        header = hdu.header

        if self.use_full_patch:
            num_rows, num_cols = header["NAXIS2"], header["NAXIS1"]
        else:
            (r, c) = self.cutout_start_pos
            num_rows = self.cutout_num_rows
            num_cols = self.cutout_num_cols
            pos = (c + num_cols//2, r + num_rows//2)

            wcs = WCS(header)
            cutout = Cutout2D(hdu.data, position=pos, size=(num_rows,num_cols), wcs=wcs)
            header = cutout.wcs.to_header()

        self.header = header
        self.num_rows = num_rows
        self.num_cols = num_cols

    def load_patch(self, to_tensor=True, save_cutout=False):
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

    ##############
    # Load coords
    ##############

    def get_world_coords(self, id, fits_uid):
        """ Get ra/dec coords from current patch and normalize.
            pix2world calculate coords in x-y order
              coords can be indexed using r-c
            @Return
              coords: 2D coordinates [npixels,2]
        """
        num_rows, num_cols = self.num_rows, self.num_cols
        xids = np.tile(np.arange(0, self.num_cols), self.num_rows)
        yids = np.repeat(np.arange(0, self.num_rows), self.num_cols)

        wcs = WCS(self.headers[fits_uid])
        ras, decs = wcs.all_pix2world(xids, yids, 0) # x-y pixel coord
        if self.use_full_fits:
            coords = np.array([ras, decs]).T
        else:
            coords = np.concatenate((
                ras.reshape((self.num_rows, self.num_cols, 1)),
                decs.reshape((self.num_rows, self.num_cols, 1)) ), axis=2)
            (r, c) = self.start_pos[id] # start position (r/c)
            coords = coords[r:r+self.num_rows,c:c+self.num_cols].reshape(-1,2)
        self.data["coords"] = coords

    # def get_pixel_coords_all_patch(self):
    #     for id, patch_uid in enumerate(self.patch_uids):
    #         num_rows, num_cols = self.num_rows[patch_uid], self.num_cols[patch_uid]
    #         self.get_mgrid_np(num_rows, num_cols)

    # def get_mgrid_np(self, num_rows, num_cols, lo=-1, hi=1, dim=2, indexing='ij', flat=True):
    #     #def get_mgrid_np(self, sidelen, lo=-1, hi=1, dim=2, indexing='ij', flat=True):
    #     """ Generates a flattened grid of (x,y,...) coords in [-1,1] (numpy version).
    #     """
    #     x = np.linspace(lo, hi, num=num_cols)
    #     y = np.linspace(lo, hi, num=num_rows)
    #     mgrid = np.stack(np.meshgrid(x, y, indexing=indexing), axis=-1)

    #     if flat: mgrid = mgrid.reshape(-1,dim) # [sidelen**2,dim]
    #     self.data["coords"] = add_dummy_dim(mgrid, **self.kwargs)

    # def get_mgrid_tensor(self, sidelen, lo=-1, hi=1, dim=2, flat=True):
    #     """ Generates a flattened grid of (x,y,...) coords in [-1,1] (Tensor version).
    #     """
    #     tensors = tuple(dim * [torch.linspace(lo, hi, steps=sidelen)])
    #     mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    #     if flat: mgrid = mgrid.reshape(-1, dim)
    #     self.data["coords"] = add_dummy_dim(mgrid, **self.kwargs)

    #############
    # Getters
    #############

    def get_header(self):
        return self.header

    def get_num_rows(self):
        """ Return num of rows of the full patch. """
        return self.num_rows

    def get_num_cols(self):
        return self.num_cols

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
        """ Get all coords [n,1,2] """
        if idx is not None:
            return self.data["coords"][idx]
        return self.data["coords"]

    ############
    # Utilities
    ############

    def convert_from_world_coords(self, ra, dec, neighbour_size, tract, patch_id, subpatch_id):
        """ Get coordinate of pixel with given ra/dec
            @Param:
               patch_id:        index of patch where the selected spectra comes from
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
        patch_uid = tract + patch_id + subpatch_id
        patch_id = self.patch_uids.index(patch_uid)

        # ra/dec values from spectra data may not be exactly the same as real coords
        # this normalized ra/dec may thus be slightly different from real coords
        # (ra_lo, ra_hi, dec_lo, dec_hi) = np.load(self.patch_obj.coords_range_fname)
        # coord_loose = ((ra - ra_lo) / (ra_hi - ra_lo),
        #                (dec - dec_lo) / (dec_hi - dec_lo))

        # get a random header
        random_name = f"calexp-HSC-G-{tract}-{patch_id}%2C{subpatch_id}.patch"
        patch_fname = join(self.input_patch_path, random_name)
        header = patch.open(patch_fname)[1].header

        # index coord from original coords array to get accurate coord
        # this only works if spectra coords is included in the loaded coords
        (r, c) = worldToPix(header, ra, dec) # image coord, r and c coord within full patch

        if self.use_full_patch:
            img_coords = np.array([r, c, patch_id])
        else:
            start_pos = self.patch_cutout_start_pos[patch_id]
            img_coords = np.array([r - start_pos[0], c - start_pos[1], patch_id])

        pixel_ids = self.get_pixel_ids(patch_uid, r, c, neighbour_size)
        grid_coords = self.get_coord(pixel_ids)
        # print(r, c, pixel_ids, coords_accurate, self.kwargs["patch_cutout_start_pos"])
        return img_coords, grid_coords, pixel_ids

    def calculate_local_id(self, r, c, index, patch_uid):
        """ Count number of pixels before given position in given patch.
        """
        if self.use_full_patch:
            r_lo, c_lo = 0, 0
            total_cols = self.num_cols[patch_uid]
        else:
            (r_lo, c_lo) = self.patch_cutout_start_pos[index]
            total_cols = self.num_cols[patch_uid]

        local_id = total_cols * (r - r_lo) + c - c_lo
        return local_id

    def calculate_global_offset(self, patch_uid):
        """ Count total number of pixels before the given patch.
            Assume given patch_uid is included in loaded patch ids which
              is sorted in alphanumerical order.
            @Return
               id: index of given patch id inside all loaded patchs
               base_count: total # pixels before current patch
        """
        id, base_count, found = 0, 0, False

        # count total number of pixels before the given patch
        for cur_patch_uid in self.patch_uids:
            if cur_patch_uid == patch_uid: found = True; break
            if self.use_full_patch:
                base_count += self.num_rows[cur_patch_uid] * self.num_cols[cur_patch_uid]
            else:
                base_count += self.num_rows[cur_patch_uid] * self.num_cols[cur_patch_uid]
            id += 1

        assert(found)
        return id, base_count

    def calculate_neighbour_ids(self, base_count, r, c, neighbour_size, index, patch_uid):
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
                local_id = self.calculate_local_id(i, j, index, patch_uid)
                ids.append(base_count + local_id)
        return ids

    def get_pixel_ids(self, patch_uid, r, c, neighbour_size):
        """ Get global id of given position based on its
              local r/c position in given patch patch.
            If neighbour_size is > 1, also find id of neighbour pixels within neighbour_size.
        """
        index, base_count = self.calculate_global_offset(patch_uid)
        if neighbour_size <= 1:
            local_id = self.calculate_local_id(r, c, index, patch_uid)
            ids = [local_id + base_count]
        else:
            ids = self.calculate_neighbour_ids(base_count, r, c, neighbour_size, index, patch_uid)
        return ids

    def evaluate(self, index, patch_uid, recon_patch, **re_args):
        """ Image evaluation function (e.g. saving, metric calculation).
            @Param:
              patch_uid:     id of current patch patch to evaluate
              recon_patch:  restored patch patch [nbands,sz,sz]
            @Return:
              metrics(_z): metrics of current model for current patch patch, [n_metrics,1,nbands]
        """
        dir = re_args["dir"]
        fname = re_args["fname"]
        verbose = re_args["verbose"]

        #if denorm_args is not None: recon_patch *= denorm_args
        # if mask is not None: # inpaint: fill unmasked pixels with gt value
        #     recon = restore_unmasked(recon, np.copy(gt), mask)
        #     if fn is not None:
        #         np.save(fn + "_restored.npy", recon)

        if re_args["log_max"]:
            #recon_min = np.round(np.min(recon_patch, axis=(1,2)), 1)
            #recon_mean = np.round(np.mean(recon_patch, axis=(1,2)), 1)
            recon_max = np.round(np.max(recon_patch, axis=(1,2)), 1)
            # log.info(f"recon. pixel min {recon_min}")
            # log.info(f"recon. pixel mean {recon_mean}")
            log.info(f"recon. pixel max {recon_max}")

        if re_args["save_locally"]:
            np_fname = join(dir, f"{patch_uid}_{fname}.npy")
            #if restore_args["recon_norm"]: np_fname += "_norm"
            if "recon_synthetic_band" in re_args and re_args["recon_synthetic_band"]:
                np_fname += "_synthetic"
            np.save(np_fname, recon_patch)

        if re_args["to_HDU"]:
            patch_fname = join(dir, f"{patch_uid}_{fname}.patch")
            generate_hdu(class_obj.headers[patch_uid], recon_patch, patch_fname)

        if "plot_func" in re_args:
            png_fname = join(dir, f"{patch_uid}_{fname}.png")
            if re_args["zscale"]:
                zscale_ranges = self.get_zscale_ranges(patch_uid)
                re_args["plot_func"](recon_patch, png_fname, zscale_ranges=zscale_ranges)
            elif re_args["match_patch"]:
                re_args["plot_func"](recon_patch, png_fname, index)
            else:
                re_args["plot_func"](recon_patch, png_fname)

        if re_args["calculate_metrics"]:
            gt_fname = self.gt_img_fnames[patch_uid] + ".npy"
            gt_patch = np.load(gt_fname)
            gt_max = np.round(np.max(gt_patch, axis=(1,2)), 1)
            log.info(f"GT. pixel max {gt_max}")

            metrics = calculate_metrics(
                recon_patch, gt_patch, re_args["metric_options"])[:,None]
            metrics_zscale = calculate_metrics(
                recon_patch, gt_patch, re_args["metric_options"], zscale=True)[:,None]
            return metrics, metrics_zscale
        return None, None

    def restore_evaluate_zoomed_patch(self, recon_patch, patch_uid, **re_args):
        """ Crop smaller cutouts from reconstructed image.
            Helpful to evaluate local reconstruction quality when recon is large.
        """
        id = re_args["cutout_patch_uids"].index(patch_uid)
        zscale_ranges = self.get_zscale_ranges(patch_uid)

        for i, (size, (r,c)) in enumerate(
                zip(re_args["cutout_sizes"][id], re_args["cutout_start_pos"][id])
        ):
            zoomed_gt = np.load(self.gt_img_fnames[patch_uid] + ".npy")[:,r:r+size,c:c+size]
            zoomed_gt_fname = str(self.gt_img_fnames[patch_uid]) + f"_zoomed_{size}_{r}_{c}"
            plot_horizontally(zoomed_gt, zoomed_gt_fname, "plot_img")

            zoomed_recon = recon_patch[:,r:r+size,c:c+size]
            zoomed_recon_fname = join(re_args["zoomed_recon_dir"],
                                      str(re_args["zoomed_recon_fname"]) + f"_{patch_uid}_{i}")
            plot_horizontally(zoomed_recon, zoomed_recon_fname,
                              "plot_img", zscale_ranges=zscale_ranges)

    def restore_evaluate_one_patch(self, index, patch_uid, num_pixels_acc, pixels, **re_args):
        if self.use_full_patch:
            num_rows, num_cols = self.num_rows[patch_uid], self.num_cols[patch_uid]
        else:
            num_rows, num_cols = self.num_rows[patch_uid], self.num_cols[patch_uid]
            # num_rows, num_cols = self.patch_cutout_sizes[index], self.patch_cutout_sizes[index]
        cur_num_pixels = num_rows * num_cols

        cur_patch = np.array(pixels[num_pixels_acc : num_pixels_acc + cur_num_pixels]).T. \
            reshape((re_args["num_bands"], num_rows, num_cols))

        if "zoom" in re_args and re_args["zoom"] and patch_uid in re_args["cutout_patch_uids"]:
            self.restore_evaluate_zoomed_patch(cur_patch, patch_uid, **re_args)

        cur_metrics, cur_metrics_zscale = self.evaluate(index, patch_uid, cur_patch, **re_args)
        num_pixels_acc += cur_num_pixels
        return num_pixels_acc, cur_metrics, cur_metrics_zscale

    def restore_evaluate_patchs(self, pixels, **re_args):
        """ Restore original PATCH/cutouts from given flattened pixels.
            Then evaluate (metric calculation) each restored PATCH/cutout image.
            @Param
               pixels: flattened pixels, [npixels, nbands]
            @Return
               pixels: list of np array of size [nbands,nrows,ncols] (saved locally)
               metrics(_z): metrics for all patchs of current model [n_metrics,1,npatchs,nbands]
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
        for index, patch_uid in enumerate(self.patch_uids):
            num_pixels_acc, cur_metrics, cur_metrics_zscale = self.restore_evaluate_one_patch(
                index, patch_uid, num_pixels_acc, pixels, **re_args)

            if re_args["calculate_metrics"]:
                metrics = np.concatenate((metrics, cur_metrics), axis=1)
                metrics_zscale = np.concatenate((metrics_zscale, cur_metrics_zscale), axis=1)

        return metrics, metrics_zscale

    ############
    # Helpers
    ############

    def read_fits_file(self):
        """ Load pixel values or variance from one PATCH file (patch_id/subpatch_id).
            Load pixel and weights separately to avoid using up mem.
        """
        cur_pixels, cur_weights = [], []

        for i in range(self.num_bands):
            if self.load_pixels:
                patch_fname = self.patch_groups[patch_uid][i]

                # u band pixel vals in first hdu, others in 2nd hdu
                is_u = "Mega-u" in patch_fname
                id = 0 if is_u else 1

                pixels = patch.open(join(self.input_patch_path, patch_fname))[id].data
                if is_u: # scale u and u* band pixel values
                    pixels /= self.u_band_scale

                if not self.use_full_patch:
                    (r, c) = self.patch_cutout_start_pos[index] # start position (r/c)
                    num_rows = self.num_rows[patch_uid]
                    num_cols = self.num_cols[patch_uid]
                    pixels = pixels[r:r+num_rows, c:c+num_cols]

                if not self.kwargs["train_pixels_norm"] == "linear":
                    pixels = normalize(pixels, self.kwargs["train_pixels_norm"], gt=pixels)
                cur_pixels.append(pixels)

            if self.load_weights: # load weights
                patch_wfname = self.patch_wgroups[patch_uid][i]
                # u band weights in first hdu, others in 4th hdu
                id = 0 if "Mega-u" in patch_wfname else 3
                var = patch.open(join(self.input_patch_path, patch_wfname))[id].data

                # u band weights stored as inverse variance, others as variance
                if id == 3: weight = var
                else:       weight = 1 / (var + 1e-6) # avoid division by 0
                if self.use_full_patch:
                    cur_data.append(weight.flatten())
                else:
                    (r, c) = self.patch_cutout_start_pos[index] # start position (r/c)
                    num_rows = self.patch_cutout_num_rows[index]
                    num_cols = self.patch_cutout_num_cols[index]
                    var = var[r:r+num_rows, c:c+num_cols].flatten()
                    cur_weights.append(var)

        if self.load_pixels:
            # save gt np img individually for each patch file
            cur_pixels = np.array(cur_pixels) # [nbands,sz,sz]
            np.save(self.gt_img_fnames[patch_uid], cur_data)

            plot_horizontally(cur_pixels, self.gt_img_fname, "plot_img")

            if self.kwargs["to_HDU"]:
                generate_hdu(self.header, cur_pixels, self.gt_img_fname + ".patch")

            if self.kwargs["plot_img_distrib"]:
                plot_horizontally(cur_pixels, self.gt_img_distrib_fname, "plot_distrib")

            # flatten pixels for ease of training [npixels,nbands]
            cur_pixels = cur_pixels.reshape(self.num_bands, -1).T

        if self.load_weights:
            cur_weights = np.sqrt(np.array(cur_weights).T) # [npixels,nbands]

        return cur_pixels, cur_weights

# PATCH class ends
#################
