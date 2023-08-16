
import torch
import pickle
import numpy as np
import logging as log

from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from os.path import join, exists
from astropy.nddata import Cutout2D
from wisp.utils.common import worldToPix
from astropy.coordinates import SkyCoord

from wisp.utils.plot import plot_horizontally
from wisp.utils.common import generate_hdu, create_patch_uid
from wisp.utils.numerical import normalize_coords, normalize, \
    calculate_metrics, calculate_zscale_ranges_multiple_patches

from wisp.datasets.patch_data import PatchData
from wisp.datasets.data_utils import set_input_path, add_dummy_dim, \
    create_selected_patches_uid, get_coords_range_fname


class FitsData:
    """ Data class for all selected patches.
    """
    def __init__(self, device, spectra_obj=None, **kwargs):
        self.kwargs = kwargs
        self.qtz = kwargs["quantize_latent"] or kwargs["quantize_spectra"]

        if not self.require_any_data(kwargs["tasks"]): return

        self.device = device
        self.spectra_obj = spectra_obj
        self.dataset_path = kwargs["dataset_path"]
        self.verbose = kwargs["verbose"]
        self.num_bands = kwargs["num_bands"]
        self.u_band_scale = kwargs["u_band_scale"]
        self.sensors_full_name = kwargs["sensors_full_name"]
        self.load_patch_data_cache = kwargs["load_patch_data_cache"]

        self.tracts = kwargs["tracts"]
        self.patches = kwargs["patches"]
        self.use_full_patch = kwargs["use_full_patch"]
        self.patch_cutout_num_rows = kwargs["patch_cutout_num_rows"]
        self.patch_cutout_num_cols = kwargs["patch_cutout_num_cols"]
        self.patch_cutout_start_pos = kwargs["patch_cutout_start_pos"]

        self.compile_patch_fnames()
        self.set_path(self.dataset_path)
        self.load_data()

    #############
    # Initializations
    #############

    def require_any_data(self, tasks):
        """ Find all required data based on given self.tasks. """
        tasks = set(tasks)

        self.load_weights = "train" in tasks and self.kwargs["weight_train"]
        self.load_pixels = len(tasks.intersection({"train","recon_img","log_pixel_value"}))
        self.load_coords = len(tasks.intersection({"train","recon_img","recon_synthetic_band","recon_gt_spectra"})) or self.kwargs["spectra_supervision"]
        self.load_spectra = self.kwargs["pretrain_codebook"] or self.kwargs["model_redshift"]

        return self.load_pixels or self.load_coords or \
            self.load_weights or self.load_spectra or \
            "recon_codebook_spectra" in tasks

    def compile_patch_fnames(self):
        self.patch_uids = [
            create_patch_uid(tract, patch)
            for tract, patch in zip(self.kwargs["tracts"], self.kwargs["patches"])
        ]

        # sort patch fnames (so we can locate a random pixel from different patch)
        self.patch_uids.sort()

        # make sure no duplicate patch ids exist if use full tile
        if self.use_full_patch:
            assert( len(self.patch_uids) == len(set(self.patch_uids)))

    def set_path(self, dataset_path):
        _, img_data_path = set_input_path(dataset_path, self.kwargs["sensor_collection_name"])
        paths = [img_data_path]

        # suffix that defines that currently selected group of image patches
        if self.kwargs["patch_selection_cho"] is None:
            # concatenate all selected patches together
            # use only with small number of selections
            suffix = create_selected_patches_uid(self, **self.kwargs)
        else:
            suffix = "_" + self.kwargs["patch_selection_cho"]

        norm_str = self.kwargs["train_pixels_norm"]
        self.coords_fname = join(img_data_path, f"coords{suffix}.npy")
        self.coords_range_fname = get_coords_range_fname(**self.kwargs)
        self.weights_fname = join(img_data_path, f"weights{suffix}.npy")
        self.headers_fname = join(img_data_path, f"headers{suffix}.txt")
        self.meta_data_fname = join(img_data_path, f"meta_data{suffix}.txt")
        self.pixels_fname = join(img_data_path, f"pixels_{norm_str}{suffix}.npy")
        self.zscale_ranges_fname = join(img_data_path, f"zscale_ranges_{norm_str}{suffix}.npy")
        self.spectra_id_map_fname = join(img_data_path, f"spectra_id_map_{norm_str}{suffix}.npy")
        self.spectra_bin_map_fname = join(
            img_data_path, f"spectra_bin_map_{norm_str}{suffix}.npy")
        self.spectra_pixel_fluxes_fname = join(
            img_data_path, f"spectra_pixel_fluxes_{norm_str}{suffix}.npy")
        self.spectra_pixel_redshift_fname = join(
            img_data_path, f"spectra_pixel_redshift_{norm_str}{suffix}.npy")

        # create path
        for path in paths:
            Path(path).mkdir(parents=True, exist_ok=True)

    def load_data(self):
        self.data = {}

        cached = self.load_patch_data_cache and \
            exists(self.headers_fname) and exists(self.meta_data_fname) and \
            (not self.load_weights or exists(self.weights_fname)) and \
            (not self.load_pixels or (exists(self.pixels_fname) and \
                                      exists(self.zscale_ranges_fname))) and \
            (not self.load_coords or (exists(self.coords_fname) and \
                                      exists(self.coords_range_fname))) and \
            (not self.load_spectra or (exists(self.spectra_id_map_fname) and \
                                       exists(self.spectra_bin_map_fname) and \
                                       exists(self.spectra_pixel_fluxes_fname) and \
                                       exists(self.spectra_pixel_redshift_fname)))

        if cached: pixels, coords, coords_range, weights, spectra_data  = self.load_cache()
        else:      pixels, coords, coords_range, weights, spectra_data = self.process_data()

        if self.load_pixels:
            pixel_max = np.round(np.max(pixels, axis=0), 3)
            pixel_min = np.round(np.min(pixels, axis=0), 3)
            log.info(f"train pixels max {pixel_max}")
            log.info(f"train pixels min {pixel_min}")
            self.data["pixels"] = torch.FloatTensor(pixels)

        if self.load_coords:
            coords = add_dummy_dim(coords, **self.kwargs)[:,None]
            self.data["coords"] = torch.FloatTensor(coords)
            self.data["coords_range"] = coords_range

        if self.load_weights:
            self.data["weights"] = torch.FloatTensor(weights)

        if self.load_spectra:
            spectra_id_map, spectra_bin_map, spectra_pixel_fluxes, \
                spectra_pixel_redshift = spectra_data
            self.data["spectra_id_map"] = spectra_id_map
            self.data["spectra_bin_map"] = spectra_bin_map
            self.data["spectra_pixel_fluxes"] = spectra_pixel_fluxes
            self.data["spectra_pixel_redshift"] = spectra_pixel_redshift

    #############
    # Getters
    #############

    def get_patch_uids(self):
        return self.patch_uids

    def get_num_rows(self):
        return self.num_rows

    def get_num_cols(self):
        return self.num_cols

    def get_gt_paths(self):
        return self.gt_paths

    def get_gt_img_fnames(self):
        return self.gt_img_fnames

    def get_num_coords(self):
        return self.data["coords"].shape[0]

    def get_pixels(self, idx=None):
        if idx is not None:
            return self.data["pixels"][idx]
        return self.data["pixels"]

    def get_coords(self, idx=None):
        """ Get all coords [n,1,2] """
        if idx is not None:
            return self.data["coords"][idx]
        return self.data["coords"]

    def get_coords_range(self):
        return self.data["coords_range"]

    def get_weights(self, idx=None):
        if idx is not None:
            return self.data["weights"][idx]
        return self.data["weights"]

    def get_spectra_id_map(self, idx=None):
        if idx is not None:
            return self.data["spectra_id_map"][idx]
        return self.data["spectra_id_map"]

    def get_spectra_bin_map(self, idx=None):
        if idx is not None:
            return self.data["spectra_bin_map"][idx]
        return self.data["spectra_bin_map"]

    def get_spectra_pixel_fluxes(self, idx=None):
        if idx is not None:
            return self.data["spectra_pixel_fluxes"][idx]
        return self.data["spectra_pixel_fluxes"]

    def get_spectra_pixel_redshift(self, idx=None):
        if idx is not None:
            return self.data["spectra_pixel_redshift"][idx]
        return self.data["spectra_pixel_redshift"]

    def get_zscale_ranges(self, patch_uid=None):
        zscale_ranges = np.load(self.zscale_ranges_fname)
        if patch_uid is not None:
            id = self.patch_uids.index(patch_uid)
            zscale_ranges = zscale_ranges[id]
        return zscale_ranges

    ############
    # Utilities
    ############

    # def world2pix(self, ra, dec, neighbour_size, tract, patch):
    #     """ Convert from world coordinate to pixel coordinate.
    #         @Param:
    #            ra/dec:         world coordinate
    #            neighbour_size: size of neighbouring area to average for spectra

    #         We can either
    #          i) directly normalize the given ra/dec with coordinates range or
    #         ii) get pixel id based on ra/dec and index from loaded coordinates.

    #         Since ra/dec values from spectra data may not be as precise as
    #           real coordinates calculated from wcs package, method i) tends to
    #           be inaccurate and method ii) is more reliable.
    #         However, if given ra/dec is not within loaded coordinates, we
    #           can only use method i)
    #     """
    #     patch_uid = create_patch_uid(tract, patch)
    #     patch_id = self.patch_uids.index(patch_uid)

    #     # ra/dec values from spectra data may not be exactly the same as real coords
    #     # this normalized ra/dec may thus be slightly different from real coords
    #     # (ra_lo, ra_hi, dec_lo, dec_hi) = np.load(self.patch_obj.coords_range_fname)
    #     # coord_loose = ((ra - ra_lo) / (ra_hi - ra_lo),
    #     #                (dec - dec_lo) / (dec_hi - dec_lo))

    #     header = self.headers[patch_uid]

    #     # index coord from original coords array to get accurate coord
    #     # this only works if spectra coords is included in the loaded coords
    #     (r, c) = worldToPix(header, ra, dec) # image coord, r and c coord within full tile

    #     img_coords = np.array([r, c, patch_id])
    #     # if self.use_full_patch:
    #     #     img_coords = np.array([r, c, patch_id])
    #     # else:
    #     #     start_pos = self.patch_cutout_start_pos[patch_id]
    #     #     img_coords = np.array([r - start_pos[0], c - start_pos[1], patch_id])

    #     pixel_ids = self.get_pixel_ids(patch_uid, r, c, neighbour_size)
    #     grid_coords = self.get_coords(pixel_ids)
    #     # print(r, c, pixel_ids, coords_accurate, self.kwargs["patch_cutout_start_pos"])
    #     return img_coords, grid_coords, pixel_ids

    def calculate_local_id(self, r, c, index, patch_uid):
        """ Count number of pixels before given position in given patch.
        """
        total_cols = self.num_cols[patch_uid]
        local_id = total_cols * r + c
        return local_id

    def calculate_global_offset(self, patch_uid):
        """ Count total number of pixels before the given patch.
            Assume given patch_uid is included in loaded patch ids which
              is sorted in alphanumerical order.
            @Return
               id: index of given patch id inside all loaded tiles
               base_count: total # pixels before current tile
        """
        id, base_count, found = 0, 0, False

        # count total number of pixels before the given tile
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
              local r/c position in given patch tile.
            If neighbour_size is > 1, also find id of neighbour pixels within neighbour_size.
        """
        index, base_count = self.calculate_global_offset(patch_uid)
        if neighbour_size <= 1:
            local_id = self.calculate_local_id(r, c, index, patch_uid)
            ids = [local_id + base_count]
        else:
            ids = self.calculate_neighbour_ids(base_count, r, c, neighbour_size, index, patch_uid)
        return ids

    def calculate_local_id_one_patch(self, r, c, start_pos, num_cols):
        """ Count number of pixels before given position in one given patch.
        """
        if not self.use_full_patch:
            start_r, start_c = start_pos
            r = r - start_r
            c = c - start_c
        return num_cols * r + c

    def calculate_neighbour_ids_one_patch(self, r, c, neighbour_size, start_pos, num_cols):
        """ Get id of pixels within neighbour_size of given position in one given patch.
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
                local_id = self.calculate_local_id_one_patch(
                    i, j, start_pos, num_cols, use_full_patch)
                ids.append(local_id)
        return ids

    def get_pixel_ids_one_patch(self, r, c, neighbour_size=1):
        """ Get id of given position in current patch.
            If neighbour_size is > 1, also find id of neighbour pixels within neighbour_size.
            @Param: r/c, img coord in terms of the full patch
            @Return: ids [n,]
        """
        start_pos = self.cutout_start_pos[patch_uid]
        num_cols = self.num_cols[patch_uid]

        if neighbour_size <= 1:
            local_id = self.calculate_local_id_one_patch(r, c, cutout_start_pos, num_cols)
            ids = [local_id]
        else:
            ids = self.calculate_neighbour_ids_one_patch(
                r, c, neighbour_size, cutout_start_pos, num_cols)
        ids = np.array(ids)
        return ids

    def evaluate(self, index, patch_uid, recon_patch, **re_args):
        """ Image evaluation function (e.g. saving, metric calculation).
            @Param:
              patch_uid:     id of current patch tile to evaluate
              recon_patch:  restored patch tile [nbands,sz,sz]
            @Return:
              metrics(_z): metrics of current model for current patch tile, [n_metrics,1,nbands]
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
            recon_min = np.round(np.min(recon_patch, axis=(1,2)), 3)
            recon_mean = np.round(np.mean(recon_patch, axis=(1,2)), 3)
            recon_max = np.round(np.max(recon_patch, axis=(1,2)), 1)
            log.info(f"recon. pixel min {recon_min}")
            log.info(f"recon. pixel mean {recon_mean}")
            log.info(f"recon. pixel max {recon_max}")

        if re_args["save_locally"]:
            np_fname = join(dir, f"{patch_uid}_{fname}.npy")
            #if restore_args["recon_norm"]: np_fname += "_norm"
            if "recon_synthetic_band" in re_args and re_args["recon_synthetic_band"]:
                np_fname += "_synthetic"
            np.save(np_fname, recon_patch)

        if re_args["to_HDU"]:
            # patch_fname = join(dir, f"{patch_uid}_{fname}.patch")
            # generate_hdu(self.headers[patch_uid], recon_patch, patch_fname)
            # requires proper saving of header
            raise NotImplementedError("hdu image generation currently not supported")

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
            gt_tile = np.load(gt_fname)
            gt_min = np.round(np.min(gt_tile, axis=(1,2)), 3)
            gt_mean = np.round(np.mean(gt_tile, axis=(1,2)), 3)
            gt_max = np.round(np.max(gt_tile, axis=(1,2)), 1)
            log.info(f"GT. pixel min {gt_min}")
            log.info(f"GT. pixel mean {gt_mean}")
            log.info(f"GT. pixel max {gt_max}")

            metrics = calculate_metrics(
                recon_patch, gt_tile, re_args["metric_options"])[:,None]
            metrics_zscale = calculate_metrics(
                recon_patch, gt_tile, re_args["metric_options"], zscale=True)[:,None]
            return metrics, metrics_zscale
        return None, None

    def restore_evaluate_zoomed_tile(self, recon_patch, patch_uid, **re_args):
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

    def restore_evaluate_one_tile(self, index, patch_uid, num_pixels_acc, pixels, **re_args):
        num_rows, num_cols = self.num_rows[patch_uid], self.num_cols[patch_uid]
        cur_num_pixels = num_rows * num_cols

        cur_patch = np.array(pixels[num_pixels_acc : num_pixels_acc + cur_num_pixels]).T. \
            reshape((re_args["num_bands"], num_rows, num_cols))

        if "zoom" in re_args and re_args["zoom"] and patch_uid in re_args["cutout_patch_uids"]:
            self.restore_evaluate_zoomed_tile(cur_patch, patch_uid, **re_args)

        cur_metrics, cur_metrics_zscale = self.evaluate(index, patch_uid, cur_patch, **re_args)
        num_pixels_acc += cur_num_pixels
        return num_pixels_acc, cur_metrics, cur_metrics_zscale

    def restore_evaluate_tiles(self, pixels, **re_args):
        """ Restore original PATCH/cutouts from given flattened pixels.
            Then evaluate (metric calculation) each restored PATCH/cutout image.
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
        for index, patch_uid in enumerate(self.patch_uids):
            num_pixels_acc, cur_metrics, cur_metrics_zscale = self.restore_evaluate_one_tile(
                index, patch_uid, num_pixels_acc, pixels, **re_args)

            if re_args["calculate_metrics"]:
                metrics = np.concatenate((metrics, cur_metrics), axis=1)
                metrics_zscale = np.concatenate((metrics_zscale, cur_metrics_zscale), axis=1)

        return metrics, metrics_zscale

    ##############
    # Helpers
    ##############

    def load_cache(self):
        if self.verbose: log.info("PATCH data cached.")
        pixels, coords, weights, spectra_data = [None]*4

        # with open(self.headers_fname, "rb") as fp:
        #     headers = pickle.load(fp)
        with open(self.meta_data_fname, "rb") as fp:
            meta_data = pickle.load(fp)
        (self.num_rows, self.num_cols, self.gt_paths, self.gt_img_fnames) = meta_data

        if self.load_pixels:
            pixels = np.load(self.pixels_fname)
            zscale_ranges = np.load(self.zscale_ranges_fname)
        if self.load_coords:
            coords = np.load(self.coords_fname)
            coords_range = np.load(self.coords_range_fname)
            coords, _ = normalize_coords(coords, coords_range=coords_range)
        if self.load_weights:
            weights = np.load(self.weights_fname)
        if self.load_spectra:
            spectra_id_map = np.load(self.spectra_id_map_fname)
            spectra_bin_map = np.load(self.spectra_bin_map_fname)
            spectra_pixel_fluxes = np.load(self.spectra_pixel_fluxes_fname)
            spectra_pixel_redshift = np.load(self.spectra_pixel_redshift_fname)
            spectra_data = (spectra_id_map, spectra_bin_map,
                            spectra_pixel_fluxes, spectra_pixel_redshift)
        return pixels, coords, coords_range, weights, spectra_data

    def process_data(self):
        pixels, coords, weights, spectra_data = [], [], [], []
        spectra_id_map, spectra_bin_map = [], []
        spectra_pixel_fluxes, spectra_pixel_redshift = [], []
        self.gt_paths, self.gt_img_fnames = {}, {}
        self.headers, self.num_rows, self.num_cols = {}, {}, {}

        for tract, patch, cutout_num_rows, cutout_num_cols, cutout_start_pos in zip(
                self.tracts, self.patches, self.patch_cutout_num_rows,
                self.patch_cutout_num_cols, self.patch_cutout_start_pos
        ):
            self.load_one_patch(
                pixels, coords, weights, spectra_id_map,
                spectra_bin_map, spectra_pixel_fluxes, spectra_pixel_redshift,
                tract, patch, cutout_num_rows, cutout_num_cols, cutout_start_pos)

        meta_data = (self.num_rows, self.num_cols, self.gt_paths, self.gt_img_fnames)

        with open(self.headers_fname, "wb") as fp:
            pickle.dump(self.headers, fp)
        with open(self.meta_data_fname, "wb") as fp:
            pickle.dump(meta_data, fp)

        if self.load_pixels:
            # apply normalization to pixels as specified
            if self.kwargs["train_pixels_norm"] == "linear":
                pixels = normalize(pixels, "linear")
            elif self.kwargs["train_pixels_norm"] == "zscale":
                pixels = normalize(pixels, "zscale", gt=pixels)

            zscale_ranges = calculate_zscale_ranges_multiple_patches(pixels)

            pixels = np.concatenate(pixels)
            np.save(self.pixels_fname, pixels)
            np.save(self.zscale_ranges_fname, zscale_ranges)

        if self.load_coords:
            coords = np.concatenate(coords)
            np.save(self.coords_fname, coords) # save un-normed coords
            coords, coords_range = normalize_coords(coords)
            np.save(self.coords_range_fname, coords_range)

        if self.load_weights:
            weights = np.concatenate(weights)
            np.save(self.weights_fname, weights)

        if self.load_spectra:
            spectra_id_map = np.concatenate(spectra_id_map)
            spectra_bin_map = np.concatenate(spectra_bin_map)
            spectra_pixel_fluxes = np.concatenate(spectra_pixel_fluxes)
            spectra_pixel_redshift = np.concatenate(spectra_pixel_redshift)

            np.save(self.spectra_id_map_fname, spectra_id_map)
            np.save(self.spectra_bin_map_fname, spectra_bin_map)
            np.save(self.spectra_pixel_fluxes_fname, spectra_pixel_fluxes)
            np.save(self.spectra_pixel_redshift_fname, spectra_pixel_redshift)
            spectra_data = (spectra_id_map, spectra_bin_map,
                            spectra_pixel_fluxes, spectra_pixel_redshift)

        return pixels, coords, coords_range, weights, spectra_data

    def load_one_patch(self, pixels, coords, weights, spectra_id_map,
                       spectra_bin_map, spectra_pixel_fluxes, spectra_pixel_redshift,
                       tract, patch, cutout_num_rows, cutout_num_cols, cutout_start_pos
    ):
        cur_patch = PatchData(
            tract, patch,
            load_pixels=self.load_pixels,
            load_coords=self.load_coords,
            load_weights=self.load_weights,
            load_spectra=self.load_spectra,
            mark_spectra_on_patch=self.kwargs["mark_spectra"],
            cutout_num_rows=cutout_num_rows,
            cutout_num_cols=cutout_num_cols,
            cutout_start_pos=cutout_start_pos,
            pixel_norm_cho=self.kwargs["train_pixels_norm"],
            full_patch=self.kwargs["use_full_patch"],
            spectra_obj=self.spectra_obj,
            **self.kwargs)

        patch_uid = cur_patch.get_patch_uid()
        self.headers[patch_uid] = cur_patch.get_header()
        self.gt_paths[patch_uid] = cur_patch.get_gt_path()
        self.num_rows[patch_uid] = cur_patch.get_num_rows()
        self.num_cols[patch_uid] = cur_patch.get_num_cols()
        self.gt_img_fnames[patch_uid] = cur_patch.get_gt_img_fname()

        if self.load_pixels:
            pixels.append(cur_patch.get_pixels())
        if self.load_coords:
            coords.append(cur_patch.get_coords())
        if self.load_weights:
            weights.append(cur_patch.get_weights())
        if self.load_spectra:
            spectra_id_map.append(cur_patch.get_spectra_id_map())
            spectra_bin_map.append(cur_patch.get_spectra_bin_map())
            spectra_pixel_fluxes.append(cur_patch.get_spectra_pixel_fluxes())
            spectra_pixel_redshift.append(cur_patch.get_spectra_pixel_redshift())

# PATCH class ends
#################
