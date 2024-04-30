
import csv
import copy
import time
import torch
import pandas
import pickle
import requests
import numpy as np
import logging as log
import matplotlib.pyplot as plt

from wisp.datasets.patch_data import PatchData
from wisp.utils.plot import plot_spectra
from wisp.utils.common import create_patch_uid, to_numpy, segment_bool_array
from wisp.utils.numerical import normalize_coords, calculate_metrics
from wisp.datasets.data_utils import set_input_path, patch_exists, \
    get_bound_id, clip_data_to_ref_wave_range, get_wave_range_fname, \
    get_coords_norm_range_fname, add_dummy_dim, wave_within_bound, \
    get_dataset_path, get_img_data_path

from tqdm import tqdm
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from os.path import join, exists
from scipy.signal import resample
from collections import defaultdict
from functools import partial, reduce
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from linetools.lists.linelist import LineList
from astropy.convolution import convolve, Gaussian1DKernel
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool


class SpectraData:
    def __init__(self, trans_obj, device, **kwargs):
        self.kwargs = kwargs
        if kwargs["space_dim"] != 3: return

        self.device = device
        self.trans_obj = trans_obj

        self.num_bands = kwargs["num_bands"]
        self.space_dim = kwargs["space_dim"]
        self.all_tracts = kwargs["spectra_tracts"]
        self.all_patches_r = kwargs["spectra_patches_r"]
        self.all_patches_c = kwargs["spectra_patches_c"]
        self.num_tracts = len(self.all_tracts)
        self.num_patches_c = len(self.all_patches_c)
        self.num_patches = len(self.all_patches_r)*len(self.all_patches_c)

        self.spectra_data_sources = kwargs["spectra_data_sources"]
        self.spectra_process_patch_info = self.kwargs["spectra_process_patch_info"]

        self.deimos_source_spectra_link = kwargs["deimos_source_spectra_link"]
        self.deimos_spectra_data_format = kwargs["deimos_spectra_data_format"]
        self.download_deimos_source_spectra = kwargs["download_deimos_source_spectra"]

        self.zcosmos_source_spectra_link = kwargs["zcosmos_source_spectra_link"]
        self.zcosmos_spectra_data_format = kwargs["zcosmos_spectra_data_format"]
        self.download_zcosmos_source_spectra = kwargs["download_zcosmos_source_spectra"]

        self.deep2_spectra_data_format = kwargs["deep2_spectra_data_format"]
        self.deep3_spectra_data_format = kwargs["deep3_spectra_data_format"]

        self.load_spectra_data_from_cache = kwargs["load_spectra_data_from_cache"]

        self.spectra_smooth_sigma = kwargs["spectra_smooth_sigma"]
        self.spectra_neighbour_size = kwargs["spectra_neighbour_size"]
        self.wave_discretz_interval = kwargs["trans_sample_interval"]
        self.trans_range = self.trans_obj.get_wave_range()
        self.process_ivar = True #kwargs["process_ivar"]
        self.convolve_spectra = kwargs["convolve_spectra"]
        self.average_neighbour_spectra = kwargs["average_neighbour_spectra"]
        self.learn_spectra_within_wave_range = kwargs["learn_spectra_within_wave_range"]

        self.supervision_wave_range = None if not self.learn_spectra_within_wave_range \
            else [kwargs["spectra_supervision_wave_lo"],
                  kwargs["spectra_supervision_wave_hi"]]

        self.spectra_config = self.get_spectra_config()
        self.dataset_path = get_dataset_path(**kwargs)
        self.set_path(self.dataset_path)

        self.load_accessory_data()
        self.load_spectra()

    def get_spectra_config(self):
        config = ""
        if self.convolve_spectra:
            config += f"_convolved_sigma_{self.spectra_smooth_sigma}"
        if self.average_neighbour_spectra:
            config += f"_average_{self.spectra_neighbour_size}_neighbours"
        if self.learn_spectra_within_wave_range:
            lo, hi = self.supervision_wave_range
            config += f"_wave_range_{lo}_to_{hi}"
        return config

    def set_path(self, dataset_path):
        """
        Create path and filename of required files.
        """
        spectra_path = join(dataset_path, "input/spectra")
        sources = sorted(self.kwargs["spectra_data_sources"])
        sources = reduce(lambda x, y: x + "_" + y, sources)
        cache_path_name = "processed_{}_{}{}_{}_spectra".format(
            self.kwargs["spectra_cho"], sources, self.spectra_config,
            self.kwargs["num_gt_spectra_upper_bound"])
        cache_path = join(spectra_path, "cache", cache_path_name)
        self.cache_path = cache_path
        Path(cache_path).mkdir(parents=True, exist_ok=True)

        self.wave_range_fname = get_wave_range_fname(**self.kwargs)
        self.coords_norm_range_fname = get_coords_norm_range_fname(**self.kwargs)
        self.input_patch_path, img_data_path = get_img_data_path(dataset_path, **self.kwargs)

        self.test_id_fname = join(self.cache_path, "test_spectra_ids.npy")
        self.validation_id_fname = join(self.cache_path, "validation_spectra_ids.npy")
        self.supervision_id_fname = join(self.cache_path, "supervision_spectra_ids.npy")

        self.emitted_wave_fname = join(cache_path, "emitted_wave.npy")
        self.emitted_wave_mask_fname = join(cache_path, "emitted_wave_mask.npy")
        self.cache_metadata_table_fname =  join(cache_path, "processed_table.tbl")

        self.gt_spectra_fname = join(cache_path, "gt_spectra.npy")
        self.gt_spectra_mask_fname = join(cache_path, "gt_spectra_mask.npy")
        self.gt_spectra_redshift_fname = join(cache_path, "gt_spectra_redshift.npy")
        self.gt_spectra_sup_bound_fname = join(cache_path, "gt_spectra_sup_bound.npy")

        if self.spectra_process_patch_info:
            self.gt_spectra_ids_fname = join(cache_path, "gt_spectra_ids.txt")
            self.gt_spectra_pixels_fname = join(cache_path, "gt_spectra_pixels.npy")
            self.gt_spectra_img_coords_fname = join(cache_path, f"gt_spectra_img_coords.npy")
            self.gt_spectra_world_coords_fname = join(
                cache_path, f"gt_spectra_world_coords.npy")

        if "deimos" in self.spectra_data_sources:
            self._set_path(
                "deimos", spectra_path,
                self.kwargs["deimos_processed_spectra_cho"],
                self.kwargs["deimos_source_spectra_fname"],
                self.deimos_spectra_data_format)

        if "zcosmos" in self.spectra_data_sources:
            self._set_path(
                "zcosmos", spectra_path,
                self.kwargs["zcosmos_processed_spectra_cho"],
                self.kwargs["zcosmos_source_spectra_fname"],
                self.zcosmos_spectra_data_format)

        if "deep2" in self.spectra_data_sources:
            self._set_path(
                "deep2", spectra_path,
                self.kwargs["deep2_processed_spectra_cho"],
                self.kwargs["deep2_source_spectra_fname"],
                self.deep2_spectra_data_format)

        if "deep3" in self.spectra_data_sources:
            self._set_path(
                "deep3", spectra_path,
                self.kwargs["deep3_processed_spectra_cho"],
                self.kwargs["deep3_source_spectra_fname"],
                self.deep3_spectra_data_format)

    def _set_path(self, data_source, spectra_path, spectra_cho,
                  source_spectra_fname, data_format
    ):
        """
        Create path and filename for the specified spectra data source.
        """
        path = join(spectra_path, data_source)
        processed_data_path = join(path, f"processed_{spectra_cho}{self.spectra_config}")

        setattr(self, f"{data_source}_emitted_wave_fname",
                join(processed_data_path, "emitted_wave"))
        setattr(self, f"{data_source}_emitted_wave_mask_fname",
                join(processed_data_path, "emitted_wave_mask"))

        setattr(self, f"{data_source}_source_spectra_path",
                join(path, f"source_spectra_{data_format}"))
        setattr(self, f"{data_source}_source_metadata_table_fname",
                join(path, source_spectra_fname))

        setattr(self, f"{data_source}_processed_spectra_path",
                join(processed_data_path, "processed_spectra"))
        setattr(self, f"{data_source}_processed_metadata_table_fname",
                join(processed_data_path, f"processed_{data_source}_table.tbl"))

        for path in [getattr(self, f"{data_source}_source_spectra_path"),
                     getattr(self, f"{data_source}_processed_spectra_path")]:
            Path(path).mkdir(parents=True, exist_ok=True)

    def load_accessory_data(self):
        self.full_wave = self.trans_obj.get_full_wave()

    def load_spectra(self):
        """ Load gt and/or dummy spectra data.
        """
        self.data = defaultdict(lambda: [])
        self.load_gt_spectra_data()
        self.set_wave_range()
        self.transform_data()

    def finalize_spectra(self):
        """ Finalize spectra data processing.
            Call this function after deriving coords norm range.
        """
        if self.kwargs["space_dim"] != 3: return
        if self.spectra_process_patch_info:
            self.process_coords()
        self.split_spectra()

    #############
    # Getters
    #############

    def get_supervision_spectra_ids(self):
        return self.sup_ids

    def get_redshift_pretrain_spectra_ids(self):
        return self.redshift_pretrain_ids

    def get_emitted_wave(self):
        """ Get range of emitted wave.
        """
        return self.data["emitted_wave"][0]

    def get_emitted_wave_mask(self):
        """ Get mask for full (used for codebook spectra plotting).
        """
        return self.data["emitted_wave_mask"]

    def get_processed_spectra_path(self):
        return self.processed_spectra_path

    def get_full_wave(self):
        return self.full_wave

    def get_num_gt_spectra(self):
        """ Get #gt spectra (doesn't count neighbours). """
        return self.num_gt_spectra

    def get_num_supervision_spectra(self):
        """ Get #supervision spectra (doesn't count neighbours). """
        return self.num_supervision_spectra

    def get_num_validation_spectra(self):
        """ Get #validation spectra (doesn't count neighbours). """
        return self.num_validation_spectra

    def get_num_test_spectra(self):
        """ Get #test spectra (doesn't count neighbours). """
        return self.num_test_spectra


    def get_test_mask(self, idx=None):
        """ Get test spectra mask for plotting. """
        if idx is None:
            return self.data["test_mask"]
        return self.data["test_mask"][idx]

    def get_test_sup_bound(self, idx=None):
        """ Get test wave bound ids. """
        if idx is None:
            return self.data["test_sup_bound"]
        return self.data["test_sup_bound"][idx]


    def get_supervision_mask(self, idx=None):
        """ Get supervision spectra mask for plotting. """
        if idx is None:
            return self.data["supervision_mask"]
        return self.data["supervision_mask"][idx]

    def get_supervision_spectra(self, idx=None):
        """ Get gt spectra (with same wave range as recon) used for supervision. """
        if idx is None:
            return self.data["supervision_spectra"]
        return self.data["supervision_spectra"][idx]

    def get_supervision_sup_bound(self, idx=None):
        """ Get supervision wave bound ids. """
        if idx is None:
            return self.data["supervision_sup_bound"]
        return self.data["supervision_sup_bound"][idx]

    def get_supervision_pixels(self, idx=None):
        """ Get pix values for pixels used for spectra supervision. """
        if idx is None:
            return self.data["supervision_pixels"]
        return self.data["supervision_pixels"][idx]

    def get_supervision_redshift(self, idx=None):
        """ Get redshift values for pixels used for spectra supervision. """
        if idx is None:
            return self.data["supervision_redshift"]
        return self.data["supervision_redshift"][idx]


    # def get_validation_spectra_ids(self, patch_uid=None):
    #     """ Get id of validation spectra in given patch.
    #         Id here is in context of all validation spectra
    #     """
    #     if patch_uid is not None:
    #         return self.data["validation_patch_ids"][patch_uid]
    #     return self.data["validation_patch_ids"]

    def get_validation_spectra(self, idx=None):
        if idx is not None:
            return self.data["validation_spectra"][idx]
        return self.data["validation_spectra"]

    def get_validation_pixels(self, idx=None):
        if idx is not None:
            return self.data["validation_pixels"][idx]
        return self.data["validation_pixels"]

    def get_validation_coords(self, idx=None):
        if idx is not None:
            return self.data["validation_coords"][idx]
        return self.data["validation_coords"]

    def get_validation_mask(self, idx=None):
        """ Get validation spectra mask for plotting. """
        if idx is None:
            return self.data["validation_mask"]
        return self.data["validation_mask"][idx]

    def get_validation_sup_bound(self, idx=None):
        """ Get validation wave bound ids. """
        if idx is None:
            return self.data["validation_sup_bound"]
        return self.data["validation_sup_bound"][idx]

    def get_validation_redshift(self, idx=None):
        if idx is not None:
            return self.data["semi_supervision_redshift"][idx]
        return self.data["semi_supervision_redshift"]


    def get_test_spectra(self, idx=None):
        """ Get gt spectra (with same wave range as recon) used for test. """
        if idx is None:
            return self.data["test_spectra"]
        return self.data["test_spectra"][idx]

    def get_test_coords(self, idx=None):
        if idx is not None:
            return self.data["test_coords"][idx]
        return self.data["test_coords"]

    def get_test_pixels(self, idx=None):
        if idx is not None:
            return self.data["test_pixels"][idx]
        return self.data["test_pixels"]

    def get_test_redshift(self, idx=None):
        if idx is not None:
            return self.data["test_redshift"][idx]
        return self.data["test_redshift"]

    #############
    # Helpers
    #############

    def process_coords(self):
        if self.kwargs["coords_type"] == "img":
            coords = self.data["gt_spectra_img_coords"]
        elif self.kwargs["coords_type"] == "world":
            coords = self.data["gt_spectra_world_coords"]
        else: raise NotImplementedError

        if self.kwargs["normalize_coords"]:
            assert exists(self.coords_norm_range_fname)
            norm_range = np.load(self.coords_norm_range_fname)
            coords, _ = normalize_coords(coords, norm_range=norm_range, **self.kwargs)

        if self.kwargs["coords_encode_method"] == "grid" and \
           self.kwargs["grid_type"] == "HashGrid" and self.kwargs["grid_dim"] == 3:
            coords = add_dummy_dim(coords, **self.kwargs)

        self.data["gt_spectra_coords"] = coords # [n,n_neighbr,2/3]

    def transform_data(self):
        self.to_tensor([
            "emitted_wave",
            "gt_spectra",
            "gt_spectra_pixels",
            "gt_spectra_redshift",
            "gt_spectra_sup_bound",
            "gt_spectra_img_coords",
            "gt_spectra_world_coords",
        ], torch.float32)
        self.to_tensor([
            "gt_spectra_mask",
            "emitted_wave_mask",
        ], torch.bool)

    def split_spectra(self):
        """
        Split spectra either
        Randomly (when doing spectra pretrain, sanity check, and generalization) or
        Patch-wise (when doing main train)
        """
        if self.spectra_process_patch_info:
            test_ids, val_ids, sup_ids = self.split_spectra_patch_wise()
        else: test_ids, val_ids, sup_ids = self.split_spectra_all_together()

        self.sup_ids = sup_ids
        self.val_ids = val_ids
        self.test_ids = test_ids
        self.num_test_spectra = len(test_ids)
        self.num_validation_spectra = len(val_ids)
        self.num_supervision_spectra = len(sup_ids)

        # log.info(f"test spectra ids: {test_ids}")
        # log.info(f"val spectra ids: {val_ids}")
        # log.info(f"sup spectra ids: {sup_ids}")
        log.info(f"spectra train/valid/test: {len(sup_ids)}/{len(val_ids)}/{len(test_ids)}")

        # supervision spectra data (used during pretrain)
        self.data["supervision_spectra"] = self.data["gt_spectra"][sup_ids]
        self.data["supervision_mask"] = self.data["gt_spectra_mask"][sup_ids]
        self.data["supervision_redshift"] = self.data["gt_spectra_redshift"][sup_ids]
        self.data["supervision_sup_bound"] = self.data["gt_spectra_sup_bound"][sup_ids]

        # valiation(and semi sup) spectra data (used during main train)
        self.data["validation_spectra"] = self.data["gt_spectra"][val_ids]
        self.data["validation_mask"] = self.data["gt_spectra_mask"][val_ids]
        self.data["semi_supervision_redshift"] = self.data["gt_spectra_redshift"][val_ids]
        self.data["validation_sup_bound"] = self.data["gt_spectra_sup_bound"][val_ids]

        # test spectra data (used during main inferrence only)
        self.data["test_spectra"] = self.data["gt_spectra"][test_ids]
        self.data["test_mask"] = self.data["gt_spectra_mask"][test_ids]
        self.data["test_redshift"] = self.data["gt_spectra_redshift"][test_ids]
        self.data["test_sup_bound"] = self.data["gt_spectra_sup_bound"][test_ids]

        if self.spectra_process_patch_info:
            if self.kwargs["pretrain_pixel_supervision"]:
                self.data["test_pixels"] = self.data["gt_spectra_pixels"][test_ids]
                self.data["validation_pixels"] = self.data["gt_spectra_pixels"][val_ids]
                self.data["supervision_pixels"] = self.data["gt_spectra_pixels"][sup_ids]

            # [n,n_neighbr**2,2/3]
            self.data["test_coords"] = self.data["gt_spectra_coords"][test_ids]
            self.data["validation_coords"] = self.data["gt_spectra_coords"][val_ids]

    def set_wave_range(self):
        """ Set wave range used for linear normalization.
            Note if the wave range used to normalize transmission wave and
              the spectra wave should be the same.
        """
        if exists(self.wave_range_fname): return
        wave = np.array(self.data["gt_spectra"][:,0])
        self.data["wave_range"] = np.array([
            int(np.floor(np.min(wave))), int(np.ceil(np.max(wave)))
        ])
        np.save(self.wave_range_fname, self.data["wave_range"])

    def load_gt_spectra_data(self):
        """ Load gt spectra data.
            The full loading workflow is:
              i) we first load source metadata table and process each spectra (
                 fluxes, wave, coords, pixels) and save processed data individually.
              ii) then we load these saved spectra and further process to gather
                  them together and save all data together.
              iii) finally we do necessary transformations.
        """
        # self.find_full_wave_bound_ids()
        spectra_data_cached = \
            exists(self.emitted_wave_fname) and \
            exists(self.emitted_wave_mask_fname) and \
            exists(self.cache_metadata_table_fname) and \
            exists(self.gt_spectra_fname) and \
            exists(self.gt_spectra_mask_fname) and \
            exists(self.gt_spectra_redshift_fname) and \
            exists(self.gt_spectra_sup_bound_fname)
        patch_info_cached = not self.spectra_process_patch_info or (
            exists(self.gt_spectra_ids_fname) and \
            exists(self.gt_spectra_pixels_fname) and \
            exists(self.gt_spectra_img_coords_fname) and \
            exists(self.gt_spectra_world_coords_fname))
        # print(spectra_data_cached, patch_info_cached)

        if self.load_spectra_data_from_cache and spectra_data_cached and patch_info_cached:
            self.load_cached_spectra_data()
        else:
            self.process_spectra()
            self.gather_processed_spectra()

        # print(len(self.data["gt_spectra"]))
        if self.kwargs["filter_redshift"]:
            self.filter_spectra_based_on_redshift()
        # print(len(self.data["gt_spectra"]))

        if self.kwargs["correct_gt_redshift_based_on_redshift_bin"]:
            self.correct_redshift_based_on_bins()

        self.num_gt_spectra = len(self.data["gt_spectra"])

    #############
    # Loading helpers
    #############

    def to_tensor(self, fields, dtype):
        for field in fields:
            self.data[field] = torch.tensor(self.data[field], dtype=dtype)

    def split_spectra_all_together(self):
        """
        Randomly split spectra into
          supervision
          validation   (same as supervision, used for sanity check) and
          test spectra ( test_spectra + supervision_spectra == all_spectra).
        """
        if exists(self.supervision_id_fname):
            sup_ids = np.load(self.supervision_id_fname)
            indices = np.arange(self.num_gt_spectra)
            test_ids = list(set(indices) - set(sup_ids))
        else:
            ids = np.arange(self.num_gt_spectra)
            np.random.seed(0)
            np.random.shuffle(ids)
            # test_ratio, val_ratio, sup_ratio = self.kwargs["spectra_split_ratios"]
            # n_test = int(self.num_gt_spectra * test_ratio)
            # n_val = int(self.num_gt_spectra * val_ratio)
            # test_ids = ids[:n_test]
            # validation_ids = ids[n_test:n_test+n_val]
            # supervision_ids = ids[n_test+n_val:]
            sup_ratio = self.kwargs["sup_spectra_ratio"]
            n_sup = int(self.num_gt_spectra * sup_ratio)
            sup_ids = ids[:n_sup]
            test_ids = ids[n_sup:]
            np.save(self.supervision_id_fname, sup_ids)

        test_ids = test_ids[:self.kwargs["generalization_max_num_spectra"]]
        sup_ids = sup_ids[:self.kwargs["num_supervision_spectra_upper_bound"]]
        val_ids = sup_ids[:self.kwargs["sanity_check_max_num_spectra"]]
        return test_ids, val_ids, sup_ids

    def split_spectra_patch_wise(self):
        """
        Use all spectra within main train patches as the test or validation set.
        Sample randomly from the rest spectra as the supervision set.
        """
        if exists(self.test_spectra_ids_fname) and \
           exists(self.validation_spectra_ids_fname) and \
           exists(self.supervision_spectra_ids_fname):
            test_ids = np.load(self.test_spectra_ids_fname)
            validation_ids = np.load(self.validation_spectra_ids_fname)
            supervision_ids = np.load(self.supervision_spectra_ids_fname)
        else:
            # reserve all spectra in main train image patch as validation or test spectra
            acc, test_ids, validation_ids = 0, [], []
            for i, (tract, patch) in enumerate(
                zip(self.kwargs["tracts"], self.kwargs["patches"])
            ):
                patch_uid = create_patch_uid(tract, patch)
                cur_spectra_id_coords = np.array(self.data["gt_spectra_ids"][patch_uid])
                cur_spectra_ids = cur_spectra_id_coords[:,0]

                if self.kwargs["train_spectra_pixels_only"] or self.kwargs["use_full_patch"]:
                    # randomly split spectra in each patch into validation and test set
                    np.random.shuffle(cur_spectra_ids)
                    num_val = int(len(cur_spectra_ids) * self.kwargs["val_spectra_ratio"])
                    cur_val_ids = cur_spectra_ids[:num_val]
                    cur_test_ids = cur_spectra_ids[num_val:]
                else:
                    # use spectra within cutout as validation set and the rest as test set
                    num_rows = self.kwargs["patch_cutout_num_rows"][i]
                    num_cols = self.kwargs["patch_cutout_num_cols"][i]
                    (r, c) = self.kwargs["patch_cutout_start_pos"][i] # top-left corner

                    coords = cur_spectra_id_coords[:,1:]
                    within_cutout = (coords[:,0] >= r) & (coords[:,0] < r + num_rows) & \
                        (coords[:,1] >= c) & (coords[:,1] < c + num_cols)
                    cur_val_ids = cur_spectra_ids[within_cutout]
                    cur_test_ids = list(set(cur_spectra_ids) - set(cur_val_ids))

                test_ids.extend(cur_test_ids)
                validation_ids.extend(cur_val_ids)
                # validation_patch_ids[patch_uid] = np.arange(acc, acc+len(cur_spectra_ids))
                # acc += len(cur_spectra_ids)

            # reserve all spectra from test patches as test spectra
            for i, (tract, patch) in enumerate(
                zip(self.kwargs["test_tracts"], self.kwargs["test_patches"])
            ):
                patch_uid = create_patch_uid(tract, patch)
                cur_spectra_id_coords = np.array(self.data["gt_spectra_ids"][patch_uid])
                cur_spectra_ids = cur_spectra_id_coords[:,0]
                test_ids.extend(cur_spectra_ids)

            test_ids = np.array(test_ids)
            validation_ids = np.array(validation_ids)
            # use the rest spectra for pretrain
            supervision_ids = np.array(
                list(set(ids)-set(validation_ids)-set(test_ids))).astype(int)

            # reupdate validation ids
            validation_ids = self.update_validation_ids(
                supervision_ids, validation_ids)
            if len(test_ids) == 0 or len(validation_ids) == 0:
                raise ValueError(
                    "Please select patches properly to make sure the number \
                    of validation and test spectra is not zero.")

        return test_ids, validation_ids, supervision_ids

    def update_validation_ids(self, supervision_ids, validation_ids):
        """
        Update validation spectra for sanity check or add more spectra for generalization.
        """
        # if self.kwargs["sample_from_codebook_pretrain_spectra"]:
        #     # select spectra for redshift pretrain from spectra used for codebook pretrain
        #     indices = np.arange(len(supervision_ids))
        #     # np.random.seed(self.kwargs["seed"])
        #     # np.random.shuffle(indices)
        #     self.redshift_pretrain_ids = indices[:self.kwargs["redshift_pretrain_num_spectra"]]
        #     validation_ids = supervision_ids[self.redshift_pretrain_ids]

        if self.kwargs["add_validation_spectra_not_in_supervision"]:
            unseen_spectra_ids = np.array(
                list(set(ids) - set(supervision_ids) - set(validation_ids)))
            # np.random.seed(0)
            # np.random.shuffle(unseen_spectra_ids)
            selected_ids = unseen_spectra_ids[:self.kwargs["num_extra_validation_spectra"]]
            validation_ids = np.append(validation_ids, selected_ids)
            assert len(set(validation_ids) & set(supervision_ids)) == 0

        return validation_ids

    def correct_redshift_based_on_bins(self):
        """
        """
        # print(self.data["gt_spectra_redshift"][:20])
        num_bins = (self.data["gt_spectra_redshift"] - self.kwargs["redshift_lo"]) // self.kwargs["redshift_bin_width"]
        num_bins = num_bins.astype(int)
        self.data["gt_spectra_redshift"] = num_bins * self.kwargs["redshift_bin_width"] + \
            self.kwargs["redshift_lo"] + self.kwargs["redshift_bin_width"] / 2
        # print(self.data["gt_spectra_redshift"][:20])

    def filter_spectra_based_on_redshift(self):
        valid_ids = np.arange(len(self.data["gt_spectra_redshift"]))
        if self.kwargs["filter_redshift_lo"] >= 0:
            valid_ids = valid_ids[
                self.data["gt_spectra_redshift"] > self.kwargs["filter_redshift_lo"]]
        if self.kwargs["filter_redshift_hi"] >= self.kwargs["filter_redshift_lo"]:
            valid_ids = valid_ids[
                self.data["gt_spectra_redshift"] < self.kwargs["filter_redshift_hi"]]
        for field in ["gt_spectra","gt_spectra_mask","gt_spectra_pixels",
                      "gt_spectra_redshift","gt_spectra_img_coords","gt_spectra_world_coords"]:
            self.data[field] = self.data[field][valid_ids]

    def load_cached_spectra_data(self):
        """ Load spectra data (which are saved together).
        """
        self.data["emitted_wave"] = np.load(self.emitted_wave_fname)
        self.data["emitted_wave_mask"] = np.load(self.emitted_wave_mask_fname)
        self.data["gt_spectra"] = np.load(self.gt_spectra_fname)
        self.data["gt_spectra_mask"] = np.load(self.gt_spectra_mask_fname)
        self.data["gt_spectra_redshift"] = np.load(self.gt_spectra_redshift_fname)
        self.data["gt_spectra_sup_bound"] = np.load(self.gt_spectra_sup_bound_fname)

        if self.spectra_process_patch_info:
            with open(self.gt_spectra_ids_fname, "rb") as fp:
                ids = pickle.load(fp)
            self.data["gt_spectra_ids"] = defaultdict(list, ids)
            self.data["gt_spectra_pixels"] = np.load(self.gt_spectra_pixels_fname)
            self.data["gt_spectra_img_coords"] = np.load(self.gt_spectra_img_coords_fname)
            self.data["gt_spectra_world_coords"] = np.load(self.gt_spectra_world_coords_fname)

    def gather_processed_spectra(self):
        """ Load processed data for each spectra and save together.
        """
        df = pandas.read_pickle(self.cache_metadata_table_fname)
        n = len(df)
        mask, spectra, redshift, sup_bound = [], [], [], []
        if self.spectra_process_patch_info:
            with open(self.gt_spectra_ids_fname, "rb") as fp:
                ids = pickle.load(fp)
            self.data["gt_spectra_ids"] = defaultdict(list, ids)
            img_coords, world_coords, pixels = [], [], []

        for i in range(n):
            redshift.append(df.iloc[i]["zspec"])
            sup_bound.append(df.iloc[i]["sup_wave_bound"])

            source = df.iloc[i]["source"]
            path = getattr(self, f"{source}_processed_spectra_path")
            fname = join(path, df.iloc[i]["mask_fname"])
            mask.append(np.load(fname))
            fname = join(path, df.iloc[i]["spectra_fname"])
            spectra.append(np.load(fname))

            if self.spectra_process_patch_info:
                fname = join(self.processed_spectra_path, df.iloc[i]["pixels_fname"])
                pixels.append(np.load(fname))
                fname = join(self.processed_spectra_path, df.iloc[i]["img_coords_fname"])
                img_coords.append(np.load(fname)[None,...])
                fname = join(self.processed_spectra_path, df.iloc[i]["world_coords_fname"])
                world_coords.append(np.load(fname)[None,...])

        np.save(self.emitted_wave_fname, self.data["emitted_wave"])
        np.save(self.emitted_wave_mask_fname, self.data["emitted_wave_mask"])

        # [n_spectra,4+2*nbands,nsmpl]
        #  (wave/flux/ivar/trans_mask/trans(nbands)/band_mask(nbands))
        self.data["gt_spectra"] = np.array(spectra).astype(np.float32)
        self.data["gt_spectra_mask"] = np.array(mask).astype(bool)
        self.data["gt_spectra_sup_bound"] = np.array(sup_bound)
        self.data["gt_spectra_redshift"] = np.array(redshift).astype(np.float32) # [n,]
        np.save(self.gt_spectra_fname, self.data["gt_spectra"])
        np.save(self.gt_spectra_mask_fname, self.data["gt_spectra_mask"])
        np.save(self.gt_spectra_redshift_fname, self.data["gt_spectra_redshift"])
        np.save(self.gt_spectra_sup_bound_fname, self.data["gt_spectra_sup_bound"])

        if self.spectra_process_patch_info:
            self.data["gt_spectra_pixels"] = np.concatenate(
                pixels, axis=0).astype(np.float32)
            self.data["gt_spectra_img_coords"] = np.concatenate(
                img_coords, axis=0).astype(np.float32) # [n,n_neighbr,2]
            self.data["gt_spectra_world_coords"] = np.concatenate(
                world_coords, axis=0).astype(np.float32) # [n,n_neighbr,2]
            np.save(self.gt_spectra_pixels_fname, self.data["gt_spectra_pixels"])
            np.save(self.gt_spectra_img_coords_fname, self.data["gt_spectra_img_coords"])
            np.save(self.gt_spectra_world_coords_fname, self.data["gt_spectra_world_coords"])

    def process_spectra(self):
        df = self.load_source_metadata()
        log.info(f"found {len(df)} source spectra")

        upper_bound = self.kwargs["num_gt_spectra_upper_bound"]
        df = self.load_source_metadata().iloc[:upper_bound]
        num_spectra = len(df)
        log.info(f"load {num_spectra} source spectra")

        emitted_wave = self.calculate_emitted_wave_range(df)
        self.data["emitted_wave"] = emitted_wave
        # self.data["emitted_wave_mask"] = self.generate_emitted_wave_mask(emitted_wave[0])

        self.trans_data = self.trans_obj.get_full_trans_data()
        self.emitted_wave_distrib = interp1d(emitted_wave[0], emitted_wave[1])

        df["mask_fname"] = ""
        df["sup_wave_bound"] = [(-1,-1)] * len(df)

        if self.spectra_process_patch_info:
            for field in ["tract","patch","pixels_fname",
                          "img_coords_fname","world_coords_fname"]:
                df[field] = "None"

            header_wcs, headers = self.load_headers(df)
            spectra_ids, spectra_to_drop = self.localize_spectra(df, header_wcs, headers)
            self.load_spectra_patch_wise(df, spectra_ids)

            if self.kwargs["spectra_drop_not_in_patch"]:
                # drop spectra not located in specified patches
                df.drop(spectra_to_drop, inplace=True)
                df.reset_index(inplace=True, drop=True)

            df.dropna(subset=["pixels_fname","img_coords_fname",
                              "world_coords_fname","spectra_fname"], inplace=True)
            df.reset_index(inplace=True, drop=True)
        else:
            self.load_spectra_all_together(df, num_spectra)

        df.to_pickle(self.cache_metadata_table_fname)

    def generate_emitted_wave_mask(self, emitted_wave):
        """ Generate mask for codebook spectra plot.
        """
        emitted_wave_mask = np.zeros(len(emitted_wave))
        (id_lo, id_hi) = get_bound_id(
            (self.kwargs["codebook_spectra_plot_wave_lo"],
             self.kwargs["codebook_spectra_plot_wave_hi"]), emitted_wave)
        emitted_wave_mask[id_lo:id_hi+1] = 1
        np.save(self.emitted_wave_mask_fname, emitted_wave_mask)
        return emitted_wave_mask

    def calculate_emitted_wave_range(self, df):
        """ Calculate coverage of emitted wave based on
              given observed wave supervision range.
        """
        redshift = list(df['zspec'])
        lo = self.kwargs["spectra_supervision_wave_lo"]
        hi = self.kwargs["spectra_supervision_wave_hi"]
        min_emitted_wave = int(lo / (1 + max(redshift)))
        max_emitted_wave = int(np.ceil(hi / (1 + min(redshift))))
        n = max_emitted_wave - min_emitted_wave + 1

        x = np.arange(min_emitted_wave, max_emitted_wave + 1)
        distrib = np.zeros(n).astype(np.int32)

        def accumulate(cur_redshift):
            distrib[ int(lo/(1+cur_redshift)) - min_emitted_wave:
                     int(hi/(1+cur_redshift)) - min_emitted_wave ] += 1

        _ = [accumulate(cur_redshift) for cur_redshift in redshift]
        distrib = np.array(distrib) # / sum(distrib))

        plt.plot(x, distrib); plt.title("Counts of spectra in restframe (emitted wave)")
        plt.xlabel("lambda"); plt.ylabel("# spectra")
        plt.savefig(self.emitted_wave_fname[:-4] + ".png")
        plt.close()

        emitted_wave = np.concatenate((x[None,:], distrib[None,:]), axis=0)
        np.save(self.emitted_wave_fname, emitted_wave)
        return emitted_wave

    def load_headers(self, df):
        """ Load headers of all image patches we have to localize each spectra later on.
        """
        header_wcs, headers = [], []
        for tract in self.all_tracts:
            cur_wcs, cur_headers = [], []
            for patch_r in self.all_patches_r:
                for patch_c in self.all_patches_c:
                    if not patch_exists(
                            self.input_patch_path, tract, f"{patch_r},{patch_c}"):
                        cur_headers.append(None)
                        cur_wcs.append(None)
                        continue

                    cur_patch = PatchData(
                        tract, f"{patch_r},{patch_c}", **self.kwargs)
                    header = cur_patch.get_header()
                    wcs = WCS(header)
                    cur_headers.append(header)
                    cur_wcs.append(wcs)

            headers.append(cur_headers)
            header_wcs.append(cur_wcs)

        log.info("spectra-data::header loaded")
        return header_wcs, headers

    def localize_spectra(self, df, header_wcs, headers):
        """ Locate tract and patch for each spectra.
        """
        n = len(df)

        if exists(self.gt_spectra_ids_fname):
            with open(self.gt_spectra_ids_fname, "rb") as fp:
                spectra_ids = pickle.load(fp)

            valid_spectra_ids = []
            for k, v in spectra_ids.items():
                # v: n*[3] (global_spectra_id/r/c)
                valid_spectra_ids.extend(np.array(v)[:,0])

            ids = np.arange(n)
            spectra_to_drop = list(set(ids) - set(valid_spectra_ids))
            spectra_ids = defaultdict(list, spectra_ids)
            log.info("spectra-data::localized spectra")
            return spectra_ids, spectra_to_drop

        spectra_to_drop = []
        spectra_ids = defaultdict(lambda: [])
        localize = partial(locate_tract_patch,
                           header_wcs, headers, self.all_tracts,
                           self.all_patches_r, self.all_patches_c)
        for idx in range(n):
            ra = df.iloc[idx]["ra"]
            dec = df.iloc[idx]["dec"]
            tract, patch, r, c = localize(ra, dec)

            # TODO: we may adapt to spectra that doesn't belong
            #       to any patches we have in the future
            if tract == -1: # current spectra doesn't belong to patches we selected
                spectra_to_drop.append(idx)
                continue

            patch_uid = create_patch_uid(tract, patch)
            spectra_ids[patch_uid].append([idx,r,c])
            df.at[idx,"tract"] = tract
            df.at[idx,"patch"] = patch

        log.info("spectra-data::localized spectra")
        with open(self.gt_spectra_ids_fname, "wb") as fp:
            pickle.dump(dict(spectra_ids), fp)
        return spectra_ids, spectra_to_drop

    def load_spectra_all_together(self, df, num_spectra):
        # [ self.process_one_spectra(None, None, df, None, idx, None)
        #   for idx in range(num_spectra) ]
        for idx in tqdm(range(num_spectra)):
        #     if df.loc[idx,"source"] == "deep3":
        #         for deep3_portion in ["b","r"]:
        #             self.process_one_spectra(
        #                 None, None, None, df, None, idx, None, deep3_portion)
        #     else:
            self.process_one_spectra(
                None, None, None, df, None, idx, None)

    def load_spectra_patch_wise(self, df, spectra_ids):
        """ Load pixels and coords for each spectra in patch-wise order.
        """
        process_each_patch = partial(
            self.process_spectra_in_one_patch, df)

        for i, tract in enumerate(self.all_tracts):
            for j, patch_r in enumerate(self.all_patches_r):
                for k, patch_c in enumerate(self.all_patches_c):
                    patch_uid = create_patch_uid(tract, f"{patch_r},{patch_c}")
                    if len(spectra_ids[patch_uid]) == 0 or \
                       not patch_exists(self.input_patch_path, tract, f"{patch_r},{patch_c}"):
                        continue

                    cur_patch = PatchData(
                        tract, f"{patch_r},{patch_c}",
                        load_pixels=True,
                        load_coords=True,
                        pixel_norm_cho=self.kwargs["train_pixels_norm"],
                        **self.kwargs)
                    process_each_patch(patch_uid, cur_patch, spectra_ids[patch_uid])

    def process_spectra_in_one_patch(self, df, patch_uid, patch, spectra_ids):
        """ Get coords and pixel values for all spectra (specified by spectra_ids)
              within the given patch.
            @Params
              df: source metadata dataframe for all spectra
              patch: patch that contains the current spectra
              spectra_ids: ids for spectra within the current patch
        """
        if len(spectra_ids) == 0: return

        log.info(f"spectra-data::processing {patch_uid}, contains {len(spectra_ids)} spectra")

        spectra_ids = np.array(spectra_ids)[:,0]

        ras = np.array(list(df.iloc[spectra_ids]["ra"]))
        decs = np.array(list(df.iloc[spectra_ids]["dec"]))
        redshift = np.array(list(df.iloc[spectra_ids]["zspec"])).astype(np.float32)

        # get img coords for all spectra within current patch
        # NOTE, these coords exclude neighbours
        wcs = WCS(patch.get_header())
        world_coords = np.concatenate((ras[:,None], decs[:,None]), axis=-1)
        img_coords = wcs.all_world2pix(world_coords, 0).astype(int) # [n,2]
        # world coords from spectra data may not be accurate in terms of
        #  wcs of each patch, here after we get img coords, we convert
        #  img coords back to world coords to get accurate values
        world_coords = wcs.all_pix2world(img_coords, 0) # [n,2]
        img_coords = img_coords[:,::-1] # xy coords to rc coords [n,2]

        cur_patch_spectra = []
        cur_patch_spectra_mask = []
        cur_patch_spectra_sup_bound = []
        process_one_spectra = partial(self.process_one_spectra,
                                      cur_patch_spectra,
                                      cur_patch_spectra_mask,
                                      cur_patch_spectra_sup_bound,
                                      df, patch)

        [ process_one_spectra(idx, img_coord)
          for i, (idx, img_coord) in enumerate(zip(spectra_ids, img_coords))]

        cur_patch_spectra_fname = join(
            self.processed_spectra_path, f"{patch_uid}.npy")
        cur_patch_mask_fname = join(
            self.processed_spectra_path, f"{patch_uid}_mask.npy")
        cur_patch_redshift_fname = join(
            self.processed_spectra_path, f"{patch_uid}_redshift.npy")
        cur_patch_sup_bound_fname = join(
            self.processed_spectra_path, f"{patch_uid}_sup_bound.npy")
        cur_patch_img_coords_fname = join(
            self.processed_spectra_path, f"{patch_uid}_img_coords.npy")
        cur_patch_world_coords_fname = join(
            self.processed_spectra_path, f"{patch_uid}_world_coords.npy")

        np.save(cur_patch_redshift_fname, redshift)
        np.save(cur_patch_img_coords_fname, img_coords)     # excl neighbours
        np.save(cur_patch_world_coords_fname, world_coords) # excl neighbours
        np.save(cur_patch_spectra_fname, np.array(cur_patch_spectra))
        np.save(cur_patch_mask_fname, np.array(cur_patch_spectra_mask))
        np.save(cur_patch_sup_bound_fname, np.array(cur_patch_spectra_sup_bound))

    def process_one_spectra(
            self, cur_patch_spectra, cur_patch_spectra_mask,
            cur_patch_spectra_sup_bound, df, patch, idx, img_coord
    ):
        """
        Get pixel and normalized coord and process spectra data for one spectra.
        @Params
          df: source metadata dataframe for all spectra
          patch: patch that contains the current spectra
          idx: spectra idx (within the df table)
          img_coord: img coord for current spectra
          deep3_portion: each deep3 spectra is recorded separately into r and b portion
        """
        spectra_source = df.iloc[idx]["source"]
        data_format = getattr(self, f"{spectra_source}_spectra_data_format")
        source_spectra_path = getattr(self, f"{spectra_source}_source_spectra_path")
        processed_spectra_path = getattr(self, f"{spectra_source}_processed_spectra_path")

        spectra_fname = df.iloc[idx][f"spectra_fname_{data_format}"]
        if data_format == "fits": fname = spectra_fname[:-5]
        elif data_format == "tbl": fname = spectra_fname[:-4]
        else: raise ValueError("Unsupported spectra data format")

        if "deep3_portion" in df:
            deep3_portion = df.loc[idx,"deep3_portion"]
            if not pandas.isna(deep3_portion): fname += f"_{deep3_portion}"

        spectra_out_fname = f"{fname}.npy"
        spectra_mask_fname = f"{fname}_mask.npy"
        spectra_sup_bound_fname = f"{fname}_sup_bound.npy"
        df.at[idx,"spectra_fname"] = spectra_out_fname
        df.at[idx,"mask_fname"] = spectra_mask_fname
        # df.at[idx,"spectra_sup_bound_fname"] = spectra_sup_bound_fname

        spectra_in_fname = join(source_spectra_path, spectra_fname)
        spectra_out_fname = join(processed_spectra_path, spectra_out_fname)
        spectra_mask_fname = join(processed_spectra_path, spectra_mask_fname)
        spectra_sup_bound_fname = join(processed_spectra_path, spectra_sup_bound_fname)

        current_spectra_processed = exists(spectra_out_fname) and \
            exists(spectra_mask_fname) and exists(spectra_sup_bound_fname)

        if not current_spectra_processed:
            spectra = unpack_gt_spectra(
                spectra_in_fname, format=data_format,
                source=spectra_source, has_ivar=True,
                deep3_portion=deep3_portion
            ) # [(2,)2/3,nsmpl]

            gt_spectra, mask, sup_bound = process_gt_spectra(
                spectra,
                spectra_out_fname,
                spectra_mask_fname,
                spectra_sup_bound_fname,
                df.loc[idx,"zspec"],
                self.emitted_wave_distrib,
                trans_data=self.trans_data,
                trans_range=self.trans_range,
                process_ivar=self.process_ivar,
                sigma=self.spectra_smooth_sigma,
                colors=self.kwargs["plot_colors"],
                max_spectra_len=self.kwargs["max_spectra_len"],
                supervision_wave_range=self.supervision_wave_range,
                upsample_scale=self.kwargs["spectra_upsample_scale"])
        else:
            if self.spectra_process_patch_info:
                gt_spectra = np.load(spectra_out_fname)
                mask = np.load(spectra_mask_fname)
            sup_bound = np.load(spectra_sup_bound_fname)

        df.at[idx,"sup_wave_bound"] = np.array(sup_bound)

        if self.spectra_process_patch_info:
            cur_patch_spectra.append(gt_spectra)
            cur_patch_spectra_mask.append(mask)
            cur_patch_spectra_sup_bound.append(sup_bound)

            pixels_fname = f"{fname}_pixels.npy"
            img_coords_fname = f"{fname}_img_coord.npy"
            world_coords_fname = f"{fname}_world_coord.npy"
            df.at[idx,"pixels_fname"] = pixels_fname
            df.at[idx,"img_coords_fname"] = img_coords_fname
            df.at[idx,"world_coords_fname"] = world_coords_fname

            pixels_fname = join(processed_spectra_path, pixels_fname)
            img_coords_fname = join(processed_spectra_path, img_coords_fname)
            world_coords_fname = join(processed_spectra_path, world_coords_fname)
            current_spectra_processed &= exists(pixels_fname) and \
                exists(img_coords_fname) and exists(world_coords_fname)
            if current_spectra_processed:
                cur_patch_spectra.append(np.load(spectra_out_fname))
                cur_patch_spectra_mask.append(np.load(spectra_mask_fname))

            pixel_ids = patch.get_pixel_ids(img_coord[0], img_coord[1])
            pixels = patch.get_pixels(pixel_ids) # [1,2]

            pixel_ids = patch.get_pixel_ids(
                img_coord[0], img_coord[1], neighbour_size=self.spectra_neighbour_size)
            img_coords = patch.get_img_coords(pixel_ids)     # mesh grid coords [n_neighbr,2]
            world_coords = patch.get_world_coords(pixel_ids) # un-normed ra/dec [n_neighbr,2]

            np.save(pixels_fname, pixels)
            np.save(img_coords_fname, img_coords)
            np.save(world_coords_fname, world_coords)

    def load_source_metadata(self):
        df = []

        if "deimos" in self.spectra_data_sources:
            cur_df = read_deimos_table(
                self.deimos_source_metadata_table_fname,
                format=self.deimos_spectra_data_format,
                download=self.download_deimos_source_spectra,
                link=self.deimos_source_spectra_link,
                path=self.deimos_source_spectra_path)
            df.append(cur_df)

        if "zcosmos" in self.spectra_data_sources:
            cur_df = read_zcosmos_table(
                self.zcosmos_source_metadata_table_fname,
                format=self.zcosmos_spectra_data_format,
                download=self.download_zcosmos_source_spectra,
                link=self.zcosmos_source_spectra_link,
                path=self.zcosmos_source_spectra_path)
            df.append(cur_df)

        if "deep3" in self.spectra_data_sources:
            cur_df = read_deep3_table(
                self.deep3_source_metadata_table_fname,
                format=self.deep3_spectra_data_format)
            df.append(cur_df)

        df = pandas.concat(df)
        df.reset_index(inplace=True, drop=True)
        if self.kwargs["random_permute_source_spectra"]:
            df = df.sample(frac=1).reset_index(drop=True)
        return df

    # def find_full_wave_bound_ids(self):
    #     """ Find id of min and max wave of supervision range in terms of
    #           the transmission wave (full_wave).
    #         Since the min and max wave for the supervision range may not
    #           coincide exactly with the trans wave, we find closest trans wave to replace
    #     """
    #     supervision_spectra_wave_bound = [
    #         self.kwargs["spectra_supervision_wave_lo"],
    #         self.kwargs["spectra_supervision_wave_hi"]
    #     ]
    #     (id_lo, id_hi) = get_bound_id(
    #         supervision_spectra_wave_bound, self.full_wave, within_bound=False)
    #     self.data["supervision_wave_bound_ids"] = [id_lo, id_hi + 1]
    #     self.data["supervision_spectra_wave_bound"] = [
    #         self.full_wave[id_lo], self.full_wave[id_hi]]

    #############
    # Utilities
    #############

    # def interpolate_spectra(self, f, spectra, masks):
    #     """ Interpolate spectra to same discretization interval as trans data
    #         @Param
    #           spectra: spectra data [bsz,2,nsmpl] (wave/flux)
    #           masks: mask out range of spectra to ignore [bsz,nsmpl] (1-keep, 0-drop)
    #     """
    #     # masks = masks[:,None] #.tile(1,2,1)
    #     interp_spectra = []
    #     print(spectra.shape, masks.shape)
    #     for (cur_spectra, cur_mask) in zip(spectra, masks):
    #         interp_spectra.append(
    #             self.interpolate_one_spectra(f, cur_spectra, cur_mask)
    #         )
    #     return interp_spectra

    # def interpolate_one_spectra(self, f, spectra, mask):
    #     # print(spectra.shape, mask.shape, spectra.dtype, mask.dtype)
    #     spectra_wave = spectra[0][mask]
    #     # print(spectra_wave.shape, spectra_wave[0], spectra_wave[-1])
    #     interp_trans = f(spectra_wave)
    #     print(interp_flux.shape)
    #     assert 0

    # def integrate_spectra_over_trans(self, spectra, trans):
    #     pass

    def calculate_spectra_metrics(self, gt_flux, recon_flux, sub_dir, axis):
        metrics = calculate_metrics(
            recon_flux, gt_flux, self.kwargs["spectra_metric_options"],
            window_width=self.kwargs["spectra_zncc_window_width"]) # [n_metrics]

        above_threshold = None
        if "zncc" in metrics:
            (zncc, zncc_sliding) = metrics["zncc"]
            zncc_sliding = np.array(zncc_sliding)
            m = len(zncc_sliding)

            thresh = self.kwargs["local_zncc_threshold"]

            if self.kwargs["plot_spectrum_according_to_zncc"]:
                above_threshold = np.full(len(recon_flux), False)
                above_threshold[:m] = zncc_sliding > thresh
                sub_dir += f"highlight_above_{thresh}_local_zncc_"

            if self.kwargs["plot_spectrum_with_sliding_zncc"]:
                sub_dir += "with_zncc_"
                axis.plot(gt_wave[:m], zncc_sliding, color="gray")

                # n = len(recon_flux)
                # los = np.arange(0, n, self.kwargs["spectra_zncc_window_width"])
                # wave_lo = min(gt_wave)
                # for val, lo in zip(zncc_sliding, los):
                #     hi = min(lo + self.kwargs["spectra_zncc_window_width"], n)
                #     val = (val + 1) / 2
                #     axis.axvspan(lo + wave_lo, hi + wave_lo, color=str(val))

            metrics.pop("zncc", None)
            metrics["zncc_global"] = zncc
            metrics["zncc_sliding_avg"] = sum(zncc_sliding) / len(zncc_sliding)
            if self.kwargs["calculate_sliding_zncc_above_threshold"]:
                zncc_sliding = zncc_sliding[zncc_sliding > thresh]
                metrics["zncc_sliding_avg_above_threshold"] = \
                    sum(zncc_sliding) / len(zncc_sliding)

        return sub_dir, metrics, above_threshold

    def normalize_one_flux(self, sub_dir, is_codebook, plot_gt_spectrum,
                           plot_recon_spectrum, flux_norm_cho, gt_flux, recon_flux
    ):
        """ Normalize one pair of gt and recon flux.
        """
        sub_dir += flux_norm_cho + "_"
        if plot_recon_spectrum:
            sub_dir = sub_dir + "with_recon_"
            if flux_norm_cho == "identity":
                pass
            elif flux_norm_cho == "max":
                recon_flux = recon_flux / np.max(recon_flux)
            elif flux_norm_cho == "sum":
                recon_flux = recon_flux / np.sum(recon_flux)
            elif flux_norm_cho == "linr":
                lo, hi = min(recon_flux), max(recon_flux)
                recon_flux = (recon_flux - lo) / (hi - lo)
            elif flux_norm_cho == "scale_gt":
                # scale gt spectra s.t. its max is same as recon
                recon_max = np.max(recon_flux)
            else: raise ValueError()

        if plot_gt_spectrum and not is_codebook:
            sub_dir = sub_dir + "with_gt_"
            # assert(np.max(gt_flux) > 0)
            if flux_norm_cho == "identity":
                pass
            elif flux_norm_cho == "max":
                gt_flux = gt_flux / np.max(gt_flux)
            elif flux_norm_cho == "sum":
                gt_flux = gt_flux / (np.sum(gt_flux) + 1e-10)
                gt_flux = gt_flux * len(gt_flux) / len(recon_flux)
            elif flux_norm_cho == "linr":
                lo, hi = min(gt_flux), max(gt_flux)
                gt_flux = (gt_flux - lo) / (hi - lo)
            elif flux_norm_cho == "scale_gt":
                gt_flux = gt_flux / np.max(gt_flux) * recon_max
            elif flux_norm_cho == "scale_recon":
                recon_flux = recon_flux / np.max(recon_flux) * np.max(gt_flux)
            else: raise ValueError()

        return sub_dir, gt_flux, recon_flux

    def plot_and_save_one_spectrum(self, name, spectra_dir, fig, axs, nrows, ncols, colors,
                                   save_spectra, calculate_metrics, linelist, idx, pargs):
        """ Plot one spectrum and save as required.
        """
        sub_dir, title, z, gt_wave, ivar, gt_flux, recon_wave, recon_flux, \
            recon_flux2, recon_loss2, recon_flux3, recon_loss3, lambdawise_losses, \
            lambdawise_weights, plot_gt_spectrum, plot_recon_spectrum = pargs

        if self.kwargs["plot_spectrum_together"]:
            if nrows == 1: axis = axs if ncols == 1 else axs[idx%ncols]
            else:          axis = axs[idx//ncols, idx%ncols]
        else: fig, axs = plt.subplots(1); axis = axs[0]

        if self.kwargs["plot_spectrum_with_trans"]:
            sub_dir += "with_" + self.kwargs["trans_norm_cho"] + "_trans_"
            self.trans_obj.plot_trans(
                axis=axis, norm_cho=self.kwargs["trans_norm_cho"], color="gray")

        if calculate_metrics:
            sub_dir, metrics, above_threshold = self.calculate_spectra_metrics(
                gt_flux, recon_flux, sub_dir, axis)
        else: metrics, above_threshold = None, None

        if title is None: title = str(idx)

        if colors is not None:
            (gt_color, recon_color, flux2_color, flux3_color) = colors
        else: gt_color, recon_color, flux2_color, flux3_color = "gray","blue","green","red"

        if plot_gt_spectrum:
            plot_spectra(fig, axis, z, gt_wave, gt_flux, gt_color,
                         "gt", "dashed", linelist, None, ivar)
        if plot_recon_spectrum:
            if above_threshold is not None: # plot recon flux according to zncc
                plot_spectra(fig, axis, z, recon_wave, recon_flux, recon_color,
                             "recon", "dotted", None, lambdawise_losses)
                segments = segment_bool_array(above_threshold)
                for (lo, hi) in segments:
                    plot_spectra(fig, axis, z, recon_wave[lo:hi], recon_flux[lo:hi],
                                 "purple", "recon", "solid", None, lambdawise_losses)
            else:
                plot_spectra(fig, axis, z, recon_wave, recon_flux, recon_color,
                             "recon", "solid", None, lambdawise_losses, None,
                             self.kwargs["plot_spectrum_with_loss"],
                             self.kwargs["plot_spectrum_color_based_on_loss"])

        if lambdawise_weights is not None:
            plot_spectra(fig, axis, z, recon_wave, lambdawise_weights, "gray",
                         "gt bin", "solid", None, None, None,)

        if recon_flux2 is not None:
            plot_spectra(fig, axis, z, recon_wave, recon_flux2, flux2_color,
                         "gt bin", "solid", None, lambdawise_losses[0], None,
                         self.kwargs["plot_spectrum_with_loss"],
                         self.kwargs["plot_spectrum_color_based_on_loss"])
            if recon_loss2 is not None: title += f": {recon_loss2:.{3}f}"
        if recon_flux3 is not None:
            plot_spectra(fig, axis, z, recon_wave, recon_flux3, flux3_color,
                         "wrong bin", "solid", None, lambdawise_losses[-1], None,
                         self.kwargs["plot_spectrum_with_loss"],
                         self.kwargs["plot_spectrum_color_based_on_loss"])
            if recon_loss3 is not None: title += f"/{recon_loss3:.{3}f}"

        axis.set_title(title)

        if sub_dir != "":
            if sub_dir[-1] == "_": sub_dir = sub_dir[:-1]
            cur_spectra_dir = join(spectra_dir, sub_dir)
        else: cur_spectra_dir = spectra_dir
        if not exists(cur_spectra_dir):
            Path(cur_spectra_dir).mkdir(parents=True, exist_ok=True)

        if not self.kwargs["plot_spectrum_together"]:
            fname = join(cur_spectra_dir, f"spectra_{idx}_{name}")
            fig.tight_layout(); plt.savefig(fname); plt.close()

        if save_spectra:
            fname = join(cur_spectra_dir, f"spectra_{idx}_{name}")
            np.save(fname, recon_flux)

        return sub_dir, metrics

    def process_recon_flux(
        self, recon_flux, recon_mask, clip, spectra_clipped, recon_wave, lambdawise_losses
    ):
        """
        Process reconstructed spectra with local averageing (flux) and clipping.
        """
        if recon_flux.ndim == 2:
            if self.average_neighbour_spectra:
                recon_flux = np.mean(recon_flux, axis=0)
            else: recon_flux = recon_flux[0]
        else: assert(recon_flux.ndim == 1)
        if clip and not spectra_clipped:
            recon_wave = recon_wave[recon_mask]
            recon_flux = recon_flux[recon_mask]
            if lambdawise_losses is not None:
                lambdawise_losses = lambdawise_losses[recon_mask]
        return recon_wave, recon_flux, lambdawise_losses

    def process_spectrum_plot_data(self, flux_norm_cho, is_codebook, clip,
                                   spectra_clipped, calculate_metrics, linelist, data):
        """ Collect data for spectrum plotting for the given spectra.
        """
        (title, z, gt_wave, ivar, gt_mask, gt_flux, recon_wave, recon_mask,
         recon_flux, recon_flux2, recon_loss2, recon_flux3, recon_loss3,
         lambdawise_losses, lambdawise_weights) = data
        """
        lambdawise_losses
          apply_gt_redshift: [nsmpl]
          brute_force:   [1/2,nsmpl] gt_bin_lambdawise_losses,wrong_bin_lambdawise_losses
        """
        sub_dir = ""
        if self.spectra_neighbour_size > 0:
            sub_dir += f"average_{self.spectra_neighbour_size}_neighbours_"
        if ivar is not None:               sub_dir += "with_ivar_"
        if linelist is not None:           sub_dir += "with_lines_"
        if self.convolve_spectra:          sub_dir += "convolved_"
        if clip or spectra_clipped:        sub_dir += "clipped_"
        if recon_flux2 is not None:        sub_dir += 'with_gt_bin_'
        if recon_flux3 is not None:        sub_dir += 'with_wrong_bin_'
        if lambdawise_losses is not None:  sub_dir += 'loss_based_color_'
        if lambdawise_weights is not None: sub_dir += 'with_weights_'

        plot_gt_spectrum = self.kwargs["plot_spectrum_with_gt"] \
            and gt_flux is not None and not is_codebook
        plot_recon_spectrum = self.kwargs["plot_spectrum_with_recon"]

        if plot_gt_spectrum and clip and not spectra_clipped:
            gt_wave = gt_wave[gt_mask]
            gt_flux = gt_flux[gt_mask]
            if ivar is not None:
                ivar = ivar[gt_mask]

        if lambdawise_weights is not None:
            if clip and not spectra_clipped:
                # print(lambdawise_weights.shape, recon_mask.shape, recon_mask.dtype)
                lambdawise_weights = lambdawise_weights[recon_mask]

        if plot_recon_spectrum:
            recon_wave_p, recon_flux, lambdawise_losses = self.process_recon_flux(
                recon_flux, recon_mask, clip, spectra_clipped, recon_wave, lambdawise_losses)
        else:
            if lambdawise_losses is not None:
                lambdawise_losses = list(lambdawise_losses)
            else: lambdawise_losses = [None,None]
            if recon_flux2 is not None: # gt bin spectra
                recon_wave_p, recon_flux2, lambdawise_losses[0] = self.process_recon_flux(
                    recon_flux2, recon_mask, clip, spectra_clipped, recon_wave,
                    lambdawise_losses[0])
            if recon_flux3 is not None: # wrong bin spectra
                recon_wave_p, recon_flux3, lambdawise_losses[-1] = self.process_recon_flux(
                    recon_flux3, recon_mask, clip, spectra_clipped, recon_wave,
                    lambdawise_losses[-1])
            lambdawise_losses = np.array(lambdawise_losses)

        plot_recon_spectrum = plot_recon_spectrum or recon_flux2 is not None or \
            recon_flux3 is not None or lambdawise_weights is not None
        if plot_recon_spectrum: recon_wave = recon_wave_p

        # recon and gt spectra differ in shape, to calculate metrics, we do interpolation
        if plot_recon_spectrum and calculate_metrics and not \
           ( recon_wave.shape == gt_wave.shape and (recon_wave == gt_wave).all() ):
            if not wave_within_bound(recon_wave, gt_wave):
                f = interp1d(gt_wave, gt_flux)
                gt_flux = f(recon_wave)
                gt_wave = recon_wave
            else:
                assert wave_within_bound(gt_wave, recon_wave)
                f = interp1d(recon_wave, recon_flux)
                recon_flux = f(gt_wave)
                recon_wave = gt_wave

        sub_dir, gt_flux, recon_flux = self.normalize_one_flux(
            sub_dir, is_codebook, plot_gt_spectrum, plot_recon_spectrum,
            flux_norm_cho, gt_flux, recon_flux)
        if recon_flux2 is not None:
            recon_flux2 = self.normalize_one_flux(
                sub_dir, is_codebook, False, plot_recon_spectrum, flux_norm_cho,
                None, recon_flux2)[-1]
        if recon_flux3 is not None:
            recon_flux3 = self.normalize_one_flux(
                sub_dir, is_codebook, False, plot_recon_spectrum, flux_norm_cho,
                None, recon_flux3)[-1]

        plot_recon_spectrum = self.kwargs["plot_spectrum_with_recon"]
        pargs = (sub_dir, title, z, gt_wave, ivar, gt_flux, recon_wave,
                 recon_flux, recon_flux2, recon_loss2, recon_flux3, recon_loss3,
                 lambdawise_losses, lambdawise_weights,
                 plot_gt_spectrum, plot_recon_spectrum)
        return pargs

    def plot_spectrum(self, spectra_dir, name, flux_norm_cho,
                      redshift, gt_wave, ivar, gt_fluxes,
                      recon_wave, recon_fluxes,
                      recon_fluxes2=None,
                      recon_losses2=None,
                      recon_fluxes3=None,
                      recon_losses3=None,
                      lambdawise_losses=None,
                      lambdawise_weights=None,
                      colors=None,
                      titles=None,
                      is_codebook=False,
                      save_spectra=False,
                      calculate_metrics=True,
                      save_spectra_together=False,
                      gt_masks=None, recon_masks=None,
                      clip=False, spectra_clipped=False
    ):
        """
        Plot all given spectra.
        @Param
          spectra_dir:   directory to save spectra
          name:          file name
          flux_norm_cho: norm choice for flux

          gt_wave:     corresponding wave for gt and recon fluxes
          ivar:        inverse variance
          gt_fluxes:   [num_spectra,nsmpl]
          recon_fluxs: [num_spectra(,num_neighbours),nsmpl]

        - clip config:
          clip: whether or not we plot spectra within certain range
          mask: not None if clip. use mask to clip flux
          spectra_clipped: whether or not spectra is already clipped to

       @Return
         metrics: [n_spectra,n_metrics]
        """
        assert not clip or (recon_masks is not None or spectra_clipped)
        calculate_metrics = calculate_metrics and not is_codebook and (clip or spectra_clipped)

        n = len(recon_wave)
        if titles is None: titles = [None]*n
        if ivar is None: ivar = [None]*n
        if redshift is None: redshift = [None]*n
        if gt_wave is None: gt_wave = [None]*n
        if gt_masks is None: gt_masks = [None]*n
        if gt_fluxes is None: gt_fluxes = [None]*n
        if recon_masks is None: recon_masks = [None]*n
        if recon_fluxes2 is None: recon_fluxes2 = [None]*n
        if recon_losses2 is None: recon_losses2 = [None]*n
        if recon_fluxes3 is None: recon_fluxes3 = [None]*n
        if recon_losses3 is None: recon_losses3 = [None]*n
        if lambdawise_losses is None: lambdawise_losses = [None]*n
        if lambdawise_weights is None: lambdawise_weights = [None]*n

        assert gt_fluxes[0] is None or \
            (len(gt_wave) == n and len(gt_fluxes) == n and len(gt_masks) == n)
        assert recon_masks[0] is None or \
            (len(recon_fluxes) == n and len(recon_masks) == n)

        recon_fluxes = to_numpy(recon_fluxes)
        if gt_fluxes[0] is not None: gt_fluxes = to_numpy(gt_fluxes)
        if recon_fluxes2[0] is not None: recon_fluxes2 = to_numpy(recon_fluxes2)
        if recon_fluxes3[0] is not None: recon_fluxes3 = to_numpy(recon_fluxes3)

        if self.kwargs["plot_spectrum_together"]:
            ncols = min(n, self.kwargs["num_spectrum_per_row"])
            nrows = int(np.ceil(n / ncols))
            fig, axs = plt.subplots(nrows, ncols, figsize=(5*ncols,5*nrows))

        if self.kwargs["plot_spectrum_with_lines"]:
            linelist = LineList("ISM")
        else: linelist = None

        process_data = partial(self.process_spectrum_plot_data,
                               flux_norm_cho, is_codebook, clip, spectra_clipped,
                               calculate_metrics, linelist)
        plot_and_save = partial(self.plot_and_save_one_spectrum,
                                name, spectra_dir, fig, axs, nrows, ncols,
                                colors, save_spectra and not save_spectra_together,
                                calculate_metrics, linelist)
        metrics = []
        for idx, cur_plot_data in enumerate(
            zip(titles, redshift, gt_wave, ivar, gt_masks, gt_fluxes, recon_wave, recon_masks,
                recon_fluxes, recon_fluxes2, recon_losses2, recon_fluxes3, recon_losses3,
                lambdawise_losses, lambdawise_weights)
        ):
            pargs = process_data(cur_plot_data)
            sub_dir, cur_metrics = plot_and_save(idx, pargs)
            if cur_metrics is not None:
                metrics.append(cur_metrics)

        if save_spectra_together:
            fname = join(spectra_dir, name)
            np.save(fname, recon_fluxes)

        if self.kwargs["plot_spectrum_together"]:
            fname = join(spectra_dir, sub_dir, f"all_spectra_{name}")
            fig.tight_layout(); plt.savefig(fname); plt.close()

        if calculate_metrics:
            metrics = np.array(metrics) # [n_spectra,n_metrics]
            return metrics
        return None

# SpectraData class ends
#############

#############
# Spectra processing
#############

def locate_tract_patch(wcs, headers, tracts, patches_r, patches_c, ra, dec):
    """ Localize the image patch and r/c coordinates of given world coord.
    """
    for i, tract in enumerate(tracts):
        for j, patch_r in enumerate(patches_r):
            for k, patch_c in enumerate(patches_c):
                l = j * len(patches_c) + k
                if headers[i][l] is None: continue

                num_rows, num_cols = headers[i][l]["NAXIS2"], headers[i][l]["NAXIS1"]
                x, y = wcs[i][l].wcs_world2pix(ra, dec, 0)
                if x >= 0 and y >= 0 and x < num_cols and y < num_rows:
                    return tract, f"{patch_r},{patch_c}", int(y), int(x) #i, l
    return -1,-1,-1,-1

# def is_in_patch(ra, dec, wcs, num_rows, num_cols):
#     x, y = wcs.wcs_world2pix(ra, dec, 0)
#     return x >= 0 and y >= 0 and x < num_cols and y < num_rows

def scale_trans(trans, source_trans):
    nbands = trans.shape[0]
    for i in range(nbands):
        cur_trans, cur_source_trans = trans[i], source_trans[i]
        # if trans sum to 0, cur band is not covered
        trans[i] = trans[i] * np.sum(cur_source_trans) / (np.sum(cur_trans) + 1e-10)

def interpolate_trans(trans_data, spectra_data, bound, sup_bound, fname=None, colors=None):
    """
    Interpolate transmission data based on wave from spectra data.
    Discretization interval for trans data is 10, which is way larger
      than that of spectra_data.
    @Param
      trans_data: [nsmpl_t,1+nbands] (wave/trans)
      spectra_data: [4,nsmpl_s] (wave,flux,ivar,weight)
      bound: defines the wave range within which the spectra is valid
      sup_bound: defines spectra supervision wave range
    @Return
      trans_mask: mask for trans (outside trans wave is 0)
      trans: interpolated transmission value
      band_mask: mask for trans of each band (outside band cover range is 0)
    """
    n = spectra_data.shape[1]
    nbands = trans_data.shape[1] - 1

    source_trans = trans_data[:,1:].T
    trans_wave = trans_data[:,0]
    spectra_wave = spectra_data[0]

    # clip spectra wave to be within transmission wave range
    trans_wave_range = [min(trans_wave), max(trans_wave)]
    (id_lo_old, id_hi_old) = sup_bound
    (id_lo_new, id_hi_new) = get_bound_id(trans_wave_range, spectra_wave)
    id_lo = max(id_lo_old, id_lo_new)
    id_hi = min(id_hi_old, id_hi_new)
    spectra_wave = spectra_wave[id_lo:id_hi+1]

    # interpolate source transmission to same discretization value as spectra wave
    spectra_trans = []
    for cur_source_trans in source_trans:
        f = interp1d(trans_wave, cur_source_trans)
        spectra_trans.append(f(spectra_wave))
    spectra_trans = np.array(spectra_trans) # [nbands,n]

    # pad interpolated transmission to the same length as original spectra data
    trans = np.full((nbands, n), 0).astype(np.float32)
    # new trans: [0,0,...,interpolated trans,0,0]
    trans[:,id_lo:id_hi+1] = spectra_trans

    # normalize new transmission data (sum to same value as before)
    # for i in range(nbands): print(sum(trans[i]))
    scale_trans(trans, source_trans)
    # for i in range(nbands): print(sum(trans[i]))

    # if spectra wave goes beyond trans wave, then we cannot interpolate
    #  for the out of bound spectra wave. we mask with 0
    trans_mask = np.zeros(n)
    # trans mask: [0,0,1(interpolated trans range)1,0,0]
    trans_mask[id_lo:id_hi+1] = 1

    # for each band, we mask wave range not covered by the band with 0
    band_mask = np.zeros((nbands, n))
    for i in range(nbands):
        band_mask[i][trans[i] != 0] = 1

    if fname is not None:
        lo_valid, hi_valid = bound
        plt.plot(spectra_data[0][lo_valid:hi_valid+1],
                 trans_mask[lo_valid:hi_valid+1], label="trans_mask")
        for j in range(nbands):
            plt.plot(spectra_data[0][lo_valid:hi_valid+1],
                     trans[j][lo_valid:hi_valid+1], color=colors[j])
        plt.savefig(fname + "_trans_mask.png")
        plt.close()

        for j in range(nbands):
            plt.plot(spectra_data[0][lo_valid:hi_valid+1],
                     band_mask[j][lo_valid:hi_valid+1], color=colors[j])
        plt.savefig(fname + "_band_mask.png")
        plt.close()

    ret = np.array([trans_mask] + list(trans) + list(band_mask))
    return ret

def wave_based_sort(spectra):
    """ Sort spectra based on wave.
    """
    ids = np.argsort(spectra[0])
    return spectra[:,ids]

def find_valid_spectra_range(spectra):
    invalid = spectra[2] <= 0 # ivar <= 0
    invalid = invalid | np.isnan(spectra[1])
    invalid = invalid | (spectra[1] == np.inf)
    invalid = invalid | (spectra[1] == -np.inf)
    return ~invalid

# def check_invalid_within_valid_range(valid):
#     """
#     Check if the given spectra has invalid observations within valid range
#      ( i.e. whether this happens: ivar [0,0,0,1,1,1,0,0,0,1,1,1,0,0,0] 0s in the middle).
#     @Params
#       valid: [n] Bool array indicating whether each lambda has valid observation
#     @Return
#       lo: start index of valid range (3 in ex above)
#       hi: end index of valid range (11 in ex above)
#       bool: whether ex above happens
#     """
#     ids = np.arange(len(valid))
#     valid_ids = ids[valid]
#     n_invalid_head = valid_ids[0]
#     n_invalid_tail = n - 1 - valid_ids[-1]
#     invalid_exist_within_valid_range = np.sum(valid) < n - n_invalid_head - n_invalid_tail
#     return valid_ids[0], valid_ids[-1], invalid_exist_within_valid_range

def handle_spectra_invalid_range(spectra, spectra_fname, plot):
    """
    Handle range of spectra where  there are no valid observations.
    """
    if plot:
        plt.plot(spectra[0],spectra[1])
        plt.savefig(spectra_fname[:-4] + "_orig.png")
        plt.close()

    n = spectra.shape[1]
    valid = find_valid_spectra_range(spectra)
    # lo, hi, invalid_embedded = check_invalid_within_valid_range(valid)
    # if invalid_embedded:
    #    pass # may want to print and check manually

    # mask nested invalid range
    ids = np.arange(n)
    valid_ids = ids[valid]
    mask = np.zeros(n)
    mask[valid] = 1
    if plot:
        plt.plot(spectra[0],spectra[1])
        plt.plot(spectra[0],mask*np.max(spectra[1]))
        plt.savefig(spectra_fname[:-4] + "_orig_w_mask.png")
        plt.close()

    # drop (instead of mask) head and tail invalid range
    mask = mask[valid_ids[0]:valid_ids[-1]+1]
    spectra = spectra[:,valid_ids[0]:valid_ids[-1]+1]
    if plot:
        plt.plot(spectra[0],spectra[1])
        plt.plot(spectra[0], mask*np.max(spectra[1]))
        plt.savefig(spectra_fname[:-4] + "_orig_cut_two_ends_w_mask.png")
        plt.close()
    return spectra, mask

def resample_spectra(spectra, upsample_scale):
    """
    Resample spectra to be regularly sampled.
    We upsample the input spectra to higher frequency first
      and then downsample back to the (close to) original frequency.
    """
    freq = np.mean(spectra[0,1:] - spectra[0,:-1])
    n = len(spectra[0])
    new_freq = freq * upsample_scale

    # upsample first
    upsampled_wave = np.linspace(spectra[0,0], spectra[0,-1], n * upsample_scale)
    assert spectra[0,0] == upsampled_wave[0]
    assert spectra[0,-1] == upsampled_wave[-1]
    upsampled_flux = np.interp(upsampled_wave, spectra[0], spectra[1])
    upsampled_ivar = np.interp(upsampled_wave, spectra[0], spectra[2])
    upsampled_spectra = np.stack([upsampled_wave, upsampled_flux, upsampled_ivar])

    # downsample to same length as before upsample (similar freq)
    # downsampled_flux = resample(upsampled_flux, n)
    # downsampled_ivar = resample(upsampled_ivar, n)
    # downsampled_wave = np.linspace(spectra[0,0], spectra[0,-1], n)
    # resampled_spectra = np.stack([downsampled_wave, downsampled_flux, downsampled_ivar])
    ids = np.arange(0, n * upsample_scale, upsample_scale)
    resampled_spectra = upsampled_spectra[:,ids]

    freq = resampled_spectra[0,1:] - resampled_spectra[0,:-1]
    mean_freq = np.mean(freq)
    assert (freq - mean_freq < 1e-8).all()
    assert spectra.shape == resampled_spectra.shape

    # plt.plot(spectra[0], spectra[1])
    # plt.savefig('tmp1.png')
    # plt.close()
    # plt.plot(resampled_spectra[0], resampled_spectra[1])
    # plt.savefig('tmp2.png')
    # plt.close()
    # assert 0

    return resampled_spectra

# def create_spectra_mask(spectra, max_spectra_len):
#     """
#     Mask out invalid and padded region of spectra.
#     """
#     m, n = spectra.shape
#     if n == max_spectra_len: mask = np.ones(max_spectra_len).astype(bool)
#     else: mask = np.zeros(max_spectra_len).astype(bool)
#     return mask

def pad_spectra(spectra, mask, max_len):
    """
    Pad spectra if shorter than max_len and update mask to 1 for un-padded region.
    """
    m, n = spectra.shape
    offset = max_len - n
    ret = np.full((m,max_len),-1).astype(spectra.dtype)
    lo, hi = offset//2, offset//2+n-1
    mask[lo:hi+1] = 1
    ret[:,lo:hi+1] = spectra
    return ret, mask, (lo, hi)

def clean_flux(spectra, mask):
    """ Replace `inf` `-inf` `nan` in flux with 0 and mask out.
    """
    ids = np.isnan(spectra[1])
    ids = ids | (spectra[1] == np.inf)
    ids = ids | (spectra[1] == -np.inf)
    mask[ids] = 0
    spectra[1][ids] = 0
    return spectra, mask

def convolve_spectra(spectra, bound, std=5, border=True, process_ivar=False):
    """ Smooth gt spectra flux and ivar with given std.
        @Param
          bound: defines range to convolve within
          border: if True, we add 1 padding at two ends when convolving
    """
    if std <= 0: return spectra

    lo, hi = bound
    n = hi - lo + 1

    discret_val = np.mean(np.diff(spectra[0][lo:hi+1])) # get discretization value of lambda
    std = std / discret_val
    kernel = Gaussian1DKernel(stddev=std)
    if border:
        nume = convolve(spectra[1][lo:hi+1], kernel) # flux
        denom = convolve(np.ones(n), kernel)
        spectra[1][lo:hi+1] = nume / denom
    else:
        spectra[1][lo:hi+1] = convolve(spectra[1][lo:hi+1], kernel)

    if process_ivar:
        mask = spectra[2][lo:hi+1] != 0
        if sum(mask) != 0:
            conved = 1/convolve(1/spectra[2][lo:hi+1][mask], kernel)
            if border:
                denom = convolve(np.ones(sum(mask)), kernel)
                spectra[2][lo:hi+1][mask] = conved / denom
            else: spectra[2][lo:hi+1][mask] = conved

    return spectra

def mask_spectra_range(spectra, mask, bound, trans_range, supervision_wave_range):
    """ Mask out spectra data beyond given wave range.
        @Param
          spectra: spectra data [3,nsmpl] (wave,flux,ivar)
          bound: defines range of valid spectra
          mask: mask to be updated [nsmpl]
          trans_range: transmission data wave range
          supervision_wave_range: spectra supervision wave range
    """
    (id_lo_old, id_hi_old) = bound

    m, n = spectra.shape
    lo1, hi1 = trans_range
    lo2, hi2 = supervision_wave_range
    wave_range = (max(lo1,lo2), min(hi1,hi2))
    (id_lo_new, id_hi_new) = get_bound_id(wave_range, spectra[0])

    id_lo = max(id_lo_old, id_lo_new)
    id_hi = min(id_hi_old, id_hi_new)

    new_mask = np.zeros(n).astype(bool)
    new_mask[id_lo:id_hi+1] = 1
    sup_mask = copy.deepcopy(mask)
    sup_mask &= new_mask
    sup_bound = (id_lo, id_hi)
    return spectra, mask, sup_mask, sup_bound

def normalize_spectra(spectra, bound, process_ivar=False):
    """ Normalize flux to be in 0-1 within supervision range (defined by bound).
    """
    (id_lo, id_hi) = bound
    flux = spectra[1][id_lo:id_hi+1]
    lo, hi = min(flux), max(flux)
    spectra[1] = (spectra[1] - lo) / (hi - lo)
    # print(np.min(spectra[2]), np.max(spectra[2]))
    if process_ivar:
        mask = spectra[2] != 0 # ivar = 0 where err is infty
        # spectra[2][mask] = (hi - lo)**2 / ( 1/spectra[2][mask] - lo)
        spectra[2][mask] = (hi - lo)**2 * spectra[2][mask]
    # print(np.min(spectra[2]), np.max(spectra[2]))
    # assert min(spectra[2]) >= 0
    return spectra

def get_wave_weight(spectra, redshift, emitted_wave_distrib, bound):
    """ Get sampling weight for spectra wave (in unmasked range).
    """
    (lo, hi) = bound
    n = spectra.shape[1]
    obs_wave = spectra[0][lo:hi+1]
    weight = np.zeros(n)
    emitted_wave = obs_wave / (1 + redshift)
    bound_weight = 1 / (emitted_wave_distrib(emitted_wave) + 1e-10)
    weight[lo:hi+1] = bound_weight
    weight = weight / max(weight)
    return weight

def process_gt_spectra(
        spectra, spectra_fname, spectra_mask_fname,
        spectra_sup_bound_fname, redshift, emitted_wave_distrib,
        sigma=-1, upsample_scale=10, trans_range=None, save=True, plot=True,
        colors=None, trans_data=None, supervision_wave_range=None, max_spectra_len=-1,
        validator=None, process_ivar=True
):
    """ Load gt spectra wave and flux for spectra supervision and
          spectrum plotting. Also smooth the gt spectra.
        Note, the gt spectra has significantly larger discretization values than
          the transmission data.

        @Param
          infname: filename of np array that stores the gt spectra data.
          spectra_fname: output filename to store processed gt spectra (wave & flux)
          spectra_mask_fname: output filename to store processed gt spectra (wave & flux)
          emitted_wave_distrib: histogram distribution of emitted wave (interpolated function)
        @Return
          spectra:  spectra data [5+2*nbands,nsmpl]
                    (wave/flux/ivar/weight/trans_mask/trans(nbands)/band_mask(nbands))
          mask:     mask out bad flux values
    """
    assert spectra.shape[1] <= max_spectra_len
    spectra = wave_based_sort(spectra)
    raise NotImplementedError()

    print(spectra.shape)
    spectra, mask = handle_spectra_invalid_range(spectra, spectra_fname, plot)
    # mask = create_spectra_mask(spectra, max_spectra_len)
    print(spectra.shape)
    spectra = resample_spectra(spectra, upsample_scale)
    # spectra, mask, bound = pad_spectra(spectra, mask, max_spectra_len)
    # spectra, mask = clean_flux(spectra, mask)
    spectra = convolve_spectra(spectra, bound, std=sigma, process_ivar=process_ivar)
    spectra, mask, sup_mask, sup_bound = mask_spectra_range(
        spectra, mask, bound, trans_range, supervision_wave_range)
    spectra = normalize_spectra(spectra, sup_bound, process_ivar=process_ivar)

    spectra, mask, bound = pad_spectra(spectra, mask, max_spectra_len)
    spectra = spectra.astype(np.float32)

    weight = get_wave_weight(spectra, redshift, emitted_wave_distrib, sup_bound)
    spectra = np.concatenate((spectra, weight[None,:]), axis=0)

    interp_trans_data = interpolate_trans(
        trans_data, spectra, bound, sup_bound, fname=spectra_fname[:-4], colors=colors)
    spectra = np.concatenate((spectra, interp_trans_data), axis=0)

    if save:
        np.save(spectra_fname, spectra)
        np.save(spectra_mask_fname, sup_mask)
        np.save(spectra_sup_bound_fname, sup_bound)

    if plot:
        # mask defines the valid range of values of the current spectra_fname
        # sup_mask defines the range we used for supervision
        plt.plot(spectra[0,mask], spectra[1,mask])
        plt.savefig(spectra_fname[:-4] + ".png")
        plt.close()

        plt.plot(spectra[0,mask], sup_mask[mask])
        plt.savefig(spectra_fname[:-4] + "_mask.png")
        plt.close()

    if validator is not None and not validator(spectra_data):
        return None, None
    return spectra, sup_mask, sup_bound

def overlay_spectrum(gt_fn, gen_wave, gen_spectra):
    gt = np.load(gt_fn)
    gt_wave, gt_spectra = gt[:,0], gt[:,1]
    gt_spectra = convolve_spectra(gt_spectra)

    gen_lo_id = np.argmax(gen_wave>gt_wave[0]) + 1
    gen_hi_id = np.argmin(gen_wave<gt_wave[-1])

    wave = gen_wave[gen_lo_id:gen_hi_id]
    gen_spectra = gen_spectra[gen_lo_id:gen_hi_id]
    f = interpolate.interp1d(gt_wave, gt_spectra)
    gt_spectra_intp = f(wave)
    return wave, gt_spectra_intp, gen_spectra

#############
# Spectra loading
#############

def read_deimos_table(fname, format, download=False, link="", path=""):
    """ Read metadata table for deimos spectra data.
    """
    df = pandas.read_table(fname, comment='#', delim_whitespace=True)
    # below commented code are used to read v1 deimos table
    # replace_cols = {"ID": "id"}
    # if format == "fits":  replace_cols["fits1d"] = "spectra_fname"
    # elif format == "tbl": replace_cols["ascii1d"] = "spectra_fname"
    # else: raise ValueError(f"invalid spectra data format: {format}")
    # df.rename(columns=replace_cols, inplace=True)
    # df.drop(columns=['sel', 'imag', 'kmag', 'Qf', 'Q',
    #                  'Remarks', 'jpg1d', 'fits2d'], inplace=True)
    # df.drop([0], inplace=True) # drop first row which is datatype
    # df.dropna(subset=["ra","dec","zspec","spectra_fname"], inplace=True)
    # df.loc[df['id'].isnull(),'id'] = 'None'
    # df.reset_index(inplace=True) # reset index after dropping
    # df.drop(columns=["index"], inplace=True)
    df["source"] = "deimos"
    df["ra"] = pandas.to_numeric(df["ra"])
    df["dec"] = pandas.to_numeric(df["dec"])
    df["zspec"] = pandas.to_numeric(df["zspec"])
    df.rename(columns={"fits1d":"spectra_fname_fits"}, inplace=True)
    col_name = f"spectra_fname_{format}"
    df.rename(columns={"spectra_fname":col_name}, inplace=True)
    if download:
        spectra_fnames = list(df[col_name])
        download_data_parallel(link, path, spectra_fnames)
        log.info("deimos source spectra data download complete")
    return df

def read_zcosmos_table(fname, format, download=False, link="", path=""):
    """ Read metadata table for zcosmos spectra data.
    """
    df = Table.read(fname).to_pandas()
    # data = fits.open(fname)[1].data
    # df = pandas.DataFrame(data)
    df.rename(columns={
        'OBJECT_ID': 'id',
        'RAJ2000':'ra',
        'DEJ2000':'dec',
        'REDSHIFT':'zspec',
        'FILENAME':'spectra_fname'
    }, inplace=True)
    df.drop(columns=['CC','IMAG_AB','FLAG_S','FLAG_X','FLAG_R','FLAG_UV'], inplace=True)
    df.dropna(subset=["ra","dec","zspec","spectra_fname"], inplace=True)
    df.reset_index(inplace=True, drop=True) # reset index after dropping
    df["ra"] = pandas.to_numeric(df["ra"])
    df["dec"] = pandas.to_numeric(df["dec"])
    df["zspec"] = pandas.to_numeric(df["zspec"])
    df["spectra_fname"] = df["spectra_fname"].str.decode("utf-8")
    df.rename(columns={"spectra_fname":"spectra_fname_fits"}, inplace=True)
    # df["spectra_fname_tbl"] = df["spectra_fname_fits"].str.replace(
    #     "zCOSMOS_BRIGHT_DR3", "sc").str.replace("fits","tbl")
    df["spectra_fname_tbl"] = df["spectra_fname_fits"].str.replace("fits","tbl")
    if download:
        col_name = f"spectra_fname_{format}"
        spectra_fnames = list(df[col_name])
        download_data_parallel(link, path, spectra_fnames)
        log.info("zcosmos source spectra data download complete")
    df["source"] = "zcosmos"
    return df

def read_deep3_table(fname, format):
    with open(fname, "rb") as fp:
        df = pickle.load(fp)
    df.dropna(subset=["zspec"], inplace=True)
    df.reset_index(inplace=True, drop=True)
    df["source"] = "deep3"
    return df

def read_manual_table(fname):
    data, colnames, datatypes = {}, None, None
    with open(fname) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                colnames = row
                for entry in row:
                    data[entry] = []
            elif line_count == 1:
                datatypes = row
            else:
                for colname, entry in zip(colnames, row):
                    data[colname].append(entry)
            line_count += 1
    for dtype, colname in zip(datatypes, colnames):
        data[colname] = np.array(data[colname]).astype(dtype)
    return data

def unpack_gt_spectra(fname, format="tbl", source="deimos", has_ivar=True, deep3_portion=None):
    if format == "tbl":
        if source == "deimos":
            data = pandas.read_table(fname, comment="#", delim_whitespace=True)
        elif source == "zcosmos":
            skip = list(np.arange(23))
            data = pandas.read_table(fname, skiprows=skip, header=None, delim_whitespace=True)
        else: raise ValueError()
        data = np.array(data)
        wave, flux = data[:,0], data[:,1]
        if has_ivar: ivar = data[:,2]
        else: ivar = np.ones(wave.shape)
    elif format == "fits":
        if source == "deep3":
            """
            header["EXTNAME"]
            1: Bxspf-B; 2: Bxspf-R; 3: Horne-B; 4: Horne-R
            """
            hdu_id = { "Bxspf-B":1, "Bxspf-R":2 }[deep3_portion]
            hdu = fits.open(fname)[hdu_id].data
            wave, flux, ivar = hdu["LAMBDA"][0], hdu["SPEC"][0], hdu["IVAR"][0]
        else:
            hdu = fits.open(fname)[1]
            header = hdu.header
            data = hdu.data[0]
            data_names = [header["TTYPE1"],header["TTYPE2"],header["TTYPE3"]]
            wave_id = data_names.index("LAMBDA")
            flux_id = data_names.index("FLUX")
            ivar_id = data_names.index("IVAR")
            wave, flux = data[wave_id], data[flux_id]
            if has_ivar: ivar = data[ivar_id]
            else: ivar = np.ones(wave.shape)
    else:
        raise ValueError(f"invalid spectra data format: {format}")
    spectra = np.array([wave, flux, ivar])
    return spectra

def download_data_parallel(http_prefix, local_path, fnames):
    fnames = list(filter(lambda fname: not exists(join(local_path, fname)), fnames))
    log.info(f"downloading {len(fnames)} spectra")

    urls = [f"{http_prefix}/{fname}" for fname in fnames]
    out_fnames = [f"{local_path}/{fname}" for fname in fnames]
    inputs = zip(urls, out_fnames)
    cpus = cpu_count()
    print(urls[0])
    assert 0
    results = ThreadPool(cpus - 1).imap_unordered(download_from_url, inputs)
    for result in results:
        if result is not None:
            # print('url:', result[0], 'time (s):', result[1])
            log.info(f"url: {result[0]}")

def download_from_url(input_):
    t0 = time.time()
    (url, fname) = input_
    try:
        r = requests.get(url)
        with open(fname, 'wb') as f:
            f.write(r.content)
        return(url, time.time() - t0)
    except Exception as e:
        print('Exception in download_url():', e)
        assert 0
        return(url)
