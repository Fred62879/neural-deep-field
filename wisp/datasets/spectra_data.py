
import csv
import time
import torch
import pandas
import pickle
import requests
import numpy as np
import logging as log
import matplotlib.pyplot as plt

from wisp.datasets.patch_data import PatchData
from wisp.utils.common import create_patch_uid
from wisp.utils.numerical import normalize_coords
from wisp.datasets.data_utils import add_dummy_dim, \
    set_input_path, patch_exists, get_bound_id, \
    clip_data_to_ref_wave_range

from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from functools import partial
from os.path import join, exists
from collections import defaultdict
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from astropy.convolution import convolve, Gaussian1DKernel
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool


class SpectraData:
    # def __init__(self, fits_obj, trans_obj, dataset_path, device, **kwargs):
    def __init__(self, trans_obj, device, **kwargs):
        self.kwargs = kwargs
        self.summarize_tasks(kwargs["tasks"])

        self.device = device
        # self.fits_obj = fits_obj
        self.trans_obj = trans_obj
        self.dataset_path = kwargs["dataset_path"]

        self.num_bands = kwargs["num_bands"]
        self.space_dim = kwargs["space_dim"]
        self.all_tracts = kwargs["spectra_tracts"]
        self.all_patches_r = kwargs["spectra_patches_r"]
        self.all_patches_c = kwargs["spectra_patches_c"]
        self.num_tracts = len(self.all_tracts)
        self.num_patches_c = len(self.all_patches_c)
        self.num_patches = len(self.all_patches_r)*len(self.all_patches_c)

        self.smooth_sigma = kwargs["spectra_smooth_sigma"]
        self.neighbour_size = kwargs["spectra_neighbour_size"]
        self.download_source_spectra = kwargs["download_source_spectra"]
        self.source_spectra_link = kwargs["source_spectra_link"]
        self.spectra_data_source = kwargs["spectra_data_source"]
        self.spectra_data_format = kwargs["spectra_data_format"] # tbl or fits
        self.wave_discretz_interval = kwargs["trans_sample_interval"]
        self.trusted_wave_range = None if not kwargs["trusted_range_only"] \
            else [kwargs["spectra_supervision_wave_lo"],
                  kwargs["spectra_supervision_wave_hi"]]

        self.load_spectra_data_from_cache = kwargs["load_spectra_data_from_cache"]

        self.set_path(self.dataset_path)
        self.load_accessory_data()
        self.load_spectra()

    def summarize_tasks(self, tasks):
        tasks = set(tasks)

        self.recon_gt_spectra = "recon_gt_spectra" in tasks or \
            "recon_gt_spectra_during_train" in tasks
        self.recon_dummy_spectra = "recon_dummy_spectra" in tasks or \
            "recon_dummy_spectra_during_train" in tasks
        self.recon_codebook_spectra = "recon_codebook_spectra" in tasks or \
            "recon_codebook_spectra_during_train" in tasks

        self.codebook_pretrain = self.kwargs["pretrain_codebook"] and \
            "codebook_pretrain" in tasks
        self.pretrain_infer = self.kwargs["pretrain_codebook"] and \
            "pretrain_infer" in tasks
        self.require_spectra_coords = self.kwargs["mark_spectra"] and \
            ("plot_redshift" in tasks or "plot_embed_map" in tasks)
        self.recon_spectra = self.recon_gt_spectra or self.recon_dummy_spectra or \
            self.recon_codebook_spectra
        self.spectra_supervision_train = "train" in tasks and self.kwargs["spectra_supervision"]
        self.spectra_valid_train = self.kwargs["train_spectra_pixels_only"] and "train" in tasks
        self.spectra_valid_infer = self.kwargs["train_spectra_pixels_only"] and "infer" in tasks

        # self.kwargs["space_dim"] == 3 and (self.codebook_pretrain or self.pretrain_infer or self.spectra_supervision_train or self.spectra_validation or self.recon_spectra or self.require_spectra_coords)

    def set_path(self, dataset_path):
        """ Create path and filename of required files.
            source_metadata_table:    ra,dec,zspec,spectra_fname
            processed_metadata_table: added pixel val and tract-patch
        """
        paths = []
        self.input_patch_path, img_data_path = set_input_path(
            dataset_path, self.kwargs["sensor_collection_name"])
        spectra_path = join(dataset_path, "input/spectra")

        if self.spectra_data_source == "manual":
            assert 0
            self.source_metadata_table_fname = join(spectra_path, "deimos_old", "spectra.csv")

        elif self.spectra_data_source == "deimos":
            path = join(spectra_path, "deimos")
            self.source_spectra_path = join(path, f"source_spectra_{self.spectra_data_format}")
            self.source_metadata_table_fname = join(path, "source_deimos_table.tbl")
            processed_data_path = join(path, "processed_"+self.kwargs["processed_spectra_cho"])
            self.processed_spectra_path = join(processed_data_path, "processed_spectra")
            self.processed_metadata_table_fname = join(
                processed_data_path, "processed_deimos_table.tbl")

        elif self.spectra_data_source == "zcosmos":
            path = join(spectra_path, "zcosmos")
            self.source_spectra_path = join(path, f"source_spectra_{self.spectra_data_format}")
            self.source_metadata_table_fname = join(path, "source_zcosmos_table.fits")
            processed_data_path = join(path, "processed_"+self.kwargs["processed_spectra_cho"])
            self.processed_spectra_path = join(processed_data_path, "processed_spectra")
            self.processed_metadata_table_fname = join(
                processed_data_path, "processed_zcosmos_table.fits")

        else: raise ValueError("Unsupported spectra data source choice.")

        self.coords_range_fname = join(img_data_path, self.kwargs["coords_range_fname"])
        self.gt_spectra_ids_fname = join(processed_data_path, "gt_spectra_ids.txt")
        self.gt_spectra_wave_fname = join(processed_data_path, "gt_spectra_wave.npy")
        self.gt_spectra_fluxes_fname = join(processed_data_path, "gt_spectra_fluxes.npy")
        self.gt_spectra_pixels_fname = join(processed_data_path, "gt_spectra_pixels.npy")
        self.gt_spectra_redshift_fname = join(processed_data_path, "gt_spectra_redshift.npy")
        self.gt_spectra_img_coords_fname = join(processed_data_path, "gt_spectra_img_coords.npy")
        self.gt_spectra_world_coords_fname = join(processed_data_path, "gt_spectra_world_coords.npy")

        for path in [processed_data_path, self.processed_spectra_path]:
            Path(path).mkdir(parents=True, exist_ok=True)

    def load_accessory_data(self):
        self.full_wave = self.trans_obj.get_full_wave()

    def load_spectra(self):
        """ Load gt and/or dummy spectra data.
        """
        self.data = defaultdict(lambda: [])

        self.load_gt_spectra_data()

        if self.recon_dummy_spectra:
            self.load_dummy_spectra_data()

        self.load_spectrum_plotting_data()

        # if self.recon_gt_spectra:
        #     self.mark_spectra_on_img()

    #############
    # Getters
    #############

    def get_processed_spectra_path(self):
        return self.processed_spectra_path

    def get_full_wave(self):
        return self.full_wave

    #def get_num_spectra_to_plot(self):
    #    return len(self.data["spectra_grid_coords"])

    def get_num_gt_spectra(self):
        """ Get #gt spectra (doesn't count neighbours). """
        return self.num_gt_spectra

    def get_num_supervision_spectra(self):
        """ Get #supervision spectra (doesn't count neighbours). """
        return self.num_supervision_spectra

    def get_num_validation_spectra(self):
        """ Get #validation spectra (doesn't count neighbours). """
        return self.num_validation_spectra

    # def get_spectra_grid_coords(self):
    #     """ Get grid (training) coords of all selected spectra (gt & dummy, incl. neighbours).
    #     """
    #     return self.data["spectra_grid_coords"]

    # def get_spectra_img_coords(self):
    #     """ Get image coords of all selected spectra (gt & dummy, incl. neighbours). """
    #     return self.data["spectra_img_coords"]

    # def get_num_spectra_coords(self):
    #     """ Get number of coords of all selected spectra
    #         (gt & dummy, incl. neighbours).
    #     """
    #     return self.get_spectra_grid_coords().shape[0]

    # def get_spectra_coord_ids(self):
    #     """ Get pixel id of all selected spectra (correspond to coords). """
    #     return self.data["spectra_coord_ids"]

    # def get_spectra_recon_wave_bound_ids(self):
    #     """ Get ids of boundary lambda of recon spectra. """
    #     return self.data["spectra_recon_wave_bound_ids"]

    # def get_recon_spectra_wave(self):
    #     """ Get lambda values (for plotting). """
    #     return self.data["recon_wave"]

    def get_gt_spectra_ids(self):
        return self.data["gt_spectra_ids"]

    def get_gt_spectra_wave(self):
        """ Get gt spectra wave (all gt spectra are clipped to the same wave range).
        """
        return self.data["gt_spectra_wave"]

    def get_gt_spectra_fluxes(self):
        """ Get gt spectra flux (trusted range only) for plotting. """
        return self.data["gt_spectra_fluxes"]

    def get_gt_spectra_pixels(self):
        return self.data["gt_spectra_pixels"]

    def get_gt_spectra_redshift(self):
        return self.data["gt_spectra_redshift"]

    def get_gt_spectra_img_coords(self):
        return self.data["gt_spectra_img_coords"]

    def get_supervision_wave_bound_ids(self):
        """ Get ids of boundary lambda of spectra supervision. """
        return self.data["supervision_wave_bound_ids"]

    def get_supervision_fluxes(self, idx=None):
        """ Get gt spectra (with same wave range as recon) used for supervision. """
        if idx is None:
            return self.data["supervision_fluxes"]
        return self.data["supervision_fluxes"][idx]

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

    def get_validation_spectra_ids(self, patch_uid=None):
        """ Get id of validation spectra in given patch.
            Id here is in context of all validation spectra
        """
        if patch_uid is not None:
            return self.data["validation_patch_ids"][patch_uid]
        return self.data["validation_patch_ids"]

    def get_validation_img_coords(self, idx=None):
        if idx is not None:
            return self.data["validation_img_coords"][idx]
        return self.data["validation_img_coords"]

    def get_validation_norm_world_coords(self, idx=None):
        if idx is not None:
            return self.data["validation_norm_world_coords"][idx]
        return self.data["validation_norm_world_coords"]

    def get_validation_fluxes(self, idx=None):
        if idx is not None:
            return self.data["validation_fluxes"][idx]
        return self.data["validation_fluxes"]

    def get_validation_pixels(self, idx=None):
        if idx is not None:
            return self.data["validation_pixels"][idx]
        return self.data["validation_pixels"]

    def get_semi_supervision_redshift(self):
        return self.data["semi_supervision_redshift"]

    #############
    # Helpers
    #############

    def load_spectrum_plotting_data(self):

        # if self.kwargs["plot_clipped_spectrum"]:
        #     recon_spectra_wave_bound = [
        #         self.kwargs["spectrum_plot_wave_lo"],
        #         self.kwargs["spectrum_plot_wave_hi"]]
        # else: recon_spectra_wave_bound = [self.full_wave[0], self.full_wave[-1]]

        # (id_lo, id_hi) = get_bound_id(
        #     recon_spectra_wave_bound, self.full_wave, within_bound=False)

        # self.data["spectrum_recon_wave"] = np.arange(
        #     self.full_wave[id_lo], self.full_wave[id_hi] + 1, self.wave_discretz_interval)
        # self.data["spectrum_recon_wave_bound_ids"] = [id_lo, id_hi + 1]
        pass

        # wave = []
        # if self.recon_gt_spectra: # and self.kwargs["plot_spectrum_with_gt"]):
        #     wave.extend(self.data["gt_recon_wave"])

        # if self.recon_dummy_spectra:
        #     wave.extend(self.data["dummy_recon_wave"])

        # self.data["recon_wave"] = wave

        # # get all spectra (gt and dummy) (grid and img) coords for inferrence
        # ids, grid_coords, img_coords = [], [], []

        # if self.recon_gt_spectra or self.spectra_supervision_train or self.require_spectra_coords:
        #     ids.extend(self.data["gt_spectra_coord_ids"])
        #     grid_coords.extend(self.data["gt_spectra_grid_coords"])
        #     img_coords.extend(self.data["gt_spectra_img_coords"])

        # if self.recon_dummy_spectra:
        #     grid_coords.extend(self.data["dummy_spectra_grid_coords"])

        # if len(ids) != 0:         self.data["spectra_coord_ids"] = np.array(ids)
        # if len(img_coords) != 0:  self.data["spectra_img_coords"] = np.array(img_coords)
        # if len(grid_coords) != 0: self.data["spectra_grid_coords"] = torch.stack(grid_coords)

    def load_dummy_spectra_data(self):
        """ Load hardcoded spectra positions for pixels without gt spectra.
            Can be used to compare with codebook spectrum.
        """
        self.data["dummy_spectra_grid_coords"] = torch.stack([
            torch.FloatTensor([0.0159,0.0159]),
            torch.FloatTensor([-1,1]),
            torch.FloatTensor([-0.9,0.12]),
            torch.FloatTensor([0.11,0.5]),
            torch.FloatTensor([0.7,-0.2]),
            torch.FloatTensor([0.45,-0.9]),
            torch.FloatTensor([1,1])])

        n = len( self.data["dummy_spectra_grid_coords"])

        self.data["dummy_spectra_grid_coords"] = add_dummy_dim(
            self.data["dummy_spectra_grid_coords"], **self.kwargs)

        recon_spectra_wave_bound = self.kwargs["dummy_spectra_clip_range"]
        id_lo, id_hi = get_bound_id(
            recon_spectra_wave_bound, self.full_wave, within_bound=False)
        self.data["spectra_recon_wave_bound_ids"].extend([[id_lo, id_hi + 1]] * n)
        self.data["dummy_recon_wave"].extend(
            [np.arange(self.full_wave[id_lo], self.full_wave[id_hi] + 1,
                       self.kwargs["trans_sample_interval"])] * n
        )

    def load_gt_spectra_data(self):
        """ Load gt spectra data.
            The full loading workflow is:
              i) we first load source metadata table and process each spectra (
                 fluxes, wave, coords, pixels) and save processed data individually.
              ii) then we load these saved spectra and further process to gather
                  them together and save all data together.
              iii) finally we do necessary transformations.
        """
        self.find_full_wave_bound_ids()
        self.coords_range = np.load(self.coords_range_fname + ".npy")

        if not self.load_spectra_data_from_cache or \
           not exists(self.gt_spectra_ids_fname) or \
           not exists(self.gt_spectra_fluxes_fname) or \
           not exists(self.gt_spectra_pixels_fname) or \
           not exists(self.gt_spectra_redshift_fname) or \
           not exists(self.gt_spectra_img_coords_fname) or \
           not exists(self.gt_spectra_world_coords_fname):

            # process and save data for each spectra individually
            if not exists(self.gt_spectra_ids_fname) or \
               not exists(self.processed_metadata_table_fname):
                self.process_spectra()

            # load data for each individual spectra and save together
            self.gather_processed_spectra()
            self.process_save_gt_spectra_data()

        else:
            # data for all spectra are saved together (small amount of spectra)
            self.load_cached_spectra_data()

        self.num_gt_spectra = len(self.data["gt_spectra_fluxes"])
        self.train_valid_split()
        self.transform_data()

    def transform_data(self):
        self.to_tensor([
            "gt_spectra_fluxes",
            "gt_spectra_pixels",
            "gt_spectra_redshift",
            "gt_spectra_world_coords"
        ])

    #############
    # Loading helpers
    #############

    def to_tensor(self, fields):
        for field in fields:
            self.data[field] = torch.FloatTensor(self.data[field])

    def split_spectra(self):
        ids = np.arange(self.num_gt_spectra)
        # n = min(self.kwargs["num_supervision_spectra"], self.num_gt_spectra)
        # np.random.shuffle(ids)
        # self.num_supervision_spectra = n
        # self.num_validation_spectra = self.num_gt_spectra - n
        # supervision_ids = ids[:n]
        # validation_ids = ids[n:]

        acc, validation_ids, validation_patch_ids = 0, [], {}
        for tract, patch in zip(self.kwargs["tracts"], self.kwargs["patches"]):
            patch_uid = create_patch_uid(tract, patch)
            cur_spectra_ids = self.data["gt_spectra_ids"][patch_uid]
            validation_ids.extend(cur_spectra_ids)
            validation_patch_ids[patch_uid] = np.arange(acc, acc+len(cur_spectra_ids))
            acc += len(cur_spectra_ids)
        validation_ids = np.array(validation_ids)

        # get supervision ids
        supervision_ids = np.array(list(set(ids) - set(validation_ids))).astype(int)
        np.random.shuffle(supervision_ids)
        supervision_ids = supervision_ids[:self.kwargs["num_supervision_spectra"]]

        self.num_validation_spectra = len(validation_ids)
        self.num_supervision_spectra = len(supervision_ids)
        self.data["validation_patch_ids"] = validation_patch_ids
        return supervision_ids, validation_ids

    def train_valid_split(self):
        sup_ids, val_ids = self.split_spectra()
        log.info(f"spectra train/valid {len(sup_ids)}/{len(val_ids)}")

        # supervision spectra data (used during pretrain)
        self.data["supervision_fluxes"] = self.data["gt_spectra_fluxes"][sup_ids]
        self.data["supervision_redshift"] = self.data["gt_spectra_redshift"][sup_ids]
        if self.kwargs["codebook_pretrain_pixel_supervision"]:
            self.data["supervision_pixels"] = self.data["gt_spectra_pixels"][sup_ids]

        # valiation(and semi sup) spectra data (used during main train)
        self.data["validation_fluxes"] = self.data["gt_spectra_fluxes"][val_ids]
        self.data["validation_pixels"] = self.data["gt_spectra_pixels"][val_ids]
        self.data["validation_img_coords"] = self.data["gt_spectra_img_coords"][val_ids]
        self.data["validation_world_coords"] = self.data["gt_spectra_world_coords"][val_ids]
        self.data["validation_norm_world_coords"] = self.data["gt_spectra_norm_world_coords"][val_ids]
        if self.kwargs["redshift_semi_supervision"]:
            self.data["semi_supervision_redshift"] = self.data["gt_spectra_redshift"][val_ids]

    def load_cached_spectra_data(self):
        """ Load spectra data (which are saved together).
        """
        with open(self.gt_spectra_ids_fname, "rb") as fp:
            self.data["gt_spectra_ids"] = defaultdict(list, pickle.load(fp))
        self.data["gt_spectra_wave"] = np.load(self.gt_spectra_wave_fname)
        self.data["gt_spectra_fluxes"] = np.load(self.gt_spectra_fluxes_fname)
        self.data["gt_spectra_pixels"] = np.load(self.gt_spectra_pixels_fname)
        self.data["gt_spectra_redshift"] = np.load(self.gt_spectra_redshift_fname)
        self.data["gt_spectra_img_coords"] = np.load(self.gt_spectra_img_coords_fname)
        self.data["gt_spectra_world_coords"] = np.load(self.gt_spectra_world_coords_fname)
        norm_coords, _ = normalize_coords(
            self.data["gt_spectra_world_coords"], coords_range=self.coords_range)
        self.data["gt_spectra_norm_world_coords"] = norm_coords

    def process_save_gt_spectra_data(self):
        """ Clip supervision spectra to trusted range.
        """
        # clip all spectra to trusted range
        clipped_wave, clipped_flux = None, []
        for (wave, flux) in self.data["gt_spectra"]:
            clipped_wave, (id_lo, id_hi) = clip_data_to_ref_wave_range(
                wave, wave, wave_range=self.data["supervision_spectra_wave_bound"])
            # (id_lo, id_hi) = get_bound_id(
            #     self.data["supervision_spectra_wave_bound"], wave, within_bound=True)
            # if clipped_wave is None:
            #     clipped_wave = wave[id_lo:id_hi+1]
            clipped_flux.append(flux[id_lo:id_hi+1][None,:])

        clipped_flux = np.concatenate(clipped_flux, axis=0) # [n,nsmpl]
        self.data["gt_spectra_wave"] = clipped_wave.astype(np.float32)
        self.data["gt_spectra_fluxes"] = clipped_flux.astype(np.float32)

        # save data for all spectra together
        np.save(self.gt_spectra_wave_fname, self.data["gt_spectra_wave"])
        np.save(self.gt_spectra_fluxes_fname, self.data["gt_spectra_fluxes"])
        np.save(self.gt_spectra_pixels_fname, self.data["gt_spectra_pixels"])
        np.save(self.gt_spectra_redshift_fname, self.data["gt_spectra_redshift"])
        np.save(self.gt_spectra_img_coords_fname, self.data["gt_spectra_img_coords"])
        np.save(self.gt_spectra_world_coords_fname, self.data["gt_spectra_world_coords"])

    def gather_processed_spectra(self):
        """ Load processed data for each spectra and save together.
        """
        df = pandas.read_pickle(self.processed_metadata_table_fname)
        with open(self.gt_spectra_ids_fname, "rb") as fp:
            self.data["gt_spectra_ids"] = defaultdict(list, pickle.load(fp))

        redshift, pixels, img_coords, world_coords, spectra = [], [], [], [], []

        n = len(df)
        for i in range(n):
            redshift.append(df.iloc[i]["zspec"])
            pixels.append(
                np.load(join(self.processed_spectra_path, df.iloc[i]["pix_fname"])))
            spectra.append(
                np.load(join(self.processed_spectra_path, df.iloc[i]["spectra_fname"])))
            img_coords.append(
                np.load(join(self.processed_spectra_path, df.iloc[i]["img_coord_fname"])))
            world_coords.append(
                np.load(join(self.processed_spectra_path, df.iloc[i]["world_coord_fname"])))

        self.data["gt_spectra"] = spectra
        self.data["gt_spectra_redshift"] = np.array(redshift).astype(np.float32)
        self.data["gt_spectra_pixels"] = np.concatenate(pixels, axis=0).astype(np.float32)
        self.data["gt_spectra_img_coords"] = np.concatenate(
            img_coords, axis=0).astype(np.int16)
        self.data["gt_spectra_world_coords"] = np.concatenate(
            world_coords, axis=0).astype(np.float32)

    def process_spectra(self):
        upper_bound = self.kwargs["num_gt_spectra"]
        df = self.load_source_metadata().iloc[:upper_bound]
        log.info(f"found {len(df)} source spectra")

        for field in ["tract","patch","pix_fname",
                      "img_coord_fname","world_coord_fname"]:
            df[field] = "None"

        if self.download_source_spectra:
            spectra_fnames = list(df["spectra_fname"])
            download_deimos_data_parallel(
                self.source_spectra_link, self.source_spectra_path, spectra_fnames)
            log.info("spectra-data::download complete")

        header_wcs, headers = self.load_headers(df)
        spectra_ids, spectra_to_drop = self.localize_spectra(df, header_wcs, headers)
        wave_ranges = self.load_spectra_patch_wise(df, spectra_ids)

        df.drop(spectra_to_drop, inplace=True)   # drop nonexist spectra
        df.reset_index(inplace=True)
        df.drop(columns=["index"], inplace=True) # drop extra index added by `reset_index`

        df.dropna(subset=[
            "pix_fname","img_coord_fname","world_coord_fname","spectra_fname"], inplace=True)
        df.reset_index(inplace=True)
        df.drop(columns=["index"], inplace=True) # drop extra index added by `reset_index`

        df.to_pickle(self.processed_metadata_table_fname)

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
                valid_spectra_ids.extend(v)

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
            tract, patch, i, j = localize(ra, dec)

            # TODO: we may adapt to spectra that doesn't belong
            #       to any patches we have in the future
            if tract == -1: # current spectra doesn't belong to patches we selected
                spectra_to_drop.append(idx)
                continue

            patch_uid = create_patch_uid(tract, patch)
            spectra_ids[patch_uid].append(idx)
            df.at[idx,"tract"] = tract
            df.at[idx,"patch"] = patch

        log.info("spectra-data::localized spectra")
        with open(self.gt_spectra_ids_fname, "wb") as fp:
            pickle.dump(dict(spectra_ids), fp)
        return spectra_ids, spectra_to_drop

    def load_spectra_patch_wise(self, df, spectra_ids):
        """ Load pixels and coords for each spectra in patch-wise order.
        """
        wave_ranges = []
        process_each_patch = partial(self.process_spectra_in_one_patch, df)

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
                    cur_wave_ranges = process_each_patch(
                        patch_uid, cur_patch, spectra_ids[patch_uid])
                    wave_ranges.extend(cur_wave_ranges)

        return np.array(wave_ranges)

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

        ras = np.array(list(df.iloc[spectra_ids]["ra"]))
        decs = np.array(list(df.iloc[spectra_ids]["dec"]))
        redshift = np.array(list(df.iloc[spectra_ids]["zspec"])).astype(np.float32)

        # get img coords for all spectra within current patch
        wcs = WCS(patch.get_header())
        world_coords = np.concatenate((ras[:,None], decs[:,None]), axis=-1)
        img_coords = wcs.all_world2pix(world_coords, 0).astype(int) # [n,2]
        # world coords from spectra data may not be accurate in terms of
        #  wcs of each patch, here after we get img coords, we convert
        #  img coords back to world coords to get accurate values
        world_coords = wcs.all_pix2world(img_coords, 0) # [n,2]
        img_coords = img_coords[:,::-1] # xy coords to rc coords

        wave_ranges, cur_patch_spectra = [], []
        process_one_spectra = partial(self.process_one_spectra,
                                      wave_ranges, cur_patch_spectra, df, patch)
        for i, (idx, img_coord) in enumerate(zip(spectra_ids, img_coords)):
            process_one_spectra(idx, img_coord)

        cur_patch_spectra_fname = join(self.processed_spectra_path, f"{patch_uid}_spectra.npy")
        cur_patch_redshift_fname = join(
            self.processed_spectra_path, f"{patch_uid}_redshift.npy")
        cur_patch_img_coords_fname = join(
            self.processed_spectra_path, f"{patch_uid}_img_coords.npy")
        cur_patch_world_coords_fname = join(
            self.processed_spectra_path, f"{patch_uid}_world_coords.npy")

        np.save(cur_patch_redshift_fname, redshift)
        np.save(cur_patch_img_coords_fname, img_coords)     # excl neighbours
        np.save(cur_patch_world_coords_fname, world_coords) # excl neighbours
        np.save(cur_patch_spectra_fname, np.array(cur_patch_spectra))
        return wave_ranges

    def process_one_spectra(self, wave_ranges, cur_patch_spectra, df, patch, idx, img_coord):
        """ Get pixel and normalized coord and
              process spectra data for one spectra.
            @Params
              df: source metadata dataframe for all spectra
              patch: patch that contains the current spectra
              idx: spectra idx (within the df table)
              img_coord: img coord for current spectra
        """
        spectra_fname = df.iloc[idx]["spectra_fname"]
        if self.spectra_data_format == "fits":
            fname = spectra_fname[:-5]
        elif self.spectra_data_format == "tbl":
            fname = spectra_fname[:-4]
        else: raise ValueError("Unsupported spectra data source")

        pix_fname = f"{fname}_pix.npy"
        img_coord_fname = f"{fname}_img_coord.npy"
        world_coord_fname = f"{fname}_world_coord.npy"
        df.at[idx,"pix_fname"] = pix_fname
        df.at[idx,"spectra_fname"] = f"{fname}.npy"
        df.at[idx,"img_coord_fname"] = img_coord_fname
        df.at[idx,"world_coord_fname"] = world_coord_fname

        pixel_ids = patch.get_pixel_ids(
            img_coord[0], img_coord[1], neighbour_size=self.neighbour_size
        )
        pixels = patch.get_pixels(pixel_ids)[None,:]
        np.save(join(self.processed_spectra_path, pix_fname), pixels)

        # `img_coord` excludes neighbours, `world_coords` include
        world_coords = patch.get_coords(pixel_ids)[None,:] # un-normed ra/dec [n,]
        np.save(join(self.processed_spectra_path, img_coord_fname), img_coord[None,:])
        np.save(join(self.processed_spectra_path, world_coord_fname), world_coords)

        # process source spectra and save locally
        gt_spectra, wave_range = process_gt_spectra(
            join(self.source_spectra_path, spectra_fname),
            join(self.processed_spectra_path, fname),
            self.full_wave,
            self.wave_discretz_interval,
            sigma=self.smooth_sigma,
            format=self.spectra_data_format
        )
        if gt_spectra is None: return

        wave_ranges.append(wave_range)

        # clip to trusted range (data collected here is for main train supervision only)
        (id_lo, id_hi) = get_bound_id(
            self.data["supervision_spectra_wave_bound"], gt_spectra[0], within_bound=True)
        cur_patch_spectra.append(gt_spectra[:,id_lo:id_hi+1])

    def load_source_metadata(self):
        if self.spectra_data_source == "manual":
            source_spectra_data = read_manual_table(self.manual_table_fname)
        elif self.spectra_data_source == "deimos":
            source_spectra_data = read_deimos_table(self.source_metadata_table_fname,
                                                    self.spectra_data_format)
        elif self.spectra_data_source == "zcosmos":
            source_spectra_data = read_zcosmos_table(self.source_metadata_table_fname,
                                                     self.spectra_data_format)
        else: raise ValueError("Unsupported spectra data source")
        return source_spectra_data

    def find_full_wave_bound_ids(self):
        """ Find id of min and max wave of supervision range in terms of
              the transmission wave (full_wave).
            Since the min and max wave for the supervision range may not
              coincide exactly with the trans wave, we find closest trans wave to replace
        """
        supervision_spectra_wave_bound = [
            self.kwargs["spectra_supervision_wave_lo"],
            self.kwargs["spectra_supervision_wave_hi"]
        ]
        (id_lo, id_hi) = get_bound_id(
            supervision_spectra_wave_bound, self.full_wave, within_bound=False)
        self.data["supervision_wave_bound_ids"] = [id_lo, id_hi + 1]
        self.data["supervision_spectra_wave_bound"] = [
            self.full_wave[id_lo], self.full_wave[id_hi]]

    #############
    # Utilities
    #############

    def normalize_one_flux(self, sub_dir, is_codebook, plot_gt_spectrum,
                           flux_norm_cho, gt_flux, recon_flux
    ):
        """ Normalize one pair of gt and recon flux.
        """
        if not is_codebook:
            sub_dir += flux_norm_cho + "_"
            if flux_norm_cho == "max":
                recon_flux = recon_flux / np.max(recon_flux)
            elif flux_norm_cho == "sum":
                recon_flux = recon_flux / np.sum(recon_flux)
            elif flux_norm_cho == "scale_gt":
                # scale gt spectra s.t. its max is same as recon
                recon_max = np.max(recon_flux)

        if plot_gt_spectrum:
            sub_dir += "with_gt_"
            # assert(np.max(gt_flux) > 0)
            if flux_norm_cho == "max":
                gt_flux = gt_flux / np.max(gt_flux)
            elif flux_norm_cho == "sum":
                gt_flux = gt_flux / (np.sum(gt_flux) + 1e-10)
            elif flux_norm_cho == "scale_gt":
                gt_flux = gt_flux / np.max(gt_flux) * recon_max
            elif flux_norm_cho == "scale_recon":
                recon_flux = recon_flux / np.max(recon_flux) * np.max(gt_flux)
        return sub_dir, gt_flux, recon_flux

    def gather_one_spectrum_plot_data(
            self, full_wave, flux_norm_cho, clip, spectra_clipped, is_codebook,
            bound_ids, gt_flux, gt_wave, recon_flux, recon_wave
    ):
        """ Collect data for spectrum plotting for the given spectra.
        """
        sub_dir = ""
        plot_gt_spectrum = self.kwargs["plot_spectrum_with_gt"] \
            and gt_flux is not None and not is_codebook

        if clip or spectra_clipped: # clip spectra to plot range
            if not spectra_clipped:
                if is_codebook or bound_ids is not None: (lo, hi) = bound_ids
                else: lo, hi = 0, recon_flux.shape[-1]
                recon_flux = recon_flux[...,lo:hi]
            sub_dir += "clipped_"

        if recon_flux.ndim == 2: # average spectra over neighbours
            if self.kwargs["average_neighbour_spectra"]:
                recon_flux = np.mean(recon_flux, axis=0)
            else: recon_flux = recon_flux[0]
        else: assert(recon_flux.ndim == 1)

        if not clip: # get wave (x-axis)
            recon_wave = full_wave
        elif is_codebook or recon_wave is not None:
            recon_wave = recon_wave
        else: recon_wave = full_wave

        sub_dir, gt_flux, recon_flux = self.normalize_one_flux(
            sub_dir, is_codebook, plot_gt_spectrum, flux_norm_cho, gt_flux, recon_flux
        )
        pargs = (sub_dir, gt_flux, gt_wave, recon_flux, recon_wave, plot_gt_spectrum)
        return pargs

    def plot_and_save_one_spectrum(
            self, name, spectra_dir, fig, axs, nrows, ncols, save_spectra, idx, pargs
    ):
        """ Plot one spectrum and save as required.
        """
        sub_dir, gt_flux, gt_wave, recon_flux, recon_wave, plot_gt_spectrum = pargs

        if self.kwargs["plot_spectrum_together"]:
            if nrows == 1: axis = axs if ncols == 1 else axs[idx%ncols]
            else:          axis = axs[idx//ncols, idx%ncols]
        else: fig, axs = plt.subplots(1); axis = axs[0]

        if self.kwargs["plot_spectrum_with_trans"]:
            sub_dir += "with_trans_"
            self.trans_obj.plot_trans(axis=axis)

        axis.set_title(idx)
        axis.plot(recon_wave, recon_flux, color="black", label="Recon.")
        if plot_gt_spectrum:
            axis.plot(gt_wave, gt_flux, color="blue", label="GT")

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

        return sub_dir

    def gather_spectrum_plotting_data(self, clip, is_codebook, mode):
        full_wave = self.get_full_wave()     # [full_nsmpl]
        gt_wave = self.get_gt_spectra_wave() # [nsmpl]
        if mode == "pretrain_infer":
            gt_fluxes = self.get_supervision_fluxes() # [n,nsmpl]
        elif mode == "infer" or mode == "main_train":
            gt_fluxes = self.get_validation_fluxes()  # [n,nsmpl]

        if clip:
            if is_codebook: clip_range = self.kwargs["codebook_spectra_clip_range"]
            else:           clip_range = self.kwargs["recon_spectra_clip_range"]
        else:               clip_range = (full_wave[0], full_wave[-1])

        (id_lo, id_hi) = get_bound_id(clip_range, full_wave, within_bound=False)
        recon_wave = np.arange(
            full_wave[id_lo], full_wave[id_hi] + 1, self.wave_discretz_interval)
        recon_wave_bound_ids = [id_lo, id_hi + 1]
        return full_wave, gt_fluxes, gt_wave, recon_wave, recon_wave_bound_ids

    def plot_spectrum(self, spectra_dir, name, recon_fluxes, flux_norm_cho,
                      mode="pretrain_infer", clip=True,
                      spectra_clipped=False, is_codebook=False,
                      save_spectra=False, save_spectra_together=False,
                      gt_spectra_ids=None, recon_spectra_ids=None
    ):
        """ Plot all given spectra.
            @Param
              recon_flux: [num_spectra(,num_neighbours),full_num_smpl]
                          in same lambda range as `full_wave`
              ids: if not None, indicates selected spectra to plot
                   (when we have large amount of spectra, we only select some to plot)
              clip: whether or not we plot spectra within certain range
              spectra_clipped: whether or not `recon_spectra` is already clipped to
        """
        n = len(recon_fluxes)
        if gt_spectra_ids is not None: n = min(n, len(gt_spectra_ids))
        if recon_spectra_ids is not None: n = min(n, len(recon_spectra_ids))

        full_wave, gt_fluxes, gt_wave, recon_wave, bound_ids = \
            self.gather_spectrum_plotting_data(clip, is_codebook, mode)

        if self.kwargs["plot_spectrum_together"]:
            ncols = min(n, self.kwargs["num_spectra_plot_per_row"])
            nrows = int(np.ceil(n / ncols))
            fig, axs = plt.subplots(nrows, ncols, figsize=(5*ncols,5*nrows))

        get_data = partial(self.gather_one_spectrum_plot_data,
                           full_wave, flux_norm_cho, clip, spectra_clipped,
                           is_codebook, bound_ids)
        plot_and_save = partial(self.plot_and_save_one_spectrum,
                                name, spectra_dir, fig, axs, nrows, ncols,
                                save_spectra and not save_spectra_together)

        if gt_spectra_ids is not None:
            gt_fluxes = gt_fluxes[gt_spectra_ids]
        if recon_spectra_ids is not None:
            recon_fluxes = recon_fluxes[recon_spectra_ids]

        for idx, (gt_flux, cur_flux) in enumerate(zip(gt_fluxes, recon_fluxes)):
            pargs = get_data(gt_flux, gt_wave, cur_flux, recon_wave)
            sub_dir = plot_and_save(idx, pargs)

        if save_spectra_together:
            fname = join(spectra_dir, name)
            np.save(fname, recon_fluxes)

        if self.kwargs["plot_spectrum_together"]:
            fname = join(spectra_dir, sub_dir, f"all_spectra_{name}")
            fig.tight_layout(); plt.savefig(fname); plt.close()

    def mark_spectra_on_img(self):
        assert 0
        markers = self.kwargs["spectra_markers"]
        spectra_img_coords = self.get_spectra_img_coords()
        spectra_fits_ids = set(spectra_img_coords[:,-1])

        for fits_id in spectra_fits_ids:
            # collect spectra in the same tile
            cur_coords, cur_markers = [], []
            for i, (r, c, cur_fits_id) in zip(
                    self.kwargs["gt_spectra_ids"], spectra_img_coords):

                if cur_fits_id == fits_id:
                    cur_coords.append([r,c])
                    cur_markers.append(markers[i])

            # mark on the corresponding tile
            self.fits_obj.mark_on_img(
                np.array(cur_coords), cur_markers, fits_id)

    def log_spectra_pixel_values(self, spectra):
        assert 0
        # gt_pixel_ids = self.get_spectra_coord_ids().flatten()
        # gt_pixels = self.fits_obj.get_pixels(idx=gt_pixel_ids).numpy()
        gt_pixels = self.data[""] # todo
        gt_pixels = np.round(gt_pixels, 2)
        np.set_printoptions(suppress = True)
        log.info(f"spectra-data::GT spectra pixel values: {gt_pixels}")

        recon_pixels = self.trans_obj.integrate(spectra)
        recon_pixels = np.round(recon_pixels, 2)
        log.info(f"spectra-data::Recon. spectra pixel values: {recon_pixels}")

# SpectraData class ends
#############

#############
# Spectra processing
#############

def locate_tract_patch(wcs, headers, tracts, patches_r, patches_c, ra, dec):
    for i, tract in enumerate(tracts):
        for j, patch_r in enumerate(patches_r):
            for k, patch_c in enumerate(patches_c):
                l = j * len(patches_c) + k
                if headers[i][l] is None: continue

                num_rows, num_cols = headers[i][l]["NAXIS2"], headers[i][l]["NAXIS1"]
                if is_in_patch(ra, dec, wcs[i][l], num_rows, num_cols):
                    return tract, f"{patch_r},{patch_c}", i, l
    return -1,-1,-1,-1

def is_in_patch(ra, dec, wcs, num_rows, num_cols):
    x, y = wcs.wcs_world2pix(ra, dec, 0)
    return x >= 0 and y >= 0 and x < num_cols and y < num_rows

def clean_flux(flux):
    """ Replace `inf` `-inf` `nan` in flux with 0.
    """
    flux[np.isnan(flux)] = 0
    flux[flux == np.inf] = 0
    flux[flux == -np.inf] = 0

def process_gt_spectra(infname, outfname, full_wave, smpl_interval,
                       sigma=-1, trusted_range=None, format="tbl",
                       interpolate=True, save=True, plot=True, validator=None):
    """ Load gt spectra wave and flux for spectra supervision and
          spectrum plotting. Also smooth the gt spectra.
        Note, the gt spectra has significantl larger discretization values than
          the transmission data. If requried, interpolate with same discretization.

        @Param
          infname: filename of np array that stores the gt spectra data.
          outfname: output filename to store processed gt spectra (wave & flux)
          full_wave: all lambda values for the transmission data.
          smpl_interval: discretization values of the transmission data.
        @Save
          gt_wave/spectra: spectra data with the corresponding lambda values.
    """
    try:
        wave, flux, ivar = unpack_gt_spectra(infname, format=format)
    except Exception as e:
        log.info(f"{e}")
        log.info(f"{infname}")
        return None, None

    clean_flux(flux)
    lo, hi = min(wave), max(wave)

    if sigma > 0: # smooth spectra
        try:
            flux = convolve_spectra(flux, std=sigma)
        except Exception as e:
            log.info(e)

    if interpolate:
        f_gt = interp1d(wave, flux)

        # make sure wave range to interpolate stay within gt spectra wave range
        # full_wave is full transmission wave
        if trusted_range is not None:
            (lo, hi) = trusted_range
        else:
            (lo_id, hi_id) = get_bound_id(
                ( min(wave),max(wave) ), full_wave, within_bound=True)
            lo = full_wave[lo_id] # lo <= full_wave[lo_id]
            hi = full_wave[hi_id] # hi >= full_wave[hi_id]

        # new gt wave with same discretization value as transmission wave
        wave = np.arange(lo, hi + 1, smpl_interval)

        # use new gt wave to get interpolated spectra
        flux = f_gt(wave)

    flux = flux.astype(np.float32)
    spectra_data = np.concatenate((
        wave[None,:], flux[None,:]), axis=0) # [2,nsmpl]

    if save:
        np.save(outfname + ".npy", spectra_data)
    if plot:
        plt.plot(wave, flux)
        plt.savefig(outfname + ".png")
        plt.close()

    if validator is not None and not validator(spectra_data):
        return None, None
    return spectra_data, (lo, hi)

def convolve_spectra(spectra, std=140, border=True):
    """ Smooth gt spectra with given std.
        If border is True, we add 1 padding at two ends when convolving.
    """
    kernel = Gaussian1DKernel(stddev=std)
    if border:
        nume = convolve(spectra, kernel)
        denom = convolve(np.ones(spectra.shape), kernel)
        return nume / denom
    return convolve(spectra, kernel)

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

def read_deimos_table(fname, format):
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
    df["ra"] = pandas.to_numeric(df["ra"])
    df["dec"] = pandas.to_numeric(df["dec"])
    df["zspec"] = pandas.to_numeric(df["zspec"])
    return df

def read_zcosmos_table(fname):
    """ Read metadata table for zcosmos spectra data.
    """
    df = Table.read(fname).to_pandas()
    df.rename(columns={
        'OBJECT_ID': 'id',
        'RAJ2000':'ra',
        'DEJ2000':'dec',
        'REDSHIFT':'zspec',
        'FILANEMS':'spectra_fname'
    }, inplace=True)
    df.drop(columns=['CC','IMAG_AB','FLAG_S','FLAG_X','FLAG_R','FLAG_UV'], inplace=True)
    df.drop([0], inplace=True) # drop first row which is datatype
    df.dropna(subset=["ra","dec","spectra_fname"], inplace=True)
    df.reset_index(inplace=True) # reset index after dropping
    df["ra"] = pandas.to_numeric(df["ra"])
    df["dec"] = pandas.to_numeric(df["dec"])
    df["zspec"] = pandas.to_numeric(df["zspec"])
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

def unpack_gt_spectra(fname, format="tbl"):
    if format == "tbl":
        data = np.array(pandas.read_table(fname, comment="#", delim_whitespace=True))
        wave, flux, ivar = data[:,0], data[:,1], data[:,2]
    elif format == "fits":
        hdu = fits.open(fname)[1]
        header = hdu.header
        data = hdu.data[0]
        data_names = [header["TTYPE1"],header["TTYPE2"],header["TTYPE3"]]
        wave_id = data_names.index("LAMBDA")
        flux_id = data_names.index("FLUX")
        ivar_id = data_names.index("IVAR")
        wave, flux, ivar = data[wave_id], data[flux_id], data[ivar_id]
    else:
        raise ValueError(f"invalid spectra data format: {format}")
    return wave, flux, ivar

def download_deimos_data_parallel(http_prefix, local_path, fnames):
    fnames = list(filter(lambda fname: not exists(join(local_path, fname)), fnames))
    log.info(f"downloading {len(fnames)} spectra")
    assert 0

    urls = [f"{http_prefix}/{fname}" for fname in fnames]
    out_fnames = [f"{local_path}/{fname}" for fname in fnames]
    inputs = zip(urls, out_fnames)
    cpus = cpu_count()
    results = ThreadPool(cpus - 1).imap_unordered(download_from_url, inputs)
    for result in results:
        if result is not None:
            print('url:', result[0], 'time (s):', result[1])
            # log.info(f"url: {result[0]}")

def download_from_url(input_):
    t0 = time.time()
    (url, fname) = input_
    try:
        r = requests.get(url)
        # open(out_fname, "wb").write(r.content)
        with open(fname, 'wb') as f:
            f.write(r.content)
        return(url, time.time() - t0)
    except Exception as e:
        print('Exception in download_url():', e)
        return(url)
