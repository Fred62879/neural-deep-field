
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
    clip_data_to_ref_wave_range, get_wave_range_fname

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
    def __init__(self, trans_obj, device, **kwargs):
        self.kwargs = kwargs

        self.device = device
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

        self.source_spectra_link = kwargs["source_spectra_link"]
        self.spectra_data_source = kwargs["spectra_data_source"]
        self.spectra_data_format = kwargs["spectra_data_format"] # tbl or fits
        self.download_source_spectra = kwargs["download_source_spectra"]
        self.load_spectra_data_from_cache = kwargs["load_spectra_data_from_cache"]

        self.smooth_sigma = kwargs["spectra_smooth_sigma"]
        self.neighbour_size = kwargs["spectra_neighbour_size"]
        self.wave_discretz_interval = kwargs["trans_sample_interval"]
        self.trusted_wave_range = None if not kwargs["learn_spectra_within_wave_range"] \
            else [kwargs["spectra_supervision_wave_lo"],
                  kwargs["spectra_supervision_wave_hi"]]
        self.trans_range = self.trans_obj.get_wave_range()

        self.set_path(self.dataset_path)
        self.load_accessory_data()
        self.load_spectra()

    def set_path(self, dataset_path):
        """ Create path and filename of required files.
            source_metadata_table:    ra,dec,zspec,spectra_fname
            processed_metadata_table: added pixel val and tract-patch
        """
        paths = []
        self.input_patch_path, img_data_path = set_input_path(
            dataset_path, self.kwargs["sensor_collection_name"])
        spectra_path = join(dataset_path, "input/spectra")

        suffix = self.kwargs["source_spectra_cho"]
        if suffix != "": suffix = "_" + suffix

        self.wave_range_fname = get_wave_range_fname(**self.kwargs)

        if self.spectra_data_source == "manual":
            assert 0
            self.source_metadata_table_fname = join(spectra_path, "deimos_old", "spectra.csv")

        elif self.spectra_data_source == "deimos":
            path = join(spectra_path, "deimos")
            self.source_spectra_path = join(
                path, f"source_spectra_{self.spectra_data_format}{suffix}")
            self.source_metadata_table_fname = join(
                path, self.kwargs["source_spectra_fname"])

            processed_data_path = join(
                path, "processed_" + self.kwargs["processed_spectra_cho"] + suffix)
            self.processed_spectra_path = join(
                processed_data_path, "processed_spectra")
            self.processed_metadata_table_fname = join(
                processed_data_path, "processed_deimos_table.tbl")

        elif self.spectra_data_source == "zcosmos":
            path = join(spectra_path, "zcosmos")
            self.source_spectra_path = join(
                path, f"source_spectra_{self.spectra_data_format}{suffix}")
            self.source_metadata_table_fname = join(
                path, "source_zcosmos_table.fits")

            processed_data_path = join(
                path, "processed_" + self.kwargs["processed_spectra_cho"] + suffix)
            self.processed_spectra_path = join(
                processed_data_path, "processed_spectra")
            self.processed_metadata_table_fname = join(
                processed_data_path, "processed_zcosmos_table.fits")

        else: raise ValueError("Unsupported spectra data source choice.")

        self.emit_wave_coverage_fname = join(
            processed_data_path,
            "emit_wave_coverage_" +
            str(self.kwargs["spectra_supervision_wave_lo"]) + "_" +
            str(self.kwargs["spectra_supervision_wave_hi"]) + ".npy")

        self.gt_spectra_fname = join(processed_data_path, "gt_spectra.npy")
        self.gt_spectra_ids_fname = join(processed_data_path, "gt_spectra_ids.txt")
        self.gt_spectra_pixels_fname = join(processed_data_path, "gt_spectra_pixels.npy")
        self.gt_spectra_redshift_fname = join(processed_data_path, "gt_spectra_redshift.npy")
        self.gt_spectra_plot_mask_fname = join(processed_data_path, "gt_spectra_plot_mask.npy")
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
        self.set_wave_range()

    #############
    # Getters
    #############

    def get_full_wave_coverage(self):
        """ Get full coverage range of emitted wave.
        """
        return self.data["full_emit_wave"]

    def get_full_wave_mask(self):
        """ Get mask for full (used for codebook spectra plotting).
        """
        return self.data["full_emit_wave_mask"]

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

    def get_supervision_plot_mask(self, idx=None):
        """ Get mask for plotting. """
        if idx is None:
            return self.data["supervision_plot_mask"]
        return self.data["supervision_plot_mask"][idx]

    def get_supervision_data(self, idx=None):
        """ Get gt spectra (with same wave range as recon) used for supervision. """
        if idx is None:
            return self.data["supervision_spectra"]
        return self.data["supervision_spectra"][idx]

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

    def get_validation_spectra(self, idx=None):
        if idx is not None:
            return self.data["validation_spectra"][idx]
        return self.data["validation_spectra"]

    def get_validation_pixels(self, idx=None):
        if idx is not None:
            return self.data["validation_pixels"][idx]
        return self.data["validation_pixels"]

    def get_validation_img_coords(self, idx=None):
        if idx is not None:
            return self.data["validation_img_coords"][idx]
        return self.data["validation_img_coords"]

    def get_validation_world_coords(self, idx=None):
        if idx is not None:
            return self.data["validation_world_coords"][idx]
        return self.data["validation_world_coords"]

    def get_semi_supervision_redshift(self):
        return self.data["semi_supervision_redshift"]

    #############
    # Helpers
    #############

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

        if not self.load_spectra_data_from_cache or \
           not exists(self.emit_wave_coverage_fname) or \
           not exists(self.gt_spectra_fname) or \
           not exists(self.gt_spectra_ids_fname) or \
           not exists(self.gt_spectra_pixels_fname) or \
           not exists(self.gt_spectra_redshift_fname) or \
           not exists(self.gt_spectra_plot_mask_fname) or \
           not exists(self.gt_spectra_img_coords_fname) or \
           not exists(self.gt_spectra_world_coords_fname):

            # process and save data for each spectra individually
            if not exists(self.gt_spectra_ids_fname) or \
               not exists(self.emit_wave_coverage_fname) or \
               not exists(self.processed_metadata_table_fname):
                self.process_spectra()
            # load data for each individual spectra and save together
            self.gather_processed_spectra()
        else:
            # data for all spectra are saved together (small amount of spectra)
            self.load_cached_spectra_data()

        self.get_full_emit_wave_mask()
        self.num_gt_spectra = len(self.data["gt_spectra"])
        self.transform_data()
        self.train_valid_split()

    def transform_data(self):
        self.to_tensor([
            "full_emit_wave",
            "gt_spectra",
            "gt_spectra_pixels",
            "gt_spectra_redshift",
            # "gt_spectra_world_coords"
        ], torch.float32)
        self.to_tensor([
            "full_emit_wave_mask",
            "gt_spectra_plot_mask",
        ], torch.bool)

    #############
    # Loading helpers
    #############

    def to_tensor(self, fields, dtype):
        for field in fields:
            self.data[field] = torch.tensor(self.data[field], dtype=dtype)

    def split_spectra(self):
        ids = np.arange(self.num_gt_spectra)

        # reserve all spectra in main train image patch as validation spectra
        acc, validation_ids, validation_patch_ids = 0, [], {}
        for tract, patch in zip(self.kwargs["tracts"], self.kwargs["patches"]):
            patch_uid = create_patch_uid(tract, patch)
            cur_spectra_ids = self.data["gt_spectra_ids"][patch_uid]
            validation_ids.extend(cur_spectra_ids)
            validation_patch_ids[patch_uid] = np.arange(acc, acc+len(cur_spectra_ids))
            acc += len(cur_spectra_ids)
        validation_ids = np.array(validation_ids)
        # validation_ids = np.array([0])
        log.info(f"validation spectra ids: {validation_ids}")

        # get supervision ids
        supervision_ids = np.array(list(set(ids) - set(validation_ids))).astype(int)
        np.random.shuffle(supervision_ids)
        supervision_ids = supervision_ids[:self.kwargs["num_supervision_spectra"]]
        # supervision_ids = np.array([14,22,31]) # 14,22 fail /31 succeed
        # log.info(f"supervision spectra ids: {supervision_ids}")

        self.num_validation_spectra = len(validation_ids)
        self.num_supervision_spectra = len(supervision_ids)
        return supervision_ids, validation_ids

    def train_valid_split(self):
        sup_ids, val_ids = self.split_spectra()
        log.info(f"spectra train/valid {len(sup_ids)}/{len(val_ids)}")

        # supervision spectra data (used during pretrain)
        self.data["supervision_spectra"] = self.data["gt_spectra"][sup_ids]
        self.data["supervision_redshift"] = self.data["gt_spectra_redshift"][sup_ids]
        self.data["supervision_plot_mask"] = self.data["gt_spectra_plot_mask"][sup_ids]
        if self.kwargs["codebook_pretrain_pixel_supervision"]:
            self.data["supervision_pixels"] = self.data["gt_spectra_pixels"][sup_ids]

        # valiation(and semi sup) spectra data (used during main train)
        self.data["validation_spectra"] = self.data["gt_spectra"][val_ids]
        self.data["validation_pixels"] = self.data["gt_spectra_pixels"][val_ids]
        self.data["validation_img_coords"] = self.data["gt_spectra_img_coords"][val_ids]
        # print(self.data["validation_img_coords"])
        self.data["validation_world_coords"] = self.data["gt_spectra_world_coords"][val_ids]
        if self.kwargs["redshift_semi_supervision"]:
            self.data["semi_supervision_redshift"] = self.data["gt_spectra_redshift"][val_ids]

    def get_full_emit_wave_mask(self):
        """ Generate mask for codebook spectra plot.
        """
        n = len(self.data["full_emit_wave"])
        self.data["full_emit_wave_mask"] = np.zeros(n)
        (id_lo, id_hi) = get_bound_id(
            (self.kwargs["codebook_spectra_plot_wave_lo"],
             self.kwargs["codebook_spectra_plot_wave_hi"]),
            self.data["full_emit_wave"]
        )
        self.data["full_emit_wave_mask"][id_lo:id_hi+1] = 1

    def load_cached_spectra_data(self):
        """ Load spectra data (which are saved together).
        """
        with open(self.gt_spectra_ids_fname, "rb") as fp:
            ids = pickle.load(fp)
        self.data["gt_spectra_ids"] = defaultdict(list, ids)

        self.data["full_emit_wave"] = np.load(self.emit_wave_coverage_fname)[0]

        self.data["gt_spectra"] = np.load(self.gt_spectra_fname)
        self.data["gt_spectra_pixels"] = np.load(self.gt_spectra_pixels_fname)
        self.data["gt_spectra_redshift"] = np.load(self.gt_spectra_redshift_fname)
        self.data["gt_spectra_plot_mask"] = np.load(self.gt_spectra_plot_mask_fname)
        self.data["gt_spectra_img_coords"] = np.load(self.gt_spectra_img_coords_fname)
        self.data["gt_spectra_world_coords"] = np.load(self.gt_spectra_world_coords_fname)

    def gather_processed_spectra(self):
        """ Load processed data for each spectra and save together.
        """
        df = pandas.read_pickle(self.processed_metadata_table_fname)
        with open(self.gt_spectra_ids_fname, "rb") as fp:
            ids = pickle.load(fp)
        self.data["gt_spectra_ids"] = defaultdict(list, ids)

        img_coords, world_coords, spectra = [], [], []
        plot_masks, redshift, pixels = [], [], []

        n = len(df)
        for i in range(n):
            redshift.append(df.iloc[i]["zspec"])
            pixels.append(
                np.load(join(self.processed_spectra_path, df.iloc[i]["pix_fname"])))
            spectra.append(
                np.load(join(self.processed_spectra_path, df.iloc[i]["spectra_fname"])))
            plot_masks.append(
                np.load(join(self.processed_spectra_path, df.iloc[i]["plot_mask_fname"])))
            img_coords.append(
                np.load(join(self.processed_spectra_path, df.iloc[i]["img_coord_fname"])))
            world_coords.append(
                np.load(join(self.processed_spectra_path, df.iloc[i]["world_coord_fname"])))

        self.data["full_emit_wave"] = np.load(self.emit_wave_coverage_fname)[0]

        # [n_spectra,4+2*nbands,nsmpl]
        #  (wave/flux/ivar/trans_mask/trans(nbands)/band_mask(nbands))
        self.data["gt_spectra"] = np.array(spectra).astype(np.float32)

        self.data["gt_spectra_plot_mask"] = np.array(plot_masks).astype(bool)
        self.data["gt_spectra_pixels"] = np.concatenate(pixels, axis=0).astype(np.float32)
        self.data["gt_spectra_redshift"] = np.array(redshift).astype(np.float32) # [n,]
        self.data["gt_spectra_img_coords"] = np.concatenate(
            img_coords, axis=0).astype(np.int16) # [n,2]
        self.data["gt_spectra_world_coords"] = np.concatenate(
            world_coords, axis=0).astype(np.float32) # [n,n_neighbr,2]

        # save data for all spectra together
        np.save(self.gt_spectra_fname, self.data["gt_spectra"])
        np.save(self.gt_spectra_pixels_fname, self.data["gt_spectra_pixels"])
        np.save(self.gt_spectra_redshift_fname, self.data["gt_spectra_redshift"])
        np.save(self.gt_spectra_plot_mask_fname, self.data["gt_spectra_plot_mask"])
        np.save(self.gt_spectra_img_coords_fname, self.data["gt_spectra_img_coords"])
        np.save(self.gt_spectra_world_coords_fname, self.data["gt_spectra_world_coords"])

    def process_spectra(self):
        upper_bound = self.kwargs["num_gt_spectra"]
        df = self.load_source_metadata().iloc[:upper_bound]
        log.info(f"found {len(df)} source spectra")

        for field in ["tract","patch","pix_fname", "plot_mask_fname",
                      "img_coord_fname","world_coord_fname"]:
            df[field] = "None"

        if self.download_source_spectra:
            spectra_fnames = list(df["spectra_fname"])
            download_deimos_data_parallel(
                self.source_spectra_link, self.source_spectra_path, spectra_fnames)
            log.info("spectra-data::download complete")

        self.trans_data = self.trans_obj.get_full_trans_data()

        if not exists(self.emit_wave_coverage_fname):
            emit_wave_coverage = self.calculate_emitted_wave_coverage(df)
        else: emit_wave_coverage = np.load(self.emit_wave_coverage_fname)
        emit_wave_distrib = interp1d(emit_wave_coverage[0], emit_wave_coverage[1])

        header_wcs, headers = self.load_headers(df)
        spectra_ids, spectra_to_drop = self.localize_spectra(df, header_wcs, headers)
        self.load_spectra_patch_wise(df, spectra_ids, emit_wave_distrib)

        df.drop(spectra_to_drop, inplace=True)   # drop nonexist spectra
        df.reset_index(inplace=True)
        df.drop(columns=["index"], inplace=True) # drop extra index added by `reset_index`

        df.dropna(subset=[
            "pix_fname","img_coord_fname","world_coord_fname","spectra_fname"], inplace=True)
        df.reset_index(inplace=True)
        df.drop(columns=["index"], inplace=True) # drop extra index added by `reset_index`

        df.to_pickle(self.processed_metadata_table_fname)

    def calculate_emitted_wave_coverage(self, df):
        """ Calculate coverage of emitted wave based on
              given observed wave supervision range.
        """
        redshift = list(df['zspec'])
        lo = self.kwargs["spectra_supervision_wave_lo"]
        hi = self.kwargs["spectra_supervision_wave_hi"]
        min_emit_wave = int(lo / (1 + max(redshift)))
        max_emit_wave = int(np.ceil(hi / (1 + min(redshift))))
        n = max_emit_wave - min_emit_wave + 1

        x = np.arange(min_emit_wave, max_emit_wave + 1)
        distrib = np.zeros(n).astype(np.int32)

        def accumulate(cur_redshift):
            distrib[ int(lo/(1+cur_redshift)) - min_emit_wave:
                     int(hi/(1+cur_redshift)) - min_emit_wave ] += 1

        _= [accumulate(cur_redshift) for cur_redshift in redshift]
        distrib = np.array(distrib) # / sum(distrib))

        plt.plot(x, distrib); plt.title("Emitted wave coverage")
        plt.savefig(self.emit_wave_coverage_fname[:-4] + ".png")
        plt.close()

        emit_wave_coverage = np.concatenate((x[None,:], distrib[None,:]), axis=0)
        np.save(self.emit_wave_coverage_fname, emit_wave_coverage)
        return emit_wave_coverage

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

    def load_spectra_patch_wise(self, df, spectra_ids, emit_wave_distrib):
        """ Load pixels and coords for each spectra in patch-wise order.
        """
        process_each_patch = partial(self.process_spectra_in_one_patch, df, emit_wave_distrib)

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

    def process_spectra_in_one_patch(self, df, emit_wave_distrib,
                                     patch_uid, patch, spectra_ids
    ):
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

        cur_patch_spectra = []
        cur_patch_spectra_plot_mask = []
        process_one_spectra = partial(self.process_one_spectra,
                                      cur_patch_spectra,
                                      cur_patch_spectra_plot_mask,
                                      df, patch, emit_wave_distrib)

        for i, (idx, img_coord) in enumerate(zip(spectra_ids, img_coords)):
            process_one_spectra(idx, img_coord)

        cur_patch_spectra_fname = join(
            self.processed_spectra_path, f"{patch_uid}.npy")
        cur_patch_plot_mask_fname = join(
            self.processed_spectra_path, f"{patch_uid}_plot_mask.npy")
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
        np.save(cur_patch_plot_mask_fname, np.array(cur_patch_spectra_plot_mask))

    def process_one_spectra(self, cur_patch_spectra,
                            cur_patch_spectra_plot_mask,
                            df, patch, emit_wave_distrib,
                            idx, img_coord
    ):
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
        plot_mask_fname = f"{fname}_plot_mask.npy"
        img_coord_fname = f"{fname}_img_coord.npy"
        world_coord_fname = f"{fname}_world_coord.npy"
        df.at[idx,"pix_fname"] = pix_fname
        df.at[idx,"spectra_fname"] = f"{fname}.npy"
        df.at[idx,"plot_mask_fname"] = plot_mask_fname
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
        gt_spectra, plot_mask = process_gt_spectra(
            join(self.source_spectra_path, spectra_fname),
            join(self.processed_spectra_path, fname),
            join(self.processed_spectra_path, plot_mask_fname),
            df.loc[idx,"zspec"],
            emit_wave_distrib,
            sigma=self.smooth_sigma,
            format=self.spectra_data_format,
            trans_range=self.trans_range,
            trusted_range=self.trusted_wave_range,
            max_spectra_len=self.kwargs["max_spectra_len"],
            colors=self.kwargs["plot_colors"],
            trans_data=self.trans_data
        )
        cur_patch_spectra.append(gt_spectra)
        cur_patch_spectra_plot_mask.append(plot_mask)

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

    def normalize_one_flux(self, sub_dir, is_codebook, plot_gt_spectrum,
                           flux_norm_cho, gt_flux, recon_flux
    ):
        """ Normalize one pair of gt and recon flux.
        """
        sub_dir += flux_norm_cho + "_"
        if not is_codebook:
            sub_dir += "with_recon_"
            if flux_norm_cho == "max":
                recon_flux = recon_flux / np.max(recon_flux)
            elif flux_norm_cho == "sum":
                recon_flux = recon_flux / np.sum(recon_flux)
            elif flux_norm_cho == "linr":
                lo, hi = min(recon_flux), max(recon_flux)
                recon_flux = (recon_flux - lo) / (hi - lo)
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
                gt_flux = gt_flux * len(gt_flux) / len(recon_flux)
            elif flux_norm_cho == "linr":
                lo, hi = min(gt_flux), max(gt_flux)
                gt_flux = (gt_flux - lo) / (hi - lo)
            elif flux_norm_cho == "scale_gt":
                gt_flux = gt_flux / np.max(gt_flux) * recon_max
            elif flux_norm_cho == "scale_recon":
                recon_flux = recon_flux / np.max(recon_flux) * np.max(gt_flux)

        return sub_dir, gt_flux, recon_flux

    def plot_and_save_one_spectrum(
            self, name, spectra_dir, fig, axs, nrows, ncols, save_spectra, idx, pargs
    ):
        """ Plot one spectrum and save as required.
        """
        sub_dir, gt_wave, gt_flux, recon_wave, recon_flux, plot_gt_spectrum, plot_recon_spectrum = pargs

        if self.kwargs["plot_spectrum_together"]:
            if nrows == 1: axis = axs if ncols == 1 else axs[idx%ncols]
            else:          axis = axs[idx//ncols, idx%ncols]
        else: fig, axs = plt.subplots(1); axis = axs[0]

        if self.kwargs["plot_spectrum_with_trans"]:
            sub_dir += "with_trans_"
            self.trans_obj.plot_trans(axis=axis)

        axis.set_title(idx)
        if plot_gt_spectrum:
            axis.plot(gt_wave, gt_flux, color="blue", label="GT")
        if plot_recon_spectrum:
            axis.plot(recon_wave, recon_flux, color="black", label="Recon.")

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

    def process_spectrum_plot_data(
            self, flux_norm_cho, is_codebook,
            clip, spectra_clipped, data):
        """ Collect data for spectrum plotting for the given spectra.
        """
        (gt_wave, gt_mask, gt_flux, recon_wave, recon_mask, recon_flux) = data

        sub_dir = ""
        plot_gt_spectrum = self.kwargs["plot_spectrum_with_gt"] \
            and gt_flux is not None and not is_codebook
        plot_recon_spectrum = self.kwargs["plot_spectrum_with_recon"]

        # average spectra over neighbours
        if recon_flux.ndim == 2:
            if self.kwargs["average_neighbour_spectra"]:
                recon_flux = np.mean(recon_flux, axis=0)
            else: recon_flux = recon_flux[0]
        else: assert(recon_flux.ndim == 1)

        if clip or spectra_clipped:
            sub_dir += "clipped_"
            if not spectra_clipped:
                if plot_gt_spectrum:
                    gt_wave = gt_wave[gt_mask]
                    gt_flux = gt_flux[gt_mask]
                recon_wave = recon_wave[recon_mask]
                recon_flux = recon_flux[recon_mask]

        sub_dir, gt_flux, recon_flux = self.normalize_one_flux(
            sub_dir, is_codebook, plot_gt_spectrum, flux_norm_cho, gt_flux, recon_flux
        )
        pargs = (sub_dir, gt_wave, gt_flux, recon_wave, recon_flux,
                 plot_gt_spectrum, plot_recon_spectrum)
        return pargs

    def plot_spectrum(self, spectra_dir, name, flux_norm_cho,
                      gt_wave, gt_fluxes,
                      recon_wave, recon_fluxes,
                      mode="pretrain_infer",
                      is_codebook=False,
                      save_spectra=False,
                      save_spectra_together=False,
                      spectra_ids=None,
                      gt_masks=None, recon_masks=None,
                      clip=False, spectra_clipped=False
    ):
        """ Plot all given spectra.
            @Param
              spectra_dir:   directory to save spectra
              name:          file name
              flux_norm_cho: norm choice for flux

              wave:        corresponding wave for gt and recon fluxes
              gt_fluxes:   [num_spectra,nsmpl]
              recon_fluxs: [num_spectra(,num_neighbours),nsmpl]

              gt/recon_spectra_ids: if not None, indicates selected spectra to plot
                   (when we have large amount of spectra, we only select some to plot)

            - clip config:
              clip: whether or not we plot spectra within certain range
              masks: not None if clip. use mask to clip flux
              spectra_clipped: whether or not spectra is already clipped to
        """
        assert not clip or (recon_masks is not None or spectra_clipped)

        # recon_fluxes *= -1

        n = len(recon_wave)
        if gt_wave is None: gt_wave = [None]*n
        if gt_masks is None: gt_masks = [None]*n
        if gt_fluxes is None: gt_fluxes = [None]*n
        if recon_masks is None: recon_masks = [None]*n

        assert gt_fluxes[0] is None or \
            (len(gt_wave) == n and len(gt_fluxes) == n and len(gt_masks) == n)
        assert recon_masks[0] is None or \
            (len(recon_fluxes) == n and len(recon_masks) == n)

        if self.kwargs["plot_spectrum_together"]:
            ncols = min(n, self.kwargs["num_spectrum_per_row"])
            nrows = int(np.ceil(n / ncols))
            fig, axs = plt.subplots(nrows, ncols, figsize=(5*ncols,5*nrows))

        process_data = partial(self.process_spectrum_plot_data,
                               flux_norm_cho, is_codebook, clip, spectra_clipped)
        plot_and_save = partial(self.plot_and_save_one_spectrum,
                                name, spectra_dir, fig, axs, nrows, ncols,
                                save_spectra and not save_spectra_together)

        for idx, cur_plot_data in enumerate(
            zip(gt_wave, gt_masks, gt_fluxes, recon_wave, recon_masks, recon_fluxes)
        ):
            pargs = process_data(cur_plot_data)
            sub_dir = plot_and_save(idx, pargs)

        if save_spectra_together:
            fname = join(spectra_dir, name)
            np.save(fname, recon_fluxes)

        if self.kwargs["plot_spectrum_together"]:
            fname = join(spectra_dir, sub_dir, f"all_spectra_{name}")
            fig.tight_layout(); plt.savefig(fname); plt.close()

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

def scale_trans(trans, source_trans):
    nbands = trans.shape[0]
    for i in range(nbands):
        cur_trans, cur_source_trans = trans[i], source_trans[i]
        # if trans sum to 0, cur band is not covered
        trans[i] = trans[i] * np.sum(cur_source_trans) / (np.sum(cur_trans) + 1e-10)

def interpolate_trans(trans_data, spectra_data, bound, fname=None, colors=None):
    """ Interpolate transmission data based on wave from spectra data.
        Discretization interval for trans data is 10, which is way larger
          than that of spectra_data.
        @Param
          trans_data: [nsmpl_t,1+nbands] (wave/trans)
          spectra_data: [4,nsmpl_s] (wave,flux,ivar,weight)
          bound: defines range within which spectra is valid
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
    (id_lo_old, id_hi_old) = bound
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
        plt.plot(spectra_data[0], trans_mask, label="trans_mask")
        for j in range(nbands):
            plt.plot(spectra_data[0], trans[j], color=colors[j])
        plt.savefig(fname + "_trans_mask.png")
        plt.close()

        for j in range(nbands):
            plt.plot(spectra_data[0], band_mask[j], color=colors[j])
        plt.savefig(fname + "_band_mask.png")
        plt.close()

    ret = np.array([trans_mask] + list(trans) + list(band_mask))
    return ret

def create_spectra_mask(spectra, max_spectra_len):
    """ Mask out padded region of spectra.
    """
    m, n = spectra.shape
    if n == max_spectra_len: mask = np.ones(max_spectra_len).astype(bool)
    else: mask = np.zeros(max_spectra_len).astype(bool)
    return mask

def wave_based_sort(spectra):
    """ Sort spectra and mask based on wave.
    """
    ids = np.argsort(spectra[0])
    return spectra[:,ids]

def pad_spectra(spectra, mask, max_len):
    """ Pad spectra if shorter than max_len.
        Create mask to ignore values in padded region.
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

def normalize_spectra(spectra, bound):
    (id_lo, id_hi) = bound
    flux = spectra[1][id_lo:id_hi+1]
    lo, hi = min(flux), max(flux)
    spectra[1,id_lo:id_hi+1] = (flux - lo) / (hi - lo)
    return spectra

def convolve_spectra(spectra, bound, std=5, border=True):
    """ Smooth gt spectra with given std.
        @Param
          bound: defines range to convolve within
          border: if True, we add 1 padding at two ends when convolving
    """
    if std <= 0: return

    lo, hi = bound
    n = hi - lo + 1

    kernel = Gaussian1DKernel(stddev=std)
    if border:
        nume = convolve(spectra[1][lo:hi+1], kernel)
        denom = convolve(np.ones(n), kernel)
        spectra[1][lo:hi+1] = nume / denom
    else:
        spectra[1][lo:hi+1] = convolve(spectra[1][lo:hi+1], kernel)
    return spectra

def mask_spectra_range(spectra, mask, bound, trans_range, trusted_range):
    """ Mask out spectra data beyond given wave range.
        @Param
          spectra: spectra data [3,nsmpl] (wave,flux,ivar)
          bound: defines range of valid spectra
          mask: mask to be updated [nsmpl]
          trans_range: transmission data wave range
          trusted_range: spectra supervision wave range
    """
    (id_lo_old, id_hi_old) = bound

    m, n = spectra.shape
    lo1, hi1 = trans_range
    lo2, hi2 = trusted_range
    wave_range = (max(lo1,lo2), min(hi1,hi2))
    (id_lo_new, id_hi_new) = get_bound_id(wave_range, spectra[0])

    id_lo = max(id_lo_old, id_lo_new)
    id_hi = min(id_hi_old, id_hi_new)

    new_mask = np.zeros(n).astype(bool)
    new_mask[id_lo:id_hi+1] = 1
    mask &= new_mask
    bound = (id_lo, id_hi)
    return spectra, mask, bound

def get_wave_weight(spectra, redshift, emit_wave_distrib, bound):
    """ Get sampling weight for spectra wave (in unmasked range).
    """
    (lo, hi) = bound
    n = spectra.shape[1]
    obs_wave = spectra[0][lo:hi+1]
    weight = np.zeros(n)
    emit_wave = obs_wave / (1 + redshift)
    bound_weight = 1 / (emit_wave_distrib(emit_wave) + 1e-10)
    weight[lo:hi+1] = bound_weight
    weight = weight / max(weight)
    return weight

def process_gt_spectra(infname, spectra_fname, plot_mask_fname,
                       redshift, emit_wave_distrib,
                       sigma=-1, format="tbl",
                       trans_range=None, trusted_range=None,
                       save=True, plot=True,
                       colors=None, trans_data=None,
                       max_spectra_len=-1, validator=None
):
    """ Load gt spectra wave and flux for spectra supervision and
          spectrum plotting. Also smooth the gt spectra.
        Note, the gt spectra has significantly larger discretization values than
          the transmission data.

        @Param
          infname: filename of np array that stores the gt spectra data.
          spectra_fname: output filename to store processed gt spectra (wave & flux)
          mask_fname: output filename to store processed gt spectra (wave & flux)
          emit_wave_distrib: histogram distribution of emitted wave (interpolated function)
        @Return
          spectra:  spectra data [5+2*nbands,nsmpl]
                    (wave/flux/ivar/weight/trans_mask/trans(nbands)/band_mask(nbands))
          mask:     mask out bad flux values
    """
    if False: #exists(spectra_fname + ".npy") and exists(plot_mask_fname):
        mask = np.load(plot_mask_fname)
        spectra = np.load(spectra_fname + ".npy")
    else:
        spectra = unpack_gt_spectra(infname, format=format) # [3,nsmpl]
        assert spectra.shape[1] <= max_spectra_len
        mask = create_spectra_mask(spectra, max_spectra_len)
        spectra = wave_based_sort(spectra)
        spectra, mask, bound = pad_spectra(spectra, mask, max_spectra_len)
        spectra, mask = clean_flux(spectra, mask)
        spectra = normalize_spectra(spectra, bound)
        spectra = convolve_spectra(spectra, bound, std=sigma)
        spectra, mask, bound = mask_spectra_range(spectra, mask, bound, trans_range, trusted_range)
        spectra = spectra.astype(np.float32)

        weight = get_wave_weight(spectra, redshift, emit_wave_distrib, bound)
        spectra = np.concatenate((spectra, weight[None,:]), axis=0)

        interp_trans_data = interpolate_trans(
            trans_data, spectra, bound, fname=spectra_fname, colors=colors)
        spectra = np.concatenate((spectra, interp_trans_data), axis=0)

        if save:
            np.save(spectra_fname + ".npy", spectra)
            np.save(plot_mask_fname, mask)

        if plot:
            plt.plot(spectra[0], spectra[1])
            plt.savefig(spectra_fname + ".png")
            plt.close()

            plt.plot(spectra[0], mask)
            plt.savefig(spectra_fname + "_plot_mask.png")
            plt.close()

    if validator is not None and not validator(spectra_data):
        return None, None
    return spectra, mask

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
    spectra = np.array([wave, flux, ivar])
    return spectra

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
