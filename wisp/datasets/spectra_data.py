
import csv
import torch
import pandas
import numpy as np
import logging as log
import matplotlib.pyplot as plt

from wisp.datasets.patch_data import PatchData
from wisp.datasets.data_utils import add_dummy_dim, \
    create_patch_uid, set_input_path, patch_exists

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
    def __init__(self, fits_obj, trans_obj, dataset_path, device, **kwargs):
        self.kwargs = kwargs
        if not self.require_any_data(kwargs["tasks"]): return

        self.device = device
        self.fits_obj = fits_obj
        self.trans_obj = trans_obj
        self.dataset_path = dataset_path

        self.num_bands = kwargs["num_bands"]
        self.space_dim = kwargs["space_dim"]
        self.num_gt_spectra = kwargs["num_gt_spectra"]
        self.all_tracts = kwargs["spectra_tracts"]
        self.all_patches_r = kwargs["spectra_patches_r"]
        self.all_patches_c = kwargs["spectra_patches_c"]
        self.num_tracts = len(self.all_tracts)
        self.num_patches_c = len(self.all_patches_c)
        self.num_patches = len(self.all_patches_r)*len(self.all_patches_c)

        self.wave_lo = kwargs["spectrum_plot_wave_lo"]
        self.wave_hi = kwargs["spectrum_plot_wave_hi"]
        self.smooth_sigma = kwargs["spectra_smooth_sigma"]
        self.neighbour_size = kwargs["spectra_neighbour_size"]
        self.spectra_data_source = kwargs["spectra_data_source"]
        self.wave_discretz_interval = kwargs["trans_sample_interval"]
        self.trusted_wave_range = None if not kwargs["trusted_range_only"] \
            else [kwargs["spectra_supervision_wave_lo"],
                  kwargs["spectra_supervision_wave_hi"]]
        self.load_spectra_data_from_cache = kwargs["load_spectra_data_from_cache"]

        self.set_path(dataset_path)
        self.load_accessory_data()
        self.load_spectra()

    def require_any_data(self, tasks):
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
        self.spectra_supervision_train = self.kwargs["spectra_supervision"] and "train" in tasks

        return self.kwargs["space_dim"] == 3 and (
            self.codebook_pretrain or \
            self.pretrain_infer or \
            self.spectra_supervision_train or \
            self.recon_spectra or self.require_spectra_coords)

    def set_path(self, dataset_path):
        """ Create path and filename of required files.
            source_metadata_table:    ra,dec,zspec,spectra_fname
            processed_metadata_table: added pixel val and tract-patch
        """
        paths = []
        self.input_patch_path, _ = set_input_path(
            dataset_path, self.kwargs["sensor_collection_name"])

        spectra_path = join(dataset_path, "input/spectra")

        if self.spectra_data_source == "manual":
            assert 0
            self.source_metadata_table_fname = join(spectra_path, "deimos_old", "spectra.csv")

        elif self.spectra_data_source == "deimos":
            path = join(spectra_path, "deimos")

            self.source_spectra_path = join(path, "source_spectra")
            self.source_metadata_table_fname = join(path, "source_deimos_table.tbl")

            processed_data_path = join(path, "processed_"+self.kwargs["processed_spectra_cho"])
            self.processed_spectra_path = join(processed_data_path, "processed_spectra")
            self.processed_metadata_table_fname = join(
                processed_data_path, "processed_deimos_table.tbl")
            self.processed_spectra_data_fname = join(
                processed_data_path, "processed_spectra.npy")

        elif self.spectra_data_source == "zcosmos":
            path = join(spectra_path, "zcosmos")

            self.source_spectra_path = join(path, "source_spectra")
            self.source_metadata_table_fname = join(path, "source_zcosmos_table.fits")

            processed_data_path = join(path, "processed_"+self.kwargs["processed_spectra_cho"])
            self.processed_spectra_path = join(processed_data_path, "processed_spectra")
            self.processed_metadata_table_fname = join(
                processed_data_path, "processed_zcosmos_table.fits")
            self.processed_spectra_data_fname = join(
                processed_data_path, "processed_spectra.npy")

        else: raise ValueError("Unsupported spectra data source choice.")

        for path in [processed_data_path, self.processed_spectra_path]:
            Path(path).mkdir(parents=True, exist_ok=True)

    def load_accessory_data(self):
        self.full_wave = self.trans_obj.get_full_wave()

    def load_spectra(self):
        """ Load gt and/or dummy spectra data.
        """
        self.data = defaultdict(lambda: [])

        if self.recon_gt_spectra or \
           self.codebook_pretrain or \
           self.pretrain_infer or \
           self.require_spectra_coords or \
           self.spectra_supervision_train:
            self.load_gt_spectra_data()

        # if self.recon_dummy_spectra:
        #     self.load_dummy_spectra_data()

        # self.load_plot_spectra_data()

        # if self.recon_gt_spectra:
        #     self.mark_spectra_on_img()

    #############
    # Getters
    #############

    def get_full_wave(self):
        return self.full_wave

    def get_num_spectra_to_plot(self):
        return len(self.data["spectra_grid_coords"])

    def get_num_gt_spectra(self):
        """ Get number of gt spectra (doesn't count neighbours). """
        if self.recon_gt_spectra or self.spectra_supervision_train:
            return len(self.kwargs["gt_spectra_ids"])
        return 0

    def get_spectra_grid_coords(self):
        """ Get grid (training) coords of all selected spectra (gt & dummy, incl. neighbours).
        """
        return self.data["spectra_grid_coords"]

    def get_spectra_img_coords(self):
        """ Get image coords of all selected spectra (gt & dummy, incl. neighbours). """
        return self.data["spectra_img_coords"]

    def get_num_spectra_coords(self):
        """ Get number of coords of all selected spectra
            (gt & dummy, incl. neighbours).
        """
        return self.get_spectra_grid_coords().shape[0]

    def get_spectra_coord_ids(self):
        """ Get pixel id of all selected spectra (correspond to coords). """
        return self.data["spectra_coord_ids"]

    def get_spectra_supervision_wave_bound_ids(self):
        """ Get ids of boundary lambda of spectra supervision. """
        return self.data["spectra_supervision_wave_bound_ids"]

    def get_spectra_recon_wave_bound_ids(self):
        """ Get ids of boundary lambda of recon spectra. """
        return self.data["spectra_recon_wave_bound_ids"]

    def get_recon_spectra_wave(self):
        """ Get lambda values (for plotting). """
        return self.data["recon_wave"]

    def get_gt_spectra(self):
        """ Get gt spectra (for plotting). """
        return self.data["gt_spectra"]

    def get_gt_spectra_pixels(self):
        return self.data["gt_spectra_pixels"]

    def get_gt_spectra_pixels(self):
        return self.data["gt_spectra_redshift"]

    def get_gt_spectra_wave(self):
        """ Get lambda values (for plotting). """
        return self.data["gt_spectra_wave"]

    def get_supervision_spectra(self, idx=None):
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

    #############
    # Helpers
    #############

    def load_plot_spectra_data(self):
        wave = []
        if self.recon_gt_spectra: # and self.kwargs["plot_spectrum_with_gt"]):
            wave.extend(self.data["gt_recon_wave"])

        if self.recon_dummy_spectra:
            wave.extend(self.data["dummy_recon_wave"])

        self.data["recon_wave"] = wave

        # get all spectra (gt and dummy) (grid and img) coords for inferrence
        ids, grid_coords, img_coords = [], [], []

        if self.recon_gt_spectra or self.spectra_supervision_train or self.require_spectra_coords:
            ids.extend(self.data["gt_spectra_coord_ids"])
            grid_coords.extend(self.data["gt_spectra_grid_coords"])
            img_coords.extend(self.data["gt_spectra_img_coords"])

        if self.recon_dummy_spectra:
            grid_coords.extend(self.data["dummy_spectra_grid_coords"])

        if len(ids) != 0:         self.data["spectra_coord_ids"] = np.array(ids)
        if len(img_coords) != 0:  self.data["spectra_img_coords"] = np.array(img_coords)
        if len(grid_coords) != 0: self.data["spectra_grid_coords"] = torch.stack(grid_coords)

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
            i) spectra data (for supervision) (used for training and inferrence):
               gt spectra
               spectra coords
           ii) spectra data (no supervision) (for spectrum plotting only)
                  (coord with gt spectra located at center of cutout)
               gt spectra
               spectra coords
               Note that:
              During training, we only load if do spectra supervision
              During inferrence, the two options give same coords but different spectra
        """
        if not self.load_spectra_data_from_cache or \
           not exists(self.gt_spectra_pixs_fname) or \
           not exists(self.gt_spectra_coords_fname) or \
           not exists(self.gt_spectra_fluxes_fname):

            # process and save data for each spectra, individually
            if not exists(self.processed_metadata_table_fname):
                self.process_spectra()
            # load data for each individual spectra
            self.load_processed_spectra()
        else:
            # save data for all spectra together (only when amount of spectra is small)
            self.load_cached_spectra_data()

        #print(self.data["pixs"].shape)
        #print(self.data["gt_spectra_grid_coords"].shape)
        #print(self.data["fluxes"].shape)

        self.transform_data()

    def load_cached_spectra_data(self):
        self.data["pixs"] = np.load(self.gt_spectra_pixs_fname)
        self.data["gt_spectra_grid_coords"] = np.load(self.gt_spectra_coords_fname)
        self.data["fluxes"] = np.load(self.gt_spectra_fluxes_fname)

    def load_processed_spectra(self):
        """ Load processed data for each spectra and save together.
        """
        df = pandas.read_pickle(self.processed_metadata_table_fname)
        num_gt_spectra = len(df)
        pixs, coords, fluxes = [], [], []
        for i in range(num_gt_spectra):
            pixs.append( np.load(join(self.processed_spectra_path, df.iloc[i]["pix_fname"])) )
            coords.append( np.load(join(self.processed_spectra_path, df.iloc[i]["coord_fname"])) )
            fluxes.append( np.load(join(self.processed_spectra_path, df.iloc[i]["spectra_fname"])) )

        self.data["pixs"] = np.concatenate(pixs, axis=0)
        self.data["gt_spectra_grid_coords"] = np.concatenate(coords, axis=0)
        self.data["fluxes"] = np.concatenate(fluxes, axis=0)

        np.save(self.gt_spectra_pixs_fname, self.data["pixs"])
        np.save(self.gt_spectra_coords_fname, self.data["gt_spectra_grid_coords"])
        np.save(self.gt_spectra_fluxes_fname, self.data["fluxes"])

    def transform_data(self):
        coord_dim = 3 if self.kwargs["coords_encode_method"] == "grid" and \
            self.kwargs["grid_dim"] == 3 else 2

        self.data["gt_spectra_grid_coords"] = torch.stack(
            self.data["gt_spectra_grid_coords"]).type(
                torch.FloatTensor)[:,:,None].view(-1,1,coord_dim) #[num_coords,num_neighbours,.]

        if self.spectra_supervision_train or self.codebook_pretrain or self.pretrain_infer:
            n = self.kwargs["num_supervision_spectra"]

            self.data["supervision_spectra"] = torch.FloatTensor(
                np.array(self.data["supervision_spectra"]))[:n]

            if self.kwargs["codebook_pretrain_pixel_supervision"]:
                pixels = torch.stack(self.data["gt_pixels"]).type(torch.FloatTensor)
                self.data["gt_pixels"] = pixels[:,:,None].view(-1,self.kwargs["num_bands"])
                self.data["supervision_pixels"] = self.data["gt_pixels"][:n]

            if self.kwargs["redshift_supervision"]:
                 self.data["gt_redshift"] = torch.FloatTensor(
                     np.array(self.data["gt_redshift"])).flatten()
                 self.data["supervision_redshift"] = self.data["gt_redshift"][:n]

        # iii) get data for for spectra supervision
        supervision_spectra_wave_bound = [
            source_spectra_data["spectra_supervision_wave_lo"][spectra_id],
            source_spectra_data["spectra_supervision_wave_hi"][spectra_id]]

        # find id of min and max lambda in terms of the transmission wave (full_wave)
        # the min and max lambda for the spectra may not coincide exactly with the
        # trans lambda, here we replace with closest trans lambda
        (id_lo, id_hi) = get_bound_id(
            supervision_spectra_wave_bound, self.full_wave, within_bound=False)
        self.data["spectra_supervision_wave_bound_ids"].append([id_lo, id_hi + 1])

        # clip gt spectra to the specified range
        supervision_spectra_wave_bound = [
            self.full_wave[id_lo], self.full_wave[id_hi]]
        (id_lo, id_hi) = get_bound_id(
            supervision_spectra_wave_bound, gt_wave, within_bound=True)
        self.data["supervision_spectra"].append(gt_spectra[id_lo:id_hi + 1])

        # iv) get data for gt spectrum plotting
        if self.kwargs["plot_clipped_spectrum"]:
            # plot only within a given range
            recon_spectra_wave_bound = [
                source_spectra_data["spectrum_plot_wave_lo"][spectra_id],
                source_spectra_data["spectrum_plot_wave_hi"][spectra_id]]
        else:
            recon_spectra_wave_bound = [ self.full_wave[0], self.full_wave[-1] ]

        (id_lo, id_hi) = get_bound_id(
            recon_spectra_wave_bound, self.full_wave, within_bound=False)

        gt_spectra
        gt_recon_wave = np.arange(self.full_wave[id_lo], self.full_wave[id_hi] + 1, smpl_interval)
        gt_spectra_wave = gt_wave
        spectra_recon_wave_bound_ids = [id_lo, id_hi + 1]

    #############
    # Data loading
    #############

    def load_source_metadata(self):
        if self.spectra_data_source == "manual":
            source_spectra_data = read_manual_table(self.manual_table_fname)
        elif self.spectra_data_source == "deimos":
            source_spectra_data = read_deimos_table(self.source_metadata_table_fname)
        elif self.spectra_data_source == "zcosmos":
            source_spectra_data = read_zcosmos_table(self.source_metadata_table_fname)
        else: raise ValueError("Unsupported spectra data source")
        return source_spectra_data

    def process_spectra(self):
        df = self.load_source_metadata().iloc[:self.num_gt_spectra]
        num_gt_spectra = len(df)
        df["tract"] = ""
        df["patch"] = ""

        # load all headers
        header_wcs, headers = [], []
        for tract in self.all_tracts:
            cur_wcs, cur_headers = [], []
            for patch_r in self.all_patches_r:
                for patch_c in self.all_patches_c:
                    if not patch_exists(
                            self.input_patch_path, tract, f"{patch_r},{patch_c}"
                    ) or not (tract == '9812' and patch_r == '1' and patch_c == '5'):
                        cur_headers.append(None)
                        cur_wcs.append(None)
                        continue

                    cur_patch = PatchData(
                        self.dataset_path, tract, f"{patch_r},{patch_c}", **self.kwargs)
                    header = cur_patch.get_header()
                    wcs = WCS(header)
                    cur_headers.append(header)
                    cur_wcs.append(wcs)

            headers.append(cur_headers)
            header_wcs.append(cur_wcs)

        # locate tract and patch for each spectra
        localize = partial(locate_tract_patch,
                           header_wcs, headers, self.all_tracts,
                           self.all_patches_r, self.all_patches_c)
        spectra_ids = [
            [ [] for i in range(self.num_patches) ]
            for j in range(self.num_tracts)
        ] # id of spectra in each patch (contain patches that dont exist)

        spectra_to_drop = []
        for idx in range(num_gt_spectra):
            ra = df.iloc[idx]["ra"]
            dec = df.iloc[idx]["dec"]
            tract, patch, i, j = localize(ra, dec)

            if tract == -1:
                spectra_to_drop.append(idx)
                continue

            # idx spectra belongs to tract i, patch j (which exists)
            spectra_ids[i][j].append(idx)
            df.at[idx,"tract"] = tract
            df.at[idx,"patch"] = patch

        # print(len(spectra_to_drop))
        # print(spectra_ids)

        # load pixels and coords for each spectra
        process_each_patch = partial(self.process_spectra_in_one_patch, df)
        for i, tract in enumerate(self.all_tracts):
            for j, patch_r in enumerate(self.all_patches_r):
                for k, patch_c in enumerate(self.all_patches_c):
                    l = j * self.num_patches_c + k

                    if len(spectra_ids[i][l]) == 0 or \
                       not patch_exists(self.input_patch_path, tract, f"{patch_r},{patch_c}"):
                        continue

                    cur_patch = PatchData(
                        self.dataset_path, tract, f"{patch_r},{patch_c}",
                        load_pixels=True,
                        load_coords=True,
                        pixel_norm_cho=self.kwargs["train_pixels_norm"],
                        **self.kwargs)
                    process_each_patch(cur_patch, spectra_ids[i][l])

        df.drop(spectra_to_drop, inplace=True) # drop nonexist spectra
        df.reset_index(inplace=True)
        df.drop(columns=["index"], inplace=True) # drop extra index added by `reset_index`
        df.to_pickle(self.processed_metadata_table_fname)

    def process_spectra_in_one_patch(self, df, patch, spectra_ids):
        """ Get coords and pixel values for all spectra (specified by spectra_ids)
              within the same given patch.
            @Params
              df: source metadata dataframe for all spectra
              patch: patch that contains the current spectra
              spectra_ids: ids for spectra within the current patch
        """
        if len(spectra_ids) == 0: return

        ras = np.array(list(df.iloc[spectra_ids]["ra"]))
        decs = np.array(list(df.iloc[spectra_ids]["dec"]))

        # get img coords for all spectra within current patch
        wcs = WCS(patch.get_header())
        radecs = np.concatenate((ras[:,None], decs[:,None]), axis=-1)
        img_coords = wcs.all_world2pix(radecs, 1).astype(int) # [n,2]

        process_one_spectra = partial(self.process_one_spectra, df, patch)
        for i, (idx, coord) in enumerate(zip(spectra_ids, img_coords)):
            process_one_spectra(idx, coord)

    def process_one_spectra(self, df, patch, idx, coord):
        """ Get pixel and normalized coord and
              process spectra data for one spectra.
            @Params
              df: source metadata dataframe for all spectra
              patch: patch that contains the current spectra
              idx: spectra idx (within the df table)
              coord: img coord for current spectra
        """
        spectra_fname = df.iloc[idx]["spectra_fname"]
        fname = spectra_fname[:-5]
        pix_fname = f"{fname}_pix.npy"
        coord_fname = f"{fname}_coord.npy"
        df.at[idx,"pix_fname"] = pix_fname
        df.at[idx,"coord_fname"] = coord_fname
        df.at[idx,"spectra_fname"] = f"{fname}.npy"

        pixel_ids = patch.get_pixel_ids(coord[0], coord[1],
                                        neighbour_size=self.neighbour_size)
        pixels = patch.get_pixels(pixel_ids)[None,:]
        np.save(join(self.processed_spectra_path, pix_fname), pixels)
        grid_coords = patch.get_coords(pixel_ids)[None,:] # [n,]
        np.save(join(self.processed_spectra_path, coord_fname), grid_coords)

        # process source spectra and save locally
        process_gt_spectra(
            join(self.source_spectra_path, spectra_fname),
            join(self.processed_spectra_path, fname),
            self.full_wave,
            self.wave_discretz_interval,
            sigma=self.smooth_sigma,
            trusted_range=self.trusted_wave_range,
            format=self.spectra_data_source
        )

    #############
    # Utilities
    #############

    def plot_spectrum(self, spectra_dir, name, recon_spectra, spectra_norm_cho,
                      save_spectra=False, save_spectra_together=False,
                      clip=True, codebook=False):
        """ Plot given spectra.
            @Param
              recon_spectra: [num_spectra(,num_neighbours),full_num_smpl]
        """
        full_wave = self.get_full_wave()
        gt_spectra = self.get_gt_spectra()

        if codebook:
            if clip:
                clip_range = self.kwargs["codebook_spectra_clip_range"]
                bound_ids = get_bound_id(clip_range, full_wave, within_bound=True)
                # print(full_wave)
                # print(full_wave[bound_ids[0]], full_wave[bound_ids[1]])
                # print(clip_range, bound_ids)
                recon_spectra_wave = np.arange(
                    full_wave[bound_ids[0]], full_wave[bound_ids[-1]],
                    self.kwargs["trans_sample_interval"])
        else:
            if clip:
                bound_ids = self.get_spectra_recon_wave_bound_ids()
            gt_spectra_wave = self.get_gt_spectra_wave()
            recon_spectra_wave = self.get_recon_spectra_wave()

        if self.kwargs["plot_spectrum_together"]:
            ncols = min(len(recon_spectra), self.kwargs["num_spectra_plot_per_row"])
            nrows = int(np.ceil(len(recon_spectra) / ncols))
            fig, axs = plt.subplots(nrows, ncols, figsize=(5*ncols,5*nrows))

        for i, cur_spectra in enumerate(recon_spectra):
            sub_dir = ""
            plot_gt_spectra = not codebook and self.kwargs["plot_spectrum_with_gt"] \
                and i < len(gt_spectra)

            if clip:
                sub_dir += "clipped_"

                if codebook:
                    (lo, hi) = bound_ids
                elif bound_ids is not None and i < len(bound_ids):
                    (lo, hi) = bound_ids[i]
                else: lo, hi = 0, cur_spectra.shape[-1]
                cur_spectra = cur_spectra[...,lo:hi]

            # average spectra over neighbours, if required
            if cur_spectra.ndim == 2:
                if self.kwargs["average_spectra"]:
                    cur_spectra = np.mean(cur_spectra, axis=0)
                else: cur_spectra = cur_spectra[0]
            else: assert(cur_spectra.ndim == 1)

            # get wave values (x-axis)
            if not clip:
                recon_wave = full_wave
            elif codebook:
                recon_wave = recon_spectra_wave
            elif recon_spectra_wave is not None and i < len(recon_spectra_wave):
                recon_wave = recon_spectra_wave[i]
            else:
                recon_wave = full_wave

            # normalize spectra within trusted range (gt spectra only)
            if 1: #not codebook:
                sub_dir += spectra_norm_cho + "_"
                # assert(np.max(cur_spectra) > 0)

                if spectra_norm_cho == "max":
                    cur_spectra = cur_spectra / np.max(cur_spectra)
                elif spectra_norm_cho == "sum":
                    cur_spectra = cur_spectra / np.sum(cur_spectra)
                elif spectra_norm_cho == "scale_gt":
                    # scale gt spectra s.t. its max is same as recon
                    cur_recon_max = np.max(cur_spectra)

            if plot_gt_spectra:
                sub_dir += "with_gt_"
                cur_gt_spectra = gt_spectra[i]
                cur_gt_spectra_wave = gt_spectra_wave[i]
                assert(np.max(cur_gt_spectra) > 0)

                if spectra_norm_cho == "max":
                    cur_gt_spectra = cur_gt_spectra / np.max(cur_gt_spectra)
                elif spectra_norm_cho == "sum":
                    cur_gt_spectra = cur_gt_spectra / np.sum(cur_gt_spectra)
                elif spectra_norm_cho == "scale_gt":
                    cur_gt_spectra = cur_gt_spectra / np.max(cur_gt_spectra) * cur_recon_max
                elif spectra_norm_cho == "scale_recon":
                    cur_spectra = cur_spectra / np.max(cur_spectra) * np.max(cur_gt_spectra)

            # plot spectra
            if self.kwargs["plot_spectrum_together"]:
                if nrows == 1:
                    if ncols == 1:
                        axis = axs
                    else:
                        axis = axs[i%ncols]
                else: axis = axs[i//ncols,i%ncols]
            else:
                fig, axs = plt.subplots(1)
                axis = axs[0]

            if self.kwargs["plot_spectrum_with_trans"]:
                sub_dir += "with_trans_"
                self.trans_obj.plot_trans(axis=axis)

            axis.set_title(i)
            axis.plot(recon_wave, cur_spectra, color="black", label="spectrum")
            # axis.set_ylim([0.003,0.007])
            if plot_gt_spectra:
                axis.plot(cur_gt_spectra_wave, cur_gt_spectra, color="blue", label="gt")

            if sub_dir != "":
                if sub_dir[-1] == "_": sub_dir = sub_dir[:-1]
                cur_spectra_dir = join(spectra_dir, sub_dir)
            else:
                cur_spectra_dir = spectra_dir

            if not exists(cur_spectra_dir):
                Path(cur_spectra_dir).mkdir(parents=True, exist_ok=True)

            if not self.kwargs["plot_spectrum_together"]:
                fname = join(cur_spectra_dir, f"spectra_{i}_{name}")
                fig.tight_layout();plt.savefig(fname);plt.close()

            if save_spectra:
                fname = join(cur_spectra_dir, f"spectra_{i}_{name}")
                np.save(fname, cur_spectra)

        if save_spectra_together:
            fname = join(cur_spectra_dir, name)
            np.save(fname, recon_spectra)

        if self.kwargs["plot_spectrum_together"]:
            fname = join(spectra_dir, sub_dir, f"all_spectra_{name}")
            fig.tight_layout();plt.savefig(fname);plt.close()

    def mark_spectra_on_img(self):
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
        log.info(f"GT spectra pixel values: {gt_pixels}")

        recon_pixels = self.trans_obj.integrate(spectra)
        recon_pixels = np.round(recon_pixels, 2)
        log.info(f"Recon. spectra pixel values: {recon_pixels}")

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

def process_gt_spectra(infname, outfname, full_wave, smpl_interval,
                       sigma=-1, trusted_range=None, format="default",
                       interpolate=True, save=True, plot=True):
    """ Load gt spectra (intensity values) for spectra supervision and
          spectrum plotting. Also smooth the gt spectra which has significantly
          larger discretization values than the transmission data.
        If requried, interpolate with same discretization value as trans data.

        @Param
          fname: filename of np array that stores the gt spectra data.
          full_wave: all lambda values for the transmission data.
          smpl_interval: discretization values of the transmission data.
        @Return
          gt_wave/spectra: spectra data with the corresponding lambda values.
                           for plotting purposes only
          gt_spectra_for_supervision:
            None if not interpolate
            o.w. gt spectra data tightly bounded by recon wave bound.
                 the range of it is identical to that of recon spectra and thus
                 can be directly compare with.
    """
    wave, flux, ivar = unpack_gt_spectra(infname, format=format)

    if sigma > 0:
        flux = convolve_spectra(flux, std=sigma)

    if interpolate:
        f_gt = interp1d(wave, flux)

        # make sure wave range to interpolate stay within gt spectra wave range
        # full_wave is full transmission wave
        if trusted_range is not None:
            (lo, hi) = trusted_range
        else:
            (lo_id, hi_id) = get_bound_id(
                ( min(gt_wave),max(gt_wave) ), full_wave, within_bound=True)
            lo = full_wave[lo_id] # lo <= full_wave[lo_id]
            hi = full_wave[hi_id] # hi >= full_wave[hi_id]

        # new gt wave range with same discretization value as transmission wave
        wave_interp = np.arange(lo, hi + 1, smpl_interval)

        # use new gt wave to get interpolated spectra
        flux_interp = f_gt(wave_interp)

    if save:
        np.save(outfname + ".npy", flux_interp[None,:])
    if plot:
        plt.plot(wave_interp, flux_interp)
        plt.savefig(outfname + ".png")
        plt.close()

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

def get_bound_id(wave_bound, source_wave, within_bound=True):
    """ Get id of lambda values in full wave that bounds or is bounded by given wave_bound
        if `within_bound`
            source_wave[id_lo] >= wave_lo
            source_wave[id_hi] <= wave_hi
        else
            source_wave[id_lo] <= wave_lo
            source_wave[id_hi] >= wave_hi
    """
    if type(source_wave).__module__ == "torch":
        source_wave = source_wave.numpy()

    wave_lo, wave_hi = wave_bound
    wave_hi = int(min(wave_hi, int(max(source_wave))))

    if within_bound:
        if wave_lo <= min(source_wave): id_lo = 0
        else: id_lo = np.argmax((source_wave >= wave_lo))

        if wave_hi >= max(source_wave): id_hi = len(source_wave) - 1
        else: id_hi = np.argmax((source_wave > wave_hi)) - 1

        assert(source_wave[id_lo] >= wave_lo and source_wave[id_hi] <= wave_hi)
    else:
        if wave_lo <= min(source_wave): id_lo = 0
        else: id_lo = np.argmax((source_wave > wave_lo)) - 1

        if wave_hi >= max(source_wave): id_hi = len(source_wave) - 1
        else: id_hi = np.argmax((source_wave >= wave_hi))

        assert(source_wave[id_lo] <= wave_lo and source_wave[id_hi] >= wave_hi)

    return [id_lo, id_hi]

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

def read_deimos_table(fname):
    """ Read metadata table for deimos spectra data.
    """
    df = pandas.read_table(fname,comment='#',delim_whitespace=True)
    df.rename(columns={"ID": "id", "fits1d": "spectra_fname"}, inplace=True)
    df.drop(columns=['sel', 'imag', 'kmag', 'Qf', 'Q',
                     'Remarks', 'ascii1d', 'jpg1d', 'fits2d'], inplace=True)
    df.drop([0], inplace=True) # drop first row which is datatype
    df.dropna(subset=["ra","dec","spectra_fname"], inplace=True)
    df.reset_index(inplace=True) # reset index after dropping
    df.drop(columns=["index"], inplace=True)
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

def unpack_gt_spectra(fname, format="default"):
    if format == "default":
        data = np.array(
            pandas.read_table(fname + ".tbl", comment="#", delim_whitespace=True)
        )
        gt_wave, gt_spectra = data[:,0], data[:,1]
    else:
        header = fits.open(fname)[1].header
        data = fits.open(fname)[1].data[0]
        data_names = [header["TTYPE1"],header["TTYPE2"],header["TTYPE3"]]
        wave_id = data_names.index("LAMBDA")
        flux_id = data_names.index("FLUX")
        ivar_id = data_names.index("IVAR")
        wave, flux, ivar = data[wave_id], data[flux_id], data[ivar_id]

    return wave, flux, ivar

def download_deimos_data_parallel(http_prefix, local_path, fnames):
    urls = [f"{http_prefix}/{fname}" for fname in fnames]
    out_fnames = [f"{local_path}/{fname}" for fname in fnames]
    inputs = zip(urls, out_fnames)
    cpus = cpu_count()
    results = ThreadPool(cpus - 1).imap_unordered(download_from_url, inputs)
    for result in results:
        print('url:', result[0], 'time (s):', result[1])

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
