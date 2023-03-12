
import csv
import torch
import numpy as np
import logging as log
import matplotlib.pyplot as plt

from pathlib import Path
from astropy.io import fits
from os.path import join, exists
from collections import defaultdict
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from astropy.convolution import convolve, Gaussian1DKernel


class SpectraData:
    def __init__(self, fits_obj, trans_obj, dataset_path, device, **kwargs):
        self.kwargs = kwargs
        if not self.require_any_data(kwargs["tasks"]): return

        self.device = device
        self.fits_obj = fits_obj
        self.trans_obj = trans_obj
        self.set_path(dataset_path)
        self.load_accessory_data()
        self.load_spectra()

    def require_any_data(self, tasks):
        tasks = set(tasks)

        self.recon_gt_spectra = "recon_gt_spectra" in tasks
        self.recon_dummy_spectra = "recon_dummy_spectra" in tasks
        self.recon_codebook_spectra = "recon_codebook_spectra" in tasks

        self.spectra_supervision_train = "train" in tasks and self.kwargs["spectra_supervision"]
        self.recon_spectra = self.recon_gt_spectra or self.recon_dummy_spectra or self.recon_codebook_spectra
        self.require_spectra_coords = self.kwargs["mark_spectra"] and ("plot_redshift" in tasks or "plot_embed_map" in tasks)

        return self.kwargs["space_dim"] == 3 and (
            self.spectra_supervision_train or \
            self.recon_spectra or self.require_spectra_coords)

    def set_path(self, dataset_path):
        input_path = join(dataset_path, "input")
        self.spectra_path = join(input_path, "spectra")
        self.input_fits_path = join(input_path, "input_fits")
        self.spectra_data_fname = join(self.spectra_path, "spectra.csv")

    def load_accessory_data(self):
        self.full_wave = self.trans_obj.get_full_wave()

    def load_spectra(self):
        """ Load gt and/or dummy spectra data.
        """
        self.data = defaultdict(lambda: None)

        # if self.recon_dummy_spectra:
        #    self.load_dummy_spectra_data()

        if self.require_spectra_coords or self.spectra_supervision_train or self.recon_gt_spectra:
            self.load_gt_spectra_data()
            self.load_plot_spectra_data()
            self.mark_spectra_on_img()

    #############
    # Getters
    #############

    def get_supervision_spectra(self):
        """ Get gt spectra (with same wave range as recon) for supervision.
        """
        return self.data["supervision_spectra"]

    def get_num_gt_spectra(self):
        """ Get number of gt spectra (doesn't count neighbours).
        """
        if self.recon_gt_spectra or self.spectra_supervision_train:
            return len(self.kwargs["gt_spectra_ids"])
        return 0

    def get_spectra_grid_coords(self):
        """ Get grid (training) coords of all selected spectra (gt & dummy, incl. neighbours).
        """
        return self.data["spectra_grid_coords"]

    def get_spectra_img_coords(self):
        """ Get image coords of all selected spectra (gt & dummy, incl. neighbours).
        """
        return self.data["spectra_img_coords"]

    def get_num_spectra_coords(self):
        """ Get number of coords of all selected spectra
            (gt & dummy, incl. neighbours).
        """
        return self.get_spectra_grid_coords().shape[0]

    def get_spectra_coord_ids(self):
        """ Get pixel id of all selected spectra (correspond to coords).
        """
        return self.data["spectra_coord_ids"]

    def get_spectra_supervision_wave_bound_ids(self):
        """ Get ids of boundary lambda of spectra supervision.
        """
        return self.data["spectra_supervision_wave_bound_ids"]

    def get_spectra_recon_wave_bound_ids(self):
        """ Get ids of boundary lambda of recon spectra.
        """
        return self.data["spectra_recon_wave_bound_ids"]

    def get_recon_spectra_wave(self):
        """ Get lambda values (for plotting).
        """
        return self.data["recon_wave"]

    def get_gt_spectra(self):
        """ Get gt spectra (for plotting).
        """
        return self.data["gt_spectra"]

    def get_gt_spectra_wave(self):
        """ Get lambda values (for plotting).
        """
        return self.data["gt_spectra_wave"]

    def get_full_wave(self):
        return self.full_wave

    #############
    # Helpers
    #############

    def load_dummy_spectra_data(self):
        """ Load hardcoded spectra positions for pixels without gt spectra.
            Can be used to compare with codebook spectrum.
        """
        spectra_ids = self.kwargs["dummy_spectra_ids"]
        smpl_interval = self.kwargs["trans_sample_interval"]
        source_spectra_data = read_spectra_data(self.dummy_spectra_data_fname)

        all_img_coords, all_grid_coords, all_ids = [], [], []

        for i, spectra_id in enumerate(spectra_ids):
            ra = source_spectra_data["ra"][spectra_id]
            dec = source_spectra_data["dec"][spectra_id]
            footprint = source_spectra_data["footprint"][spectra_id]
            tile_id = source_spectra_data["tile_id"][spectra_id]
            subtile_id = source_spectra_data["subtile_id"][spectra_id]
            fits_uid = f"{footprint}{tile_id}{subtile_id}"

            img_coords, grid_coords, ids = self.fits_obj.convert_from_world_coords(
                ra, dec, self.kwargs["spectra_neighbour_size"],
                footprint, tile_id, subtile_id)

            all_ids.append(ids)                 # [num_neighbours,1]
            all_img_coords.append(img_coords)
            all_grid_coords.append(grid_coords) # [num_neighbours,2/3]

        self.data["dummy_spectra_coords"] = coords

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
        spectra_ids = self.kwargs["gt_spectra_ids"]
        smpl_interval = self.kwargs["trans_sample_interval"]
        source_spectra_data = read_spectra_data(self.spectra_data_fname)

        recon_waves, gt_waves = [], []
        spectra, supervision_spectra = [], []
        spectra_recon_wave_bound_ids = []
        spectra_supervision_wave_bound_ids = []
        all_ids, all_img_coords, all_grid_coords = [], [], []

        for i, spectra_id in enumerate(spectra_ids):
            ra = source_spectra_data["ra"][spectra_id]
            dec = source_spectra_data["dec"][spectra_id]
            footprint = source_spectra_data["footprint"][spectra_id]
            tile_id = source_spectra_data["tile_id"][spectra_id]
            subtile_id = source_spectra_data["subtile_id"][spectra_id]
            fits_uid = f"{footprint}{tile_id}{subtile_id}"

            log.info(f'spectra: {spectra_id}, {ra}/{dec}')

            # i) get img coord, grid coord, and pixel ids of selected gt spectra
            img_coords, grid_coords, ids = self.fits_obj.convert_from_world_coords(
                ra, dec, self.kwargs["spectra_neighbour_size"],
                footprint, tile_id, subtile_id)

            all_ids.append(ids)                 # [num_neighbours,1]
            all_img_coords.append(img_coords)
            all_grid_coords.append(grid_coords) # [num_neighbours,2/3]

            if not self.spectra_supervision_train and not self.recon_spectra: continue

            # ii) load actual spectra data
            fname = join(self.spectra_path, fits_uid,
                         source_spectra_data["spectra_fname"][spectra_id] + ".npy")

            gt_wave, gt_spectra = load_gt_spectra(
                fname, self.full_wave, smpl_interval,
                interpolate=self.spectra_supervision_train,
                sigma=self.kwargs["spectra_smooth_sigma"])

            # iii) get data for for spectra supervision
            if self.spectra_supervision_train:
                supervision_spectra_wave_bound = [
                    source_spectra_data["spectra_supervision_wave_lo"][spectra_id],
                    source_spectra_data["spectra_supervision_wave_hi"][spectra_id]]

                # the specified range may not coincide exactly with the trans lambda
                # here we replace with closest trans lambda
                (id_lo, id_hi) = get_bound_id(
                    supervision_spectra_wave_bound, self.full_wave, within_bound=False)
                spectra_supervision_wave_bound_ids.append([id_lo, id_hi + 1])
                supervision_spectra_wave_bound = [
                    self.full_wave[id_lo], self.full_wave[id_hi]]

                # clip gt spectra to the specified range
                (id_lo, id_hi) = get_bound_id(
                    supervision_spectra_wave_bound, gt_wave, within_bound=True)
                gt_spectra_for_supervision = gt_spectra[id_lo:id_hi + 1]
                supervision_spectra.append(gt_spectra_for_supervision)

            # iv) get data for gt spectrum plotting
            if self.recon_spectra:
                gt_waves.append(gt_wave)
                spectra.append(gt_spectra)

                if self.kwargs["plot_clipped_spectrum"]:
                    # plot only within a given range
                    recon_spectra_wave_bound = [
                        source_spectra_data["spectrum_plot_wave_lo"][spectra_id],
                        source_spectra_data["spectrum_plot_wave_hi"][spectra_id]]
                else:
                    recon_spectra_wave_bound = [ self.full_wave[0], self.full_wave[-1] ]

                (id_lo, id_hi) = get_bound_id(
                    recon_spectra_wave_bound, self.full_wave, within_bound=False)

                spectra_recon_wave_bound_ids.append([id_lo, id_hi + 1])
                recon_waves.append(
                    np.arange(self.full_wave[id_lo], self.full_wave[id_hi] + 1, smpl_interval)
                )

        coord_dim = 3 if self.kwargs["coords_encode_method"] == "grid" and self.kwargs["grid_dim"] == 3 else 2
        self.data["gt_spectra_coord_ids"] = all_ids  # [num_coords,num_neighbours,1]
        self.data["gt_spectra_img_coords"] = all_img_coords
        self.data["gt_spectra_grid_coords"] = torch.stack(all_grid_coords).type(
            torch.FloatTensor)[:,:,None].view(-1,1,coord_dim) # [num_coords,num_neighbours,.]

        if self.spectra_supervision_train:
            n = self.kwargs["num_supervision_spectra"]
            self.data["supervision_spectra"] = torch.FloatTensor(
                np.array(supervision_spectra))[:n]
            self.data["spectra_supervision_wave_bound_ids"] = spectra_supervision_wave_bound_ids

        if self.recon_spectra:
            self.data["gt_spectra"] = spectra
            self.data["gt_spectra_wave"] = gt_waves
            self.data["gt_recon_wave"] = recon_waves
            self.data["spectra_recon_wave_bound_ids"] = spectra_recon_wave_bound_ids

        ## tmp, dummy redshift
        # if self.kwargs["redshift_supervision"]:
        #     dummy_redshift = torch.arange(1, 1+len(all_ids), dtype=torch.float)
        #     positions = np.array(all_ids).flatten()
        #     self.fits_obj.data["redshift"][positions] = dummy_redshift
        ## ends

    def load_plot_spectra_data(self):
        wave = []
        if (self.recon_gt_spectra and self.kwargs["plot_spectrum_with_gt"]): # or \
           #self.spectra_supervision_train:
            wave.extend(self.data["gt_recon_wave"])

        if self.recon_dummy_spectra:
            wave.extend(self.data["dummy_recon_wave"])

        #if self.recon_codebook_spectra:
        # wave.extend(self.data["codebook_recon_wave"])
        #    wave.extend(self.data["gt_recon_wave"])

        self.data["recon_wave"] = wave

        # get all spectra (gt and dummy) (grid and img) coords for inferrence
        ids, grid_coords, img_coords = [], [], []

        if self.recon_gt_spectra or self.spectra_supervision_train or self.require_spectra_coords:
            ids.extend(self.data["gt_spectra_coord_ids"])
            grid_coords.extend(self.data["gt_spectra_grid_coords"])
            img_coords.extend(self.data["gt_spectra_img_coords"])

        if self.recon_dummy_spectra:
            ids.extend(self.data["dummy_spectra_coord_ids"])
            grid_coords.extend(self.data["dummy_spectra_coords"])

        if len(ids) != 0:         self.data["spectra_coord_ids"] = np.array(ids)
        if len(img_coords) != 0:  self.data["spectra_img_coords"] = np.array(img_coords)
        if len(grid_coords) != 0: self.data["spectra_grid_coords"] = torch.stack(grid_coords)

    #############
    # Utilities
    #############

    def plot_spectrum(self, spectra_dir, name, recon_spectra, save_spectra=False, clip=True, codebook=False):
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
                recon_spectra_wave = np.arange(
                    full_wave[bound_ids[0]], full_wave[bound_ids[-1]],
                    self.kwargs["trans_sample_interval"])
        else:
            if clip:
                bound_ids = self.get_spectra_recon_wave_bound_ids()
            gt_spectra_wave = self.get_gt_spectra_wave()
            recon_spectra_wave = self.get_recon_spectra_wave()

        for i, cur_spectra in enumerate(recon_spectra):
            sub_dir = ""

            # clip spectra use bound, if given
            if clip:
                sub_dir += "clipped_"

                if codebook:
                    (lo, hi) = bound_ids
                elif codebook or (bound_ids is not None and i < len(bound_ids)):
                    (lo, hi) = bound_ids[i]
                else: lo, hi = 0, -1
                cur_spectra = cur_spectra[...,lo:hi]

            # average spectra over neighbours, if required
            if cur_spectra.ndim == 2:
                if self.kwargs["average_spectra"]:
                    cur_spectra = np.mean(cur_spectra, axis=0)
                else: cur_spectra = cur_spectra[0]
            else:
                assert(cur_spectra.ndim == 1)
            cur_spectra /= np.max(cur_spectra)

            # get wave values (x-axis)
            if not clip:
                recon_wave = full_wave
            elif codebook:
                recon_wave = recon_spectra_wave
            elif recon_spectra_wave is not None and i < len(recon_spectra_wave):
                recon_wave = recon_spectra_wave[i]
            else:
                recon_wave = full_wave

            if self.kwargs["plot_spectrum_with_trans"]:
                sub_dir += "with_trans_"
                self.trans_obj.plot_trans()

            plt.plot(recon_wave, cur_spectra, color="black", label="spectrum")

            if not codebook and self.kwargs["plot_spectrum_with_gt"] and i < len(gt_spectra):
                sub_dir += "with_gt"

                cur_gt_spectra = gt_spectra[i]
                cur_gt_spectra_wave = gt_spectra_wave[i]
                cur_gt_spectra /= np.max(cur_gt_spectra)
                plt.plot(cur_gt_spectra_wave, cur_gt_spectra, color="blue", label="gt")

            if sub_dir != "":
                if sub_dir[-1] == "_": sub_dir = sub_dir[:-1]
                cur_spectra_dir = join(spectra_dir, sub_dir)
            else:
                cur_spectra_dir = spectra_dir

            if not exists(cur_spectra_dir):
                Path(cur_spectra_dir).mkdir(parents=True, exist_ok=True)

            fname = join(cur_spectra_dir, f"spectra_{i}_{name}")
            plt.savefig(fname)
            plt.close()

            if save_spectra:
                np.save(fname, cur_spectra)

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
        gt_pixel_ids = self.get_spectra_coord_ids().flatten()
        gt_pixels = self.fits_obj.get_pixels(ids=gt_pixel_ids).numpy()
        gt_pixels = np.round(gt_pixels, 2)
        np.set_printoptions(suppress = True)
        log.info(f"GT spectra pixel values: {gt_pixels}")

        recon_pixels = self.trans_obj.integrate(spectra)
        recon_pixels = np.round(recon_pixels, 2)
        log.info(f"Recon. spectra pixel values: {recon_pixels}")

# SpectraData class ends
#############

def read_spectra_data(fname):
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

def load_gt_spectra(fname, full_wave, smpl_interval, interpolate=False, sigma=-1):

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
    gt = np.load(fname)
    gt_wave, gt_spectra = gt[:,0], gt[:,1]
    if sigma != -1:
        gt_spectra = convolve_spectra(gt_spectra, std=sigma)

    if interpolate:
        f_gt = interp1d(gt_wave, gt_spectra)

        # make sure wave range to interpolate stay within gt spectra wave range
        # full_wave is full transmission wave
        (lo_id, hi_id) = get_bound_id( ( min(gt_wave),max(gt_wave) ), full_wave, within_bound=True)
        lo = full_wave[lo_id] # lo <= full_wave[lo_id]
        hi = full_wave[hi_id] # hi >= full_wave[hi_id]

        # new gt wave range with same discretization value as transmission wave
        gt_wave = np.arange(lo, hi + 1, smpl_interval)

        # interpolate new gt wave to get interpolated spectra
        gt_spectra = f_gt(gt_wave)

    return gt_wave, gt_spectra

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
