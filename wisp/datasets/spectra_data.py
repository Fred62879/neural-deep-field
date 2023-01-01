
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from os.path import join, exists
from scipy.interpolate import interp1d
from wisp.utils.common import worldToPix
from astropy.convolution import convolve, Gaussian1DKernel


class SpectraData:
    def __init__(self, fits_obj, trans_obj, dataset_path, device, **kwargs):
        if not self.require_any_data(kwargs["tasks"]): return

        self.kwargs = kwargs
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
        self.spectra_supervision_train = "spectra_supervision" in tasks
        return self.recon_gt_spectra or self.recon_dummy_spectra or \
            self.spectra_supervision_train

    def set_path(self, dataset_path):
        input_path = join(dataset_path, "input")
        self.spectra_path = join(input_path, "spectra")
        self.input_fits_path = join(input_path, "input_fits")
        self.spectra_data_fname = join(self.spectra_path, "spectra.csv")

    def load_accessory_data(self):
        self.use_full_fits = self.kwargs["use_full_fits"]
        self.fits_ids = self.fits_obj.get_fits_ids()
        self.num_rows = self.fits_obj.get_num_rows()
        self.num_cols = self.fits_obj.get_num_cols()
        self.fits_cutout_sizes = self.fits_obj.get_fits_cutout_sizes()
        self.fits_cutout_start_pos = self.fits_obj.get_fits_cutout_start_pos()

        self.full_wave = self.trans_obj.get_full_wave()
        self.full_wave_bound = self.trans_obj.get_full_wave_bound()

    def load_spectra(self):
        """ Load gt and/or dummy spectra data.
        """
        self.data = {}
        if self.spectra_supervision_train or self.recon_gt_spectra:
            self.load_gt_spectra_data()
        if self.recon_dummy_spectra:
            self.load_dummy_spectra_data()
        self.load_plot_spectra_data()

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
            return len(self.kwargs["gt_spectra_choices"])
        return 0

    '''
    def get_gt_spectra_coords(self):
        """ Get coords of all (not only supervision) gt spectra.
        """
        if self.spectra_supervision_train:
            return self.data["gt_spectra_coords"]
        return None
    '''

    def get_spectra_coords(self):
        """ Get coords of all selected spectra (gt & dummy, incl. neighbours).
        """
        return self.data["spectra_coords"]

    def get_num_spectra_coords(self):
        """ Get number of coords of all selected spectra
            (gt & dummy, incl. neighbours).
        """
        return self.get_spectra_coords().shape[0]

    def get_spectra_coord_ids(self):
        """ Get pixel id of all selected spectra (correspond to coords).
        """
        return self.data["spectra_coord_ids"]

    def get_recon_wave_bound_ids(self):
        """ Get ids of boundary lambda of recon spectra
            (for supervision and plotting).
        """
        return self.data["recon_wave_bound_ids"]

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

    #############
    # Helpers
    #############

    def load_dummy_spectra_data(self):
        """ Load hardcoded spectra positions for pixels without gt spectra.
            Can be used to compare with codebook spectrum.
        """
        size = self.kwargs["spectrum_neighbour_size"] // 2
        acc, coords = 0, []
        # for each fits (or cutout), select multiple coords
        positions = [[32,32],[10,21],
                     [10,21],[2,2]]

        # for i, fits_id in enumerate(self.fits_ids):
        #     pixel_ids = []
        #     for (r,c) in positions:
        #         neighbours = [ [r-i,c-j] for i in range(size) for j in range(size)]
        #         if self.use_full_fits:
        #         cur_ids = np.array([c[0] * config["img_sz"] + c[1] for r,c in pos])

        #         # w/ neighbour averaging
        #         pixel_ids += acc

        #     coords.append(self.data["coords"][pixel_ids])

        #     # accumulatively count number of pixels
        #     if self.use_full_fits:
        #         acc += self.num_rows[fits_id] * self.cols[fits_id]
        #     else: acc += self.fits_cutout_sizes[i]**2

        self.data["dummy_spectra_coords"] = coords

    def load_gt_spectra_data(self):
        """ Load gt spectra data. Note that:
              During training, we only consider w/ supervision
              During inferrence, the two options give same coords but different spectra

            i) spectra data (w/ supervision) (used for training and inferrence):
               gt spectra
               spectra coords
           ii) spectra data (w/ o/ supervision) (for spectrum plotting only)
                  (coord with gt spectra located at center of cutout)
               gt spectra
               spectra coords
        """
        choices = self.kwargs["gt_spectra_choices"]
        smpl_interval = self.kwargs["trans_sample_interval"]
        source_spectra_data = read_spectra_data(self.spectra_data_fname)

        all_coords, all_ids = [], []
        recon_bound_ids = []
        recon_waves, gt_waves = [], []
        spectra, supervision_spectra = [], []

        for i, choice in enumerate(choices):
            footprint = source_spectra_data["footprint"][choice]
            tile_id = source_spectra_data["tile_id"][choice]
            subtile_id = source_spectra_data["subtile_id"][choice]
            fits_id = f"{footprint}{tile_id}{subtile_id}"

            # get (neighbour) coord & id of pixel(s) with gt spectra
            coords, ids = self.get_gt_spectra_pixel_coords(
                source_spectra_data, choice, self.kwargs["spectra_neighbour_size"])
            all_ids.append(ids)       # [num_neighbours,1]
            all_coords.append(coords) # [num_neighbours,2]

            # get range of wave that we reconstruct spectra
            recon_wave_bound = [source_spectra_data["trusted_wave_lo"][choice],
                                source_spectra_data["trusted_wave_hi"][choice]]
            (lo, hi) = recon_wave_bound
            recon_wave = np.arange(lo, hi + 1, smpl_interval)
            recon_wave_bound_id = get_bound_id(recon_wave_bound, self.full_wave)
            recon_waves.append(recon_wave)
            recon_bound_ids.append(recon_wave_bound_id)

            # load gt spectra data
            fname = join(self.spectra_path, fits_id,
                         source_spectra_data["spectra_fname"][choice] + ".npy")
            gt_wave, gt_spectra, gt_spectra_for_supervision = load_gt_spectra(
                fname, self.full_wave_bound, recon_wave_bound, smpl_interval,
                interpolate=self.spectra_supervision_train)

            gt_waves.append(gt_wave)
            spectra.append(gt_spectra)
            if self.spectra_supervision_train:
                supervision_spectra.append(gt_spectra_for_supervision)

        self.data["gt_spectra"] = spectra
        self.data["gt_spectra_wave"] = gt_waves
        self.data["gt_recon_wave"] = recon_waves
        self.data["gt_spectra_coord_ids"] = all_ids  # [num_coords,num_neighbours,1]
        self.data["recon_wave_bound_ids"] = recon_bound_ids

        # [num_coords,num_neighbours,1,2]
        self.data["gt_spectra_coords"] = torch.stack(all_coords).type(
            torch.FloatTensor).to(self.device)[:,:,None]
        self.data["gt_spectra_coords"] = self.data["gt_spectra_coords"].view(-1,1,2)

        if self.spectra_supervision_train:
            n = self.kwargs["num_supervision_spectra"]
            self.data["supervision_spectra"] = torch.FloatTensor(
                np.array(supervision_spectra)).to(self.device)[:n]

    def load_plot_spectra_data(self):
        wave = []
        if self.recon_gt_spectra or self.spectra_supervision_train:
            wave.extend(self.data["gt_recon_wave"])
        if self.recon_dummy_spectra:
            wave.extend(self.data["dummy_recon_wave"])
        self.data["recon_wave"] = wave

        ids = []
        if self.recon_gt_spectra or self.spectra_supervision_train:
            ids.extend(self.data["gt_spectra_coord_ids"])
        if self.recon_dummy_spectra:
            ids.extend(self.data["dummy_spectra_coord_ids"])
        self.data["spectra_coord_ids"] = np.array(ids)

        # get all spectra (gt and dummy) coords for inferrence
        coords = []
        if self.recon_gt_spectra or self.spectra_supervision_train:
            coords.extend(self.data["gt_spectra_coords"])
        if self.recon_dummy_spectra:
            coords.extend(self.data["dummy_spectra_coords"])
        self.data["spectra_coords"] = torch.stack(coords)

    def get_gt_spectra_pixel_coords(self, spectra_data, choice, neighbour_size):
        """ Get coordinate of pixel with gt spectra. We can either
             i) directly normalize the given ra/dec with coordinates range or
            ii) get pixel id based on ra/dec and index from loaded coordinates.

            Since ra/dec values from spectra data may not be as precise as
              real coordinates calculated from wcs package, method i) tends to
              be inaccurate and method ii) is more reliable.
            However, if given ra/dec is not within loaded coordinates, we
              can only use method i)

           Neighbour_Size specify the neighbourhood of the given spectra to average.
        """
        ra, dec = spectra_data["ra"][choice], spectra_data["dec"][choice]
        footprint = spectra_data["footprint"][choice]
        tile_id = spectra_data["tile_id"][choice]
        subtile_id = spectra_data["subtile_id"][choice]
        fits_id = footprint + tile_id + subtile_id

        # ra/dec values from spectra data may not be exactly the same as real coords
        # this normalized ra/dec may thsu be slightly different from real coords
        # (ra_lo, ra_hi, dec_lo, dec_hi) = np.load(self.fits_obj.coords_range_fname)
        # coord_loose = ((ra - ra_lo) / (ra_hi - ra_lo),
        #                (dec - dec_lo) / (dec_hi - dec_lo))

        # get random header
        random_name = f"calexp-HSC-G-{footprint}-{tile_id}%2C{subtile_id}.fits"
        fits_fname = join(self.input_fits_path, random_name)
        header = fits.open(fits_fname)[1].header

        # index coord from original coords array to get accurate coord
        # this only works if spectra coords is included in the loaded coords
        (r, c) = worldToPix(header, ra, dec)
        pixel_ids = self.fits_obj.get_pixel_ids(fits_id, r, c, neighbour_size)
        print(ra, dec, r, c)
        coords_accurate = self.fits_obj.get_coord(pixel_ids)
        return coords_accurate, pixel_ids

    #############
    # Utilities
    #############

    def plot_spectrum(self, spectra_dir, name, recon_spectra):
        """ Plot given spectra.
            @Param
              recon_spectra: [num_spectra,num_neighbours,full_num_smpl]
        """
        for i, cur_spectra in enumerate(recon_spectra):
            (lo, hi) = self.get_recon_wave_bound_ids()[i]
            cur_spectra = cur_spectra[...,lo:hi]

            if self.kwargs["average_spectra"]:
                cur_spectra = np.mean(cur_spectra, axis=0)
            else: cur_spectra = cur_spectra[0]
            cur_spectra /= np.max(cur_spectra)

            if self.kwargs["plot_spectrum_with_trans"]:
                self.trans_obj.plot_trans()

            plt.plot(self.get_recon_spectra_wave()[i], cur_spectra,
                     color="black", label="spectrum")

            if i < self.get_num_gt_spectra():
                cur_gt_spectra = self.get_gt_spectra()[i]
                cur_gt_spectra_wave = self.get_gt_spectra_wave()[i]
                cur_gt_spectra /= np.max(cur_gt_spectra)
                plt.plot(cur_gt_spectra_wave, cur_gt_spectra, color="blue", label="gt")

            fname = join(spectra_dir, f"spectra_{i}_{name}.png")
            plt.savefig(fname)
            plt.close()

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

def convolve_spectra(spectra, std=50, border=True):
    """ Smooth gt spectra with given std.
        If border is True, we add 1 padding at two ends when convolving.
    """
    kernel = Gaussian1DKernel(stddev=std)
    if border:
        nume = convolve(spectra, kernel)
        denom = convolve(np.ones(spectra.shape), kernel)
        return nume / denom
    return convolve(spectra, kernel)

def get_bound_id(wave_bound, full_wave):
    """ Get id of lambda values in full wave that tightly bounds given range.
        full_wave[id_lo] >= wave_lo
        full_wave[id_hi] <= wave_hi (before adding 1)
    """
    if type(full_wave).__module__ == "torch":
        full_wave = full_wave.numpy()

    wave_lo, wave_hi = wave_bound
    wave_hi = int(min(wave_hi, int(max(full_wave))))
    wave_id_hi = np.argmin((full_wave < wave_hi))
    wave_id_lo = np.argmax((full_wave >= wave_lo))
    return [wave_id_lo, wave_id_hi + 1]

def load_gt_spectra(fname, trans_wave_bound, recon_wave_bound, smpl_interval, interpolate=False):
    """ Load gt spectra (intensity values) for spectra supervision and
          spectrum plotting. Also smooth the gt spectra which has significantly
          larger discretization values than the transmission data.
        If requried, interpolate with same discretization value as trans data.

        @Param
          fname: filename of np array that stores the gt spectra data.
          trans_wave_bound: lower and upper lambda value for the transmission data.
          recon_wave_bound: bound for wave range of the reconstructed spectra.
          smpl_interval: discretization values of the transmission data.
        @Return
          gt_wave/spectra: spectra data with the corresponding lambda values.
          gt_spectra_for_supervision:
            None if not interpolate
            o.w. gt spectra data tightly bounded by recon wave bound.
                 we range of it is identical to that of recon spectra and thus
                 can be directly compare with.
    """
    gt = np.load(fname)
    gt_wave, gt_spectra = gt[:,0], gt[:,1]
    gt_spectra = convolve_spectra(gt_spectra)
    gt_spectra_for_supervision = None

    if interpolate:
        f_gt = interp1d(gt_wave, gt_spectra)

        # make sure wave range to interpolate stay within gt wave range
        (lo, hi) = trans_wave_bound
        lo = max(lo, min(gt_wave))
        hi = min(hi, max(gt_wave))

        # new gt wave range with same discretization value as trans
        gt_wave = np.arange(lo, hi + 1, smpl_interval)

        # interpolate new gt wave to get interpolated spectra
        gt_spectra = f_gt(gt_wave)

        (lo, hi) = get_bound_id(recon_wave_bound, gt_wave)
        gt_spectra_for_supervision = gt_spectra[lo:hi]

    gt_spectra /= np.max(gt_spectra)
    return gt_wave, gt_spectra, gt_spectra_for_supervision

def overlay_spectrum(gt_fn, gen_wave, gen_spectra):
    gt = np.load(gt_fn)
    gt_wave, gt_spectra = gt[:,0], gt[:,1]
    gt_spectra = convolve_spectra(gt_spectra)

    gen_lo_id = np.argmax(gen_wave>gt_wave[0]) + 1
    gen_hi_id = np.argmin(gen_wave<gt_wave[-1])
    #print(gen_lo_id, gen_hi_id)
    #print(gt_wave[0], gt_wave[-1], np.min(gen_wave), np.max(gen_wave))

    wave = gen_wave[gen_lo_id:gen_hi_id]
    gen_spectra = gen_spectra[gen_lo_id:gen_hi_id]
    f = interpolate.interp1d(gt_wave, gt_spectra)
    gt_spectra_intp = f(wave)
    return wave, gt_spectra_intp, gen_spectra
