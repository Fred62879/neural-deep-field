
import csv
import torch
import numpy as np
import scipy.interpolate as interpolate

from astropy.io import fits
from os.path import join, exists
from wisp.utils.common import worldToPix
from astropy.convolution import convolve, Gaussian1DKernel


class SpectraData:
    def __init__(self, fits_obj, trans_obj, dataset_path, tasks, device, **kwargs):
        if not self.require_any_data(tasks): return

        self.kwargs = kwargs
        self.device = device
        self.fits_obj = fits_obj
        self.trans_obj = trans_obj
        self.set_path(dataset_path)
        self.load_accessory_data()
        self.load_spectra()

    def require_any_data(self, tasks):
        self.recon_spectra = "recon_spectra" in tasks
        self.plot_dummy_spectrum = "plot_dummy_spectrum" in tasks
        self.spectra_supervision = "spectra_supervision" in tasks
        return self.recon_spectra or self.plot_dummy_spectrum or \
            self.spectra_supervision

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

    def load_spectra(self):
        """ Load gt and/or dummy spectra data.
        """
        self.data = {}
        if self.plot_dummy_spectrum:
            self.load_dummy_spectra()
        if self.spectra_supervision or self.recon_spectra:
            self.load_gt_spectra()

    ################
    # Helpers
    ################

    def load_dummy_spectra(self):
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

    def load_gt_spectra(self):
        """ Load gt spectra data depending on given tasks.
            We support loading only one of two sets of data to avoid confusion.

            i) spectra supervision data (used for training and inferrence):
               gt spectra
               spectra coords
            ii) centerize gt spectra data (for spectrum plotting only)
                  (coord with gt spectra located at center of cutout)
                gt spectra
                spectra coords
        """
        choices = self.kwargs["gt_spectra_choices"]
        source_spectra_data = read_spectra_data(self.spectra_data_fname)

        smpl_interval = self.kwargs["trans_sample_interval"]
        wave_range = [self.kwargs["trusted_wave_lo"], self.kwargs["trusted_wave_hi"]]

        spectra = []
        coords = torch.zeros(len(choices), 2)
        for i, choice in enumerate(choices):
            footprint = source_spectra_data["footprint"][choice]
            tile_id = source_spectra_data["tile_id"][choice]
            subtile_id = source_spectra_data["subtile_id"][choice]
            fits_id = f"{footprint}{tile_id}{subtile_id}"

            # generate gt spectra fnames
            fname = join(self.spectra_path, fits_id,
                         source_spectra_data["spectra_fname"][choice] + ".npy")

            # load gt spectra data
            gt_spectra = load_supervision_gt_spectra(fname, wave_range, smpl_interval)
            spectra.append(gt_spectra)

            # get coords of gt spectra pixels
            coords[i] = self.get_gt_spectra_pixel_coords(source_spectra_data, choice)

            # get id bound of trusted wave range
            if self.spectra_supervision:
                self.data["trusted_wave_range_id"] = self.trans_obj.get_bound_id(wave_range)

        self.data["gt_spectra"] = torch.FloatTensor(np.array(spectra))
        self.data["gt_spectra_coords"] = torch.FloatTensor(coords).to(self.device)
        if self.spectra_supervision:
            self.data["supervision_gt_spectra"] = torch.FloatTensor(spectra).to(self.device)

    def get_gt_spectra_pixel_coords(self, spectra_data, choice):
        """ Get coordinate of pixel with gt spectra. We can either
             i) directly normalize the given ra/dec with coordinates range or
            ii) get pixel id based on ra/dec and index from loaded coordinates.

            Since ra/dec values from spectra data may not be as precise as
              real coordinates calculated from wcs package, method i) tends to
              be inaccurate and method ii) is more reliable.
            However, if given ra/dec is not within loaded coordinates, we
              can only use method i)
        """
        ra, dec = spectra_data["ra"][choice], spectra_data["dec"][choice]
        footprint = spectra_data["footprint"][choice]
        tile_id = spectra_data["tile_id"][choice]
        subtile_id = spectra_data["subtile_id"][choice]

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
        pixel_id = self.get_pixel_id(footprint, tile_id, subtile_id, r, c)
        coord_accurate = self.fits_obj.get_coord(pixel_id)
        return coord_accurate

    def get_pixel_id(self, footprint, tile_id, subtile_id, r, c):
        """ Get global id of given position based on its
              local r/c position in given fits tile.
        """
        fits_id = footprint + tile_id + subtile_id
        i, count, found = 0, 0, False

        # count total number of pixels before the given tile
        for cur_fits_id in self.fits_ids:
            if cur_fits_id == fits_id: found = True; break
            if self.use_full_fits:
                count += self.num_rows[cur_fits_id] * self.num_cols[cur_fits_id]
            else: count += self.fits_cutout_sizes[i]**2
            i += 1
        assert(found)

        # count number of pixels before given position in given tile
        if self.use_full_fits:
            count += r * self.num_cols[fits_id] + c
        else:
            (start_r, start_c) = self.fits_cutout_start_pos[i]
            size = self.fits_cutout_sizes[i]
            count += size * (r - start_r) + c - start_c
        return count

    def load_spectrum_plotting_data(self):
        if not config["plot_spectrum"] and not config["plot_cdbk_spectrum"]:
            return

        config["spectrum_labels"] = ["g", "r", "i", "z", "y", "nb387", "nb816", "nb921","u","u*"]
        config["spectrum_colors"] = ["green","red","blue","gray","yellow","gray","red","blue","yellow","blue"]
        config["spectrum_styles"] = ["solid","solid","solid","solid","solid","dashed","dashed","dashed","dashdot","dashdot"]

        # get coordinates (pixl id or ra/dec form) for all specified spectra pixel
        # together with the corspd gt spectra data filename
        if config["is_test"]:
            fn = "fake_spectrum"+config["sensor_collection_name"]+str(config["fake_spectra_cho"])+".npy"
            config["spectra_coords"] = [{"coords": config["fake_coord"],"radec":False}]
            config["gt_spectra_fns"] = [join(config["spectra_dir"], fn)]

        elif config["spectra_supervision"]:
            add_spectra_supervision_args(config)

        elif config["plot_centerize_spectrum"]:
            add_centerize_gt_spectra_args(config)

        else: # hardcode pixel r/c position to plot spectrum w/o gt spectra
            pass

    #############
    # Getters
    #############

    def get_num_spectra_coords(self):
        return self.get_spectra_coords().shape[0]

    def get_dummy_spectra_coords(self):
        return self.data["dummy_spectra_coords"]

    def get_spectra_coords(self):
        return self.data["gt_spectra_coords"]

    def get_supervision_gt_spectra(self):
        return self.data["supervision_gt_spectra"]

    def get_gt_spectra(self):
        return self.data["gt_spectra"]

    def get_trusted_wave_range_id(self):
        return self.data["trusted_wave_range_id"]

# SpectraData class ends
#############

#############
# Utilities
#############

def read_spectra_data(fname):
    data = {}
    colnames = None
    datatypes = None
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

def process_gt_spectra(gt_fname):
    """ Process gt spectra for spectrum plotting.
    """
    gt = np.load(gt_fname)
    gt_wave, gt_spectra = gt[:,0], gt[:,1]
    lo, hi = np.min(gt_spectra), np.max(gt_spectra)
    if hi != lo:
        gt_spectra = (gt_spectra - lo) / (hi - lo)
    gt_spectra = convolve_spectra(gt_spectra)
    return gt_wave, gt_spectra

def load_supervision_gt_spectra(fname, wave_range, smpl_interval):
    """ Load gt spectra (intensity values) for spectra supervision.
        Interpolate original spectra data to have same discretization value as trans data.
        @Param
          fname: filename of np array that stores gt spectra data
          wave_range: trusted wave lower and upper bound (range that we interpolate)
          smpl_interval: discretization interval of transmission data.
    """
    gt = np.load(fname)
    gt_wave, gt_spectra = gt[:,0], gt[:,1]
    gt_spectra = convolve_spectra(gt_spectra)
    f_gt = interpolate.interp1d(gt_wave, gt_spectra)

    # assume lo, hi is within range of gt wave
    (lo, hi) = wave_range
    trusted_wave = np.arange(lo, hi + 1, smpl_interval)
    smpl_spectra = f_gt(trusted_wave)
    smpl_spectra /= np.max(smpl_spectra)
    return smpl_spectra

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

#############
# Plotting functions
#############

def plot_spectrum(model_id, i, spectrum_dir, spectra, spectrum_wave,
                  orig_wave, orig_transms, colors, lbs, styles):
    """ Plot spectrum with sensor transmission as background.
        spectra [bsz,nsmpl]
    """
    for j, cur_spectra in enumerate(spectra):
        for k, trans in enumerate(orig_transms):
            plt.plot(orig_wave, trans, color=colors[k], label=lbs[k], linestyle=styles[k])

        if spectrum_wave.ndim == 3: # bandwise
            cur_s_wave = spectrum_wave[j].flatten()
            cur_s_wave, ids = torch.sort(cur_s_wave)
            cur_spectra = cur_spectra[ids]
        else:
            cur_s_wave = spectrum_wave

        plot_fn = join(spectrum_dir, str(model_id) + "_" + str(i) + "_" + str(j) + ".png")
        plt.plot(cur_s_wave, cur_spectra/np.max(cur_spectra), color="black", label="spectrum")
        #plt.xlabel("wavelength");plt.ylabel("intensity");plt.legend(loc="upper right")
        plt.title("Spectrum for pixel{}".format(i))
        plt.savefig(plot_fn);plt.close()

def plot_spectrum_gt(model_id, i, gt_fn, spectrum_dir, spectra, spectrum_wave,
                     orig_wave, orig_transms, colors, lbs, styles):
    def helper(nm, cur_spectra):
        for j, trans in enumerate(orig_transms):
            plt.plot(orig_wave, trans, color=colors[j], label=lbs[j], linestyle=styles[j])

        if spectrum_wave.ndim == 3: # bandwise
            cur_s_wave = spectrum_wave[i].flatten()
            cur_s_wave, ids = torch.sort(cur_s_wave)
            cur_spectra = cur_spectra[ids]
        else:
            cur_s_wave = spectrum_wave

        plot_fn = join(spectrum_dir, "gt_" + nm)
        plt.plot(cur_s_wave, cur_spectra/np.max(cur_spectra), color="black", label="spectrum")
        plt.plot(gt_wave, gt_spectra/np.max(gt_spectra),label="gt")
        plt.savefig(plot_fn);plt.close()

        """
        wave, gt_spectra_ol, gen_spectra_ol = overlay_spectrum(gt_fn, cur_s_wave, spectra)
        print(gt_spectra_ol.shape, gen_spectra_ol.shape)
        sam = calculate_sam_spectrum(gt_spectra_ol/np.max(gt_spectra_ol), gen_spectra_ol/np.max(gen_spectra_ol))
        cur_sam.append(sam)
        """

    cur_sam = []
    avg_spectra = np.mean(spectra, axis=0)
    gt_wave, gt_spectra = process_gt_spectra(gt_fn)
    helper(str(model_id) + "_" + str(i), avg_spectra)
    return cur_sam

def recon_spectrum_(model_id, batch_coords, covars, spectrum_wave, orig_wave,
                    orig_transms, net, trans_args, spectrum_dir, args):
    """ Generate spectra for pixels specified by coords using given net
        Save, plot spectrum, and calculate metrics
        @Param
          coords: list of n arrays, each array can be of size
                  [bsz,nsmpl,3] / [bsz,nbands,nsmpl_per_band,2] / [bsz,nsmpl,2]
    """
    sams = []
    wave_fn = join(spectrum_dir, "wave.npy")
    np.save(wave_fn, spectrum_wave)
    gt_fns = args.gt_spectra_fns

    wave_hi = int(min(args.wave_hi, int(np.max(spectrum_wave))))
    id_lo = np.argmax(spectrum_wave > args.wave_lo)
    id_hi = np.argmin(spectrum_wave < wave_hi)
    spectrum_wave = spectrum_wave[id_lo:id_hi]

    for i, coord in enumerate(batch_coords):
        print(coord)
        spectra = generate_spectra(args.mc_cho, coord, None, net, trans_args) # None is covar[i]
        #np.save(join(spectrum_dir, str(model_id) + "_" + str(i)), spectra)

        if args.mc_cho == "mc_hardcode":
            pix = np.load(args.hdcd_trans_fn)@spectra[0] / args.num_trans_smpl
        else: pix = np.load(args.full_trans_fn)@spectra[0] / np.load(args.nsmpl_within_bands_fn)

        spectra = spectra[:,id_lo:id_hi]
        if gt_fns is None:
            plot_spectrum(model_id, i, spectrum_dir, spectra, spectrum_wave, orig_wave, orig_transms,
                          args.spectrum_colors, args.spectrum_labels, args.spectrum_styles)
        else:
            plot_spectrum_gt(model_id, i, gt_fns[i], spectrum_dir, spectra, spectrum_wave, orig_wave,
                             orig_transms, args.spectrum_colors, args.spectrum_labels, args.spectrum_styles)
    return np.array(sams)

'''
## abandoned
def generate_spectra(mc_cho, coord, covar, net, trans_args, get_eltws_prod=False):
    """ Generate spectra profile for spectrum plotting
        @Param
          coord:  [bsz,3] / [bsz,nbands,nsmpl_per_band,2] / [bsz,nsmpl,2]
        @Return
          output: [bsz,nsmpl]
    """
    bsz = len(coord) # bsz/1
    if mc_cho == "mc_hardcode":
        net_args = [coord, covar, None]
    elif mc_cho == "mc_bandwise":
        wave, trans = trans_args # [bsz,nsmpl,1]/[nbands,nsmpl]
        net_args = [coord, covar, wave[:bsz], trans]
    elif mc_cho == "mc_mixture":
        #wave = trans_args[0][:bsz] # [bsz,nsmpl,1]
        wave = trans_args[0] # [nsmpl,1]
        net_args = [coord, covar, wave, None, None]
    else:
        raise("Unsupported monte carlo choice")

    with torch.no_grad():
        (spectra, _, _, _, _) = net(net_args)

    if spectra.ndim == 3: # bandwise
        spectra = spectra.flatten(1,2)
    spectra = spectra.detach().cpu().numpy() # [bsz,nsmpl]
    return spectra
'''
