
import torch
import pickle
import numpy as np
import logging as log
import matplotlib.pyplot as plt

from pathlib import Path
from os.path import join, exists
from scipy.interpolate import interp1d
#from astroquery.svo_fps import SvoFps
#from unagi import filters as unagi_filters

from wisp.utils.plot import plot_save
from wisp.utils.numerical import normalize_data
from wisp.datasets.data_utils import get_wave_range_fname, get_bound_id


class TransData:

    def __init__(self, device, **kwargs):

        if kwargs["space_dim"] != 3:
            return

        self.kwargs = kwargs
        self.device = device
        self.verbose = kwargs["verbose"]
        self.plot = kwargs["plot_trans"]

        self.filters = kwargs["filters"]
        self.filter_ids = kwargs["filter_ids"]
        self.sample_method = kwargs["trans_sample_method"]
        self.uniform_sample = kwargs["uniform_sample_wave"]
        self.learn_trusted_spectra = kwargs["learn_spectra_within_wave_range"]

        self.wave_lo = kwargs["wave_lo"]
        self.wave_hi = kwargs["wave_hi"]
        self.u_scale = kwargs["u_band_scale"]
        self.smpl_interval = kwargs["trans_sample_interval"]
        assert(self.smpl_interval == 10)

        if kwargs["on_cedar"]:
            self.dataset_path = kwargs["cedar_dataset_path"]
        else: self.dataset_path = kwargs["dataset_path"]
        self.set_path(self.dataset_path)

        self.init_trans()

    #############
    # Initializations
    #############

    def set_path(self, dataset_path):
        input_path = join(dataset_path, "input")
        source_wave_path = join(input_path, "wave")

        self.wave_range_fname = get_wave_range_fname(**self.kwargs)
        self.trans_dir = join(input_path, self.kwargs['sensor_collection_name'], 'transmission')

        self.source_wave_fname = join(source_wave_path, "source_wave.txt")
        self.source_trans_fname = join(source_wave_path, "source_trans.txt")
        self.nsmpl_within_bands_fname = join(self.trans_dir, "nsmpl_within_bands.npy")
        self.band_coverage_range_fname = join(self.trans_dir, "band_coverage_range.npy")

        self.processed_wave_fname = join(self.trans_dir, "processed_wave.txt")
        self.processed_trans_fname = join(self.trans_dir, "processed_trans.txt")

        self.full_wave_fname = join(self.trans_dir, f"full_wave_{self.smpl_interval}")
        self.full_trans_fname = join(self.trans_dir, f"full_trans_{self.smpl_interval}")
        self.full_distrib_fname = join(self.trans_dir, f"full_distrib_{self.smpl_interval}")
        self.full_uniform_distrib_fname = join(self.trans_dir, f"full_uniform_distrib_{self.smpl_interval}")
        if self.learn_trusted_spectra:
            name = str(self.kwargs["spectra_supervision_wave_lo"]) + "_" + \
                str(self.kwargs["spectra_supervision_wave_hi"])
            self.full_wave_masks_fname = join(
                self.trans_dir, f"full_wave_masks_{name}_{self.smpl_interval}"
            )
        self.encd_ids_fname = join(self.trans_dir, f"encd_ids_{self.smpl_interval}.npy")

        self.bdws_wave_fname = join(self.trans_dir, f"bdws_wave_{self.smpl_interval}")
        self.bdws_trans_fname = join(self.trans_dir, f"bdws_trans_{self.smpl_interval}")
        self.bdws_distrib_fname = join(self.trans_dir, f"bdws_distrib_{self.smpl_interval}")
        self.bdws_uniform_distrib_fname = join(self.trans_dir, f"bdws_uniform_distrib_{self.smpl_interval}")

        hdcd_nsmpls = self.kwargs["hardcode_num_trans_samples"]
        self.hdcd_wave_fname = join(self.trans_dir, f"hdcd_wave_{hdcd_nsmpls}")
        self.hdcd_trans_fname = join(self.trans_dir, f"hdcd_trans_{hdcd_nsmpls}")

        self.flat_trans_fname = join(self.trans_dir, "flat_trans")

        # create path
        for path in [source_wave_path, self.trans_dir]:
            Path(path).mkdir(parents=True, exist_ok=True)

    def init_trans(self):
        # hardcoded transmission range
        self.trans_range = {
            "g": [7,194],
            "r": [6,214],
            "i": [0,220],
            "z": [5,150],
            "y": [2,191],
            "nb387": [0,17],
            "nb816": [95,141],
            "nb921": [0,51],
            "u" : [0,48],
            "u*": [1,210]
        }

        if self.learn_trusted_spectra:
            self.trusted_wave_bound = [
                self.kwargs["spectra_supervision_wave_lo"],
                self.kwargs["spectra_supervision_wave_hi"]]

        self.data = {}
        self.process_wave_trans()
        if self.kwargs["trans_sample_method"] == "mixture":
            self.load_full_wave_trans()
        elif self.kwargs["trans_sample_method"] == "hardcode":
            self.load_hdcd_wave_trans()
        self.load_sampling_trans_data()
        self.set_wave_range()

    #############
    # getters
    #############

    def get_source_wave(self):
        return self.data["wave"]

    def get_source_trans(self):
        return self.data["trans"]

    def get_hdcd_wave(self):
        return self.data["hdcd_wave"]

    def get_hdcd_trans(self):
        return self.data["hdcd_trans"]

    def get_hdcd_nsmpl(self):
        return self.data["hdcd_nsmpl"]

    def get_full_wave(self):
        return self.data["full_wave"]

    def get_wave_range(self):
        """ Get wave range used for lambda value normalziation.
        """
        return self.data["wave_range"]

    def get_trans_wave_range(self):
        """ Get min and max value of transmission lambda.
        """
        return (min(self.data["full_wave"]), max(self.data["full_wave"]))

    def get_full_trans_data(self):
        return self.data["trans_data"]

    def get_full_wave_masks(self):
        """ Get masks for supervised wave range.
        """
        assert self.learn_trusted_spectra
        return self.data["full_wave_masks"]

    def get_num_samples_within_bands(self):
        return np.load(self.nsmpl_within_bands_fname)

    def get_band_coverage_range(self):
        return np.load(self.band_coverage_range_fname)

    def get_transmission_interpolation_function(self):
        # print(self.data["full_wave"].shape, self.data["full_trans"].shape)
        # print(self.data["full_wave"][0], self.data["full_wave"][-1])
        # wave [nsmpl], trans [nbands, nsmpl]
        f = interp1d(self.data["full_wave"], self.data["full_trans"], axis=1)
        return f

    def get_interpolated_transmission(self, f):
        interp_trans = f(self.data["full_wave"])
        return interp_trans

    def sample_wave(self, batch_size, num_samples=-1, use_all_wave=False):
        """ Sample lambda and transmission data for given sampling methods.
            @Return
              wave:  [bsz,nsmpl,1]/[bsz,nbands,nsmpl,1]
              trans: [bsz,nbands,nsmpl]
              nsmpl: [bsz,nbands]/None
        """
        smpl_ids = None
        nsmpl_within_each_band = None

        if use_all_wave:
            smpl_trans = self.data["full_trans"]
            smpl_wave = self.data["full_wave"][None,:,None]
            nsmpl_within_each_band = self.data["nsmpl_within_bands"]

            if type(smpl_wave).__module__ == "numpy":
                smpl_wave = np.tile(smpl_wave, (batch_size,1,1))
            elif type(smpl_wave).__module__ == "torch":
                smpl_wave = smpl_wave.tile(batch_size,1,1)
            else: raise ValueError()

        elif self.sample_method == "hardcode":
            smpl_wave, smpl_trans = None, self.trans

        elif self.sample_method == "bandwise":
            assert num_samples != -1
            smpl_wave, smpl_trans, _ = batch_sample_wave_bandwise(
                batch_size, num_samples, self.trans_data, waves=self.wave, sort=False)

        elif self.sample_method == "mixture":
            assert num_samples != -1
            smpl_wave, smpl_trans, smpl_ids, nsmpl_within_each_band = batch_sample_wave(
                batch_size, num_samples, self.trans_data, **self.kwargs)
        else:
            raise Exception("Unsupported monte carlo choice")

        return smpl_wave, smpl_trans, smpl_ids, nsmpl_within_each_band

    def get_flat_trans(self):
        if exists(self.flat_trans_fname):
            flat_trans = np.load(self.flat_trans_fname)
        else:
            range = self.data["full_wave"][-1] - self.data["full_wave"][0]
            avg_inte = np.mean(integration)
            lo = min([min(cur_wave) for cur_wave in self.data["wave"]])
            hi = max([max(cur_wave) for cur_wave in self.data["wave"]])

            full_nsmpl = len(self.data["full_wave"])
            flat_trans = np.full((full_nsmpl), avg_inte/range).reshape(1,-1)
            np.save(self.flat_trans_fname, flat_trans)
            if self.plot:
                plt.plot(self.data["full_wave"], self.data["full_trans"].T)
                plt.plot(self.data["full_wave"], flat_trans.T)
                plt.savefig(self.flat_trans_fname)
                plt.close()

        self.data["flat_trans"] = flat_trans

    #############
    # Helper methods
    #############

    def set_wave_range(self):
        """ Set wave range used for linear normalization.
            Note if the wave range used to normalize transmission wave and
              the spectra wave should be the same.
        """
        if exists(self.wave_range_fname):
            wave_range = np.load(self.wave_range_fname)
        else:
            if self.kwargs["trans_sample_method"] == "hardcode":
                wave_range = (min(self.data["hdcd_wave"]), max(self.data["hdcd_wave"]))
            else:
                wave_range = (min(self.data["full_wave"]), max(self.data["full_wave"]))
        self.data["wave_range"] = wave_range
        np.save(self.wave_range_fname, wave_range)

    def load_source_wave_trans(self):
        """ Load source lambda and transmission data.
            Assume sensors are in the following order:
              ['g','r','i','z','y','nb387','nb816','nb921','u','u*']
            (NOTE: this function should be called locally to create
              base_wave and base_trans. the unagi_filters package on
              is raising error to matplotlib)
        """
        if exists(self.source_wave_fname) and exists(self.source_trans_fname):
            print(self.source_wave_fname, self.source_trans_fname)
            assert 0
            log.info(f"wave and transmission data cached.")
            with open(self.source_wave_fname, 'rb') as fp:
                source_wave = pickle.load(fp)
            with open(self.source_trans_fname, 'rb') as fp:
                source_trans = pickle.load(fp)
        else:
            assert(False)
            source_wave, source_trans, wave_lo, wave_hi = [], [], np.Infinity, 0

            # grizy band
            hsc_filter_total = unagi_filters.hsc_filters(use_saved=False)
            for filter in hsc_filter_total:
                cf  = filter['short']
                if not cf in self.filters: continue
                wave, trans = filter['wave'], filter['trans']
                wave_lo = min(wave_lo, min(wave))
                wave_hi = max(wave_hi, max(wave))
                source_wave.append(wave); source_trans.append(trans)

            if 'u' in self.filters: # mega-u band
                data = SvoFps.get_transmission_data('CFHT/MegaCam.u')
                wave = list(data['Wavelength'])[1:] # trans for 3000 is 0
                trans = list(data['Transmission'])[1:]
                wave_lo = min(wave_lo, min(wave))
                wave_hi = max(wave_hi, max(wave))
                source_wave.append(wave); source_trans.append(trans)

            if 'us' in self.filters: # u* band
                data = SvoFps.get_transmission_data('CFHT/MegaCam.u_1')
                wave = list(data['Wavelength'])
                trans = list(data['Transmission'])
                wave_lo = min(wave_lo, min(wave))
                wave_hi = max(wave_hi, max(wave))
                source_wave.append(wave); source_trans.append(trans)

            with open(self.source_wave_fname, 'wb') as fp:
                pickle.dump(source_wave, fp)
            with open(self.source_trans_fname, 'wb') as fp:
                pickle.dump(source_trans, fp)

        for i in range(len(source_wave)):
            plt.plot(source_wave[i], source_trans[i])
        plt.savefig(self.source_trans_fname[:-4]+'.png'); plt.close()

        source_wave =  { filter:source_wave[i]
                         for filter, i in zip(self.filters,self.filter_ids) }
        source_trans = { filter:source_trans[i]
                         for filter, i in zip(self.filters,self.filter_ids) }
        return source_wave, source_trans

    def process_wave_trans(self):
        """ Process source wave and transmission.
        """
        if exists(self.processed_wave_fname) and exists(self.processed_trans_fname):
            with open(self.processed_wave_fname, "rb") as fp:
                wave = pickle.load(fp)
            with open(self.processed_trans_fname, "rb") as fp:
                trans = pickle.load(fp)
        else:
            source_wave, source_trans = self.load_source_wave_trans()

            wave, trans = remove_zero_pad(
                source_wave, source_trans, self.filters, self.trans_range)

            # unify discretization interval for all bands
            unify_discretization_interval(wave, trans, self.filters, self.smpl_interval)
            scale_trans(trans, source_trans, self.filters)
            wave, trans = map2list(wave, trans, self.filters)

            with open(self.processed_wave_fname, "wb") as fp:
                pickle.dump(wave, fp)
            with open(self.processed_trans_fname, "wb") as fp:
                pickle.dump(trans, fp)

        # coverage range of lambda (in angstrom) for each band [nbands]
        band_coverage_range = [cur_wave[-1] - cur_wave[0] for cur_wave in wave]
        np.save(self.band_coverage_range_fname, band_coverage_range)

        # number of lambda samples for each band [nbands]
        cur_trans_range = [v for k,v in self.trans_range.items()
                           if k in self.kwargs["filters"]]
        nsmpl_within_bands = count_avg_nsmpl(cur_trans_range)
        np.save(self.nsmpl_within_bands_fname, nsmpl_within_bands)

        integration = integrate_trans(wave, trans)
        if self.verbose:
            log.info(f"trans integration value: { np.round(integration, 2) }")
            log.info(f"sensor cover range: { band_coverage_range }")

        self.data["wave"] = wave
        self.data["trans"] = trans
        self.data["integration"] = integration
        self.data["band_coverage_range"] = band_coverage_range
        self.data["nsmpl_within_bands"] = torch.FloatTensor(nsmpl_within_bands) #.to(self.device)

    def load_sampling_trans_data(self):
        """ Get trans data depending on sampling method.
        """
        if self.sample_method == "mixture":
            self.trans_data = (self.data["full_wave"],self.data["full_trans"],
                               self.data["distrib"],self.data["encd_ids"])
        elif self.sample_method == "bandwise":
            self.trans_data = self.load_bandwise_wave_trans(norm_wave, trans)
        elif self.sample_method == "hardcode":
            self.trans_data = (self.data["hdcd_wave"], self.data["hdcd_trans"])
        else:
            raise ValueError("Unrecognized transmission sampling method.")

    def load_full_wave_trans(self):
        """ Load wave, trans, and distribution for mixture sampling.
        """
        if not exists(self.full_wave_fname) or \
           not exists(self.full_trans_fname) or \
           (not self.uniform_sample and not exists(self.full_distrib_fname)) or \
           (self.learn_trusted_spectra and not exists(self.full_wave_masks_fname)):

            # average all bands to get probability for mixture sampling
            trans_pdf = pnormalize(self.data["trans"])

            full_wave, distrib = average(self.data["wave"], trans_pdf)
            trans_dict = convert_to_dict(self.data["wave"], self.data["trans"])
            full_trans, encd_ids = convert_trans(full_wave, trans_dict)

            if self.learn_trusted_spectra:
                (lo, hi) = get_bound_id(self.trusted_wave_bound, full_wave)
                full_wave_masks = np.zeros(len(full_wave)).astype(bool)
                full_wave_masks[lo:hi+1] = 1

            np.save(self.encd_ids_fname, encd_ids)
            np.save(self.full_wave_fname, full_wave)
            np.save(self.full_wave_masks_fname, full_wave_masks)
            np.save(self.full_trans_fname, full_trans)
            np.save(self.full_distrib_fname, distrib)
            plot_save(self.full_trans_fname, full_wave, full_trans.T)
            plot_save(self.full_distrib_fname, full_wave, distrib)

        else:
            full_wave = np.load(self.full_wave_fname)
            full_trans = np.load(self.full_trans_fname)
            if not self.uniform_sample:
                distrib = np.load(self.full_distrib_fname)
            if self.learn_trusted_spectra:
                full_wave_masks = np.load(self.full_wave_masks_fname)

        if self.uniform_sample:
            distrib = np.ones(len(full_wave)).astype(np.float64)
            distrib /= len(distrib)

        trans_data = np.concatenate((full_wave[:,None],full_trans.T),axis=-1)

        self.data["trans_data"] = trans_data
        self.data["encd_ids"] = torch.FloatTensor(encd_ids)
        self.data["full_wave"] = full_wave.astype('float32')
        self.data["full_trans"] = torch.FloatTensor(full_trans)
        if self.learn_trusted_spectra: self.data["full_wave_masks"] = full_wave_masks
        if not self.uniform_sample: self.data["distrib"] = torch.FloatTensor(distrib)
        else: self.data["distrib"] = None

    def load_hdcd_wave_trans(self):
        """ Load wave, trans, and distribution for hardcode sampling.
        """
        if exists(self.hdcd_wave_fname + ".npy") and exists(self.hdcd_trans_fname + ".npy"):
            self.data["hdcd_wave"] = np.load(self.hdcd_wave_fname + ".npy")
            self.data["hdcd_trans"] = np.load(self.hdcd_trans_fname + ".npy")
            plot_save(self.hdcd_trans_fname + ".jpg",
                      self.data["hdcd_wave"],
                      self.data["hdcd_trans"].T)
            self.data["hdcd_wave"] = torch.FloatTensor(self.data["hdcd_wave"])
            self.data["hdcd_trans"] = torch.FloatTensor(self.data["hdcd_trans"])
            self.data["hdcd_nsmpl"] = torch.FloatTensor([self.kwargs["hardcode_num_trans_samples"]])
        else: assert(False)

        self.data["trans_data"] = np.concatenate((
            self.data["hdcd_wave"][:,None], self.data["hdcd_trans"].T), axis=-1)

    def load_bandwise_wave_trans(self, wave, trans):
        """ Load wave, trans, and distribution for bandwise sampling.
        """
        if not exists(self.bdws_wave_fname+".txt"):
            with open(self.bdws_wave_fname+".txt", "wb") as fp:
                pickle.dump(wave, fp)

        if not exists(self.bdws_trans_fname+".txt"):
            with open(self.bdws_trans_fname+".txt", "wb") as fp:
                pickle.dump(trans, fp)

        if self.uniform_sample:
            if not exists(self.bdws_uniform_distrib_fname):
                distrib = [np.ones(len(cur_trans)) / len(cur_trans) for cur_trans in trans]
                with open(self.bdws_uniform_distrib_fname, "wb") as fp:
                    pickle.dump(distrib, fp)
            else:
                with open(self.bdws_uniform_distrib_fname, "rb") as fp:
                    distrib = pickle.load(fp)
            distrib_fn = self.bdws_uniform_distrib_fname
        else:
            distrib = trans
            distrib_fn = self.bdws_distrib_fname

        if self.plot:
            for cur_wave, cur_trans in zip(wave, trans):
                plt.plot(cur_wave, cur_trans)
            plt.savefig(self.bdws_trans_fname);plt.close()

            for cur_wave, cur_distrib in zip(wave, self.distrib):
                plt.plot(cur_wave, cur_distrib)
            plt.savefig(distrib_fn);plt.close()

        wave = torch.FloatTensor(wave)
        trans = torch.FloatTensor(trans)
        distrib = torch.FloatTensor(distrib)
        return wave, trans, distrib

    #############
    # Utilities
    #############

    def plot_trans(self, axis=None, norm_cho="linr", color=None):
        trans = self.get_source_trans()
        wave = self.get_source_wave()
        for j, (cur_wave, cur_trans) in enumerate(zip(wave, trans)):
            cur_trans = normalize_data(cur_trans, norm_cho)
            cur_color = self.kwargs["plot_colors"][j] if color is None else color
            if axis is not None:
                axis.plot(cur_wave, cur_trans, color=cur_color,
                          label=self.kwargs["plot_labels"][j],
                          linestyle=self.kwargs["plot_styles"][j])
            else:
                plt.plot(cur_wave, cur_trans, color=cur_color,
                         label=self.kwargs["plot_labels"][j],
                         linestyle=self.kwargs["plot_styles"][j])

    def integrate(self, spectra, spectra_masks=None, all_wave=True, interpolate=False):
        """ Integrate spectra over transmission.
            TODO: deal with cases where spectra pixel has multiple neighbours
            @Param
              spectra: spectra data [bsz,2,nsmpl] (wave/flux)
              spectra_masks: mask out range of spectra to ignore [bsz,nsmpl] (1-keep, 0-drop)
        """
        if interpolate:
            spectra = self.interpolate_spectra(spectra, spectra_masks)
        elif all_wave:
            trans = self.data["full_trans"].numpy()
            nsmpl = self.data["nsmpl_within_bands"].numpy()
            spectra = spectra[:,0]
            recon_pixels = np.einsum("ij,kj->ik", spectra, trans) / nsmpl
        else:
            raise ValueError("Not Implemented")
        return recon_pixels

# TransData class ends
#################

def increment_repeat(counts, ids):
    bins = torch.bincount(ids)
    sids, _ = ids.sort()
    incr = bins[np.nonzero(bins)].flatten()
    bu = torch.unique(sids)
    counts[bu] += incr

def batch_sample_wave(bsz, nsmpls, trans_data,
                      use_all_wave=False,
                      wave_sample_method="uniform",
                      sort=False, **kwargs):
    """ Sample wave and trans for all bands together (mixture sampling)
        @Param  wave        [nsmpl_full]
                trans       [nbands,nsmpl_full]
                distrib     [nsmpl_full]
                band_masks  [nbands,nsmpl_full] if mixture o.w. None
        @Return smpl_wave   [bsz,nsmpl,1]
                smpl_trans  [bsz,nbands,nsmpl]
                avg_nsmpl   [nbands]
                ids         [bsz,nsmpls]
    """
    (wave, trans, distrib, band_mask) = trans_data
    (nbands, nsmpl_full) = trans.shape

    if use_all_wave:
        ids = torch.arange(nsmpl_full)
        ids = ids[None,:].tile((bsz,1))
    else:
        if wave_sample_method == "uniform": # kwargs["uniform_sample_wave"]:
            ids = torch.zeros(bsz,nsmpls).uniform_(0,nsmpl_full).to(torch.long)
        elif wave_sample_method == "importance":
            distrib = distrib[None,:].tile(bsz,1)
            ids = torch.multinomial(distrib, nsmpls, replacement=True)
        else: raise ValueError("Unsupported wave sampling method!")

    if band_mask is None:
        avg_nsmpl = torch.zeros(bsz, nbands).type(trans.dtype)
    elif kwargs["mixture_avg_per_band"]:
        # count number of samples falling within response range of each band
        avg_nsmpl = [torch.sum(band_mask[ids], dim=1) for band_mask in band_mask]
        avg_nsmpl = torch.stack(avg_nsmpl).T # [bsz,nbands]
        avg_nsmpl[avg_nsmpl==0] = 1 # avoid dividing by 0
    elif use_all_wave:
        assert(False)
        # TODO: should use covr_rnge of each band if train with all wave for mixture
        avg_nsmpl = torch.full((bsz, nbands), nsmpl_full)
    else:
        avg_nsmpl = nsmpls

    # sort sampled waves (True only for spectrum plotting)
    if sort: ids, _ = torch.sort(ids, dim=1) # [bsz,nsmpls]

    smpl_wave = wave[ids][...,None] # [bsz,nsmpls,1]
    smpl_trans = torch.stack([cur_trans[ids] for cur_trans in trans])
    smpl_trans = smpl_trans.permute(1,0,2)   # [bsz,nbands,nsmpls]
    return smpl_wave, smpl_trans, ids, avg_nsmpl

def batch_sample_wave_bandwise(bsz, nsmpls, trans_data, waves=None, sort=True):
    """ Sample wave and trans for each band independently
          (use for bandwise mc training)
        @Param  norm_waves: list of tensor nbands*[nsmpl_cur_band]
                transs:     list of tensor nbands*[nsmpl_cur_band]
                distribs:   None OR list of tensor nbands*[nsmpl_cur_band]
                counts:     None OR list of tensor nbands*[nsmpl_cur_band]
                trans_ones: if True - dot product is equivalent to summing
                                      up spectra intensity
        @Return wave:  [bsz,n_bands,nsmpl_per_band,1]
                trans: [bsz,n_bands,nsmpl_per_band]
    """
    nbands = args.num_bands
    unismpl = args.uniform_smpl
    nsmpl = args.nsmpl_per_band
    float_tensor = args.float_tensor

    smpl_trans = torch.ones((bsz, nbands, nsmpl))
    smpl_norm_wave = torch.zeros((bsz, nbands, nsmpl))
    smpl_wave = None if waves is None else torch.zeros((bsz, nbands, nsmpl))

    for i in range(nbands):
        trans = torch.tensor(transs[i])
        norm_wave = torch.tensor(norm_waves[i])
        batch_distrib = torch.tensor(distribs[i])[None,:].tile(bsz, 1)
        ids = torch.multinomial(batch_distrib, nsmpl, replacement=True) # [bsz,nsmpl]
        if sort: ids, _ = torch.sort(ids, dim=1)

        smpl_norm_wave[:,i] = norm_wave[ids] # [bsz,nsmpl_per_band]
        if unismpl: smpl_trans[:,i] = trans[ids]
        if counts is not None: increment_repeat(counts[i], ids.flatten())
        if waves is not None:  # [bsz,nsmpl_per_band]
            smpl_wave[:,i] = torch.tensor(waves[i])[ids]

    smpl_trans = smpl_trans.type(float_tensor)
    smpl_norm_wave = smpl_norm_wave[...,None].type(float_tensor)
    return smpl_norm_wave, smpl_trans, smpl_wave

def sample_trans(norm_wave, trans, avg_distrib, nsmpl, wave=None, sort=True, counts=None):
    """ Sample wave and trans for MC mixture estimate
        output wave [nsmpl], trans[nbands,nsmpl]
        ct if not None is torch tensor [full_nsmpl]
    """
    ids = torch.multinomial(avg_distrib, nsmpl, replacement=True)
    if counts is not None: increment_repeat(counts, ids)
    if sort: ids, _ = torch.sort(ids)

    sorted_nm_wave, sorted_trans = norm_wave[ids], trans[:,ids]
    sorted_wave = None if wave is None else torch.tensor(wave)[ids]
    sorted_wave = sorted_wave.detach().cpu().numpy()
    #sorted_wave = None if wave is None else torch.tensor(wave)[ids]
    return sorted_nm_wave, sorted_trans, sorted_wave

#############
# Trans processing functions
#############

def get_bandwise_coverage_range(wave, trans, threshold, uniform_sample=False):
    """ Calculate coverage range for each band.
        Used as sampling prob for bandwise uniform sampling.
    """
    if not uniform_sample:
        coverage_range = None
    else:
        coverage_range = count_avg_nsmpl(waves, transs, threshold)
        #coverage_range = measure_all(waves, transs, threshold)/1000
    return coverage_range

def get_bandwise_prob(unagi_wave_fn, unagi_trans_fn, threshold):
    with open(unagi_wave_fn, "rb") as fp:
        wave = pickle.load(fp)
    with open(unagi_trans_fn, "rb") as fp:
        trans = pickle.load(fp)
    #print("calculating prob, size of base wave ", len(waves[0]))

    wave_range = measure_all_bands(wave, trans, threshold=threshold)
    prob = 1/wave_range
    rela_prob = prob/max(prob)
    return rela_prob

def remove_zero_pad(wave, trans, bands, trans_range):
    """ Remove zero transmission values from two ends (hardcoded).
    """
    trimmed_wave, trimmed_trans = {}, {}
    print(wave.keys())
    for band in bands:
        cur_wave, cur_trans = wave[band], trans[band]
        (lo, hi) = trans_range[band]
        if hi is None: hi = len(cur_trans)
        trimmed_wave[band] = cur_wave[lo:hi]
        trimmed_trans[band] = cur_trans[lo:hi]
    return trimmed_wave, trimmed_trans

def count_avg_nsmpl(trans_range):
    return np.array([measure_one_band(cur_range) for cur_range in trans_range])

# def count_avg_nsmpl(wave, trans, threshold):
#     """ For each band, count # of wave samples with above-thresh trans val.
#     """
#     return np.array([measure_one_band(cur_wave, cur_trans, threshold, 0)
#                      for cur_wave, cur_trans in zip(wave, trans)])

def measure_one_band(trans_range):
    lo, hi = trans_range
    return hi - lo

# def measure_one_band(wave, trans, threshold, cho):
#     """ Measure num/range of lambda where band has above-thresh trans val.
#     """
#     start, n = -1, len(trans)
#     for i in range(n):
#         if start == -1:
#             if trans[i] > threshold: start = i
#         else:
#             if trans[i] < threshold or i == n - 1:
#                 if cho == 0: val = int(i - start)
#                 else: val = int(wave[i] - wave[start])
#                 return val
#     assert(False)

# def measure_all_bands(wave, trans, threshold):
#     """ For each band, measure num/range of lambda with above-thresh trans val.
#     """
#     return np.array([measure_one_band(wave, trans, threshold, 1)
#                      for cur_wave, cur_trans in zip(wave, trans)])

def measure_all_bands(trans_range):
    """ For each band, measure num/range of lambda with above-thresh trans val.
    """
    return np.array([measure_one_band(cur_range) for cur_range in trans_range])

# def trim_wave_trans(wave, trans, bands, trans_threshold):
#     """ Trim away range where transmission value < threshold for each band.
#     """
#     trimmed_wave, trimmed_trans = {}, {}
#     for band in bands:
#         cur_wave, cur_trans = wave[band], trans[band]
#         start, n = -1, len(cur_wave)
#         for i in range(n):
#             if start == -1:
#                 if cur_trans[i] > trans_threshold: start = i
#             else:
#                 if cur_trans[i] <= trans_threshold or i == n - 1:
#                     trimmed_wave[band] = cur_wave[start:i]
#                     trimmed_trans[band] = cur_trans[start:i]
#                     break
#     return trimmed_wave, trimmed_trans

def unify_discretization_interval(wave, trans, bands, new_smpl_interval):
    """ Make sample interval the same for all bands.
        Currently supports a 10-band collection - grizy,3*nb,u,u*
    """
    lo_wave, hi_wave = np.inf, 0
    for band in bands:
        if band == "us" or band == "u": continue
        lo, hi = wave[band][0], wave[band][-1]
        if lo < lo_wave: lo_wave = lo
        if hi > hi_wave: hi_wave = hi

    full_wave = np.arange(lo_wave, hi_wave+1, new_smpl_interval)

    for i, band in enumerate(bands):
        if band == "u": continue

        orig_smpl_interval = wave[band][1] - wave[band][0]
        interval =  new_smpl_interval // orig_smpl_interval
        id_lo = np.where(full_wave >= wave[band][0])[0][0]
        id_hi = np.where(full_wave <= wave[band][-1])[0][-1]

        id_lo = np.where(wave[band] == full_wave[id_lo])[0][0]
        id_hi = np.where(wave[band] == full_wave[id_hi])[0][0]
        ids = np.arange(id_lo, id_hi+1, interval).astype(int)

        wave[band] = np.array(wave[band])[ids]
        trans[band] = np.array(trans[band])[ids]

    # all bands except the u band has a uniform discretization interval
    if 'u' in bands:
        interpolate_u_band(wave, trans, full_wave, new_smpl_interval)

'''
def downsample_us(wave, trans):
    """ Downsample u* band from 20 to 10. Source u* wave interval is 20.
    """
    ids = np.arange(0, len(wave["us"]), 2)
    wave["us"] = np.array(wave["us"])[ids]
    trans["us"] = np.array(trans["us"])[ids]
'''

def interpolate_u_band(wave, trans, full_wave, smpl_interval):
    """ Interpolate u band with discretization value 10.
        Source wave for u band is not uniformly spaced.
    """
    fu = interpolate.interp1d(wave["u"], trans["u"])
    lo = np.where(full_wave >= wave["u"][0])[0][0]
    u_wave = np.arange(lo, wave["u"][-1] + 1, smpl_interval)
    u_trans = fu(u_wave)
    wave["u"] = u_wave
    trans["u"] = u_trans

def scale_trans(trans, source_trans, bands):
    """ Scale transmission value for each band s.t. integration of transmission
          before and after interpolation remains the same.
    """
    for band in bands:
        cur_trans, cur_source_trans = trans[band], source_trans[band]
        trans[band] = np.array(trans[band]) * np.sum(cur_source_trans) / np.sum(cur_trans)

def scaleu(trans):
    """ Scale u band transmission.
        (Not used. We scale u band pixel value instead).
    """
    if "u" in trans:
        trans["u"] = np.array(trans["u"]) * u_scale
    if "us" in trans:
        trans["us"] = np.array(trans["us"]) * u_scale

def map2list(wave, trans, bands):
    """ Convert map to list. """
    lwave, ltrans = [], []
    for band in bands:
        lwave.append(wave[band])
        ltrans.append(trans[band])
    return lwave, ltrans

def integrate_trans(wave, trans, cho=0):
    """ MC estimate of transmisson integration.
        Assume transmissions have close-to-0 ranges being trimmed.
    """
    widths = [cur_wave[-1] - cur_wave[0] for cur_wave in wave]
    if cho == 0: # \inte T_b(lambda)
        inte = [ integrate(cur_wave, cur_trans, width)
                 for cur_wave, cur_trans, width in zip(wave, trans, widths)]
    elif cho == 1: # \inte lambda * T_b(lambda)
        inte = [ integrate_enrgy(wave, trans, width)
                 for cur_wave, cur_trans, width in zip(wave, trans, widths)]
    else: raise Exception("Unsupported integration choice")
    return np.array(inte)

def convert_to_dict(wave, trans):
    dicts = []
    for cur_wave, cur_trans in zip(wave, trans):
        dict = {}
        for lbd, intensity in zip(cur_wave, cur_trans):
            dict[lbd] = intensity
        dicts.append(dict)
    return dicts

def convert_trans(full_wave, trans_dict):
    """ Append 0 to transmission for each band s.t. trans for all bands
          cover same lambda range.
        Generate a 0-1 array where 1 indicates non-zero trans in
          corresponding position.
    """
    n, m = len(full_wave), len(trans_dict)
    full_trans = np.zeros((m, n)) # [nbands,nsmpl_full]
    encd_ids = np.zeros((m, n))   # 1 indicate non-zero trans

    for i, wave in enumerate(full_wave):
        for chnl in range(m):
            cur_trans = trans_dict[chnl].get(wave, 0)
            full_trans[chnl,i] = cur_trans
            encd_ids[chnl,i] = cur_trans != 0
    return full_trans, encd_ids

def pnormalize(trans):
    """ Normalize transmission of each band to sum to 1 (pdf).
    """
    return [cur_trans/sum(cur_trans) for cur_trans in trans]

def inormalize(trans, integration):
    """ Normalize transmission of each band to integrate to 1.
    """
    return [cur_trans/inte for cur_trans, inte in zip(trans, integration)]

def integrate(wave, trans, width):
    return width * np.mean(np.array(trans))

def integrate_enrgy(wave, trans, width):
    inte = sum(np.array(trans) * np.array(wave)/1000)
    inte = width * inte / len(trans)
    return round(inte, 2)

def average(wave, trans):
    """ Calculate mixture sampling distribution function.
        Add up transmission value (thru all bands) at each lambda and average.
    """
    pdf = {}
    for cur_wave, cur_trans in zip(wave, trans):
        for lbd, intensity in zip(cur_wave, cur_trans):
            prev = pdf.get(lbd, [])
            prev.append(intensity)
            pdf[lbd] = prev

    pdf = dict(sorted(pdf.items()))
    full_wave, distrib = [], []
    for key in pdf.keys():
        full_wave.append(key)
        avg = sum(pdf[key])/len(pdf[key])
        distrib.append(avg)
    return np.array(full_wave), np.array(distrib)
