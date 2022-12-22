
import torch
import pickle
import numpy as np
import logging as log
import matplotlib.pyplot as plt

from os.path import join, exists
from wisp.utils.plot import plot_save


class TransData:

    def __init__(self, dataset_path, **kwargs):

        self.kwargs = kwargs
        self.verbose = kwargs["verbose"]
        self.plot = kwargs["plot_trans"]

        self.filters = kwargs["filters"]
        self.filter_ids = kwargs["filter_ids"]
        self.sample_method = kwargs["trans_sample_method"]
        self.uniform_sample = kwargs["uniform_sample_trans"]

        self.wave_lo = kwargs["wave_lo"]
        self.wave_hi = kwargs["wave_hi"]
        self.u_scale = kwargs["u_band_scale"]
        self.trans_threshold = kwargs["trans_threshold"]
        self.smpl_interval = kwargs["trans_smpl_interval"]
        assert(self.smpl_interval == 10)

        self.set_log_path(dataset_path)
        self.init_trans()

    #############
    # Initializations
    #############

    def set_log_path(self, dataset_path):
        input_path = join(dataset_path, "input")
        source_trans_path = join(input_path, "transmission")
        self.trans_dir = join(input_path, self.kwargs['sensor_collection_name'], 'transmission')

        self.source_wave_fname = join(source_trans_path, "source_wave.txt")
        self.source_trans_fname = join(source_trans_path, "source_trans.txt")
        self.nsmpl_within_bands_fname = join(self.trans_dir, "nsmpl_within_bands.npy")
        self.band_coverage_range_fname = join(self.trans_dir, "band_coverage_range.npy")

        self.processed_wave_fname = join(self.trans_dir, "processed_wave.txt")
        self.processed_trans_fname = join(self.trans_dir, "processed_trans.txt")

        self.full_wave_fname = join(self.trans_dir, "full_wave")
        self.full_trans_fname = join(self.trans_dir, "full_trans")
        self.full_distrib_fname = join(self.trans_dir, 'full_distrib')
        self.full_uniform_distrib_fname = join(self.trans_dir, 'full_uniform_distrib')
        self.encd_ids_fname = join(self.trans_dir, "encd_ids.npy")

        self.bdws_wave_fname = join(self.trans_dir, "bdws_wave")
        self.bdws_trans_fname = join(self.trans_dir, "bdws_trans")
        self.bdws_distrib_fname = join(self.trans_dir, 'bdws_distrib')
        self.bdws_uniform_distrib_fname = join(self.trans_dir, 'bdws_uniform_distrib')

        hdcd_nsmpls = self.kwargs["hardcode_num_trans_samples"]
        self.hdcd_wave_fname = join(self.trans_dir, f"hdcd_wave_{hdcd_nsmpls}.npy")
        self.hdcd_trans_fname = join(self.trans_dir, f"hdcd_trans_{hdcd_nsmpls}.npy")

        self.flat_trans_fname = join(self.trans_dir, "flat_trans")

    def init_trans(self):
        bands = self.kwargs["filters"]
        band_ids = self.kwargs["filter_ids"]

        if not exists(self.processed_wave_fname) or not exists(self.processed_trans_fname):
            source_wave, source_trans = self.load_source_wave_trans()
            wave, trans, integration = self.preprocess_wave_trans(source_wave, source_trans)
        else:
            with open(self.processed_wave_fname, "rb") as fp:
                wave = pickle.load(fp)
            with open(self.processed_trans_fname, "rb") as fp:
                trans = pickle.load(fp)
            integration = integrate_trans(wave, trans)
            if self.verbose:
                log.info(f"trans integration value: { np.round(integration, 2) }")

        self.full_wave, self.full_trans, self.full_distrib, encd_ids = \
            self.load_full_wave_trans(wave, trans)

        lo, hi = min(self.full_wave), max(self.full_wave)
        self.full_norm_wave = (self.full_wave - lo) / (hi - lo)

        self.full_nsmpl = len(self.full_wave)

        if self.sample_method == "mixture":
            self.trans_data = (self.full_norm_wave, self.full_trans, self.full_distrib, encd_ids)
        elif self.sample_method == "bandwise":
            self.trans_data = self.load_bandwise_wave_trans(norm_wave, trans)
        elif self.sample_method == "hardcode":
            self.trans_data = self.load_hdcd_wave_trans(self.trans_dir, hdcd_wave, hdcd_trans)
        else:
            raise ValueError("Unrecognized transmission sampling method.")

    #############
    # getters
    #############

    def sample_wave_trans(self, batch_size, num_samples):
        """ Sample lambda and transmission data for given sampling methods.
            @Return
              wave:  [bsz,nsmpl,1]/[bsz,nbands,nsmpl,1]
              trans: [bsz,nbands,nsmpl]
              nsmpl: [bsz,nbands]/None
        """
        nsmpl_within_each_band = None
        if self.sample_method == "hardcode":
            smpl_wave, smpl_trans = None, self.trans

        elif self.sample_method == "bandwise":
            smpl_wave, smpl_trans, _ = batch_sample_trans_bandwise(
                batch_size, num_samples, self.trans_data, waves=self.wave, sort=False)

        elif self.sample_method == "mixture":
            smpl_wave, smpl_trans, _, nsmpl_within_each_band = batch_sample_trans(
                batch_size, num_samples, self.trans_data, **self.kwargs)
        else:
            raise Exception("Unsupported monte carlo choice")
        return smpl_wave, smpl_trans, nsmpl_within_each_band

    def get_flat_trans(self):
        if exists(self.flat_trans_fname):
            flat_trans = np.load(self.flat_trans_fname)
        else:
            range = self.full_wave[-1] - self.full_wave[0]
            avg_inte = np.mean(integration)
            lo = min([min(cur_wave) for cur_wave in self.wave])
            hi = max([max(cur_wave) for cur_wave in self.wave])

            flat_trans = np.full((self.full_nsmpl), avg_inte/range).reshape(1,-1)
            np.save(self.flat_trans_fname, flat_trans)
            if self.plot:
                plt.plot(self.full_wave, full_trans.T)
                plt.plot(self.full_wave, flat_trans.T)
                plt.savefig(self.flat_trans_fname)
                plt.close()

        self.flat_trans = flat_trans

    def get_num_samples_within_bands(self):
        return np.load(self.nsmpl_within_bands_fname)

    def get_band_coverage_range(self):
        return np.load(self.band_coverage_range_fname)

    #############
    # Helper methods
    #############

    def load_source_wave_trans(self):
        """ Load source lambda and transmission data.
            Assume sensors are in the following ordered:
              ['g','r','i','z','y','nb387','nb816','nb921','u','u*']
            (NOTE: this function should be called locally to create
              base_wave and base_trans. the unagi_filters package on
              is raising error to matplotlib)
        """
        if exists(self.source_wave_fname) and exists(self.source_trans_fname):
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

        source_wave =  { filter:source_wave[i]  for filter, i in zip(self.filters,self.filter_ids) }
        source_trans = { filter:source_trans[i] for filter, i in zip(self.filters,self.filter_ids) }
        return source_wave, source_trans

    def preprocess_wave_trans(self, source_wave, source_trans):
        """ Preprocess source wave and transmission.
        """
        # transmission trimming
        wave, trans = trim_wave_trans(source_wave, source_trans, self.filters, self.trans_threshold)

        # unify discretization interval as 10 for all bands
        if 'us' in self.filters: downsample_us(wave, trans)
        if 'u' in self.filters: interpolate_u_band(wave, trans, self.smpl_interval)
        scale_trans(trans, source_trans, self.filters)
        wave, trans = map2list(wave, trans, self.filters)

        integration = integrate_trans(wave, trans)
        band_coverage_range = [cur_wave[-1] - cur_wave[0] for cur_wave in wave]
        nsmpl_within_bands = count_avg_nsmpl(wave, trans, self.trans_threshold)
        np.save(self.nsmpl_within_bands_fname, nsmpl_within_bands)

        if self.verbose:
            log.info(f"trans integration value: { np.round(integration, 2) }")
            log.info(f"sensor cover range: { band_coverage_range }")

        return wave, trans, integration

    def load_full_wave_trans(self, wave, trans):
        """ Load wave, trans, and distribution for mixture sampling.
        """
        if not exists(self.full_wave_fname) or not exists(self.full_trans_fname) \
           or (not self.uniform_sample and not exists(self.full_distrib_fname)):

            # average all bands to get probability for mixture sampling
            trans_pdf = pnormalize(trans)

            full_wave, distrib = average(wave, trans_pdf)
            trans_dict = convert_to_dict(wave, trans)
            full_trans, encd_ids = convert_trans(full_wave, trans_dict)

            np.save(self.encd_ids_fname, encd_ids)
            np.save(self.full_wave_fname, full_wave)
            plot_save(self.full_trans_fname, full_wave, full_trans.T)
            plot_save(self.full_distrib_fname, full_wave, distrib)

        else:
            full_wave = np.load(self.full_wave_fname)
            full_trans = np.load(self.full_trans_fname)
            if not self.uniform_sample:
                distrib = np.load(self.full_distrib_fname)

        if self.uniform_sample:
            distrib = np.ones(len(full_wave)).astype(np.float64)
            distrib /= len(distrib)

        distrib = torch.FloatTensor(distrib)
        encd_ids = torch.FloatTensor(encd_ids)
        full_wave = torch.FloatTensor(full_wave)
        full_trans = torch.FloatTensor(full_trans)
        return full_wave, full_trans, distrib, encd_ids

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

        norm_wave = [(cur_wave - min(cur_wave)) / (max(cur_wave) - min(cur_wave))
                     for cur_wave in wave]

        wave = torch.FloatTensor(norm_wave)
        trans = torch.FloatTensor(trans)
        distrib = torch.FloatTensor(distrib)
        return wave, trans, distrib

    def load_hdcd_wave_trans(self, wave, trans):
        """ Load wave, trans, and distribution for hardcode sampling.
        """
        if not exists(self.hdcd_wave_fname) or not exists(self.hdcd_trans_fname):
            wave = np.load(self.hdcd_wave_fname)
            trans = np.load(self.hdcd_trans_fname)
            if self.plot:
                plot_save(self.hdcd_trans_fname, wave, trans)
        else: assert(False)

        wave = torch.FloatTensor(wave)
        trans = torch.FloatTensor(trans)
        return wave, trans, None

# TransData class ends
#################

#############
# Utilities
#############

def increment_repeat(counts, ids):
    bins = torch.bincount(ids)
    sids, _ = ids.sort()
    incr = bins[np.nonzero(bins)].flatten()
    bu = torch.unique(sids)
    counts[bu] += incr

def batch_sample_trans(bsz, nsmpls, trans_data, use_all_wave=False, sort=False, **kwargs):
    """ Sample wave and trans for all bands together (mixture sampling)
        @Param  wave        [nsmpl_full]
                trans       [nbands,nsmpl_full]
                distrib     [nsmpl_full]
                encd_ids    [nbands,nsmpl_full] if mixture o.w. None
        @Return smpl_wave   [bsz,nsmpl]
                smpl_trans  [bsz,nbands,nsmpl]
                avg_nsmpl   [nbands]
                ids         [nsmpls]
    """
    (wave, trans, distrib, encd_ids) = trans_data
    (nbands, nsmpl_full) = trans.shape

    if use_all_wave:
        # use all lambda [bsz,nsmpl_full]
        ids = torch.arange(nsmpl_full)
        #ids = np.arange(0, nsmpl_full)
        ids = ids[None,:].tile((bsz,1))
    else:
        # sample #nsmpl lambda [bsz,nsmpl]
        distrib = distrib[None,:].tile(bsz,1)
        ids = torch.multinomial(distrib, nsmpls, replacement=True)

    if encd_ids is None:
        avg_nsmpl = torch.zeros(bsz, nbands).type(trans.dtype)
    elif kwargs["mixture_avg_per_band"]:
        # count number of samples falling within response range of each band
        avg_nsmpl = [torch.sum(encd_id[ids], dim=1) for encd_id in encd_ids]
        avg_nsmpl = torch.stack(avg_nsmpl).T # [bsz,nbands]
        avg_nsmpl[avg_nsmpl==0] = 1 # avoid dividing by 0
    elif use_all_wave:
        assert(False)
        # TODO: we probably should use covr_rnge of each band if we train with all wave for mixture
        avg_nsmpl = torch.full((bsz, nbands), nsmpl_full)
    else:
        avg_nsmpl = nsmpls

    # sort sampled waves (True only for spectrum plotting)
    if sort: ids, _ = torch.sort(ids, dim=1) # [bsz,nsmpls]

    smpl_wave = wave[ids] # [bsz,nsmpls]
    smpl_trans = torch.stack([cur_trans[ids] for cur_trans in trans])
    smpl_trans = smpl_trans.permute(1,0,2)   # [bsz,nbands,nsmpls]
    return smpl_wave, smpl_trans, ids, avg_nsmpl

def batch_sample_trans_bandwise(bsz, nsmpls, trans_data, waves=None, sort=True):
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

def trim_wave_trans(wave, trans, bands, trans_threshold):
    """ Trim away range where transmission value < threshold for each band.
    """
    trimmed_wave, trimmed_trans = {}, {}

    for band in bands:
        cur_wave, cur_trans = wave[band], trans[band]
        start, n = -1, len(cur_wave)
        for i in range(n):
            if start == -1:
                if cur_trans[i] > trans_threshold: start = i
            else:
                if cur_trans[i] <= trans_threshold or i == n - 1:
                    trimmed_wave[band] = cur_wave[start:i]
                    trimmed_trans[band] = cur_trans[start:i]
                    break

    return trimmed_wave, trimmed_trans

def downsample_us(wave, trans):
    """ Downsample u* band from 20 to 10. Source u* wave interval is 20.
    """
    ids = np.arange(0, len(wave["us"]), 2)
    wave["us"] = np.array(wave["us"])[ids]
    trans["us"] = np.array(trans["us"])[ids]

def interpolate_u_band(wave, trans, smpl_interval):
    """ Interpolate u band with discretization value 10.
        Source wave for u band is not uniformly spaced.
    """
    fu = interpolate.interp1d(wave["u"], trans["u"])
    u_wave = np.arange(wave["u"][0], wave["u"][-1] + 1, smpl_interval)
    u_trans = fu(u_wave)
    wave["u"] = u_wave
    trans["u"] = u_trans

def scale_trans(trans, source_trans, bands):
    """ Scale transmission value for each band s.t. integration of transmission
          before and after scaling remains the same.
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

def count_avg_nsmpl(wave, trans, threshold):
    """ For each band, count # of wave samples with above-thresh trans val.
    """
    return np.array([measure_one_band(cur_wave, cur_trans, threshold, 0)
                     for cur_wave, cur_trans in zip(wave, trans)])

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

def measure_one_band(wave, trans, threshold, cho):
    """ Measure num/range of lambda where band has above-thresh trans val.
    """
    start, n = -1, len(trans)
    for i in range(n):
        if start == -1:
            if trans[i] > threshold: start = i
        else:
            if trans[i] < threshold or i == n - 1:
                if cho == 0: val = int(i - start)
                else: val = int(wave[i] - wave[start])
                return val
    assert(False)

def measure_all_bands(wave, trans, threshold):
    """ For each band, measure num/range of lambda with above-thresh trans val.
    """
    return np.array([measure_one_band(wave, trans, threshold, 1)
                     for cur_wave, cur_trans in zip(wave, trans)])

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
    return full_wave, np.array(distrib)
