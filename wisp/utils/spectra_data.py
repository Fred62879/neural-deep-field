

class SpectraData:

    def __init__(self):
        pass

    def generate_spectra(mc_cho, coord, covar, net, trans_args, get_eltws_prod=False):
        ''' Generate spectra profile for spectrum plotting
            @Param
              coord:  [bsz,3] / [bsz,nbands,nsmpl_per_band,2] / [bsz,nsmpl,2]
            @Return
              output: [bsz,nsmpl]
        '''
        bsz = len(coord) # bsz/1
        if mc_cho == 'mc_hardcode':
            net_args = [coord, covar, None]
        elif mc_cho == 'mc_bandwise':
            wave, trans = trans_args # [bsz,nsmpl,1]/[nbands,nsmpl]
            net_args = [coord, covar, wave[:bsz], trans]
        elif mc_cho == 'mc_mixture':
            #wave = trans_args[0][:bsz] # [bsz,nsmpl,1]
            wave = trans_args[0] # [nsmpl,1]
            net_args = [coord, covar, wave, None, None]
        else:
            raise('Unsupported monte carlo choice')

        with torch.no_grad():
            (spectra, _, _, _, _) = net(net_args)

        if spectra.ndim == 3: # bandwise
            spectra = spectra.flatten(1,2)
        spectra = spectra.detach().cpu().numpy() # [bsz,nsmpl]
        return spectra

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

    def convolve_spectra(spectra, std=50, border=True):
        kernel = Gaussian1DKernel(stddev=std)
        if border:
            nume = convolve(spectra, kernel)
            denom = convolve(np.ones(spectra.shape), kernel)
            return nume / denom
        return convolve(spectra, kernel)

    def process_gt_spectra(gt_fn):
        ''' Process gt spectra for spectrum plotting '''
        gt = np.load(gt_fn)
        gt_wave, gt_spectra = gt[:,0], gt[:,1]
        lo, hi = np.min(gt_spectra), np.max(gt_spectra)
        if hi != lo:
            gt_spectra = (gt_spectra - lo) / (hi - lo)
        gt_spectra = convolve_spectra(gt_spectra)
        return gt_wave, gt_spectra

    def load_supervision_gt_spectra(fn, trusted_wave_range, smpl_interval):
        ''' Load gt spectra for spectra supervision
            @Param
              fn: filename of np array that stores gt spectra data
              smpl_wave: sampled wave/lambda for spectrum plotting
        '''
        gt = np.load(fn)
        gt_wave, gt_spectra = gt[:,0], gt[:,1]
        gt_spectra = convolve_spectra(gt_spectra)
        f_gt = interpolate.interp1d(gt_wave, gt_spectra)

        # assume lo, hi is within range of gt wave
        (lo, hi) = trusted_wave_range
        trusted_wave = np.arange(lo, hi + 1, smpl_interval)
        smpl_spectra = f_gt(trusted_wave)
        smpl_spectra /= np.max(smpl_spectra)
        print(smpl_spectra.shape)
        return smpl_spectra

    def load_supervision_gt_spectra_all(fns, trusted_wave_range,
                                        trans_smpl_interval, float_tensor):
        ret = np.array([load_supervision_gt_spectra(fn, trusted_wave_range, trans_smpl_interval)
                        for fn in fns])
        return torch.tensor(ret).type(float_tensor)

    #############
    # Plotting functions
    #############

    def plot_spectrum(model_id, i, spectrum_dir, spectra, spectrum_wave,
                      orig_wave, orig_transms, colors, lbs, styles):
        ''' Plot spectrum with sensor transmission as background.
            spectra [bsz,nsmpl]
        '''
        for j, cur_spectra in enumerate(spectra):
            for k, trans in enumerate(orig_transms):
                plt.plot(orig_wave, trans, color=colors[k], label=lbs[k], linestyle=styles[k])

            if spectrum_wave.ndim == 3: # bandwise
                cur_s_wave = spectrum_wave[j].flatten()
                cur_s_wave, ids = torch.sort(cur_s_wave)
                cur_spectra = cur_spectra[ids]
            else:
                cur_s_wave = spectrum_wave

            plot_fn = join(spectrum_dir, str(model_id) + '_' + str(i) + '_' + str(j) + '.png')
            plt.plot(cur_s_wave, cur_spectra/np.max(cur_spectra), color='black', label='spectrum')
            #plt.xlabel('wavelength');plt.ylabel('intensity');plt.legend(loc="upper right")
            plt.title('Spectrum for pixel{}'.format(i))
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

            plot_fn = join(spectrum_dir, 'gt_' + nm)
            plt.plot(cur_s_wave, cur_spectra/np.max(cur_spectra), color='black', label='spectrum')
            plt.plot(gt_wave, gt_spectra/np.max(gt_spectra),label='gt')
            plt.savefig(plot_fn);plt.close()

            '''
            wave, gt_spectra_ol, gen_spectra_ol = overlay_spectrum(gt_fn, cur_s_wave, spectra)
            print(gt_spectra_ol.shape, gen_spectra_ol.shape)
            sam = calculate_sam_spectrum(gt_spectra_ol/np.max(gt_spectra_ol), gen_spectra_ol/np.max(gen_spectra_ol))
            cur_sam.append(sam)
            '''

        cur_sam = []
        avg_spectra = np.mean(spectra, axis=0)
        gt_wave, gt_spectra = process_gt_spectra(gt_fn)
        helper(str(model_id) + '_' + str(i), avg_spectra)
        return cur_sam

    def recon_spectrum_(model_id, batch_coords, covars, spectrum_wave, orig_wave,
                        orig_transms, net, trans_args, spectrum_dir, args):
        ''' Generate spectra for pixels specified by coords using given net
            Save, plot spectrum, and calculate metrics
            @Param
              coords: list of n arrays, each array can be of size
                      [bsz,nsmpl,3] / [bsz,nbands,nsmpl_per_band,2] / [bsz,nsmpl,2]
        '''
        sams = []
        wave_fn = join(spectrum_dir, 'wave.npy')
        np.save(wave_fn, spectrum_wave)
        gt_fns = args.gt_spectra_fns

        wave_hi = int(min(args.wave_hi, int(np.max(spectrum_wave))))
        id_lo = np.argmax(spectrum_wave > args.wave_lo)
        id_hi = np.argmin(spectrum_wave < wave_hi)
        spectrum_wave = spectrum_wave[id_lo:id_hi]

        for i, coord in enumerate(batch_coords):
            print(coord)
            spectra = generate_spectra(args.mc_cho, coord, None, net, trans_args) # None is covar[i]
            #np.save(join(spectrum_dir, str(model_id) + '_' + str(i)), spectra)

            if args.mc_cho == 'mc_hardcode':
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
