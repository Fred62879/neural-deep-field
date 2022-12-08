
import numpy as np

from os.path import join, exists
from wisp.utils.common import get_grid, get_gt_spectra_pix_pos


def add_spectra_supervision_args(config):
    assert(not config['plot_centerize_spectrum'])

    config['trusted_spectra_wave_range'] = [config['trusted_wave_lo'],
                                            config['trusted_wave_hi']]
    cho = config['spectra_supervision_cho']
    subtile_ids = ['5','5','5','5','5','5']
    assert(config['tile_id'] == '1')
    assert(config['footprint'] == '9812')
    assert(config['subtile_id'] == subtile_ids[cho])
    '''
    spectra_fns = [['spec1d.mn44.000.lid_737','spec1d.mn44.008.lid_738'], # 4
                   ['spec1d.mn44.012.lid_736'], # 5
                   ['spec1d.mn44.066.3203'],    # 5
                   ['spec1d.mn44.067.1943'],    # 5
                   ['spec1d.m41be.051.4972']]   # 7
    '''
    ras   = [[149.408493, 149.408493], # img_sz 64, start_r/c 86/188
             [149.38211, 149.38211], # img_sz 192, start_r/c 1900/772
             [149.38211, 149.3744965], # img_sz 192, start_r/c 1900/772
             [149.3744965, 149.38211], #  duplication for sanity check
             [149.3820953, 149.38211, 149.3744965, 149.3755035]  # img_sz 320, start_r/c 1908/708
             ][cho]
    decs  = [[2.325505, 2.325505],
             [2.4109753, 2.4109753],
             [2.4109753, 2.4152501],
             [2.4152501, 2.4109753],
             [2.409508, 2.4109753, 2.4152501, 2.423475] # id:[3278,13198,42801,99099
             ][cho]

    # add pixl id of gt spectra
    pixl_ids = []
    fits_fn = join(config['data_dir'], config['dr'] + '_input',
                   config['dr'] + '_dud', config['groups'][config['img_id']][0])
    for ra, dec, in zip(ras, decs):
        r, c = get_gt_spectra_pix_pos(fits_fn, ra, dec)
        r -= config['start_r']
        c -= config['start_c']
        pixl_ids.append(r*config['img_sz'] + c)

    config['spectra_supervision_gt_spectra_pixl_ids'] = pixl_ids

    spectra_fns = [['spec1d.mn44.012.lid_736', 'spec1d.mn44.012.lid_736'],
                   ['spec1d.mn44.046.serendip', 'spec1d.mn44.046.serendip'],
                   ['spec1d.mn44.046.serendip', 'spec1d.mn44.067.1943'],
                   ['spec1d.mn44.067.1943', 'spec1d.mn44.046.serendip'],
                   ['spec1d.mn44.046.MIPS_35174','spec1d.mn44.046.serendip',
                   'spec1d.mn44.067.1943','spec1d.mn44.043.MIPS_30328']][cho]
    fns = [join(config['spectra_dir'], fn + '.npy') for fn in spectra_fns]
    config['spectra_supervision_gt_spectra_fns'] = fns

    # add argument for spectrum plotting
    config['gt_spectra_fns'] = fns
    # pass each coord individually thru the network (sanity check)
    coords = [{'coords': [id], 'radec': False} for id in pixl_ids]
    config['spectra_coords'] = coords

def add_centerize_gt_spectra_args(config):
    ''' Add hardcoded definition for gt spectra whose corresponding
          pixel is at the center of the train img.
    '''
    assert(not config['spectra_supervision'] and config['gt_spectra_cho'] != -1)

    img_sz = config['img_sz']
    cho = config['gt_spectra_cho']
    if cho not in set([0,1,2,3,4,5]): return

    # currently only has spectra from calexp-9812-*-1%2C[4,5,7]
    subtile_ids = ['4','4','5','5','5','7']
    assert(config['tile_id'] == '1')
    assert(config['footprint'] == '9812')
    assert(config['subtile_id'] == subtile_ids[cho])

    ''' get position of pixel with gt spectra
        exact_coord: ra/dec coordinates
        rough_coord: r/c position closest to exact_coord
        negb_coord:  neighbouring pixels surrounding rough coord
    '''
    ras  = [149.410507,149.417206,149.408493,149.377594,149.3744965,149.373750]
    decs = [2.282275,2.318980,2.325505,2.3632109,2.4152501,2.776508]
    ra, dec = ras[cho], decs[cho]

    fits_fn = join(config['data_dir'], config['dr'] + '_input',
              config['dr'] + '_dud', config['groups'][config['img_id']][0])
    r, c = get_gt_spectra_pix_pos(fits_fn, ras[cho], decs[cho])

    # redefine cropping start row and column to make sure
    # the gt spectra pixl is at the cutout centre
    start_r = config['start_r'] = int(r) - img_sz//2
    start_c = config['start_c'] = int(c) - img_sz//2
    assert(config['start_r'] >= 0 and config['start_c'] >= 0)
    r, c = r - start_r, c - start_c

    rough_id = [r * config['img_sz'] + c]
    exact_coord = np.array([[ra, dec]])
    # 4 negibhouring coordinates surrounding given pixel
    negb_rc_coords = get_grid(r, r + 2, c, c + 2).reshape((-1, 2))
    negb_ids = np.array([c[0] * config['img_sz'] + c[1] for c in negb_rc_coords])

    config['spectra_coords'] = [
        {'coords':exact_coord,'radec':True},
        {'coords':rough_id,'radec':False},
        {'coords':negb_ids,'radec':False}
    ]

    # add gt spectra fns
    spectra_fns = ['spec1d.mn44.000.lid_737','spec1d.mn44.008.lid_738',
                   'spec1d.mn44.012.lid_736','spec1d.mn44.066.3203',
                   'spec1d.mn44.067.1943','spec1d.m41be.051.4972']
    config['gt_spectra_fn'] = spectra_fns[cho] + '.npy'
    config['gt_spectra_fns'] = [join(config['spectra_dir'], config['gt_spectra_fn'])]*3

def add_spectrum_plotting_args(config):
    if not config['plot_spectrum'] and not config['plot_cdbk_spectrum']:
        return

    config['spectrum_labels'] = ['g', 'r', 'i', 'z', 'y', 'nb387', 'nb816', 'nb921','u','u*']
    config['spectrum_colors'] = ['green','red','blue','gray','yellow','gray','red','blue','yellow','blue']
    config['spectrum_styles'] = ['solid','solid','solid','solid','solid','dashed','dashed','dashed','dashdot','dashdot']

    # get coordinates (pixl id or ra/dec form) for all specified spectra pixel
    # together with the corspd gt spectra data filename
    if config['is_test']:
        fn = 'fake_spectrum'+config['sensor_collection_name']+str(config['fake_spectra_cho'])+'.npy'
        config['spectra_coords'] = [{'coords': config['fake_coord'],'radec':False}]
        config['gt_spectra_fns'] = [join(config['spectra_dir'], fn)]

    elif config['spectra_supervision']:
        add_spectra_supervision_args(config)

    elif config['plot_centerize_spectrum']:
        add_centerize_gt_spectra_args(config)

    else: # hardcode pixel r/c position to plot spectrum w/o gt spectra
        if config['recon_img_sz'] == 64:
            negb_rc_coords = np.array([[32,32],[32,32]])
        else: raise Exception('Unsupported rsz for spectrum plotting')
        negb_ids = np.array([c[0] * config['img_sz'] + c[1] for c in negb_rc_coords])
        config['spectra_coords'] = [{'coords': negb_ids, 'radec':False}]
        config['gt_spectra_fns'] = None

def add_trans_args(config):
    ''' add transsmission and wave filename '''
    mc_cho = config['mc_cho']
    verbose = config['verbose']
    base_dir = config['base_dir']
    trans_dir = config['trans_dir']
    trans_cho = config['trans_cho']
    nsmpl = config['num_trans_smpl']

    if mc_cho == 'mc_hardcode':
        config['infer_use_all_wave'] = False

    nsmpl_within_bands_str = 'nsmpl_within_bands'

    full_trans_str = 'full_trans.npy'
    flat_trans_str = 'flat_trans.npy'
    bdws_trans_str = 'bdws_trans.txt'
    hdcd_trans_str = 'hdcd_trans'+str(nsmpl)+'.npy'

    full_inorm_trans_str = 'full_inorm_trans.npy'
    flat_inorm_trans_str = 'flat_inorm_trans.npy'
    bdws_inorm_trans_str = 'bdws_inorm_trans.txt'
    hdcd_inorm_trans_str = 'hdcd_inorm_trans'+str(nsmpl)+'.npy'

    config['base_wave_fn'] = join(base_dir, 'base_wave.txt')
    config['base_trans_fn'] = join(base_dir, 'base_trans.txt')

    config['encd_ids_fn'] = join(trans_dir, 'encd_ids.npy')
    config['bdws_wave_fn'] = join(trans_dir, 'bdws_wave.txt')
    config['full_wave_fn'] = join(trans_dir, 'full_wave.npy')
    config['hdcd_trans_dim'] = 35 if config['num_bands'] <= 5 else 40
    config['hdcd_wave_fn'] = join(trans_dir, 'hdcd_wave' + str(nsmpl) + '.npy')
    config['nsmpl_within_bands_fn'] = join(trans_dir, 'nsmpl_within_bands.npy')

    #config['full_nsmpl'] = len(np.load(config['full_wave_fn']))
    config['nsmpl_per_band'] = config['num_trans_smpl']//config['num_bands']

    if trans_cho == 'orig_trans':
        #if verbose:
        #    print('original trans (with same discretization interval)')
        config['full_trans_fn'] = join(trans_dir, full_trans_str)
        config['flat_trans_fn'] = join(trans_dir, flat_trans_str)
        config['hdcd_trans_fn'] = join(trans_dir, hdcd_trans_str)
        config['bdws_trans_fn'] = join(trans_dir, bdws_trans_str)

    elif trans_cho == 'norm_trans':
        assert(False)
        #if verbose:
        #    print('normed trans (with diff discretization interval)')
        config['full_trans_fn'] = join(trans_dir, full_trans_str)
        config['flat_trans_fn'] = join(trans_dir, flat_trans_str)
        config['hdcd_trans_fn'] = join(trans_dir, hdcd_trans_str)
        config['bdws_trans_fn'] = join(trans_dir, bdws_trans_str)

    elif trans_cho == 'inte_norm_trans':
        assert(False)
        #if verbose: print('trans integrates to 1')
        config['full_trans_fn'] = join(trans_dir, full_inorm_trans_str)
        config['flat_trans_fn'] = join(trans_dir, flat_inorm_trans_str)
        config['hdcd_trans_fn'] = join(trans_dir, hdcd_inorm_trans_str)
        config['bdws_trans_fn'] = join(trans_dir, bdws_inorm_trans_str)

    #config['covr_rnge'] = None if not config['uniform_smpl'] else \
    #    get_bandwise_covr_rnge(config['bdws_wave_fn'],
    #                           config['bdws_trans_fn'],1e-3).type(config['float_tensor'])

    if config['recon_synthetic']: # use only in multi-band recon
        if verbose: print('= synthetic band sampling for recon')
        config['synthetic_distrib_fn'] = \
            join(trans_dir, 'distrib_uniform.npy')

    distrib_fn = None
    if mc_cho == 'mc_hardcode':
        if verbose: print('= use hardcoded samples')
        pass
    elif config['uniform_smpl']:
        if mc_cho == 'mc_bandwise':
            if verbose: print('= uniform bandwise sampling')
            distrib_fn = join(trans_dir, 'distrib_bdws_uniform.txt')
        elif mc_cho == 'mc_mixture':
            if verbose: print('= uniform across-band sampling')
            distrib_fn = join(trans_dir, 'distrib_uniform.npy')
        else:
            raise Exception('Invalid input choice for uniform sampling')
    else:
        if mc_cho == 'mc_bandwise':
            if verbose: print('= non-uniform bandwise sampling')
            distrib_fn = config['bdws_trans_fn']
        elif mc_cho == 'mc_mixture':
            if verbose: print('= non-uniform mixture sampling')
            distrib_fn = join(trans_dir, 'distrib.npy')
        else:
            raise Exception('Invalid input choice for sampling')

    config['distrib_fn'] = distrib_fn
    if verbose: print('= sampling distrib fn', distrib_fn)

def add_trans_spectra_args(config):
    if config['space_dim'] == 2: return
    add_trans_args(config)
    add_spectrum_plotting_args(config)
