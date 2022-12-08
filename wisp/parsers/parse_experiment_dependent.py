
from pathlib import Path
from os.path import join


def add_output_paths(config):
    output_dir = join \
        (config['data_dir'], 'pdr3' +'_output', 'iu_output',
         config['sensor_collection_name'],
         'trail_'+ config['trail_id'], config['experiment_id'])

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if config['verbose']:
        print('= output dir', output_dir)

    for path_nm, folder_nm, in zip(
            ['model_dir','recon_dir','metric_dir', 'spectrum_dir',
             'cdbk_spectrum_dir', 'cutout_dir','embd_map_dir','latent_dir',
             'latent_embd_dir'],
            ['models','recons','metrics','spectrum','cdbk_spectrum',
             'cutout','embd_map','latent','latent_embd_dir']
    ):
        path = join(output_dir, folder_nm)
        config[path_nm] = path
        Path(path).mkdir(parents=True, exist_ok=True)

    config['counts_fn'] = join(output_dir, 'counts.')
    config['grad_fn'] = join(output_dir, 'gradient.png')
    config['train_loss_fn'] = join(output_dir, 'loss.npy')
    config['sample_hist_fn'] = join(output_dir, 'sample_hist.png')
    config['model_fns'] = [join(config['model_dir'], str(i) + '.pth')
                           for i in range(config['num_model_smpls'])]
    #config['recon_png_fn'] = join(config['recon_dir'], 'recon.png')
    config['embd_map_fn'] = join(config['recon_dir'], 'embd_map')
    config['cutout_png_fn'] = join(config['recon_dir'], 'cutout')
    if config['mask_config'] != 'region':
        mask_str = '_' + str(float(100 * config['sample_ratio']))
    else:
        mask_str = '_' + str(config['m_start_r'])+'_'+str(config['m_start_c'])+'_'+str(config['mask_sz'])

    if config['inpaint_cho'] == 'spectral_inpaint':
        #config['sampled_pixl_id_fn'] = join \
        config['mask_fn'] = join(config['mask_dir'], str(config['img_sz']) + mask_str + '.npy')
        config['masked_pixl_id_fn'] = join(config['mask_dir'], str(config['img_sz']) + mask_str + '_masked_id.npy')
    else: config['mask_fn'], config['masked_pixl_id_fn'] = None, None


def setup_pixl_output_norm(config):
    if config['output_norm_cho'] == 'arcsinh' or config['train_pixels_norm'] == 'arcsinh':
        config['output_norm_cho'] = 'arcsinh'
        config['train_pixels_norm'] = 'arcsinh'
        # infer should remove arcsinh layer unless rnorm specified
        config['infer_pixl_norm_cho'] = 'arcsinh' if config['recon_norm'] else 'identity'
        config['infer_output_norm_cho'] = 'arcsinh' if config['recon_norm'] else 'identity'
    else:
        config['infer_pixl_norm_cho'] = config['train_pixels_norm']
        config['infer_output_norm_cho'] = config['output_norm_cho']

    #config['infer_pixls_fn'] = join(config['img_data_dir'], 'pixls_'+ str(config['infer_pixl_norm_cho']) + '_' + config['suffx'])
    #config['train_pixls_fn'] = join(config['img_data_dir'], 'pixls_'+ str(config['train_pixl_norm_cho']) + '_' + config['suffx'])


def add_model_args(config):

    # i) coord pe before latent generation
    if config['coord_pe_cho'] == 'rand_gaus':
        coord_pe_args = [config['coord_dim'], config['coord_pe_dim'], config['coord_pe_omega'],
                         config['coord_pe_sigma'], config['coord_pe_bias'],
                         config['float_tensor'], config['verbose'], config['coord_pe_seed']]
        config['coord_pe_args'] = coord_pe_args
    else:
        assert(False)

    # ii) encoder for latent generation
    if config['encode']:
        dim_in = config['coord_pe_dim'] if config['pe_coord'] else config['coord_dim']
        dim_out = config['latent_dim']
        if config['encoder_output_scaler']: dim_out += 1
        if config['encoder_output_redshift']: dim_out += 1

        if config['encoder_cho'] == 'relumlp':
            relu_args = [dim_in, config['encoder_dim_hidden'],
                         dim_out, config['encoder_num_hidden_layers'],
                         config['encoder_mlp_seed']]
            config['encoder_mlp_args'] = relu_args

        elif config['encoder_cho'] == 'siren':
            siren_args = [dim_in, config['encoder_dim_hidden'],
                          dim_out, config['encoder_num_hidden_layers'],
                          config['encoder_siren_last_linr'], config['encoder_first_w0'],
                          config['encoder_hidden_w0'], config['encoder_coords_scaler'],
                          config['encoder_mlp_seed'], config['float_tensor']]
            config['encoder_mlp_args'] = siren_args

        elif config['encoder_cho'] == 'pemlp':
            pemlp_args = [config['encoder_pe_dim'], config['encoder_dim_hidden'],
                          dim_out, config['encoder_num_hidden_layers'], config['encoder_mlp_seed']]
            config['encoder_mlp_args'] = pemlp_args

        else: # not support mfn yet
            assert(False)
    else:
        config['pe_wave'] = False
        config['pe_coord'] = False
        config['encoder_quantize'] = False
        config['encoder_output_scaler'] = False

    if not config['encoder_quantize']:
        config['plot_latent_embd'] = False
        config['plot_cdbk_spectrum'] = False
        config['plot_embd_map_during_recon'] = False
        config['plot_embd_map_during_train'] = False

    # ~ pe wave
    if config['pe_wave']:
        if config['wave_pe_cho'] == 'pe':
            config['wave_pe_dim'] = 2 * (config['wave_pe_max_deg'] - \
                                         config['wave_pe_min_deg']) * config['wave_dim']
            pe_args = [config['wave_pe_min_deg'], config['wave_pe_max_deg'],
                       config['float_tensor'], config['verbose'], config['wave_pe_seed']]
            config['wave_pe_args'] = pe_args

        elif config['wave_pe_cho'] == 'rand_gaus':
            pe_args = [1, config['wave_pe_dim'], config['wave_pe_omega'],
                       config['wave_pe_sigma'], config['wave_pe_bias'],
                       config['float_tensor'], config['verbose'], config['wave_pe_seed']]
            config['wave_pe_args'] = pe_args

    # iii) main mlp
    # ~ pe args for main mlp
    if config['mlp_cho'] == 'pemlp':
        pe_cho = config['pe_cho']
        valid_pe_chos = set(config['valid_pe_chos'])
        assert(pe_cho in valid_pe_chos)

        if pe_cho == 'pe':
            config['pe_dim'] = (config['pe_max_deg']-config['pe_min_deg']) \
                * config['dim'] * 2
            config['pe_args'] = pe_args

        elif config['pe_cho'] == 'rand_gaus':
            pe_args = [config['coord_dim'], config['pe_dim'], config['pe_omega'],
                       config['pe_sigma'], config['pe_bias'], config['float_tensor'],
                       config['verbose'], config['pe_seed']]
            config['pe_args'] = pe_args
        else:
            assert(False)
    else:
        config['pe_cho'] = 'None'
        config['pe_args'] = None

    # ~ dim_in for mlp
    if config['mlp_cho'] == 'pemlp':
        mlp_dim_in = config['pe_dim']
    elif config['encode']:
        mlp_dim_in = config['latent_dim']
        if config['pe_wave']:
            mlp_dim_in += config['wave_pe_dim']
        elif config['dim'] == 3:
            mlp_dim_in += 1
    else:
        mlp_dim_in = config['dim']

    # ~ dim_out for mlp
    if config['dim'] == 2:
        mlp_dim_out = config['num_bands']
    else: mlp_dim_out = 1

    # ~ mlp args
    if config['mlp_cho'] == 'pemlp':
        pemlp_args = [mlp_dim_in, config['mlp_dim_hidden'], mlp_dim_out,
                      config['mlp_num_hidden_layers'], config['mlp_seed']]
        config['mlp_args']  = pemlp_args

    elif config['mlp_cho'] == 'siren':
        siren_args = [mlp_dim_in, config['mlp_dim_hidden'], mlp_dim_out,
                      config['mlp_num_hidden_layers'], config['last_linear'],
                      config['first_w0'], config['hidden_w0'], config['coords_scaler'],
                      config['mlp_seed'], config['float_tensor']]
        config['mlp_args']  = siren_args
    else:
        assert(False)

def add_experiment_dependent_args(config):
    #add_model_args(config)
    setup_pixl_output_norm(config)
    #add_output_paths(config)
