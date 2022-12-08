

def process_config(config):
    ''' redefine some args to guarantee consistency '''
    config['num_bands'] = len(config['sensors_full_name'])

    # define img id and generate all source fns
    #config['img_id'], config['groups'], config['wgroups'] = \
    #    generate_source_fns \
    #    (config['footprint'], config['tile_id'], config['subtile_id'],
    #     config['sensors_full_name'], config['verbose'])

    # str definition for mask dir
    config['mask_bands'] = ''
    for b in config['inpaint_bands']:
        config['mask_bands'] += str(b)

def check_choice_args(config):
    #assert(config['train_pixl_norm_cho'] in set(config['valid_pixl_norm_chos']))
    assert(config['loss_cho'] in set(config['valid_loss_chos']))
    assert(config['mask_config'] in set(config['valid_mask_configs']))
    assert(config['trans_cho'] in set(config['valid_trans_chos']))
    assert(config['inpaint_cho'] in set(config['valid_inpaint_chos']))

    #assert(config['output_norm_cho'] in set(config['valid_output_norm_chos']))
    assert(config['pe_cho'] in set(config['valid_pe_chos']))
    assert(config['mlp_cho'] in set(config['valid_mlp_chos']))
    assert(config['ipe_cho'] in set(config['valid_ipe_chos']))
    assert(config['mc_cho'] in set(config['valid_mc_chos']))
    assert(config['integration_cho'] in set(config['valid_integration_chos']))

def add_hardcoded_vals(config):
    # valid train args choices
    config['valid_pixl_norm_chos'] = ['identity','arcsinh']
    config['valid_loss_chos'] = ['l1','l2','rnerf_mean','rnerf_sum']
    config['valid_mask_configs'] = ['rand_diff','rand_same','region']
    config['valid_trans_chos'] = ['orig_trans','norm_trans','inte_norm_trans']
    config['valid_inpaint_chos'] = ['no_inpaint','spatial_inpaint','spectral_inpaint']

    config['update_ipe_covar_chos'] = ['global_rand','local_rand','global_rand_predefined_schedule','global_rand_schedule']
    # valid model arg choices
    config['valid_output_norm_chos'] = ['identity','arcsinh','sinh','None']
    config['valid_mc_chos'] = ['mc_hardcode','mc_bandwise','mc_mixture','None']
    config['valid_mlp_chos'] = ['pemlp','siren','fourier_mfn','gabor_mfn','None']
    config['valid_pe_chos'] = ['pe','pe_mix_dim','pe_rand','rand_gaus','rand_gaus_linr','None']
    config['valid_ipe_chos'] = ['global_constant','local_constant','global_rand','local_rand','global_schedule','global_rand_predefined_schedule','global_rand_schedule','None']
    config['valid_integration_chos'] = ['identity', 'dot_product','trapezoid','simpson']

def add_basic_args(config):
    add_hardcoded_vals(config)
    check_choice_args(config)
    process_config(config)
