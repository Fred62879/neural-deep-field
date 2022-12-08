

def parse_dataset_args(config):
    process_config(config)

def process_config(config):
    ''' redefine some args to guarantee consistency '''
    config['num_bands'] = len(config['sensors_full_name'])

    # define img id and generate all source fns
    config['img_id'], config['groups'], config['wgroups'] = \
        generate_source_fns \
        (config['footprint'], config['tile_id'], config['subtile_id'],
         config['sensors_full_name'], config['verbose'])

    # str definition for mask dir
    config['mask_bands'] = ''
    for b in config['inpaint_bands']:
        config['mask_bands'] += str(b)

def parse_data_spec(config):
    num_bands = config['num_bands']
    num_pixls = config['num_imgs']*config['img_sz']**2

    # inpaint specification
    if ['inpaint_cho'] != 'no_inpaint':
        if config['mask_bandset_cho'] == '10':
            config['mask_bandset'] = [0,1,2,3,4,5,6,7,8,9]
        elif config['mask_bandset_cho'] == '5grizy':
            config['mask_bandset'] = [0,1,2,3,4]
        else:
            raise Exception('Unsupported mask bandsets')
