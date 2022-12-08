import torch
import numpy as np

from pathlib import Path
from os.path import join

def add_train_infer_args(config):
    num_bands = config['num_bands']
    num_epochs = config['num_epochs']
    num_pixls = config['num_imgs']*config['img_sz']**2

    config['cuda'] = config['cuda'] and torch.cuda.is_available()
    if config['cuda']:
        config['device'] = torch.device('cuda')
        config['float_tensor'] = torch.cuda.FloatTensor
        config['double_tensor'] = torch.cuda.DoubleTensor
    else:
        config['device'] =  torch.device('cpu')
        config['float_tensor'] = torch.FloatTensor
        config['double_tensor'] = torch.DoubleTensor

    config['plot_spectrum'] = config['plot_spectrum'] and config['space_dim'] != 2
    config['recon_synthetic'] = config['recon_synthetic'] \
        and config['space_dim'] != 2 and config['inpaint_cho'] == 'no_inpaint'

    # train and infer args
    if config['recon_img_sz'] is None:
        config['recon_img_sz'] = config['img_sz']

    if config['is_test'] and config['fake_coord'] is None:
        config['fake_coord'] = [0,0]

    if config['inpaint_cho'] == 'spatial_inpaint':
        config['npixls'] = int(num_pixls * config['sample_ratio'])
        config['train_num_pixls_per_epoch'] = int(
            num_pixls * config['train_pixl_ratio_per_epoch'] * config['sample_ratio'])
    else:
        config['npixls'] = num_pixls
        config['train_num_pixls_per_epoch'] = int(
            num_pixls * config['train_pixl_ratio_per_epoch'])

    config['recon_bsz'] = int(min(num_pixls, config['recon_bsz_hi']))
    '''
    config['train_bsz'] = int(min(config['train_bsz_hi'],
                                  config['num_train_pixls']))
    config['train_nb'] = int(np.ceil(config['num_train_pixls']
                                     /config['train_bsz']))
    '''
    model_smpl_intvl = max(1, num_epochs // config['num_model_checkpoint'])
    config['loss_smpl_intvl'] = max(1, num_epochs // 20)
    config['model_smpl_intvl'] = model_smpl_intvl
    config['num_loss_smpls'] = num_epochs // config['loss_smpl_intvl'] + 1

    # save model before ~ epoch starts (0 based)
    smpls = list(np.arange(0, num_epochs + 1, model_smpl_intvl)[1:]) # 1 based
    if smpls[0] != 1: smpls = [1] + smpls
    smpls = [val - 1 for val in smpls] # convert to 0 based
    config['smpl_epochs'] = smpls
    config['num_model_smpls'] = len(config['smpl_epochs'])

    # multi-band image recon args
    config['metric_options'] = ['mse','psnr','ssim']
    config['metric_names'] = ['mse','psnr','ssim']

    # inpaint specification
    if ['inpaint_cho'] != 'no_inpaint':
        if config['mask_bandset_cho'] == '10':
            config['mask_bandset'] = [0,1,2,3,4,5,6,7,8,9]
        elif config['mask_bandset_cho'] == '5grizy':
            config['mask_bandset'] = [0,1,2,3,4]
        else:
            raise Exception('Unsupported mask bandsets')

def add_input_paths(config):
    dr = config['dr']
    img_id = config['img_id']
    data_dir = config['dataset_path']

    dim = str(config['space_dim'])
    img_sz = str(config['img_sz'])
    start_r = str(config['start_r'])
    start_c = str(config['start_c'])
    num_bands = str(config['num_bands'])
    sensor_col_nm = config['sensor_collection_name']
    suffx = img_id +'_'+ img_sz +'_'+ start_r +'_'+ start_c + '.npy'

    input_dir = join(data_dir, dr +'_input')
    spectra_dir = join(input_dir, 'spectra', str(config['tile_id'])+'2c'+str(config['subtile_id']))
    base_dir = join(input_dir, 'transmission')
    img_data_dir = join(input_dir, sensor_col_nm, 'img_data')
    trans_dir = join(input_dir, sensor_col_nm, 'transmission')

    config['img_data_dir'] = img_data_dir

    dirs = [input_dir, spectra_dir, img_data_dir, trans_dir]

    if config['inpaint_cho'] == 'spectral_inpaint':
        mask_dir = join(input_dir, 'sampled_pixl_ids',
                        'cutout_' + suffx[:-4] + '_mask_'
                        + str(config['mask_config']) + '_'
                        + str(config['mask_bandset_cho']) +'_'
                        + str(config['mask_bands']) + '_'
                        + str(config['mask_seed']))
        dirs.append(mask_dir)
        config['mask_dir'] = mask_dir

    for path in dirs:
        Path(path).mkdir(parents=True, exist_ok=True)

    config['suffx'] = suffx
    config['data_dir'] = data_dir
    config['base_dir'] = base_dir
    config['input_dir'] = input_dir
    config['trans_dir'] = trans_dir
    config['spectra_dir'] = spectra_dir
    config['img_data_dir'] = img_data_dir
    config['dud_dir'] = join(input_dir, dr+'_dud')

    config['name_fn'] = join(input_dir, 'name_' + dr +'.txt')
    config['gt_img_fn'] = join(img_data_dir, str(config['gt_img_norm_cho']) + '_gt_img_'+ suffx)
    print(config['gt_img_fn'])
    if config['cutout_based_train']:
        config['coords_fn'] = join(img_data_dir, 'coords_cutout'+ suffx)
        config['weights_fn'] = join(img_data_dir, 'weights_cutout'+ suffx)
    else:
        config['coords_fn'] = join(img_data_dir, 'coords_'+ suffx)
        config['weights_fn'] = join(img_data_dir, 'weights_'+ suffx)

    config['groups_fn'] = join(input_dir, sensor_col_nm, 'groups.npy')
    config['coords_rnge_fn'] = join(img_data_dir, 'coords_rnge_'+ suffx)
    config['gt_fit_fn'] = join(img_data_dir, 'gt_img_'+ suffx[:-4]+'.fits')

    # filename of pixls, redefined finally if need to plot spectrum
    config['infer_pixls_fn'] = join(img_data_dir, 'pixls_'+
                                    (config['infer_pixl_norm_cho']) + '_' + suffx)
    config['train_pixls_fn'] = join(img_data_dir, 'cutout_'
                                    if config['cutout_based_train'] else 'pixls_')
    config['train_pixls_fn'] += str(config['train_pixl_norm_cho']) + '_' + suffx
    #config['norm_args_fn'] = join(img_data_dir, 'norm_args_'+ img_id + '_' + norm_cho + '.npy')

def add_input_args(config):
    #add_input_paths(config)
    #add_train_infer_args(config)
    pass
