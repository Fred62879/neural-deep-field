
import argparse
import configargparse

import sys
sys.path.insert(0, './parsers')

from os.path import join
from wisp.parsers.parse_yaml import parse_input_config
from wisp.parsers.parse_basic import add_basic_args
from wisp.parsers.parse_input import add_input_args
from wisp.parsers.setup_experiment import setup_experiments
from wisp.parsers.parse_trans_spectra import add_trans_spectra_args
from wisp.parsers.parse_experiment_dependent import add_experiment_dependent_args


def finalize(config):
    ''' Finalize arguments that changed during parsing '''

    # pixl filename
    suffx = config['img_id'] +'_'+ str(config['img_sz']) +'_'+ \
        str(config['start_r']) +'_'+ str(config['start_c']) + '.npy'
    config['infer_pixls_fn'] = join(config['img_data_dir'], 'pixls_'+
                                    (config['infer_pixl_norm_cho']) + '_' + suffx)
    config['train_pixls_fn'] = join(config['img_data_dir'], 'cutout_'
                                    if config['cutout_based_train'] else 'pixls_')
    config['train_pixls_fn'] += str(config['train_pixl_norm_cho']) + '_' + suffx
    #config['gt_img_fn'] = join(config['img_data_dir'], 'gt_img_'+ suffx)

def parse_args():
    ''' Parse all command arguments and generate all needed ones. '''
    config, args_str = parse_input_config()

    add_basic_args(config)
    add_input_args(config)
    add_trans_spectra_args(config)
    setup_experiments(config)
    add_experiment_dependent_args(config)
    #finalize(config)

    args = argparse.Namespace(**config)
    return args, args_str
