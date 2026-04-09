#!/usr/bin/env python3
"""
Environment utilities for BigVGAN.
"""

import argparse
import json
from collections import OrderedDict


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)


def create_hparams():
    """Create default hyperparameters."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None)
    parser.add_argument('-e', '--exp_name', type=str, default='default')
    parser.add_argument('-m', '--model', type=str, default='bigvgan')
    args = parser.parse_args()
    
    config_path = args.config
    if config_path is None:
        # Default config
        hparams = AttrDict({
            # Model parameters
            'resblock': '1',
            'resblock_kernel_sizes': [3, 7, 11],
            'resblock_dilation_sizes': [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            'upsample_rates': [8, 8, 2, 2],
            'upsample_kernel_sizes': [16, 16, 4, 4],
            'upsample_initial_channel': 512,
            'activation': 'snake',
            'snake_logscale': 1.0,
            'gpt_dim': 1024,
            'num_mels': 100,
            'speaker_embedding_dim': 192,
            'feat_upsample': True,
            'cond_d_vector_in_each_upsampling_layer': True,
            
            # Additional parameters
            'discriminator_channel_mult': 1,
            'use_spectral_norm': False,
            'mpd_reshapes': [2, 3, 5, 7, 11],
            'resolutions': [[1024, 256, 1024], [2048, 512, 1024], [512, 128, 512]],
            
            # CUDA kernel
            'use_cuda_kernel': False,
        })
    else:
        with open(config_path, 'r') as f:
            data = f.read()
        hparams = AttrDict(json.loads(data))
    
    return hparams


def load_hparams_from_json(path):
    """Load hyperparameters from JSON file."""
    with open(path) as f:
        data = f.read()
    return AttrDict(json.loads(data))


def save_hparams_to_json(hparams, path):
    """Save hyperparameters to JSON file."""
    with open(path, 'w') as f:
        json.dump(hparams, f, indent=4)