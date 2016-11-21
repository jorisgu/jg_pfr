# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

from datasets.pascal_voc import pascal_voc
from datasets.coco import coco
from datasets.nyud import nyud

import numpy as np

# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test','fake']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

# Set up nyud_<year>_<split>_<chanel>
for year in ['v1', 'v2']:
    for split in [  'test',
                    'train',
                    'val',
                    'trainval',

                    'test_i_50_10',
                    'train_i_50_10',
                    'val_i_50_10',
                    'trainval_i_50_10',

                    'test_i_100_60',
                    'train_i_100_60',
                    'val_i_100_60',
                    'trainval_i_100_60',

                    'test_i_all',
                    'train_i_all',
                    'val_i_all',
                    'trainval_i_all',

                    'testGupta',
                    'trainGupta',
                    'valGupta',
                    'trainvalGupta']:

        for chanel in [ 'd_raw_focus_8bits',
                        'd_raw_focus_16bits',
                        'd_raw_normal_8bits',
                        'd_raw_normal_16bits',
                        'd_raw_HHA_8bits',
                        'd_raw_jet_8bits',
                        'd_raw_cubehelix_8bits',
                        'd_raw_HHA_focus_8bits',
                        'd_raw_jet_focus_8bits',
                        'd_raw_cubehelix_focus_8bits',
                        'd_raw_histeqBack_8bits',
                        'd_raw_histeqBack_16bits',
                        'd_raw_histeqFront_8bits',
                        'd_raw_histeqFront_16bits',
                        'd_raw_histeqRandom_8bits',
                        'd_raw_histeqRandom_16bits',
                        'rgb_raw_8bits',
                        'rgb_i_10_8bits',
                        'rgb_i_20_8bits',
                        'rgb_i_30_8bits',
                        'rgb_i_40_8bits',
                        'rgb_i_50_8bits',
                        'rgb_i_60_8bits',
                        'rgb_i_70_8bits',
                        'rgb_i_80_8bits',
                        'rgb_i_90_8bits',
                        'rgb_iRange_100_60_8bits',
                        'rgb_iRange_50_10_8bits',
                        'rgb_iAll_8bits']:
            name = 'nyud_{}_{}_{}'.format(year, split, chanel)
            __sets[name] = (lambda split=split, year=year, chanel=chanel: nyud(split, year, chanel))

# Set up coco_2014_<split>
for year in ['2014']:
    for split in ['train', 'val', 'minival', 'valminusminival']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
    for split in ['test', 'test-dev']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
