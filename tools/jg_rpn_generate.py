#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Faster R-CNN network using alternating optimization.
This tool implements the alternating optimization algorithm described in our
NIPS 2015 paper ("Faster R-CNN: Towards Real-time Object Detection with Region
Proposal Networks." Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun.)
"""

import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
from rpn.generate import imdb_proposals
import argparse
import pprint
import numpy as np
import sys, os
import multiprocessing as mp
import cPickle
import shutil

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a RPN')
    parser.add_argument('--gpu_id', dest='gpu_id', type=int)
    parser.add_argument('--path_net_proto', dest='path_net_proto', type=str)
    parser.add_argument('--path_net_weights', dest='path_net_weights', type=str)
    parser.add_argument('--imdb_name', dest='imdb_name',type=str)
    parser.add_argument('--path_cfg', dest='path_cfg',type=str)
    parser.add_argument('--output_dir', dest='output_dir', type=str)
    parser.add_argument('--output_file_name', dest='output_file_name', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def get_roidb(imdb_name, rpn_file=None):
    print 'Requiring dataset `{:s}` for training'.format(imdb_name)
    imdb = get_imdb(imdb_name)
    print 'Loaded dataset `{:s}` for training'.format(imdb.name)
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
    if rpn_file is not None:
        imdb.config['rpn_file'] = rpn_file
    roidb = get_training_roidb(imdb)
    return roidb, imdb


def _init_caffe(cfg):
    """Initialize pycaffe in a training process.
    """

    import caffe
    # fix the random seeds (numpy and caffe) for reproducibility
    np.random.seed(cfg.RNG_SEED)
    caffe.set_random_seed(cfg.RNG_SEED)
    # set up caffe
    caffe.set_mode_gpu()
    caffe.set_device(cfg.GPU_ID)



if __name__ == '__main__':
    args = parse_args()

    if args.path_cfg is not None:
        cfg_from_file(args.path_cfg)
    cfg.GPU_ID = args.gpu_id


    print 'RPN model: {}'.format(args.path_net_weights)
    print('Using config:')
    pprint.pprint(cfg)

    import caffe
    _init_caffe(cfg)


    imdb = get_imdb(args.imdb_name)
    print 'Loaded dataset `{:s}` for proposal generation'.format(imdb.name)

    # Load RPN and configure output directory
    rpn_net = caffe.Net(args.path_net_proto, args.path_net_weights, caffe.TEST)
    #output_dir = get_output_dir(imdb)
    print 'Output will be saved to `{:s}`'.format(args.output_dir)
    # Generate proposals on the imdb
    rpn_proposals = imdb_proposals(rpn_net, imdb)
    # Write proposals to disk and send the proposal file path through the
    # multiprocessing queue

    rpn_proposals_path = os.path.join(args.output_dir, args.output_file_name)
    with open(rpn_proposals_path, 'wb') as f:
        cPickle.dump(rpn_proposals, f, cPickle.HIGHEST_PROTOCOL)
    print 'Wrote RPN proposals to {}'.format(rpn_proposals_path)
