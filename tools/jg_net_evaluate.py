#!/usr/bin/env python

import _init_paths
#from fast_rcnn.test import test_net
import fast_rcnn.test as frcnnt
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import time, os, sys

# from lib faster_rcnn.test
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
from utils.timer import Timer
import numpy as np
import cv2
from fast_rcnn.nms_wrapper import nms
import cPickle
from utils.blob import im_list_to_blob


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a RPN')
    parser.add_argument('--imdb_name', dest='imdb_name',type=str)
    parser.add_argument('--output_dir', dest='output_dir', type=str)
    parser.add_argument('--input_file_name', dest='input_file_name', type=str)
    parser.add_argument('--output_file_name', dest='output_file_name', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if os.path.exists(os.path.join(args.output_dir, args.input_file_name)):
        with open(os.path.join(args.output_dir, args.input_file_name), 'rb') as fid:
            all_boxes = cPickle.load(fid)
    else:
        print('File not found in net evaluate')

    imdb = get_imdb(args.imdb_name)
    imdb.output_dir = args.output_dir
    #print 'Evaluating detections'
    imdb.evaluate_detections(all_boxes, os.path.join(args.output_dir,'results'))
