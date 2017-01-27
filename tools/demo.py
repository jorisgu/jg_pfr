#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
from matplotlib.pyplot import cm


CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}
def save_detections(im, scores, boxes, thresh, nms_tresh, images_path_results):
    """Save detected bounding boxes."""

    colors=iter(cm.rainbow(np.linspace(0,1,len(CLASSES))))



    # image and fig
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    # adding each class on image
    for cls_ind, cls in enumerate(CLASSES[1:]):
        chosen_color=next(colors)
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, nms_tresh)
        dets = dets[keep, :]
        #vis_detections(im, cls, dets, thresh=CONF_THRESH)


        inds = np.where(dets[:, -1] >= thresh)[0]
        if len(inds) == 0:
            continue


        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]

            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor=chosen_color, linewidth=3.5)
                )
            ax.text(bbox[0], bbox[1] - 2,
                    '{:s} {:.3f}'.format(cls, score),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=14, color='white')


    plt.axis('off')
    plt.tight_layout()
    #plt.draw()
    plt.savefig(images_path_results)
    plt.close()

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    #plt.draw()
    plt.savefig()

def saving_demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    images_path = '/home/jguerry/Documents/GuiOnTheFloor/'
    images_path_results = '/home/jguerry/Documents/GuiOnTheFloor/results/'
    # Load the demo image
    im_file = os.path.join(images_path, image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    save_detections(im, scores, boxes, CONF_THRESH, NMS_THRESH, os.path.join(images_path_results, image_name))

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    images_path = '/home/jguerry/Documents/GuiOnTheFloor/'
    images_path_results = '/home/jguerry/Documents/GuiOnTheFloor/results/'
    # Load the demo image
    im_file = os.path.join(images_path, image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
                '001763.jpg', '004545.jpg', '0000873-000002922482.jpg']
    im_names = ['0000144-000000482010.jpg','0000155-000000518852.jpg',
                '0000166-000000555763.jpg','0000177-000000592591.jpg',
                '0000188-000000629454.jpg','0000199-000000666324.jpg',
                '0000210-000000703232.jpg','0000221-000000740060.jpg',
                '0000232-000000776946.jpg','0000243-000000813790.jpg',
                '0000254-000000850658.jpg','0000265-000000887519.jpg',
                '0000276-000000924421.jpg','0000287-000000961311.jpg',
                '0000298-000000998112.jpg','0000309-000001034987.jpg',
                '0000320-000001071842.jpg','0000331-000001108712.jpg',
                '0000342-000001145573.jpg','0000353-000001182461.jpg',
                '0000364-000001219370.jpg','0000375-000001256238.jpg',
                '0000386-000001293046.jpg','0000397-000001329942.jpg',
                '0000408-000001366851.jpg','0000419-000001403646.jpg',
                '0000430-000001440521.jpg','0000441-000001477370.jpg',
                '0000452-000001514256.jpg','0000463-000001551142.jpg',
                '0000474-000001587957.jpg','0000485-000001624824.jpg',
                '0000496-000001661691.jpg','0000507-000001698556.jpg',
                '0000518-000001735463.jpg','0000529-000001772347.jpg',
                '0000540-000001809255.jpg','0000551-000001846131.jpg',
                '0000562-000001883009.jpg','0000573-000001919886.jpg',
                '0000584-000001956800.jpg','0000595-000001993561.jpg',
                '0000606-000002030437.jpg','0000617-000002067238.jpg',
                '0000628-000002104129.jpg','0000639-000002140959.jpg',
                '0000650-000002177823.jpg','0000661-000002214693.jpg',
                '0000672-000002251552.jpg','0000683-000002288418.jpg',
                '0000694-000002325341.jpg','0000705-000002362150.jpg',
                '0000716-000002399036.jpg','0000727-000002435954.jpg',
                '0000738-000002472920.jpg','0000749-000002509619.jpg',
                '0000760-000002546550.jpg','0000771-000002583474.jpg',
                '0000782-000002620366.jpg','0000793-000002657069.jpg',
                '0000804-000002693990.jpg','0000815-000002730825.jpg',
                '0000826-000002767680.jpg','0000837-000002804558.jpg',
                '0000848-000002841489.jpg','0000859-000002878281.jpg',
                '0000870-000002915197.jpg','0000881-000002952034.jpg',
                '0000892-000002988910.jpg','0000903-000003025745.jpg',
                '0000914-000003062603.jpg','0000925-000003099477.jpg',
                '0000936-000003136336.jpg','0000947-000003173195.jpg',
                '0000958-000003210061.jpg','0000969-000003246930.jpg',
                '0000980-000003283795.jpg','0000991-000003320780.jpg',
                '0001002-000003357600.jpg','0001013-000003394458.jpg',
                '0001024-000003431378.jpg','0001035-000003468256.jpg',
                '0001046-000003505079.jpg','0001057-000003541853.jpg',
                '0001068-000003578727.jpg','0001079-000003615596.jpg','0001090-000003652450.jpg']
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        saving_demo(net, im_name)
        #demo(net, im_name)

    plt.show()
