# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import caffe
import numpy as np


class BasicBatchingLayer(caffe.Layer):
    """
    Generate new batch from a blob (ie. a features space) (eg. conv5_3) and rois
    from (images_batch_size,512,13,13)
    to   (num_rois, 512, roi_height, roi_width)
    """

    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.debug = params['debug']

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        top[0].reshape(1, 5)


    def forward(self, bottom, top):

        cfg_key = str(self.phase) # either 'TRAIN' or 'TEST'
        # todo either train (all image in ROIs) or test (from a RPN ?)

        im_info = bottom[0].data[0, :] # c h w
        gt_boxes = bottom[1].data
        data_shape = bottom[2].data.shape # batch_size c h w eg. 2 512 13 13
        batch_size = data_shape[0]

        scale_reduction_w = data_shape[3]/im_info[2]
        scale_reduction_h = data_shape[2]/im_info[1]

        W = self.W
        H = self.H

        #generateROISwithParams()
        # todo generation of better bounding boxe for full ranging of the image
        # todo take into account the scaling (automatically)
        # todo a roi layer wich is just a "rebatching layer"
        N = batch_size*self.number_ROIs
        blob = np.zeros((N,5))
        for i in range(batch_size): #for each image
            for j in range(self.number_ROIs):
                x1 = j*2
                y1 = j*2
                x2 = x1 + 100
                y2 = y1 + 100
                blob[i*self.number_ROIs+j,:] = [i,x1,y1,x2,y2]
        top[0].reshape(*(blob.shape))
        top[0].data[...] = blob


    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
