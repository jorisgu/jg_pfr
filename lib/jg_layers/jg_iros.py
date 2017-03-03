from utils.cython_bbox import bbox_overlaps
from nms.gpu_nms import gpu_nms
from nms.cpu_nms import cpu_nms

import os
import caffe
import numpy as np
import numpy.random as npr
import yaml
from PIL import Image
import cPickle
import xml.etree.ElementTree as ET
import sklearn.metrics #for confusion matrix
import math #for view generation

def get_classes(classes_file_path):
    classes = []
    with open(classes_file_path, 'r') as fp:
        for line in fp:
            if line[0]=='#' or line[0]=='\n':
                pass
            else:
                classes.append(line[:-1])
    return classes

def prepareViews(img_w, img_h, ratio_for_stride=0.5, min_size_min=256, min_size_max=256, min_size_step=16, verbose=False):
    min_size_max +=1
    if verbose:
        print('img_size : ' + str(img_w) + ',' +str(img_h))
    list_min_size = range(min_size_min,min_size_max,min_size_step)
    if verbose:
        print('Looking for these sizes of patch : ' + str(list_min_size))

    views = []
    for min_size in list_min_size:
        stride = int(math.ceil(ratio_for_stride*min_size))
        i_w = 0
        i_h = 0
        if verbose:
            print('Looking for this size of patch : ' + str(min_size))
        while i_w + min_size-1 < img_w:
            i_h = 0
            while i_h + min_size-1 < img_h:
                if verbose:
                    print('Find : ' + str(i_w)+','+ str(i_h))
                # views.append([i_w, i_h, min_size, min_size]) # x y w h
                views.append([i_w, i_h, i_w + min_size, i_h + min_size]) #x1 y1 x2 y2
                i_h += stride
            i_w += stride
    if verbose:
        print('Number of patchs in this image : ' + str(len(views)))
    return views

def bbox_transform(ex_rois, gt_rois):
    # --------------------------------------------------------
    # Fast R-CNN
    # Copyright (c) 2015 Microsoft
    # Licensed under The MIT License [see LICENSE for details]
    # Written by Ross Girshick
    # --------------------------------------------------------
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets

def bbox_transform_inv(boxes, deltas):
    # --------------------------------------------------------
    # Fast R-CNN
    # Copyright (c) 2015 Microsoft
    # Licensed under The MIT License [see LICENSE for details]
    # Written by Ross Girshick
    # --------------------------------------------------------
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    # --------------------------------------------------------
    # Fast R-CNN
    # Copyright (c) 2015 Microsoft
    # Licensed under The MIT License [see LICENSE for details]
    # Written by Ross Girshick
    # --------------------------------------------------------
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes

def _compute_targets_anchor(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)

def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2**np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """

    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in xrange(ratio_anchors.shape[0])])
    return anchors

def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors

def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

# def bbox_overlaps(boxes,query_boxes):
#     """
#     Parameters
#     ----------
#     boxes: (N, 4) ndarray of float
#     query_boxes: (K, 4) ndarray of float
#     Returns
#     -------
#     overlaps: (N, K) ndarray of overlap between boxes and query_boxes
#     """
#     N = boxes.shape[0]
#     K = query_boxes.shape[0]
#     overlaps = np.zeros((N, K), dtype=np.float)
#     for k in range(K):
#         box_area = ((query_boxes[k, 2] - query_boxes[k, 0] + 1) * (query_boxes[k, 3] - query_boxes[k, 1] + 1))
#         for n in range(N):
#             iw = (min(boxes[n, 2], query_boxes[k, 2]) - max(boxes[n, 0], query_boxes[k, 0]) + 1)
#             if iw > 0:
#                 ih = (min(boxes[n, 3], query_boxes[k, 3]) - max(boxes[n, 1], query_boxes[k, 1]) + 1 )
#                 if ih > 0:
#                     ua = float((boxes[n, 2] - boxes[n, 0] + 1) *(boxes[n, 3] - boxes[n, 1] + 1) + box_area - iw * ih)
#                     overlaps[n, k] = iw * ih / ua
#     return overlaps

def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret

def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep

# def nms(dets, thresh):
#     x1 = dets[:, 0]
#     y1 = dets[:, 1]
#     x2 = dets[:, 2]
#     y2 = dets[:, 3]
#     scores = dets[:, 4]
#
#     areas = (x2 - x1 + 1) * (y2 - y1 + 1)
#     order = scores.argsort()[::-1]
#
#     ndets = dets.shape[0]
#     suppressed = np.zeros((ndets), dtype=np.int)
#
#     keep = []
#     for _i in range(ndets):
#         i = order[_i]
#         if suppressed[i] == 1:
#             continue
#         keep.append(i)
#         ix1 = x1[i]
#         iy1 = y1[i]
#         ix2 = x2[i]
#         iy2 = y2[i]
#         iarea = areas[i]
#         for _j in range(_i + 1, ndets):
#             j = order[_j]
#             if suppressed[j] == 1:
#                 continue
#             xx1 = max(ix1, x1[j])
#             yy1 = max(iy1, y1[j])
#             xx2 = min(ix2, x2[j])
#             yy2 = min(iy2, y2[j])
#             w = max(0.0, xx2 - xx1 + 1)
#             h = max(0.0, yy2 - yy1 + 1)
#             inter = w * h
#             ovr = inter / (iarea + areas[j] - inter)
#             if ovr >= thresh:
#                 suppressed[j] = 1
#
#     return keep

def nms(dets, thresh, gpu=False):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    if gpu:
        return gpu_nms(dets, thresh, device_id=0)
    else:
        return cpu_nms(dets, thresh)

def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    BBOX_INSIDE_WEIGHTS=(1.0, 1.0, 1.0, 1.0)
    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights

def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    BBOX_NORMALIZE_TARGETS_PRECOMPUTED = False
    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    if BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    return np.hstack(
            (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

def _sample_rois(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    FG_THRESH = 0.5
    BG_THRESH_HI = 0.5
    BG_THRESH_LO = 0.1
    # overlaps: (rois x gt_boxes)
    overlaps = bbox_overlaps(np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),np.ascontiguousarray(gt_boxes[:, 1:], dtype=np.float))
    # overlaps = bbox_overlaps(np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, 0]
    # labels = gt_boxes[gt_assignment, 4]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < BG_THRESH_HI) &
                       (max_overlaps >= BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    rois = all_rois[keep_inds]

    bbox_target_data = _compute_targets(rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], 1:], labels)

    bbox_targets, bbox_inside_weights = _get_bbox_regression_labels(bbox_target_data, num_classes)

    return labels, rois, bbox_targets, bbox_inside_weights

class jg_iros_input_layer(caffe.Layer):
    def setup(self, bottom, top):
        """Setup the layer."""

        layer_params = yaml.load(self.param_str_)

        self.dummy_data = layer_params['dummy_data']
        self.data_way = layer_params['data_way']
        self.batch_size = layer_params['batch_size']


        self.images_folder_0 = layer_params['images_folder_0']
        self.image_file_extension_0 = layer_params['image_file_extension_0']

        self.segmentations_folder = layer_params['segmentations_folder']
        self.segmentation_file_extension = layer_params['segmentation_file_extension']

        self.bbox_folder = layer_params['bbox_folder']
        self.bbox_file_extension = layer_params['bbox_file_extension']

        self.set_file_path = layer_params['set_file_path']
        self.classes_file_path = layer_params['classes_file_path']

        # classes
        self.classes = []
        self.classes = get_classes(self.classes_file_path)
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))

        # images
        self.list_images = open(self.set_file_path, 'r').read().splitlines()
        self.nb_images = len(self.list_images)
        print "Number of image :",self.nb_images
        print "Separated in", self.num_classes, "classes :", self._class_to_ind

        self.max_rois_per_view = 50

        # shuffling at each epoch
        self.shuffle = layer_params['shuffle']
        self.permutation = []
        self.countEpoch = 0
        if self.shuffle:
            print "Shuffling activated."
            self.permutation = np.random.permutation(self.nb_images)
        else:
            self.permutation = range(self.nb_images)

        #tmp image in layer
        self.views = []
        self.image_0 = np.zeros((1,3,256,256), dtype=np.float32)
        self.segmentation = np.zeros((1,3,256,256), dtype=np.float32)
        self.bbox = np.zeros((1,5), dtype=np.uint16)
        # init batch data
        self.batch_data_0 = np.zeros((int(self.batch_size),3,256,256), dtype=np.float32)
        self.batch_segmentation = np.zeros((int(self.batch_size),1,256,256), dtype=np.uint8)
        self.batch_bbox = np.zeros((int(self.batch_size),self.max_rois_per_view,5), dtype=np.uint16) #[batch_size,Nrois,5](label, x1, y1, x2, y2)
        self.batch_iminfo = np.asarray([480,640,1],dtype=np.uint16) # h w scale
        # counters
        self.views_processed = 0
        self.images_processed = 0
        self.patches_processed = 0
        self.batches_processed = 0
        self.nb_epochs = 0  # setup a counter for iteration
        self.needToChangeImage = True

    def update(self,newParams):
        for key in newParams.keys():
            setattr(self, key, newParams[key])

        self.classes = get_classes(self.classes_file_path)
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))

        # images
        self.list_images = open(self.set_file_path, 'r').read().splitlines()
        self.nb_images = len(self.list_images)
        #print "Number of image :",self.nb_images
        #print "Separated in", self.num_classes, "classes :", self._class_to_ind

    def forward(self, bottom, top):
        """
        Top order :
            #data
            #data_
            #segmentation
        """
        # fill blob with data
        top[0].data[...] = self.batch_data_0
        top[1].data[...] = self.batch_segmentation
        top[2].data[...] = self.batch_bbox
        top[3].data[...] = self.batch_iminfo
        # Update number of batches processed
        self.batches_processed += 1

    def backward(self, top, propagate_down, bottom):
        pass

    def reshape(self, bottom, top):
        if self.data_way=='views':
            self.load_data_view()
        elif self.data_way=='full':
            self.load_data_full()
        # reshape net
        top[0].reshape(*self.batch_data_0.shape)
        top[1].reshape(*self.batch_segmentation.shape)
        top[2].reshape(*self.batch_bbox.shape)
        top[3].reshape(*self.batch_iminfo.shape)

    def load_data_view(self):
        """
        Load input image and preprocess for Caffe:
        - extract only a sub crop of the image
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """

        for i in range(int(self.batch_size)):
            if self.views_processed>=len(self.views):
                self.needToChangeImage = True
            if self.needToChangeImage:
                self.needToChangeImage = False
                self.views_processed=0
                self.images_processed+=1

                # if dataset entirely done :
                if self.images_processed>=self.nb_images:
                    self.images_processed = 0
                    self.countEpoch += 1
                    if self.shuffle:
                        self.permutation = np.random.permutation(self.nb_images)

                ## DATA LOADING
                img_0 = Image.open('{}{}.{}'.format(self.images_folder_0, self.list_images[self.permutation[self.images_processed]],self.image_file_extension_0))
                img_0 = np.array(img_0, dtype=np.float32)
                if len(img_0.shape)==3:
                    img_0 = img_0[:,:,::-1]
                    #img -= self.mean_bgr
                    img_0 = img_0.transpose((2,0,1))
                elif len(img_0.shape)==2:
                    img_0 = img_0[np.newaxis, ...]
                else:
                    print "Error : shape not accepted for img_0."
                # recompute views coordinates if the shape of image changed
                if self.image_0.shape!=img_0.shape or len(self.views)==0:
                    self.views = prepareViews(img_0.shape[2], img_0.shape[1], ratio_for_stride=0.8, min_size_min=256, min_size_max=256, min_size_step=64, verbose=False)

                ## SEGMENTATION LOADING
                seg = Image.open('{}{}.{}'.format(self.segmentations_folder, self.list_images[self.permutation[self.images_processed]],self.segmentation_file_extension))
                seg = np.array(seg, dtype=np.uint8)
                seg = seg[np.newaxis, ...]

                ## ROIS LOADING
                filename = '{}{}.{}'.format(self.bbox_folder, self.list_images[self.permutation[self.images_processed]],self.bbox_file_extension)
                tree = ET.parse(filename)
                objs = tree.findall('object')
                only_in_cls_obj = [obj for obj in objs if obj.find('name').text.lower().strip() in self.classes]
                objs = only_in_cls_obj
                num_objs = len(objs)
                img_rois = np.zeros((num_objs, 5), dtype=np.uint16)

                # Load object bounding boxes into a data frame.
                for ix, obj in enumerate(objs):
                    bbox = obj.find('bndbox')
                    # Make pixel indexes 0-based
                    x1 = float(bbox.find('xmin').text) - 1
                    y1 = float(bbox.find('ymin').text) - 1
                    x2 = float(bbox.find('xmax').text) - 1
                    y2 = float(bbox.find('ymax').text) - 1
                    label = self._class_to_ind[obj.find('name').text.lower().strip()]
                    img_rois[ix, :] = [label, x1, y1, x2, y2]


                assert img_rois.shape[0] <= self.max_rois_per_view
                # print("img_rois",img_rois)


                # tmp saving :
                self.bbox = img_rois
                self.image_0 = img_0
                self.segmentation = seg

            img_patch_0 = self.image_0[:,self.views[self.views_processed][1]:self.views[self.views_processed][3],self.views[self.views_processed][0]:self.views[self.views_processed][2]]
            seg_patch = self.segmentation[:,self.views[self.views_processed][1]:self.views[self.views_processed][3],self.views[self.views_processed][0]:self.views[self.views_processed][2]]
            copyOfBbox=np.zeros((50,5),dtype=np.uint16)
            # print("view:",self.views[self.views_processed][0],self.views[self.views_processed][1],self.views[self.views_processed][2],self.views[self.views_processed][3])
            for id_bb,bb in enumerate(self.bbox):
                #x1 = min(max(x1,xview1),xview2)
                copyOfBbox[id_bb,1]=min(max(bb[1],self.views[self.views_processed][0]),self.views[self.views_processed][2])
                #x2 = min(max(x2,xview1),xview2)
                copyOfBbox[id_bb,3]=min(max(bb[3],self.views[self.views_processed][0]),self.views[self.views_processed][2])
                #y1 = min(max(y1,yview1),yview2)
                copyOfBbox[id_bb,2]=min(max(bb[2],self.views[self.views_processed][1]),self.views[self.views_processed][3])
                #y2 = min(max(y2,yview1),yview2)
                copyOfBbox[id_bb,4]=min(max(bb[4],self.views[self.views_processed][1]),self.views[self.views_processed][3])

                copyOfBbox[id_bb,0]=bb[0]

            # print("copyOfBbox:",copyOfBbox)
            self.batch_data_0[i,...] = img_patch_0
            self.batch_segmentation[i,...] = seg_patch
            self.batch_bbox[i,...] = np.zeros((50,5),dtype=np.uint16)
            self.batch_bbox[i,...] = copyOfBbox
            self.batch_iminfo = [img_0.shape[2],img_0.shape[3],1]


            self.views_processed+=1
            self.patches_processed+=1

    def load_data_full(self):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        ## DATA LOADING
        img_0 = Image.open('{}{}.{}'.format(self.images_folder_0, self.list_images[self.permutation[self.images_processed]],self.image_file_extension_0))
        img_0 = np.array(img_0, dtype=np.float32)
        if len(img_0.shape)==3:
            img_0 = img_0[:,:,::-1]
            #img_0 -= self.mean_bgr
            img_0 = img_0.transpose((2,0,1))
        elif len(img_0.shape)==2:
            img_0 = img_0[np.newaxis, ...]
        else:
            print "Error : shape not accepted for img."

        ## SEGMENTATION LOADING
        seg = Image.open('{}{}.{}'.format(self.segmentations_folder, self.list_images[self.permutation[self.images_processed]],self.segmentation_file_extension))
        seg = np.array(seg, dtype=np.uint8)
        seg = seg[np.newaxis, ...]

        ## ROIS LOADING
        filename = '{}{}.{}'.format(self.bbox_folder, self.list_images[self.permutation[self.images_processed]],self.bbox_file_extension)
        tree = ET.parse(filename)
        objs = tree.findall('object')
        only_in_cls_obj = [obj for obj in objs if obj.find('name').text.lower().strip() in self.classes]
        objs = only_in_cls_obj
        num_objs = len(objs)
        img_rois = np.zeros((num_objs, 5), dtype=np.uint16)
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            label = self._class_to_ind[obj.find('name').text.lower().strip()]
            img_rois[ix, :] = [label, x1, y1, x2, y2]


        self.batch_data_0 = img_0[np.newaxis, ...]
        self.batch_segmentation = seg[np.newaxis, ...]
        self.batch_bbox = img_rois[np.newaxis, ...]
        self.batch_iminfo = np.asarray([img_0.shape[1],img_0.shape[2],1])

        self.images_processed+=1
        if self.images_processed>=self.nb_images:
            self.images_processed = 0
            self.countEpoch += 1
            if self.shuffle:
                self.permutation = np.random.permutation(self.nb_images)

class jg_bbox_rpn(caffe.Layer):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """
    def setup(self, bottom, top):
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)

        self._feat_stride = layer_params['feat_stride']
        anchor_scales = layer_params.get('scales', (8, 16, 32))
        self._anchors = generate_anchors(scales=np.array(anchor_scales))
        self._num_anchors = self._anchors.shape[0]

        self.RPN_PRE_NMS_TOP_N = 12000 # Number of top scoring boxes to keep before apply NMS to RPN proposals
        self.RPN_POST_NMS_TOP_N = 2000 # Number of top scoring boxes to keep after applying NMS to RPN proposals
        self.RPN_NMS_THRESH = 0.7 # NMS threshold used on RPN proposals
        self.RPN_MIN_SIZE = 16 # Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)



        self.img_width = int(layer_params.get('img_width', 256))
        self.img_height = int(layer_params.get('img_height', 256))
        self.img_scale = float(layer_params.get('img_scale', 1))
        self.stride = int(layer_params.get('stride', 16))
        # self.width = int(self.img_width/self.stride)
        # self.height = int(self.img_height/self.stride)
        self.height, self.width = bottom[1].data.shape[-2:]

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        top[0].reshape(1, 5)

    def forward(self, bottom, top):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)

        assert bottom[0].data.shape[0] == 1, 'Only single item batches are supported'

        im_info = bottom[2].data
        self.img_width = im_info[1]
        self.img_height = im_info[0]
        self.img_scale = im_info[2]


        pre_nms_topN  = self.RPN_PRE_NMS_TOP_N
        post_nms_topN = self.RPN_POST_NMS_TOP_N
        nms_thresh    = self.RPN_NMS_THRESH
        min_size      = self.RPN_MIN_SIZE

        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs, which we want
        scores = bottom[0].data[:, self._num_anchors:, :, :]
        bbox_deltas = bottom[1].data



        # 1. Generate proposals from bbox deltas and shifted anchors
        # height, width = scores.shape[-2:]



        # Enumerate all shifts
        shift_x = np.arange(0, self.width) * self._feat_stride
        shift_y = np.arange(0, self.height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),shift_x.ravel(), shift_y.ravel())).transpose()

        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]
        anchors = self._anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        anchors = anchors.reshape((K * A, 4))

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        #
        # bbox deltas will be (1, 4 * A, H, W) format
        # transpose to (1, H, W, 4 * A)
        # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
        # in slowest to fastest order
        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

        # Same story for the scores:
        #
        # scores are (1, A, H, W) format
        # transpose to (1, H, W, A)
        # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
        scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas)

        # 2. clip predicted boxes to image
        proposals = clip_boxes(proposals, [self.img_width,self.img_height])

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in self.img_scale)
        keep = _filter_boxes(proposals, min_size * self.img_scale)
        proposals = proposals[keep, :]
        scores = scores[keep]

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        order = scores.ravel().argsort()[::-1]
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals = proposals[order, :]
        scores = scores[order]

        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        keep = nms(np.hstack((proposals, scores)), nms_thresh)
        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
        proposals = proposals[keep, :]
        scores = scores[keep]

        # Output rois blob
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
        top_proposal_shape_before = top[0].data.shape
        #print "reshaping : top_proposal_shape_before =",top_proposal_shape_before
        top[0].reshape(*(blob.shape))
        top_proposal_shape_after = top[0].data.shape
        #print "reshaping : top_proposal_shape_after =",top_proposal_shape_after
        top[0].data[...] = blob

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

class jg_rpn_gt(caffe.Layer):
    """ Convert bbox to delta compared to specifics anchors"""

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        self.batch_size = 1 #layer_params['batch_size']
        self.batch_delta = np.zeros((1,4), dtype=np.float32)
        self.num_classes = layer_params['num_classes']

        self._allowed_border = layer_params.get('allowed_border', 0)
        anchor_scales = layer_params.get('scales', (8, 16, 32))
        self._anchors = generate_anchors(scales=np.array(anchor_scales))
        self._num_anchors = self._anchors.shape[0]
        self._feat_stride = layer_params['feat_stride']

        self.RPN_NEGATIVE_OVERLAP = 0.2
        self.RPN_POSITIVE_OVERLAP = 0.7
        self.RPN_FG_FRACTION = 0.5
        self.RPN_BATCHSIZE = 200
        self.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0) # Deprecated (outside weights)
        self.RPN_POSITIVE_WEIGHT = -1.0 # Set to -1.0 to use uniform example weighting

        self.img_width = int(layer_params.get('img_height', 256))
        self.img_height = int(layer_params.get('img_height', 256))
        self.img_scale = float(layer_params.get('img_scale', 1))
        self.stride = int(layer_params.get('stride', 16))
        self.height, self.width = bottom[1].data.shape[-2:]
        # self.width = int(self.img_width/self.stride)
        # self.height = int(self.img_height/self.stride)

        A = self._num_anchors
        # # labels
        top[0].reshape(1, 1, A * self.height, self.width)
        # bbox_targets
        top[1].reshape(1, A * 4, self.height, self.width)
        # bbox_inside_weights
        top[2].reshape(1, A * 4, self.height, self.width)
        # bbox_outside_weights
        top[3].reshape(1, A * 4, self.height, self.width)

    def forward(self, bottom, top):
        """ Top(s): 0#delta
        """
        #top[0].data[...] = self.batch_delta
        im_info = bottom[2].data
        self.img_width = im_info[1]
        self.img_height = im_info[0]
        self.img_scale = im_info[2]


        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors
        # measure GT overlap

        gt_boxes = bottom[0].data[0]

        # 1. Generate proposals from bbox deltas and shifted anchors
        shift_x = np.arange(0, self.width) * self._feat_stride
        shift_y = np.arange(0, self.height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),shift_x.ravel(), shift_y.ravel())).transpose()
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]
        all_anchors = (self._anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        all_anchors = all_anchors.reshape((K * A, 4))
        total_anchors = int(K * A)
        #
        #
        # print("Anchors before",all_anchors)
        # print("shifts",shifts)
        # print("Total anchors",total_anchors)
        # only keep anchors inside the image
        inds_inside = np.where(
            (all_anchors[:, 0] >= -self._allowed_border) &
            (all_anchors[:, 1] >= -self._allowed_border) &
            (all_anchors[:, 2] < self.img_width + self._allowed_border) &  # width
            (all_anchors[:, 3] < self.img_height + self._allowed_border)    # height
        )[0]

        # print("inds_inside",inds_inside)
        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = np.empty((len(inds_inside), ), dtype=np.float32)
        labels.fill(-1)
        # overlaps between the anchors and the gt boxes
        # overlaps (ex, gt)

        overlaps = bbox_overlaps(np.ascontiguousarray(anchors, dtype=np.float),np.ascontiguousarray(gt_boxes, dtype=np.float))
        # overlaps = bbox_overlaps(anchors,gt_boxes)

        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps,np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        # assign bg labels first so that positive labels can clobber them [priority to objectness instead of background]
        labels[max_overlaps < self.RPN_NEGATIVE_OVERLAP] = 0

        # fg label: for each gt, anchor with highest overlap
        labels[gt_argmax_overlaps] = 1

        # fg label: above threshold IOU
        labels[max_overlaps >= self.RPN_POSITIVE_OVERLAP] = 1

        # # assign bg labels last so that negative labels can clobber positives
        # labels[max_overlaps < self.RPN_NEGATIVE_OVERLAP] = 0

        # subsample positive labels if we have too many
        num_fg = int(self.RPN_FG_FRACTION * self.RPN_BATCHSIZE)
        fg_inds = np.where(labels == 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1

        # subsample negative labels if we have too many
        num_bg = self.RPN_BATCHSIZE - np.sum(labels == 1)
        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1

        bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
        bbox_targets = _compute_targets_anchor(anchors, gt_boxes[argmax_overlaps, :])

        bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        bbox_inside_weights[labels == 1, :] = np.array(self.RPN_BBOX_INSIDE_WEIGHTS)
        bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        if self.RPN_POSITIVE_WEIGHT < 0:
            # uniform weighting of examples (given non-uniform sampling)
            num_examples = np.sum(labels >= 0)
            positive_weights = np.ones((1, 4)) * 1.0 / num_examples
            negative_weights = np.ones((1, 4)) * 1.0 / num_examples
        else:
            assert ((self.RPN_POSITIVE_WEIGHT > 0) & (self.RPN_POSITIVE_WEIGHT < 1))
            positive_weights = (self.RPN_POSITIVE_WEIGHT / np.sum(labels == 1))
            negative_weights = ((1.0 - self.RPN_POSITIVE_WEIGHT) / np.sum(labels == 0))
        bbox_outside_weights[labels == 1, :] = positive_weights
        bbox_outside_weights[labels == 0, :] = negative_weights

        # map up to original set of anchors
        labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

        # labels
        labels = labels.reshape((1, self.height, self.width, A)).transpose(0, 3, 1, 2)
        labels = labels.reshape((1, 1, A * self.height, self.width))
        top[0].reshape(*labels.shape)
        top[0].data[...] = labels

        # bbox_targets
        bbox_targets = bbox_targets.reshape((1, self.height, self.width, A * 4)).transpose(0, 3, 1, 2)
        top[1].reshape(*bbox_targets.shape)
        top[1].data[...] = bbox_targets

        # bbox_inside_weights
        bbox_inside_weights = bbox_inside_weights.reshape((1, self.height, self.width, A * 4)).transpose(0, 3, 1, 2)
        assert bbox_inside_weights.shape[2] == self.height
        assert bbox_inside_weights.shape[3] == self.width
        top[2].reshape(*bbox_inside_weights.shape)
        top[2].data[...] = bbox_inside_weights

        # bbox_outside_weights
        bbox_outside_weights = bbox_outside_weights.reshape((1, self.height, self.width, A * 4)).transpose(0, 3, 1, 2)
        assert bbox_outside_weights.shape[2] == self.height
        assert bbox_outside_weights.shape[3] == self.width
        top[3].reshape(*bbox_outside_weights.shape)
        top[3].data[...] = bbox_outside_weights

    def backward(self, top, propagate_down, bottom):
        """ Bottoms: """
        pass

    def reshape(self, bottom, top):
        """A delta is made of 4 values, the batch size correspond to its input number of rois or number of anchor"""
        pass

class jg_prediction_gt(caffe.Layer):
    """ Convert bbox to delta compared to specifics anchors"""

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        self.batch_size = 1 #layer_params['batch_size']
        self.batch_delta = np.zeros((1,4), dtype=np.float32)
        self.num_classes = layer_params['num_classes']

        self._allowed_border = layer_params.get('allowed_border', 0)
        anchor_scales = layer_params.get('scales', (8, 16, 32))
        self._anchors = generate_anchors(scales=np.array(anchor_scales))
        self._num_anchors = self._anchors.shape[0]
        self._feat_stride = layer_params['feat_stride']

        self.BATCH_SIZE = 256
        self.FG_FRACTION = 0.25

        self.img_width = int(layer_params.get('img_width', 256))
        self.img_height = int(layer_params.get('img_height', 256))
        self.img_scale = float(layer_params.get('img_scale', 1))
        self.stride = int(layer_params.get('stride', 16))
        self.width = int(self.img_width/self.stride)
        self.height = int(self.img_height/self.stride)

        self._num_classes = int(layer_params.get('_num_classes', 37))

        # sampled rois (0, x1, y1, x2, y2)
        top[0].reshape(1, 5)
        # labels
        top[1].reshape(1, 1)
        # bbox_targets
        top[2].reshape(1, self._num_classes * 4)
        # bbox_inside_weights
        top[3].reshape(1, self._num_classes * 4)
        # bbox_outside_weights
        top[4].reshape(1, self._num_classes * 4)

    def forward(self, bottom, top):

        im_info = bottom[2].data
        self.img_width = im_info[1]
        self.img_height = im_info[0]
        self.img_scale = im_info[2]


        # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        all_rois = bottom[0].data
        # GT boxes (x1, y1, x2, y2, label)
        # TODO(rbg): it's annoying that sometimes I have extra info before
        # and other times after box coordinates -- normalize to one format
        gt_boxes = bottom[1].data[0]

        # Include ground-truth boxes in the set of candidate rois
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)

        all_rois = np.vstack((all_rois, np.hstack((zeros, gt_boxes[:, :-1]))))

        # Sanity check: single batch only
        assert np.all(all_rois[:, 0] == 0), 'Only single item batches are supported'

        num_images = 1
        rois_per_image = self.BATCH_SIZE / num_images
        fg_rois_per_image = int(np.round(self.FG_FRACTION * rois_per_image))

        # Sample rois with classification labels and bounding box regression targets
        # print("all_rois",all_rois)
        # print("gt_boxes",gt_boxes)
        # print("fg_rois_per_image",fg_rois_per_image)
        # print("rois_per_image",rois_per_image)
        # print("self._num_classes",self._num_classes)
        # exit()
        labels, rois, bbox_targets, bbox_inside_weights = _sample_rois(all_rois, gt_boxes, fg_rois_per_image,rois_per_image, self._num_classes)

        # sampled rois
        top[0].reshape(*rois.shape)
        top[0].data[...] = rois

        # classification labels
        top[1].reshape(*labels.shape)
        top[1].data[...] = labels

        # bbox_targets
        top[2].reshape(*bbox_targets.shape)
        top[2].data[...] = bbox_targets

        # bbox_inside_weights
        top[3].reshape(*bbox_inside_weights.shape)
        top[3].data[...] = bbox_inside_weights

        # bbox_outside_weights
        top[4].reshape(*bbox_inside_weights.shape)
        top[4].data[...] = np.array(bbox_inside_weights > 0).astype(np.float32)

    def backward(self, top, propagate_down, bottom):
        """ Bottoms: """
        pass

    def reshape(self, bottom, top):
        """A delta is made of 4 values, the batch size correspond to its input number of rois or number of anchor"""
        pass

class jg_map_rpn(caffe.Layer):
    """ Compute score for bbox prediction and class attribution"""

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        self.batch_size = 1 #layer_params['batch_size']
        # anchor_scales = layer_params.get('scales', (8, 16, 32))
        self.num_classes = 37 # layer_params['num_classes']
        self.num_images = 1 #5050#len(imdb.image_index)
        self.tresh_NMS = 0.3
        self.ovthresh=0.5
        top[0].reshape(1)

    def forward(self, bottom, top):
        ## BOTTOMS :
        # 0 : rois
        # 1 : cls_prob
        # 2 : bbox_deltas
        # 3 : bbox_gt
        # 4 : im_info

        # all detections are collected into:
        #    all_boxes[cls][image] = N x 5 array of detections in
        #    (x1, y1, x2, y2, score)
        all_boxes = [[[] for _ in xrange(self.batch_size)] for _ in xrange(self.num_classes)]


        # rois = net.blobs['rois'].data.copy()
        rois = bottom[0].data.copy()
        # unscale back to raw image space
        boxes = rois[:, 1:5] # / im_scales[0]

        # scores = blobs_out['cls_prob']
        scores = bottom[1].data.copy()


        # box_deltas = blobs_out['bbox_pred']
        box_deltas = bottom[2].data.copy()

        im_info = bottom[4].data
        pred_boxes = bbox_transform_inv(boxes, box_deltas)


        # print boxes.shape
        # print boxes
        # print box_deltas.shape
        # print box_deltas
        # print pred_boxes.shape
        # print pred_boxes
        # print "im_info"
        # print im_info.shape
        # print im_info
        # print im_info[:2]
        # exit()
        pred_boxes = clip_boxes(pred_boxes, im_info[:2])

        gt_boxes = bottom[3].data.copy()

        # skip j = 0, because it's the background class
        # for j in xrange(1, imdb.num_classes):
        ## multibatch : change i to make a loop on patch
        # i = 0
        thresh=0.05
        for i in xrange(self.batch_size):
            # for each prob vector if the class prob is over tresh then it is kept (even for a same proposed roi)
            for j in xrange(self.num_classes):
                inds = np.where(scores[:, j] > thresh)[0] # for all detections...
                cls_scores = scores[inds, j]
                cls_boxes = pred_boxes[inds, j*4:(j+1)*4]
                cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) #.astype(np.float32, copy=False)
                # print "cls_boxes"
                # print cls_boxes
                # print cls_boxes.shape
                # print "cls_dets"
                # print cls_dets
                # print cls_dets.shape
                # print "cls_scores"
                # print cls_scores
                # print cls_scores.shape

                keep = nms(cls_dets, self.tresh_NMS)
                cls_dets = cls_dets[keep, :]
                all_boxes[j][i] = cls_dets

            # Limit to max_per_image detections *over all classes*
            max_per_image = 2000
            if max_per_image > 0:
                image_scores = np.hstack([all_boxes[j][i][:, -1] for j in xrange(1, self.num_classes)])
                if len(image_scores) > max_per_image:
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    for j in xrange(1, self.num_classes):
                        keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                        all_boxes[j][i] = all_boxes[j][i][keep, :]

        ##evaluate_detections(all_boxes, output_dir)

        # available : all_boxes with 2000 detections max and only if score is above tresh
        # format : all_boxes[0] = [box, score]

        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True
        # for i in xrange(self.num_classes):
        for i in xrange(self.batch_size):
            # for each prob vector if the class prob is over tresh then it is kept (even for a same proposed roi)
            for j in xrange(self.num_classes):
                # if cls == '__background__':
                #     continue
                ###################################################################################
                # BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
                cls_dets = all_boxes[j][i]
                if len(cls_dets)==0:
                    continue
                BB = cls_dets[:,:4] # 0 BASED         but we can do : +1 # here we go in 1based pixel for pascal evaluation

                # confidence = np.array([float(x[1]) for x in splitlines])
                confidence = cls_dets[:,4]
                # sort by confidence
                sorted_ind = np.argsort(-confidence)
                sorted_scores = np.sort(-confidence)
                BB = BB[sorted_ind, :]
                # image_ids = [image_ids[x] for x in sorted_ind] #only usefull

                # go down dets and mark TPs and FPs
                # nd = len(image_ids)
                nd = len(cls_dets)

                tp = np.zeros(nd)
                fp = np.zeros(nd)

                BBGT = gt_boxes[i,1:] #
                already_detected = [False] * len(BBGT)
                for d in range(nd):
                    # R = class_recs[image_ids[d]]
                    # BBGT = R['bbox'].astype(float)
                    # bb = BB[d, :].astype(float)
                    bb = BB[d, :]

                    ovmax = -np.inf

                    if BBGT.size > 0:
                        # compute overlaps
                        # intersection
                        ixmin = np.maximum(BBGT[:, 0], bb[0])
                        iymin = np.maximum(BBGT[:, 1], bb[1])
                        ixmax = np.minimum(BBGT[:, 2], bb[2])
                        iymax = np.minimum(BBGT[:, 3], bb[3])
                        iw = np.maximum(ixmax - ixmin + 1., 0.)
                        ih = np.maximum(iymax - iymin + 1., 0.)
                        inters = iw * ih

                        # union
                        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                               (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                               (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                        overlaps = inters / uni
                        ovmax = np.max(overlaps)
                        jmax = np.argmax(overlaps)

                    if ovmax > self.ovthresh:
                        if not already_detected[jmax]:
                            tp[d] = 1.
                            already_detected[jmax] = True
                        else:
                            fp[d] = 1.
                    else:
                        fp[d] = 1.

                # compute precision recall
                fp = np.cumsum(fp)
                tp = np.cumsum(tp)
                rec = tp / float(len(BBGT))
                # avoid divide by zero in case the first detection matches a difficult
                # ground truth
                prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
                ap = voc_ap(rec, prec, use_07_metric)

                ##############################################################################
                aps += [ap]



        # sampled rois
        mAP = np.mean(aps)
        # mAP = 0.55*np.ones((1))
        top[0].reshape(*mAP.shape)
        top[0].data[...] = mAP

    def backward(self, top, propagate_down, bottom):
        """ Bottoms: """
        pass

    def reshape(self, bottom, top):
        """aaaaa"""
        pass














def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
