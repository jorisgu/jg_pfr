import os
import caffe
import numpy as np
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

def prepareViews(img_w, img_h, ratio_for_stride=0.5, min_size_min=128, min_size_max=128, min_size_step=16, verbose=False):
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
        self.image_0 = np.zeros((1,3,128,128), dtype=np.float32)
        self.segmentation = np.zeros((1,3,128,128), dtype=np.float32)

        # init batch data
        self.batch_data_0 = np.zeros((int(self.batch_size),3,128,128), dtype=np.float32)
        self.batch_segmentation = np.zeros((int(self.batch_size),1,128,128), dtype=np.uint8)
        self.batch_bbox = np.zeros((47,6), dtype=np.float32) #[Nrois,6](img_id,class_id, x, y, w, h) do a layer of conversion to bbox reg target [bbox_target = convertFrom(vt_roi,stride)]
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
                    print "Epoch really done :",self.countEpoch
                    print "Average views by image :", float(self.patches_processed)/(self.countEpoch*self.nb_images)
                    if self.shuffle:
                        self.permutation = np.random.permutation(self.nb_images)

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

                seg = Image.open('{}{}.{}'.format(self.segmentations_folder, self.list_images[self.permutation[self.images_processed]],self.segmentation_file_extension))
                seg = np.array(seg, dtype=np.uint8)
                seg = seg[np.newaxis, ...]

                # recompute views coordinates if the shape of image
                if self.image_0.shape!=img_0.shape or len(self.views)==0:
                    self.views = prepareViews(img_0.shape[2], img_0.shape[1], ratio_for_stride=0.8, min_size_min=128, min_size_max=128, min_size_step=64, verbose=False)

                self.image_0 = img_0
                self.segmentation = seg

            img_patch_0 = self.image_0[:,self.views[self.views_processed][1]:self.views[self.views_processed][3],self.views[self.views_processed][0]:self.views[self.views_processed][2]]
            seg_patch = self.segmentation[:,self.views[self.views_processed][1]:self.views[self.views_processed][3],self.views[self.views_processed][0]:self.views[self.views_processed][2]]

            self.batch_data_0[i,...] = img_patch_0
            self.batch_segmentation[i,...] = seg_patch

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
        img_0 = Image.open('{}{}.{}'.format(self.images_folder_0, self.list_images[self.permutation[self.images_processed%self.nb_images]],self.image_file_extension_0))
        img_0 = np.array(img_0, dtype=np.float32)
        if len(img_0.shape)==3:
            img_0 = img_0[:,:,::-1]
            #img_0 -= self.mean_bgr
            img_0 = img_0.transpose((2,0,1))
        elif len(img_0.shape)==2:
            img_0 = img_0[np.newaxis, ...]
        else:
            print "Error : shape not accepted for img."


        seg = Image.open('{}{}.{}'.format(self.segmentations_folder, self.list_images[self.permutation[self.images_processed%self.nb_images]],self.segmentation_file_extension))
        seg = np.array(seg, dtype=np.uint8)
        seg = seg[np.newaxis, ...]

        self.batch_data_0 = img_0[np.newaxis, ...]
        self.batch_segmentation = seg[np.newaxis, ...]

        self.images_processed+=1
        if self.images_processed>=self.nb_images:
            self.countEpoch += 1
            if self.shuffle:
                self.permutation = np.random.permutation(self.nb_images)


class jg_segmentation2objectness(caffe.Layer):
    """ Convert segmentation blob to objectness map for specifics anchors"""
    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        self.batch_size = 1 #layer_params['batch_size']
        self.batch_label = np.zeros((1,1,72,8), dtype=np.float32) #[Nrois,6](img_id,class_id, x, y, w, h) do a layer of conversion to bbox reg target [bbox_target = convertFrom(vt_roi,stride)]

    def forward(self, bottom, top):
        """ Top(s): 0#objectness label"""
        #top[0].data[...] = self.batch_label

    def backward(self, top, propagate_down, bottom):
        """ Bottoms: """
        pass

    def reshape(self, bottom, top):
        top[0].reshape(*self.batch_label.shape)

class jg_segmentation2label(caffe.Layer):
    """ Convert segmentation blob to classe labels map for specifics bbox wrt to segmentation_gt"""
    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        self.batch_size = 1 #layer_params['batch_size']
        self.batch_label = np.zeros((200,1), dtype=np.float32)

    def forward(self, bottom, top):
        """ Top(s): 0#class label
        """
        #top[0].data[...] = self.batch_label

    def backward(self, top, propagate_down, bottom):
        """ Bottoms: """
        pass

    def reshape(self, bottom, top):
        top[0].reshape(*self.batch_label.shape)

class jg_bbox2delta(caffe.Layer):
    """ Convert bbox to delta compared to specifics anchors"""
    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        self.batch_size = 1 #layer_params['batch_size']
        self.batch_delta = np.zeros((1,4), dtype=np.float32)
        self.delta_mode = layer_params['delta_mode']
        self.num_classes = layer_params['num_classes']

    def forward(self, bottom, top):
        """ Top(s): 0#delta
        """
        #top[0].data[...] = self.batch_delta

    def backward(self, top, propagate_down, bottom):
        """ Bottoms: """
        pass

    def reshape(self, bottom, top):
        """A delta is made of 4 values, the batch size correspond to its input number of rois or number of anchor"""
        if self.delta_mode=='objectness':
            height = 8 #TODO compute it from bottoms
            width = 8
            num_anchors = 9 #self.num_anchors

            # bbox_targets
            top[0].reshape(1, num_anchors * 4, height, width)
            # bbox_inside_weights
            top[1].reshape(1, num_anchors * 4, height, width)
            # bbox_outside_weights
            top[2].reshape(1, num_anchors * 4, height, width)
        else:
            # top[0].reshape(bottom[0].shape[0],4)

            # bbox_targets
            top[0].reshape(bottom[0].shape[0], self.num_classes * 4)
            # bbox_inside_weights
            top[1].reshape(bottom[0].shape[0], self.num_classes * 4)
            # bbox_outside_weights
            top[2].reshape(bottom[0].shape[0], self.num_classes * 4)

class jg_bbox_selection(caffe.Layer):
    """Make a fusion of bbox from rpn and gt"""
    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        self.batch_size = 1 #layer_params['batch_size']
        self.batch_bbox = np.zeros((200,6), dtype=np.float32) #[Nrois,6](img_id,class_id, x, y, w, h) do a layer of conversion to bbox reg target [bbox_target = convertFrom(vt_roi,stride)]

    def forward(self, bottom, top):
        """ Top(s): 0#bbox
        """
        #top[0].data[...] = self.batch_bbox

    def backward(self, top, propagate_down, bottom):
        """ Bottoms: """
        pass

    def reshape(self, bottom, top):
        top[0].reshape(*self.batch_bbox.shape)
