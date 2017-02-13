import os
import caffe
import numpy as np
import yaml
from PIL import Image
import cPickle
import xml.etree.ElementTree as ET
import sklearn.metrics #for confusion matrix
import math #for view generation
#import time
#import scipy.io
#from multiprocessing import Process, Queue

class dataset:
    """
    - keep list of images
    - keep list of crops for each images
    - reload data if already computed
    - give a batch on demand (with a unique id making it reproductible)
    """

    def __init__(self):
        pass

    def __call__(self):
        pass


def prepareViews(img_w, img_h, ratio_for_stride=0.5, min_size_min=64, min_size_max=128, min_size_step=16, verbose=False):
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
                views.append([i_w, i_h, i_w + min_size-1, i_h + min_size-1]) #x1 y1 x2 y2
                i_h += stride
            i_w += stride
    if verbose:
        print('Number of patchs in this image : ' + str(len(views)))
    return views

class jg_dummy_layer(caffe.Layer):

    def setup(self, bottom, top):
        pass

    def forward(self, bottom, top):
        pass

    def backward(self, top, propagate_down, bottom):
        pass

    def reshape(self, bottom, top):
        pass

class jg_input_voc_layer(caffe.Layer):
    """works with only on image for now by batch"""
    #improvement here : https://github.com/BVLC/caffe/blob/master/examples/pycaffe/layers/pascal_multilabel_datalayers.py

    def get_classes(self):
        with open(self.classes_file_path, 'r') as fp:
            for line in fp:
                if line[0]=='#' or line[0]=='\n':
                    pass
                else:
                    self.classes.append(line[:-1])
        #print "Classes :", self.classes

    def setup(self, bottom, top):
        """Setup the layer."""


        # tops: check configuration
        if len(top) != 3:
            raise Exception("Need to define {} tops for all outputs.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        layer_params = yaml.load(self.param_str_)

        self.iter_counter = 0  # setup a counter for iteration

        self.unique_key = layer_params['unique_key']
        self.batch_size = layer_params['batch_size']
        self.width = layer_params['width']
        self.height = layer_params['height']

        self.images_folder = layer_params['images_folder']
        self.image_file_extension = layer_params['image_file_extension']

        self.annotations_folder = layer_params['annotations_folder']
        self.annotation_file_extension = layer_params['annotation_file_extension']

        self.segmentations_folder = layer_params['segmentations_folder']
        self.segmentation_file_extension = layer_params['segmentation_file_extension']

        self.set_file_path = layer_params['set_file_path']
        self.classes_file_path = layer_params['classes_file_path']
        self.cache_folder = layer_params['cache_folder']
        if not os.path.isdir(self.cache_folder):
            os.mkdir(self.cache_folder)

        # shuffling at each epoch
        self.shuffle = layer_params['shuffle']


        self.recorded_loss = [] #todo in backward prop keep values of loss to get hard examples

        self.max_width = layer_params['max_width']
        self.max_height = layer_params['max_height']

        self.classes = []
        self.get_classes()
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))


        self.batch_data = np.zeros((1,3,320,240), dtype=np.float32)
        self.batch_segmentation = np.zeros((1,3,320,240), dtype=np.uint8)
        self.batch_rois = np.zeros((47,6), dtype=np.float32) #[Nrois,6](img_id,class_id, x, y, w, h) do a layer of conversion to bbox reg target [bbox_target = convertFrom(vt_roi,stride)]
        self.batch_ready = False

        self.list_images = open(self.set_file_path, 'r').read().splitlines()
        self.nb_images = len(self.list_images)
        print "Number of image :",self.nb_images
        print "Separated in", self.num_classes, "classes :", self._class_to_ind
        print "Training with a batch_size of :",self.batch_size
        im = Image.open('{}{}.{}'.format(self.images_folder, self.list_images[0],self.image_file_extension))
        first_im = np.array(im, dtype=np.float32)
        print "First im dim = ", first_im.shape
        #self.define_new_batch()

    def forward(self, bottom, top):
        """
        Top order :
            #data
            #segmentation
            #rois
        """

        # fill blob with data
        top[0].data[...] = self.batch_data
        top[1].data[...] = self.batch_segmentation
        top[2].data[...] = self.batch_rois

        # Update number of forward pass done
        self.iter_counter += 1
        self.iter_counter=self.iter_counter%self.nb_images

    def backward(self, top, propagate_down, bottom):
        #bottom[0].diff[...] = 10 * top[0].diff
        #self.loss = top[0].diff
        pass

    def reshape(self, bottom, top):
        #while !self.batch_ready:
        #    pass

        # make batch
        self.batch_data = self.load_img_data(self.iter_counter)[np.newaxis, ...]
        self.batch_segmentation = self.load_img_segmentation(self.iter_counter)[np.newaxis, ...]
        self.batch_rois = self.load_img_rois(self.iter_counter)[np.newaxis, ...]

        # reshape net
        top[0].reshape(*self.batch_data.shape)
        top[1].reshape(*self.batch_segmentation.shape)
        top[2].reshape(*self.batch_rois.shape)
        #top[0].reshape(1, *self.batch_data)
        #top[1].reshape(1, *self.batch_segmentation)
        #top[2].reshape(1, *self.batch_rois)

    def load_img_data(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open('{}{}.{}'.format(self.images_folder, self.list_images[idx],self.image_file_extension))
        in_ = np.array(im, dtype=np.float32)
        if len(in_.shape)==3:
            in_ = in_[:,:,::-1]
        #in_ -= self.mean_bgr
            in_ = in_.transpose((2,0,1))
        elif len(in_.shape)==2:
            in_ = in_[np.newaxis, ...]
        else:
            print "Error : shape not accepted."
        return in_

    def load_img_segmentation(self, idx,delta=False):
        """
        Load segmentation image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        im = Image.open('{}{}.{}'.format(self.segmentations_folder, self.list_images[idx],self.segmentation_file_extension))
        segmentation = np.array(im, dtype=np.uint8)
        if delta:
            segmentation = segmentation - 1
        segmentation = segmentation[np.newaxis, ...]
        return segmentation

    def load_img_rois(self, idx):
        """
        Load bounding boxes info from pascal voc template data
        """
        filename = '{}{}.{}'.format(self.annotations_folder, self.list_images[idx],self.annotation_file_extension)
        tree = ET.parse(filename)
        objs = tree.findall('object')
        #if not self.config['use_diff']:
        ## Exclude the samples labeled as difficult
        #    non_diff_objs = [
        #        obj for obj in objs if int(obj.find('difficult').text) == 0]
        #    objs = non_diff_objs

        only_in_cls_obj = [obj for obj in objs if obj.find('name').text.lower().strip() in self.classes]
        objs = only_in_cls_obj

        num_objs = len(objs)

        img_rois = np.zeros((num_objs, 5), dtype=np.uint32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            img_rois[ix, :] = [cls, x1, y1, x2, y2]

        img_rois = img_rois[np.newaxis, ...]
        return img_rois

    def load_batch_data(self):
        """
        Load data in a batch
        """

    def load_batch_segmentation(self):
        """
        Load segmentations in a batch
        """

    def load_batch_rois(self):
        """
        Load rois in a batch
        """

    def define_new_batch(self):
        for batch in [self.list_images[x:x+self.batch_size] for x in xrange(0, len(self.list_images), self.batch_size)]:
            new_shape = (len(batch),) + tuple([3,480,640])
            print new_shape
            #if net.blobs['data'].data.shape != new_shape:
            #    net.blobs['data'].reshape(*new_shape)
            #for index, image in enumerate(chunk):
            #    image_data = transformer.preprocess('data', image)
            #    net.blobs['data'].data[index] = image_data
            #output = net.forward()[net.outputs[-1]]

    def check_params(params):
        required = ['batch_size']
        for r in required:
            assert r in params.keys(), 'Params must include {}'.format(r)

class jg_input_test_layer(caffe.Layer):
    """works with only on image for now by batch"""

    def get_classes(self):
        self.classes = []
        with open(self.classes_file_path, 'r') as fp:
            for line in fp:
                if line[0]=='#' or line[0]=='\n':
                    pass
                else:
                    self.classes.append(line[:-1])

    def setup(self, bottom, top):
        """Setup the layer."""


        # tops: check configuration
        if len(top) != 1:
            raise Exception("Need to define {} tops for all outputs.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        layer_params = yaml.load(self.param_str_)

        self.iter_counter = 0  # setup a counter for iteration

        self.images_folder = layer_params['images_folder']
        self.image_file_extension = layer_params['image_file_extension']

        self.segmentations_folder = layer_params['segmentations_folder']
        self.segmentation_file_extension = layer_params['segmentation_file_extension']

        self.set_file_path = layer_params['set_file_path']
        self.classes_file_path = layer_params['classes_file_path']

        # classes
        self.classes = []
        self.get_classes()
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))

        # images
        self.list_images = open(self.set_file_path, 'r').read().splitlines()
        self.nb_images = len(self.list_images)
        print "Number of image :",self.nb_images
        print "Separated in", self.num_classes, "classes :", self._class_to_ind

        self.batch_data = np.zeros((1,3,400,400), dtype=np.float32)
        self.batch_segmentation = np.zeros((1,3,400,400), dtype=np.uint8)

        # shuffling at each epoch
        self.shuffle = layer_params['shuffle']
        self.permutation = []
        self.countEpoch = 0
        if self.shuffle:
            print "Shuffling activated."
            self.permutation = np.random.permutation(self.nb_images)
        else:
            self.permutation = range(self.nb_images)

    def update(self,newParams):
        for key in newParams.keys():
            setattr(self, key, newParams[key])

        self.get_classes()
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
            #segmentation
        """

        # fill blob with data
        top[0].data[...] = self.batch_data
        #top[1].data[...] = self.batch_segmentation

        # Update number of forward pass done
        self.iter_counter += 1
        if self.iter_counter>=self.nb_images:
            self.countEpoch += 1
            print "New epoch [total epoch :", self.countEpoch,"]."
            self.iter_counter=0
            if self.shuffle:
                print "Reshuffling !"
                self.permutation = np.random.permutation(self.nb_images)

    def backward(self, top, propagate_down, bottom):
        pass

    def reshape(self, bottom, top):
        # make batch
        self.batch_data = self.load_img_data(self.iter_counter)[np.newaxis, ...]
        #self.batch_segmentation = self.load_img_segmentation(self.iter_counter)[np.newaxis, ...]

        # reshape net
        top[0].reshape(*self.batch_data.shape)
        #top[1].reshape(*self.batch_segmentation.shape)

    def load_img_data(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open('{}{}.{}'.format(self.images_folder, self.list_images[self.permutation[idx]],self.image_file_extension))
        in_ = np.array(im, dtype=np.float32)
        if len(in_.shape)==3:
            in_ = in_[:,:,::-1]
        #in_ -= self.mean_bgr
            in_ = in_.transpose((2,0,1))
        elif len(in_.shape)==2:
            in_ = in_[np.newaxis, ...]
        else:
            print "Error : shape not accepted."
        return in_

    def load_img_segmentation(self, idx,delta=False):
        """
        Load segmentation image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        im = Image.open('{}{}.{}'.format(self.segmentations_folder, self.list_images[self.permutation[idx]],self.segmentation_file_extension))
        segmentation = np.array(im, dtype=np.uint8)
        if delta:
            segmentation = segmentation - 1
        segmentation = segmentation[np.newaxis, ...]
        return segmentation

class jg_input_fuse_segmentation_layer(caffe.Layer):
    """works with only on image for now by batch"""

    def get_classes(self):
        self.classes = []
        with open(self.classes_file_path, 'r') as fp:
            for line in fp:
                if line[0]=='#' or line[0]=='\n':
                    pass
                else:
                    self.classes.append(line[:-1])

    def setup(self, bottom, top):
        """Setup the layer."""


        # tops: check configuration
        if len(top) != 3:
            raise Exception("Need to define {} tops for all outputs.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        layer_params = yaml.load(self.param_str_)

        self.iter_counter = 0  # setup a counter for iteration

        self.dummy_data = layer_params['dummy_data']
        self.data_way = layer_params['data_way']


        self.images_folder = layer_params['images_folder']
        self.image_file_extension = layer_params['image_file_extension']
        self.images_folder_ = layer_params['images_folder_']
        self.image_file_extension_ = layer_params['image_file_extension_']

        self.segmentations_folder = layer_params['segmentations_folder']
        self.segmentation_file_extension = layer_params['segmentation_file_extension']

        self.set_file_path = layer_params['set_file_path']
        self.classes_file_path = layer_params['classes_file_path']

        # classes
        self.classes = []
        self.get_classes()
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))

        # images
        self.list_images = open(self.set_file_path, 'r').read().splitlines()
        self.nb_images = len(self.list_images)
        print "Number of image :",self.nb_images
        print "Separated in", self.num_classes, "classes :", self._class_to_ind

        self.batch_data = np.zeros((1,3,128,128), dtype=np.float32)
        self.batch_data_ = np.zeros((1,3,128,128), dtype=np.float32)
        self.batch_segmentation = np.zeros((1,3,128,128), dtype=np.uint8)

        # shuffling at each epoch
        self.shuffle = layer_params['shuffle']
        self.permutation = []
        self.countEpoch = 0
        if self.shuffle:
            print "Shuffling activated."
            self.permutation = np.random.permutation(self.nb_images)
        else:
            self.permutation = range(self.nb_images)

        #views
        self.views = []
        self.images_processed = 0
        self.views_processed = 0
        self.image=np.zeros((1,3,32,32), dtype=np.float32)
        self.image_=np.zeros((1,3,32,32), dtype=np.float32)
        self.segmentation=np.zeros((1,3,32,32), dtype=np.float32)

    def update(self,newParams):
        for key in newParams.keys():
            setattr(self, key, newParams[key])

        self.get_classes()
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
        top[0].data[...] = self.batch_data
        top[1].data[...] = self.batch_data_
        top[2].data[...] = self.batch_segmentation

        # Update number of forward pass done
        self.iter_counter += 1
        if self.iter_counter>=self.nb_images:
            self.countEpoch += 1
            print "New epoch [total epoch :", self.countEpoch,"]."
            self.iter_counter=0
            if self.shuffle:
                print "Reshuffling !"
                self.permutation = np.random.permutation(self.nb_images)

    def backward(self, top, propagate_down, bottom):
        pass

    def reshape(self, bottom, top):
        if self.data_way=='views':
            self.load_data_view()
        elif self.data_way=='full':
            self.load_data_full(self.iter_counter)
        # reshape net
        top[0].reshape(*self.batch_data.shape)
        top[1].reshape(*self.batch_data_.shape)
        top[2].reshape(*self.batch_segmentation.shape)

    def load_data_view(self):
        """
        Load input image and preprocess for Caffe:
        - extract only a sub crop of the image
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        needToChangeImage = False
        if self.views_processed>=len(self.views):
            needToChangeImage = True



        if needToChangeImage:
            self.images_processed+=1
            self.views_processed=0

            img = Image.open('{}{}.{}'.format(self.images_folder, self.list_images[self.permutation[self.images_processed%self.nb_images]],self.image_file_extension))
            img = np.array(img, dtype=np.float32)
            if len(img.shape)==3:
                img = img[:,:,::-1]
                #img -= self.mean_bgr
                img = img.transpose((2,0,1))
            elif len(img.shape)==2:
                img = img[np.newaxis, ...]
            else:
                print "Error : shape not accepted for img."

            img_ = Image.open('{}{}.{}'.format(self.images_folder_, self.list_images[self.permutation[self.images_processed%self.nb_images]],self.image_file_extension_))
            img_ = np.array(img_, dtype=np.float32)
            if len(img_.shape)==3:
                img_ = img_[:,:,::-1]
                #img -= self.mean_bgr
                img_ = img_.transpose((2,0,1))
            elif len(img_.shape)==2:
                img_ = img_[np.newaxis, ...]
            else:
                print "Error : shape not accepted for img_."

            seg = Image.open('{}{}.{}'.format(self.segmentations_folder, self.list_images[self.permutation[self.images_processed%self.nb_images]],self.segmentation_file_extension))
            seg = np.array(seg, dtype=np.uint8)
            seg = seg[np.newaxis, ...]


            if self.image.shape!=img.shape:
                self.views = prepareViews(img.shape[2], img.shape[1], ratio_for_stride=0.5, min_size_min=128, min_size_max=128, min_size_step=64, verbose=False)

            self.image = img
            self.image_ = img_
            self.segmentation = seg



        img_view = self.image[:,self.views[self.views_processed][1]:self.views[self.views_processed][3],self.views[self.views_processed][0]:self.views[self.views_processed][2]]
        img_view_ = self.image_[:,self.views[self.views_processed][1]:self.views[self.views_processed][3],self.views[self.views_processed][0]:self.views[self.views_processed][2]]
        seg_view = self.segmentation[:,self.views[self.views_processed][1]:self.views[self.views_processed][3],self.views[self.views_processed][0]:self.views[self.views_processed][2]]

        self.batch_data = img_view[np.newaxis, ...]
        self.batch_data_ = img_view_[np.newaxis, ...]
        self.batch_segmentation = seg_view[np.newaxis, ...]

        self.views_processed+=1

    def load_data_full(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        img = Image.open('{}{}.{}'.format(self.images_folder, self.list_images[self.permutation[idx]],self.image_file_extension))
        img = np.array(img, dtype=np.float32)
        if len(img.shape)==3:
            img = img[:,:,::-1]
            #img -= self.mean_bgr
            img = img.transpose((2,0,1))
        elif len(img.shape)==2:
            img = img[np.newaxis, ...]
        else:
            print "Error : shape not accepted for img."


        img_ = Image.open('{}{}.{}'.format(self.images_folder_, self.list_images[self.permutation[idx]],self.image_file_extension_))
        img_ = np.array(img_, dtype=np.float32)
        if len(img_.shape)==3:
            img_ = img_[:,:,::-1]
            #img -= self.mean_bgr
            img_ = img_.transpose((2,0,1))
        elif len(img_.shape)==2:
            img_ = img_[np.newaxis, ...]
        else:
            print "Error : shape not accepted for img_."

        seg = Image.open('{}{}.{}'.format(self.segmentations_folder, self.list_images[self.permutation[idx]],self.segmentation_file_extension))
        seg = np.array(seg, dtype=np.uint8)
        seg = seg[np.newaxis, ...]

        self.batch_data = img[np.newaxis, ...]
        self.batch_data_ = img_[np.newaxis, ...]
        self.batch_segmentation = seg[np.newaxis, ...]

    def load_img_data(self, idx,data_=False):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        if data_:
            im = Image.open('{}{}.{}'.format(self.images_folder_, self.list_images[self.permutation[idx]],self.image_file_extension_))
        else:
            im = Image.open('{}{}.{}'.format(self.images_folder, self.list_images[self.permutation[idx]],self.image_file_extension))
        in_ = np.array(im, dtype=np.float32)
        if len(in_.shape)==3:
            in_ = in_[:,:,::-1]
        #in_ -= self.mean_bgr
            in_ = in_.transpose((2,0,1))
        elif len(in_.shape)==2:
            in_ = in_[np.newaxis, ...]
        else:
            print "Error : shape not accepted."
        if self.dummy_data:
            return np.ones((3,64,64),dtype=np.float32)
        else:
            return in_
        self.batch_data = self.load_img_data(self.iter_counter)[np.newaxis, ...]
        self.batch_data_ = self.load_img_data(self.iter_counter,True)[np.newaxis, ...]
        self.batch_segmentation = self.load_img_segmentation(self.iter_counter)[np.newaxis, ...]

    def load_img_segmentation(self, idx,delta=False):
        """
        Load segmentation image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        im = Image.open('{}{}.{}'.format(self.segmentations_folder, self.list_images[self.permutation[idx]],self.segmentation_file_extension))
        segmentation = np.array(im, dtype=np.uint8)
        if delta:
            segmentation = segmentation - 1
        segmentation = segmentation[np.newaxis, ...]

        if self.dummy_data:
            return np.ones((1,64,64),dtype=np.float32)
        else:
            return segmentation

class jg_input_segmentation_layer(caffe.Layer):
    """works with only on image for now by batch"""

    def get_classes(self):
        self.classes = []
        with open(self.classes_file_path, 'r') as fp:
            for line in fp:
                if line[0]=='#' or line[0]=='\n':
                    pass
                else:
                    self.classes.append(line[:-1])

    def setup(self, bottom, top):
        """Setup the layer."""


        # tops: check configuration
        if len(top) != 2:
            raise Exception("Need to define {} tops for all outputs.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        layer_params = yaml.load(self.param_str_)

        self.iter_counter = 0  # setup a counter for iteration

        self.images_folder = layer_params['images_folder']
        self.image_file_extension = layer_params['image_file_extension']

        self.segmentations_folder = layer_params['segmentations_folder']
        self.segmentation_file_extension = layer_params['segmentation_file_extension']

        self.set_file_path = layer_params['set_file_path']
        self.classes_file_path = layer_params['classes_file_path']

        # classes
        self.classes = []
        self.get_classes()
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))

        # images
        self.list_images = open(self.set_file_path, 'r').read().splitlines()
        self.nb_images = len(self.list_images)
        print "Number of image :",self.nb_images
        print "Separated in", self.num_classes, "classes :", self._class_to_ind

        self.batch_data = np.zeros((1,3,400,400), dtype=np.float32)
        self.batch_segmentation = np.zeros((1,3,400,400), dtype=np.uint8)

        # shuffling at each epoch
        self.shuffle = layer_params['shuffle']
        self.permutation = []
        self.countEpoch = 0
        if self.shuffle:
            print "Shuffling activated."
            self.permutation = np.random.permutation(self.nb_images)
        else:
            self.permutation = range(self.nb_images)

    def update(self,newParams):
        for key in newParams.keys():
            setattr(self, key, newParams[key])

        self.get_classes()
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
            #segmentation
        """

        # fill blob with data
        top[0].data[...] = self.batch_data
        top[1].data[...] = self.batch_segmentation

        # Update number of forward pass done
        self.iter_counter += 1
        if self.iter_counter>=self.nb_images:
            self.countEpoch += 1
            print "New epoch [total epoch :", self.countEpoch,"]."
            self.iter_counter=0
            if self.shuffle:
                print "Reshuffling !"
                self.permutation = np.random.permutation(self.nb_images)

    def backward(self, top, propagate_down, bottom):
        pass

    def reshape(self, bottom, top):
        # make batch
        self.batch_data = self.load_img_data(self.iter_counter)[np.newaxis, ...]
        self.batch_segmentation = self.load_img_segmentation(self.iter_counter)[np.newaxis, ...]

        # reshape net
        top[0].reshape(*self.batch_data.shape)
        top[1].reshape(*self.batch_segmentation.shape)

    def load_img_data(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open('{}{}.{}'.format(self.images_folder, self.list_images[self.permutation[idx]],self.image_file_extension))
        in_ = np.array(im, dtype=np.float32)
        if len(in_.shape)==3:
            in_ = in_[:,:,::-1]
        #in_ -= self.mean_bgr
            in_ = in_.transpose((2,0,1))
        elif len(in_.shape)==2:
            in_ = in_[np.newaxis, ...]
        else:
            print "Error : shape not accepted."
        return in_

    def load_img_segmentation(self, idx,delta=False):
        """
        Load segmentation image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        im = Image.open('{}{}.{}'.format(self.segmentations_folder, self.list_images[self.permutation[idx]],self.segmentation_file_extension))
        segmentation = np.array(im, dtype=np.uint8)
        if delta:
            segmentation = segmentation - 1
        segmentation = segmentation[np.newaxis, ...]
        return segmentation

class jg_rebatching_layer(caffe.Layer):
    """ Define a new batch from two blobs :
    - the feature blobs
    - the rois blobs
    """

    def setup(self, bottom, top):
        """Setup the layer."""

        layer_params = yaml.load(self.param_str_)

        self.strategy = layer_params['strategy']
        self.batch_size = layer_params['batch_size']



    def forward(self, bottom, top):
        """
        Top order :
            #data
        """

        # fill blob with data
        top[0].data[...] = self.batch_data


    def backward(self, top, propagate_down, bottom):
        """
        RELU BACKWARD :
          if (propagate_down[0]) {
            const Dtype* bottom_data = bottom[0]->cpu_data();
            const Dtype* top_diff = top[0]->cpu_diff();
            Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
            const int count = bottom[0]->count();
            Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
            for (int i = 0; i < count; ++i) {
              bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
                  + negative_slope * (bottom_data[i] <= 0));
            }
            https://github.com/BVLC/caffe/blob/master/src/caffe/layers/reshape_layer.cpp
            https://github.com/BVLC/caffe/blob/master/src/caffe/layers/concat_layer.cpp
        """

        bottom[0].diff[...] = 10 * top[0].diff
        #bottom[0].diff[...] = 10 * top[0].diff
        #self.loss = top[0].diff
        pass


    def reshape(self, bottom, top):
        #while !self.batch_ready:
        #    pass

        # make batch
        self.batch_data = self.rebatch_features()

        # reshape net
        top[0].reshape(*self.batch_data.shape)

    def rebatch_features(self):
        """
        Rebatch
        """
        # get size of feature blob
        # newbatch has same deepness

        newbatch = np.zeros([256, 3, 480, 640])

        im = Image.open('{}{}.{}'.format(self.images_folder, self.list_images[idx],self.image_file_extension))
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        #in_ -= self.mean_bgr
        in_ = in_.transpose((2,0,1))
        return newbatch

    def load_img_segmentation(self, idx=0):
        """
        Load segmentation image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        im = Image.open('{}{}.{}'.format(self.segmentations_folder, self.list_images[idx],self.segmentation_file_extension))
        segmentation = np.array(im, dtype=np.uint8)
        segmentation = segmentation[np.newaxis, ...]
        return segmentation

    def load_img_rois(self, idx):
        """
        Load bounding boxes info from pascal voc template data
        """
        filename = '{}{}.{}'.format(self.annotations_folder, self.list_images[idx],self.annotation_file_extension)
        tree = ET.parse(filename)
        objs = tree.findall('object')
        #if not self.config['use_diff']:
        ## Exclude the samples labeled as difficult
        #    non_diff_objs = [
        #        obj for obj in objs if int(obj.find('difficult').text) == 0]
        #    objs = non_diff_objs

        only_in_cls_obj = [obj for obj in objs if obj.find('name').text.lower().strip() in self.classes]
        objs = only_in_cls_obj

        num_objs = len(objs)

        img_rois = np.zeros((num_objs, 5), dtype=np.uint32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            img_rois[ix, :] = [cls, x1, y1, x2, y2]

        img_rois = img_rois[np.newaxis, ...]
        return img_rois

    def load_batch_data(self):
        """
        Load data in a batch
        """

    def load_batch_segmentation(self):
        """
        Load segmentations in a batch
        """

    def load_batch_rois(self):
        """
        Load rois in a batch
        """

    def define_new_batch(self):
        for batch in [self.list_images[x:x+self.batch_size] for x in xrange(0, len(self.list_images), self.batch_size)]:
            new_shape = (len(batch),) + tuple([3,480,640])
            print new_shape
            #if net.blobs['data'].data.shape != new_shape:
            #    net.blobs['data'].reshape(*new_shape)
            #for index, image in enumerate(chunk):
            #    image_data = transformer.preprocess('data', image)
            #    net.blobs['data'].data[index] = image_data
            #output = net.forward()[net.outputs[-1]]
