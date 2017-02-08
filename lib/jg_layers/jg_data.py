import os
import caffe
import numpy as np
import yaml
from PIL import Image
import cPickle
import xml.etree.ElementTree as ET
import sklearn.metrics #for confusion matrix

#import scipy.io
#from multiprocessing import Process, Queue

class dataset:
    """
    - keep list of images
    - keep list of crops for each images
    - reload data if already computed
    - give a batch on demand (with a unique id making it reproductible)
    """

    def prepareCrops(img_w, img_h, overlap_max=0.5, crop_size=100):

        tresh_h = max(3,int(0.10*crop_size))
        delta = img_h
        smax_h = crop_size
        smin_h = 0
        #for s from smax to smin:
        n_h = -1
        s_h = -1
        d_h = -1
        for s in range(smax_h,smin_h,-1):
            n_step_max = int(float((img_h-crop_size))/s+1)
            for n in range(n_step_max,0,-1):
                delta = int(float(img_h - crop_size - s*(n-1)) /2 + 0.5)
                if delta<tresh_h:
                    n_h = n
                    s_h = s
                    d_h = delta
                    #print "H",img_h,"h",crop_size,"s",s,"n",n,"delta",delta, "tresh",tresh_h
                    #print "h + (n-1)*s+2d =",crop_size + (n_h-1)*s_h+2*d_h
                    break
            if delta<tresh_h:
                break


        tresh_w = max(3,int(0.10*crop_size))
        delta = img_w
        smax_w = crop_size
        smin_w = 0
        #for s from smax to smin:
        n_w = -1
        s_w = -1
        d_w = -1
        for s in range(smax_w,smin_w,-1):
            n_step_max = int(float((img_w-crop_size))/s+1)
            for n in range(n_step_max,0,-1):
                delta = int(float(img_w - crop_size - s*(n-1)) /2 + 0.5)
                if delta<tresh_w:
                    n_w = n
                    s_w = s
                    d_w = delta
                    #print "H",img_h,"h",crop_size,"s",s,"n",n,"delta",delta, "tresh",tresh_h
                    #print "w + (n-1)*s+2d =",crop_size + (n_w-1)*s_w+2*d_w
                    break
            if delta<tresh_h:
                break

        crops = np.zeros([n_w*n_h,4],dtype=np.uint32)
        pos = 0
        for i_w in range(n_w):
            for i_h in range(n_h):

                crops[pos,:] = [d_w+i_w*s_w, d_h+i_h*s_h, d_w+i_w*s_w+crop_size, d_h+i_h*s_h+crop_size]
                pos = pos + 1

        return crops

    def __init__(self):
        pass

    def __call__(self, crop_id):
        """Return the ..."""
        pass

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

        self.batch_data = np.zeros((1,3,400,400), dtype=np.float32)
        self.batch_data_ = np.zeros((1,3,400,400), dtype=np.float32)
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
        # make batch
        self.batch_data = self.load_img_data(self.iter_counter)[np.newaxis, ...]
        self.batch_data_ = self.load_img_data(self.iter_counter,True)[np.newaxis, ...]
        self.batch_segmentation = self.load_img_segmentation(self.iter_counter)[np.newaxis, ...]

        # reshape net
        top[0].reshape(*self.batch_data.shape)
        top[1].reshape(*self.batch_data_.shape)
        top[2].reshape(*self.batch_segmentation.shape)

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

# class PythonConfMat(caffe.Layer):
#     """
#     Compute the Accuracy with a Python Layer
#     """
#
#     def setup(self, bottom, top):
#         # check input pair
#         if len(bottom) != 2:
#             raise Exception("Need two inputs.")
#
#         self.num_labels = bottom[0].channels
#         params = json.loads(self.param_str)
#         self.test_iter = params['test_iter']
#         self.conf_matrix = np.zeros((self.num_labels, self.num_labels))
#         self.current_iter = 0
#
#     def reshape(self, bottom, top):
#         # bottom[0] are the net's outputs
#         # bottom[1] are the ground truth labels
#
#         # Net outputs and labels must have the same number of elements
#         if bottom[0].num != bottom[1].num:
#             raise Exception("Inputs must have the same number of elements.")
#
#         # accuracy output is scalar
#         top[0].reshape(1)
#
#     def forward(self, bottom, top):
#         self.current_iter += 1
#
#         # predicted outputs
#         pred = np.argmax(bottom[0].data, axis=1)
#         accuracy = np.sum(pred == bottom[1].data).astype(np.float32) / bottom[0].num
#         top[0].data[...] = accuracy
#
#         # compute confusion matrix
#         self.conf_matrix += sklearn.metrics.confusion_matrix(bottom[1].data, pred, labels=range(self.num_labels))
#
#         if self.current_iter == self.test_iter:
#             self.current_iter = 0
#             sys.stdout.write('\nCAUTION!! test_iter = %i. Make sure this is the correct value' % self.test_iter)
#             sys.stdout.write('\n"param_str: \'{"test_iter":%i}\'" has been set in the definition of the PythonLayer' % self.test_iter)
#             sys.stdout.write('\n\nConfusion Matrix')
#             sys.stdout.write('\t'*(self.num_labels-2)+'| Accuracy')
#             sys.stdout.write('\n'+'-'*8*(self.num_labels+1))
#             sys.stdout.write('\n')
#             for i in range(len(self.conf_matrix)):
#                 for j in range(len(self.conf_matrix[i])):
#                     sys.stdout.write(str(self.conf_matrix[i][j].astype(np.int))+'\t')
#                 sys.stdout.write('| %3.2f %%' % (self.conf_matrix[i][i]*100 / self.conf_matrix[i].sum()))
#                 sys.stdout.write('\n')
#             sys.stdout.write('Number of test samples: %i \n\n' % self.conf_matrix.sum())
#             # reset conf_matrix for next test phase
#             self.conf_matrix = np.zeros((self.num_labels, self.num_labels))
#
#     def backward(self, top, propagate_down, bottom):
#         pass
