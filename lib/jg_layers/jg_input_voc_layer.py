import os
import caffe
import numpy as np
import yaml
from PIL import Image
import cPickle
import xml.etree.ElementTree as ET

#import scipy.io
#from multiprocessing import Process, Queue

class patches:
    """
    Class to make patches of one image (patch can be called by instance(patch_id) from 0 to instance.nb_views)
    """

    def prepareViews(self, img_w, img_h, stride=112, crop_size=224):

        list_x0 = range(0,img_w,stride)


>>> for n in range(int(math.ceil(float(500)/float(224))),500-224+1):
...     s =int(math.floor((float(500)-float(224))/(float(n)*float(224))))
...     r = s/224.
...     print s, r

        views = []
        for min_size in list_x0:
            stride = int(math.ceil(ratio_for_stride*min_size))
            i_w = 0
            i_h = 0
            while i_w + min_size-1 < img_w:
                i_h = 0
                while i_h + min_size-1 < img_h:
                    views.append([min_size, i_w, i_w + min_size-1, i_h, i_h + min_size-1])
                    i_h += stride
                i_w += stride
        return views

    def __init__(self, img_in_path, edge_length_max = 640., verbose=False):
        self.img = io.imread(img_in_path)
        self.img_w = self.img.shape[0]
        self.img_h = self.img.shape[1]

        self.edge_length_max = edge_length_max

	if self.img.ndim == 2:
             self.img_c = 1
	     self.img = np.expand_dims(self.img, axis=2)
	elif self.img.shape[2] == 4:
             self.img = self.img[:, :, :3]
             self.img_c = 3
	elif self.img.shape[2] == 3:
             self.img_c = 3
	else:
             print('ERROR IN images.py patches.__init__ : file has not a good format (must be RGB, RGBA or grayscale')

        need_downsampling = False
        if self.img_w > edge_length_max:
	    if verbose:
	       print('w too big')
	       print('old : ' + str(self.img_w) + 'x' + str(self.img_h))

	    self.img_h = int(1.*self.img_h/self.img_w*self.edge_length_max)
            self.img_w = self.edge_length_max
            need_downsampling = True
	    if verbose:
	       print('new : ' + str(self.img_w) + 'x' + str(self.img_h))

        if self.img_h > edge_length_max:
	    if verbose:
	       print('h too big')
	       print('old : ' + str(self.img_w) + 'x' + str(self.img_h))
	    self.img_w = int(1.*self.img_w/self.img_h*self.edge_length_max)
            self.img_h = self.edge_length_max
            need_downsampling = True
	    if verbose:
	       print('new : ' + str(self.img_w) + 'x' + str(self.img_h))

        if need_downsampling:
            self.img = resize(self.img,(self.img_w,self.img_h), preserve_range=True)





        self.views = self.prepareViews(self.img_w,self.img_h)
        self.nb_views = len(self.views)

    def __call__(self, patch_id, toresize = True):
        selected_view = self.views[patch_id]
        selected_img = np.zeros((selected_view[0],selected_view[0],self.img_c),dtype=np.int8)
        selected_img = self.img[selected_view[1]:selected_view[2],selected_view[3]:selected_view[4],:]
        if toresize:
             resized_img = resize(selected_img,(256,256))
             return resized_img
        else:
             return selected_img

    def show(self):
        patches_id = range(0, self.nb_views)
        ic = io.ImageCollection(patches_id, load_func=self.__call__)
        collV = CollectionViewer(ic)
        collV.show()

    def nb_views_max(self, img_w=-1, img_h=-1, ratio_for_stride=0.5, crop_size_min=32, crop_size_max=257, crop_size_step=32, verbose=False):
	if img_w==-1:
	   img_w = self.edge_length_max
	if img_h==-1:
	   img_h = self.edge_length_max
    	return len(self.prepareViews(img_w, img_h, ratio_for_stride, crop_size_min, crop_size_max, crop_size_step, verbose))

class jg_input_voc_layer(caffe.Layer):


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
        print "Number of image :",len(self.list_images)
        print "Separated in", self.num_classes, "classes :", self._class_to_ind
        print "Training with a batch_size of :",self.batch_size
        self.define_new_batch()

    def forward(self, bottom, top):
        """
        Top order :
            #data
            #segmentation
            #rois
        """

        #while !self.batch_ready:
        #    pass

        # make batch
        self.batch_data = self.load_img_data(0)[np.newaxis, ...]
        self.batch_segmentation = self.load_img_segmentation(0)[np.newaxis, ...]
        self.batch_rois = self.load_img_rois(0)[np.newaxis, ...]

        # reshape net
        top[0].reshape(*self.batch_data.shape)
        top[1].reshape(*self.batch_segmentation.shape)
        top[2].reshape(*self.batch_rois.shape)
        #top[0].reshape(1, *self.batch_data)
        #top[1].reshape(1, *self.batch_segmentation)
        #top[2].reshape(1, *self.batch_rois)

        # fill blob with data
        top[0].data[...] = self.batch_data
        top[1].data[...] = self.batch_segmentation
        top[2].data[...] = self.batch_rois

        # Update number of forward pass done
        self.iter_counter += 1

    def backward(self, top, propagate_down, bottom):
        pass

    def reshape(self, bottom, top):
        pass

    def load_img_data(self, idx=0):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open('{}{}.{}'.format(self.images_folder, self.list_images[idx],self.image_file_extension))
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        #in_ -= self.mean_bgr
        in_ = in_.transpose((2,0,1))
        return in_

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

        img_rois = np.zeros((num_objs, 5), dtype=np.uint16)

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
