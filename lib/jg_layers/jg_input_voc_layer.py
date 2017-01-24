import caffe
import numpy as np
import yaml
from PIL import Image
#import scipy.io
#from multiprocessing import Process, Queue

class jg_input_voc_layer(caffe.Layer):


    def get_classes(self):
        with open(self.classes_file_path, 'r') as fp:
            for line in fp:
                if line[0]=='#' or line[0]=='\n':
                    pass
                else:
                    self.classes.append(line[:-1])
        print "Classes :", self.classes

    def setup(self, bottom, top):
        """Setup the layer."""



        layer_params = yaml.load(self.param_str_)

        self.iter_counter = 0  # setup a counter for iteration
        self.images_folder = layer_params['images_folder']
        self.image_file_extension = layer_params['image_file_extension']
        self.annotations_folder = layer_params['annotations_folder']
        self.segmentations_folder = layer_params['segmentations_folder']
        self.set_file_path = layer_params['set_file_path']
        self.classes_file_path = layer_params['classes_file_path']

        # shuffling at each epoch
        self.shuffle = layer_params['shuffle']


        self.recorded_loss = [] #todo in backward prop keep values of loss to get hard examples

        self.max_width = layer_params['max_width']
        self.max_height = layer_params['max_height']


        self.classes = []
        self.get_classes()
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))


        self.batch_data = []
        self.batch_vt_segmentation = []
        self.batch_vt_rois = [] #[Nrois,6](img_id,class_id, x, y, w, h) do a layer of conversion to bbox reg target [bbox_target = convertFrom(vt_roi,stride)]
        self.batch_ready = False

        #read list of images :
        # load indices for images and labels
        self.list_images = open(self.set_file_path, 'r').read().splitlines()
        print "Number of image :",len(self.list_images)



    def forward(self, bottom, top):
        self.iter_counter += 1
        pass
        #data
        top[0].data[...] = self.load_image()
        #vt_segmentation
        top[1].data[...] = self.load_image()
        #vt_roi
        top[2].data[...] = self.load_image()
        pass


    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        #print "reshaping..."
        aaaaaaaa = self.load_image()
        #print aaaaaaaa.shape
        top[0].reshape(1, *aaaaaaaa.shape)
        top[1].reshape(1, *aaaaaaaa.shape)
        top[2].reshape(1, *aaaaaaaa.shape)
        #print "done reshaping"


    def load_image(self, idx=0):
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
