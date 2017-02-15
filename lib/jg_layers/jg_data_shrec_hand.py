from __future__ import print_function
import os
import caffe
import numpy as np
import yaml
from PIL import Image
import cPickle
import xml.etree.ElementTree as ET
import sklearn.metrics #for confusion matrix
import math #for view generation

class jg_input_shrec_layer(caffe.Layer):

    def setup(self, bottom, top):
        """Setup the layer."""

        layer_params = yaml.load(self.param_str_)
        self.batch_size = layer_params['batch_size']
        self.num_classes = int(layer_params['num_classes'])
        self.nb_key_frame = layer_params['nb_key_frame']
        self.root_dataset = layer_params['root_dataset'] #'/c16/THESE.JORIS/datasets/HandGestureDataset_SHREC2017/'
        self.folder_path_canvas = self.root_dataset + 'gesture_{}/finger_{}/subject_{}/essai_{}/'
        self.set_file_path = self.root_dataset + layer_params['set_file_path']
        self.shuffle = layer_params['shuffle']

        self.data_w = 224
        self.data_h = 224



        self.logfile = open('/tmp/log_jg_shrec_layer.txt','w+')


        # images
        self.list_images = np.loadtxt(self.set_file_path)
        self.nb_images = len(self.list_images)

        self.permutation = []
        if self.shuffle:
            print("Shuffling activated.", file=self.logfile)
            self.permutation = np.random.permutation(self.nb_images)
        else:
            self.permutation = range(self.nb_images)

        # init batch data
        self.batch_data = np.zeros((int(self.batch_size),self.nb_key_frame,self.data_h,self.data_w), dtype=np.float32)
        # self.batch_label = np.zeros((int(self.batch_size),1,self.num_classes), dtype=np.uint8)
        self.batch_label = np.zeros((int(self.batch_size)), dtype=np.uint8)

        # counters
        self.images_processed = 0
        self.batches_processed = 0
        self.nb_epochs = 0




        print("This is a test",file=self.logfile)
        print("Number of image.", self.nb_images, file=self.logfile)
        print("Number of classes.", self.num_classes, file=self.logfile)

    def update(self,newParams):
        for key in newParams.keys():
            setattr(self, key, newParams[key])

        self.list_images = open(self.set_file_path, 'r').read().splitlines()
        self.nb_images = len(self.list_images)

    def forward(self, bottom, top):
        top[0].data[...] = self.batch_data
        top[1].data[...] = self.batch_label
        self.batches_processed += 1

    def backward(self, top, propagate_down, bottom):
        pass

    def reshape(self, bottom, top):
        self.load_data()
        top[0].reshape(*self.batch_data.shape)
        top[1].reshape(*self.batch_label.shape)

    def load_data(self):
        for i in range(int(self.batch_size)):
            if self.images_processed>=self.nb_images:
                self.images_processed = 0
                self.nb_epochs += 1
                if self.shuffle:
                    self.permutation = np.random.permutation(self.nb_images)

            idx = self.permutation[self.images_processed]
            sequence_folder_path = self.folder_path_canvas.format(*[str(int(x)) for x in self.list_images[idx,:4]])
            data = np.zeros([self.nb_key_frame,self.data_h,self.data_w]).astype(np.float32)
            for j in range(self.nb_key_frame):
                depth_img = Image.open(sequence_folder_path+str(int(math.floor(self.list_images[idx,6]*(1+j)/(self.nb_key_frame+1))))+'_depth.png')
                depth_img = depth_img.resize((self.data_h,self.data_w),Image.ANTIALIAS)
                data[j,...]=depth_img   #depths[int(math.floor(self.nb_images*(1+j)/(self.nb_key_frame+1))),:]

            if self.num_classes==14:
                label = int(self.list_images[idx,4])-1
                # label = np.zeros(14).astype(np.uint8); label[int(self.list_images[idx,4])-1] = 1
            elif self.num_classes==28:
                label = int(self.list_images[idx,5])-1
                # label = np.zeros(28).astype(np.uint8); label[int(self.list_images[idx,5])-1] = 1
            else:
                print("Error for number of classe (14 or 28)", file=self.logfile)

            # img = img[np.newaxis, ...]
            # label = label[np.newaxis, ...]


            self.batch_data[i,...] = data
            self.batch_label[i,...] = label

            self.images_processed+=1
