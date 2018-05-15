from __future__ import print_function

import caffe
import numpy as np
import yaml
import os
import sys
import time
from PIL import Image
from skimage import transform 
from scipy import ndimage
import matplotlib.pyplot as plt
import pykitti
import re

from blob_fetcher import BlobFetcher, KittiBlobFetcher

class InputImageTuple(caffe.Layer):
        
    def setup(self, bottom, top):
        assert len(bottom) == 0
        assert len(top) > 1
        """Setup the layer"""
        self._tuple_len = 4
        
        layer_params = yaml.load(self.param_str)
        self._data_shape = tuple(layer_params['data_shape']) #(1, 227, 227)
        self._base_dir = layer_params['base_dir']
        self._keys_file = layer_params['keys_file']
        self._label_file = layer_params['label_file']
        self._batch_size = layer_params['batch_size']

        # Read keys
        self._key_index = 0
        self._data_keys = np.genfromtxt(self._keys_file, "U")
        self._data_labels = np.genfromtxt(self._label_file, 'i')
        self._num_images = len(top) - 1
        assert len(self._data_keys) == len(self._data_labels)
        for key in self._data_keys:
            assert self._num_images == len(key)
            
        if 'shuffle' in layer_params and layer_params['shuffle_tuple'] == True:
            idxs = np.arange(len(self._data_keys))
            np.random.shuffle(idxs)
            self._data_keys = self._data_keys[idxs]
            self._data_labels = self.data_labels[idxs]
            
        # Reshape image tops
        for i in range(self._num_images):
            top[i].reshape(*((self._batch_size,) + self._data_shape))
            
        top[-1].reshape(self._batch_size)
        
        self._direct = False
        if 'direct' in layer_params:
           self._direct = layer_params['direct']
           
        if not self._direct:
            self._fetcher = KittiBlobFetcher(self._base_dir)
        else:
            self._fetcher = BlobFetcher(self._base_dir)
        
        # Start blobfetcher process and fetch first batch        
        self._fetcher.start()
        self.prepare_next_minibatch()

    def prepare_next_minibatch(self):
        self._next_labels = np.zeros((self._batch_size)) 
        for b in range(self._batch_size):
            if self._key_index >= len(self._data_keys):
                self._key_index = 0
                
            batch_keys = self._data_keys[self._key_index]
            #should_flip_image = True if (self._lr_flip_aug and np.random.randint(2) == 1) else False;
            for t in range(self._num_images):
                im_name = batch_keys[t];

                im_path = os.path.join(self._base_dir, 'extracted', im_name)
                #im_datas[t][b,:] = self.load_image(im_path, shape=(self._batch_size,) + self._data_shape)
                self._fetcher.req(im_path, shape=(self._batch_size,) + self._data_shape)
            
            self._next_labels[b] = self._data_labels[self._key_index]
            
            self._key_index += 1
            
    def get_next_minibatch(self): 
        im_datas = np.zeros((self._num_images, self._batch_size) + self._data_shape, dtype=np.float32)
        labels = self._next_labels
        for b in range(self._batch_size):
            for t in range(self._num_images):
                im_datas[t][b,:] = self._fetcher.get()
            
            #labels[b] = self._data_labels[self._key_index]    
                        
        return im_datas, labels

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        
        start = time.time()
        
        #if hasattr(self, '_last_time'):
            #print("since last {} seconds", start - self._last_time)
        
        # get current batch
        im_datas, labels = self.get_next_minibatch()
        
        for c in range(self._num_images):
            top[c].data[:] = im_datas[c].astype(np.float32, copy=False)
            
        top[-1].data[:] = labels
                
        # prepare next one
        assert self._fetcher.valid()
        self.prepare_next_minibatch()
        
        #print("reading took {} seconds".format(time.time() - start))
        self._last_time = time.time()

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
    

