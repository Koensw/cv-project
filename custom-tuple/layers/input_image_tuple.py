from __future__ import print_function

import caffe
import numpy as np
import yaml
import os
from PIL import Image
from skimage import transform 
from scipy import ndimage
import matplotlib.pyplot as plt

class InputImageTuple(caffe.Layer):
    #def rgb_jitter_image(self, im):
        #assert im.shape[2] == 3; #we want a rgb or bgr image (NOT in caffe order)
        #assert im.dtype == np.float32
        #for ch in range(3):
            #thisRand = np.random.uniform(0.8, 1.2000000001); 
            #im[:,:,ch] *= thisRand;
        #shiftVal = np.random.randint(0,6);
        #if np.random.randint(2) == 1:
            #shiftVal = -shiftVal;
        #im += shiftVal;
        #im = im.astype(np.uint8); #cap values to [0,255]
        #im = im.astype(np.float32)
        #return im;
        
    def setup(self, bottom, top):
        assert len(bottom) == 0
        assert len(top) > 1
        """Setup the layer"""
        self._tuple_len = 4
        self._data_shape = (3, 227, 227)
        
        layer_params = yaml.load(self.param_str)
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
            
        # Reshape image tops
        for i in range(self._num_images):
            top[i].reshape(*((self._batch_size,) + self._data_shape))
            
        top[-1].reshape(self._batch_size)

    def load_image(self, imname, shape):
        # Load image
        im = np.array(Image.open(imname))
        if im is None:
            print('could not read image: {}'.format(imname))
                                
        # Resize if necessary
        h = im.shape[0]
        w = im.shape[1]
        if shape[2] != h or shape[3] != w:
            # apply anti aliasing
            factors = (np.asarray(im.shape, dtype=float) /
                       np.asarray(shape[2:] + (shape[1],), dtype=float))
            anti_aliasing_sigma = np.maximum(0, (factors - 1) / 2)
            
            im = ndimage.gaussian_filter(im, anti_aliasing_sigma)
            
            # resize to requested size
            im = transform.resize(im, shape[2:], mode = 'reflect')
           
        # Convert to caffe style channels
        im = np.transpose(im, axes = (2, 0, 1))
                
        #im -= immean;
        
        #if self._rgb_jitter_aug:
            #im = self.rgb_jitter_image(im);
        #if flip_image:
            #im = im[:,::-1,:]
        #if caffe_order:
            #im = np.transpose(im, axes = (2, 0, 1))
        return im;


    def get_next_minibatch(self):
        im_datas = np.zeros((self._num_images, self._batch_size) + self._data_shape, dtype=np.float32)
        labels = np.zeros((self._batch_size)) 
         
        for b in range(self._batch_size):
            if self._key_index >= len(self._data_keys):
                self._key_index = 0
                
            batch_keys = self._data_keys[self._key_index]
            #should_flip_image = True if (self._lr_flip_aug and np.random.randint(2) == 1) else False;
            for t in range(self._num_images):
                im_name = batch_keys[t];

                im_path = os.path.join(self._base_dir, im_name)
                im_datas[t][b,:] = self.load_image(im_path, shape=(self._batch_size,) + self._data_shape)
            
            labels[b] = self._data_labels[self._key_index]
            
            self._key_index += 1
            
        return im_datas, labels

    def forward(self, bottom, top):
        #"""Get blobs and copy them into this layer's top blob vector."""
        im_datas, labels = self.get_next_minibatch()
        
        #if self._key_index == 1:
            #plt_img = np.transpose(im_datas[0][0], axes=(1, 2, 0))
            #plt.imshow(plt_img)
            #plt.show()
        
        for c in range(self._num_images):
            top[c].data[:] = im_datas[c].astype(np.float32, copy=False)
            
        top[-1].data[:] = labels


    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
