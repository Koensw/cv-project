import caffe
import numpy as np
import yaml
#from multiprocessing import Process, Queue
#import multiprocessing
#import h5py
#import math
#import code
#import traceback as tb
import os
from PIL import Image
#import cv2
import scipy.misc
#from multiprocessing.sharedctypes import Array as sharedArray
#import ctypes
#import atexit
import time
import sys
import operator
from functools import reduce

class InputImageTuple(caffe.Layer):
    #def setup_prefetch(self):
        #self._slots_used = Queue(self._max_queue_size);
        #self._slots_filled = Queue(self._max_queue_size);
        #global shared_mem_list
        #shared_mem_list = [[] for t in range(self._num_tops)]
        #for t in range(self._num_tops):
            #for c in range(self._max_queue_size):
                #shared_mem = sharedArray(ctypes.c_float, self._blob_counts[t]);
                #with shared_mem.get_lock():
                    #s = np.frombuffer(shared_mem.get_obj(), dtype=np.float32);
                    #assert(s.size == self._blob_counts[t]), '{} {}'.format(s.size, self._blob_counts)
                #shared_mem_list[t].append(shared_mem);
        #self._shared_mem_shape = self._data_shapes;
        ##start the process now
        #self._prefetch_process_name = "data prefetcher"
        #self._prefetch_process_id_q = Queue(1);
        #self._prefetch_process = BlobFetcher(self._prefetch_process_name, self._prefetch_process_id_q,\
                                    #self._slots_used, self._slots_filled,\
                                    #self._shared_mem_shape, self._num_tops, self.get_next_minibatch_helper)
        #for c in range(self._max_queue_size):
            #self._slots_used.put(c);
        #self._prefetch_process.start();
        #self._prefetch_process_id = self._prefetch_process_id_q.get();
        #print('prefetching enabled: %d'%(self._prefetch_process_id));
        #print('setting up prefetcher with queue size: %d'%(self._max_queue_size));
        #def cleanup():
            #print('terminate BlobFetcher')
            #self._prefetch_process.terminate()
            #self._prefetch_process.join();
        #atexit.register(cleanup)

    #def check_prefetch_alive(self):
        #try:
            #os.kill(self._prefetch_process_id, 0) #not killing just poking to see if alive
        #except err:
            ##will raise exception if process is dead
            ##can do something more intelligent here rather than raise the same error ...
            #raise err

    #def crop_ms_augment_image(self, imPath, im_shape, should_flip_image):
        #if self._dataset == 'ucf101' or self._dataset == 'hmdb51':
            ##caffe read image automatically adds in RGB jittering
            #if self._dataset == 'ucf101':
                #orig_im = self.caffe_read_image(imPath, shape=[1,3,256,340], immean=self._image_mean, caffe_order=False);
            #else:
                #orig_im = self.caffe_read_image(imPath, shape=None, immean=self._image_mean, caffe_order=False);
                #self.setup_crop_ms_aug(im_size=[orig_im.shape[0], orig_im.shape[1]]);
            ## tb.print_stack();namespace = globals().copy();namespace.update(locals());code.interact(local=namespace)
            ##multiscale
            #ms_ind = np.random.randint(0, len(self._crop_sizes));
            #crop_ind = np.random.randint(0, len(self._crop_offsets));
            #crop_ht = self._crop_sizes[ms_ind][0]; crop_wid = self._crop_sizes[ms_ind][1]
            #ht_off = self._crop_offsets[crop_ind][0]; wid_off = self._crop_offsets[crop_ind][1];
            #crop_im = orig_im[ht_off:(ht_off+crop_ht), wid_off:(wid_off+crop_wid), :];
            ## tb.print_stack();namespace = globals().copy();namespace.update(locals());code.interact(local=namespace)
            #if crop_im.shape[0] != self._nw_size or crop_im.shape[1] != self._nw_size:
                #aug_im = cv2.resize(crop_im,(self._nw_size, self._nw_size), interpolation = cv2.INTER_LINEAR).astype(np.float32)
            #else:
                #aug_im = crop_im;
            ##flip?
            #if should_flip_image == True:
                #aug_im = aug_im[:,::-1,:];
            ##caffe_order
            #aug_im = np.transpose(aug_im, axes = (2,0,1));
            #return aug_im;

    #def setup_crop_ms_aug(self, im_size=None):
        #if self._dataset == 'ucf101' or self._dataset == 'hmdb51':
            #self._scale_ratios = [1.0];
            #self._nw_size = self._data_shapes[0][2];
            #if im_size is None:
                #self._orig_im_size = [256, 340]
            #else:
                #self._orig_im_size = im_size
            #height_off = int((self._orig_im_size[0] - self._nw_size)/4.0);
            #wid_off = int((self._orig_im_size[1] - self._nw_size)/4.0);
            #self._crop_offsets = [];
            #self._crop_offsets.append([0, 0])
            #self._crop_offsets.append([0, 4*wid_off])
            #self._crop_offsets.append([4*height_off, 0])
            #self._crop_offsets.append([4*height_off, 4*wid_off])
            #self._crop_offsets.append([2*height_off, 2*wid_off])

            ##crop_sizes
            #self._crop_sizes = [];
            #base_size = min(self._orig_im_size[0], self._orig_im_size[1])
            #for ii, h in enumerate(self._scale_ratios):
                #crop_h = int(base_size * h);
                #crop_h = self._nw_size if (abs(crop_h - self._nw_size) < 3) else crop_h;
                #for jj, w in enumerate(self._scale_ratios):
                    #crop_w = int(base_size * w);
                    #crop_w = self._nw_size if (abs(crop_w - self._nw_size) < 3) else crop_w;
                    #if abs(ii-jj)<=1:
                        #self._crop_sizes.append([crop_h, crop_w])

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

    def load_image(self, imname, shape):
        # Load image
        im = np.array(Image.open(imname))
        if im is None:
            print('could not read image: {}'.format(imname))
        im = im[:,:,::-1]
        
        print(im.shape, shape)
        
        # Resize if necessary
        h = im.shape[0]
        w = im.shape[1]
        if (shape[2] != h or shape[3] != w or len(im.shape)!=3 or im.shape[2]!=3):
            im = scipy.misc.imresize(im, (int(shape[2]), int(shape[3]), shape[1]), interp='bilinear');
        # print 'resize', tt.toc();
        im = im.astype(np.float32)
        im -= immean;
        
        #if self._rgb_jitter_aug:
            #im = self.rgb_jitter_image(im);
        #if flip_image:
            #im = im[:,::-1,:]
        #if caffe_order:
            #im = np.transpose(im, axes = (2, 0, 1))
        return im;

    def setup(self, bottom, top):
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
            
        top[-1].reshape(self._batch_size, 1, 1, 1)

    def get_next_minibatch(self):
        im_datas = np.zeros((self._num_images, self._batch_size) + self._data_shape, dtype=np.float32)
            
        for b in range(self._batch_size):
            if self._key_index >= len(self._data_keys):
                self._key_index = 0
                
            batch_keys = self._data_keys[self._key_index]
            #should_flip_image = True if (self._lr_flip_aug and np.random.randint(2) == 1) else False;
            for t in range(self._num_images):
                im_name = batch_keys[t];

                im_path = os.path.join(self._base_dir, im_name);
                #if self._crop_ms_aug:
                    #im_datas[t][b,...] = self.crop_ms_augment_image(imPath, self._single_batch_shapes[t],\
                                                                    #should_flip_image);
                #else:
                #caffe_read_image automatically adds in RGB jittering if enabled
                im_datas[t][b,:] = self.load_image(im_path, shape=self._single_batch_shapes[t])
            self._key_index += 1
        return im_datas;

    def forward(self, bottom, top):
        raise Exception("Not implemented")
        #"""Get blobs and copy them into this layer's top blob vector."""
        im_datas = self.get_next_minibatch();
        for im_data in im_datas:
            top[c].data[:] = im_data
        
        #tb.print_stack();namespace = globals().copy();namespace.update(locals());code.interact(local=namespace)
        #time.sleep(2);


    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
