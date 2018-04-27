from __future__ import print_function

import caffe
import numpy as np
import yaml
import os
from PIL import Image
from skimage import transform 
from scipy import ndimage
import matplotlib.pyplot as plt

class Visualize(caffe.Layer):
    def setup(self, bottom, top):
        assert len(bottom) > 0
        assert len(top) == 0
        """Setup the layer"""
        self._tuple_len = 4
        self._data_shape = (3, 227, 227)
        
        layer_params = yaml.load(self.param_str)
        self._types = layer_params['types']
        self._plot_iter = layer_params['plot_iter']

        self._save = False
        if "save_path" in layer_params:
            self._save_path = layer_params['save_path']
	    self._save = True
        
        assert len(self._types) == len(bottom)
        self._batch_size = bottom[0].shape[0]
        
        self._cur_iter = -1
        
        self._type_size = 0
        for i in range(len(self._types)):
            tp = self._types[i]
            
            if tp == 'txt': 
                pass
            elif tp == 'img': 
                self._type_size += 1
            else:
                raise Exception("Visualize: invalid type")

    def forward(self, bottom, top):
        self._cur_iter += 1
        if (self._cur_iter % self._plot_iter) != 0: return

        fig, axarr = plt.subplots(self._type_size + 1, self._batch_size, squeeze = False, sharex = True, sharey = True)

        # Fill
        strs = np.empty((self._batch_size,), dtype="U32")
        for i in range(len(self._types)):
            tp = self._types[i]
            
            if tp == 'img':
                for j in range(self._batch_size):
                    im = np.transpose(bottom[i].data[j], axes=(1, 2, 0))
                    axarr[i, j].imshow(im)
                    axarr[i, j].get_xaxis().set_visible(False)
                    axarr[i, j].get_yaxis().set_visible(False)
            elif tp == 'txt':
                #print(bottom[i].data)
                for j in range(self._batch_size):
                    if len(strs[j]) != 0: strs[j] += "\n"
                    strs[j] += str(bottom[i].data[j].flatten())
          
        for j in range(self._batch_size):
            data = strs[j]
            axarr[-1, j].text(0.5, 0.5, str(data), horizontalalignment='center',
                              verticalalignment='center', transform=axarr[-1, j].transAxes)
        
        # Plot
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
	if not self._save:
	        mng = plt.get_current_fig_manager()
	        mng.window.showMaximized()
	        plt.show()
	else:
		plt.savefig(os.path.join(self._save_path, "vis_{}.png".format(self._cur_iter)), dpi=300)
		plt.close()


    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
