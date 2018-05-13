import os
import numpy as np
from PIL import Image
from skimage import transform 
from scipy import ndimage
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue

class KittiBlobFetcher(Process):
    def __init__(self, basedir):
        super(KittiBlobFetcher, self).__init__(name='blob_fetcher')
        
        self._iqueue = Queue()
        self._oqueue = Queue()
        
        self._basedir = basedir
        
    def channel_split(self, im):
        assert im.shape[2] == 3
        
        chn = np.random.randint(3)
        return im[:,:,chn:chn+1]

    def jitter_image(self, im):
        assert im.shape[2] == 3
                
        # multiply channel
        for ch in range(3):
            mult = np.random.uniform(0.8, 1.2)
            im[:,:,ch] *= mult
            
        # shift bias   
        shiftVal = np.random.uniform(-0.2, 0.2) 
        im += shiftVal
        
        # add sptial jitter
        nim = im
        #for i in range(im.shape[0]):
            #for j in range(im.shape[1]):
                #if np.random.randint(2) == 1:
                    #r = np.random.randint(-3, 3)
                    #c = np.random.randint(-3, 3)
                    #if 0 <= i+r < im.shape[0] and 0 <= j+c < im.shape[1]:
                        #nim[i+r, j+c] = im[i, j]
                
        # bring values to [0,1] range
        #im = (nim - np.min(nim)) / (np.max(nim) - np.min(nim))
        im = nim
        im[im < 0] = 0
        im[im > 1] = 1
        
        return im

    def load_image(self, imname, shape):
        mtch = re.match("^[a-zA-Z/]+/([0-9_]+)_drive_([0-9]+).*/([0-9]+).[a-z]{3,4}", imname)
        date, drive, frame = mtch.groups()
        frame = int(frame)
        
        # Load image
        #if self._data_shape[0] == 1:
            #im = np.array(Image.open(imname).convert('L'))
            #im = im[:,:,None]
        #else:
        dataset = self._datasets[date + "_" + drive]
        im = np.array(dataset.get_rgb(frame)[0]) #np.array(Image.open(imname))
           
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
        
        # Jitter and split image
        im = im.astype(np.float32)
        
        im = self.jitter_image(im)
        #if shape[1] == 1:
        im = self.channel_split(im)
           
        # Convert to caffe style channels
        im = np.transpose(im, axes = (2, 0, 1))
          
        # Load velo data and transform to image view
        velo = dataset.get_velo(frame)
        conv = np.matmul(dataset.calib.P_rect_00, np.matmul(dataset.calib.R_rect_00, dataset.calib.T_cam0_velo))
        velo_data = np.c_[velo[:, 0:3], np.ones(velo.shape[0])]
        res = np.transpose(np.matmul(conv, np.transpose(velo_data)))
        res[res[:, 2] < 1e-4, 2] = 1e-4
        lid_img = res[:, 0:2] / res[:, 2, None]

        nim = np.zeros((im.shape[0] + 2,) + im.shape[1:])
        nim[:im.shape[0], :] = im
        for i in range(lid_img.shape[0]):
            r = int(lid_img[i, 1] / h * shape[2])
            c = int(lid_img[i, 0] / w * shape[3])
            if not (0 <= r < shape[2] and 0 <= c < shape[3]): continue
            
            nim[im.shape[0], r, c] = res[i, 2]
            nim[im.shape[0]+1, r, c] = velo[i, 3]

        im = nim
        
        #plt.figure()
        #plt.imshow(nim[0])
        #plt.figure()
        #plt.imshow(nim[1])
        #plt.figure()
        #print(h, w, shape[2], shape[3])
        #plt.scatter(lid_img[:,0] / w * shape[3], lid_img[:,1] / h * shape[2], c=res[:, 2])
        #plt.xlim([0, shape[3]])
        #plt.ylim([shape[2], 0])
        #plt.show()
        #sys.exit(0)
        
        #ax[1, 1].scatter(lid_img[:,0], lid_img[:,1], c=res[:, 2]) #first_velo[:, 3]
                
        #im -= immean;
        
        #if self._rgb_jitter_aug:
            #im = self.rgb_jitter_image(im);
        #if flip_image:
            #im = im[:,::-1,:]
        #if caffe_order:
            #im = np.transpose(im, axes = (2, 0, 1))
        return im
    
    def valid(self):
        return self._iqueue.empty() and self._oqueue.empty()
    
    def get(self):
        return self._oqueue.get()
                
    def req(self, loc, shape):
        self._iqueue.put((loc, shape))
            
    def run(self):
        # load dataset
        dirs = sorted(os.listdir(self._basedir + '/extracted'))
        self._datasets = {}
        print('loading datasets: ', end='')
        for dr in dirs:
            mtch = re.match("([0-9_]+)_drive_([0-9]+)_sync", dr)
            if mtch is None: continue
            date, drive = mtch.groups()

            print('.', end='')
            sys.stdout.flush()
        
            self._datasets[date + "_" + drive] = pykitti.raw(self._basedir, date, drive)
        print()
        
        # fetching blobs
        while True:
            loc, shape = self._iqueue.get()
            self._oqueue.put(self.load_image(loc, shape))
            #print('image processed {} todo'.format(self._iqueue.qsize()))
