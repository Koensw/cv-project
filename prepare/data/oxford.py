from __future__ import print_function, absolute_import
import matplotlib.pyplot as plt
import numpy as np
import os, sys
#if len(sys.argv) != 2:
    #print('ERROR: pass base location as argument')
    #sys.exit(1)

## ADD PATHS
def addPath(path):
    if path not in sys.path:
        sys.path.insert(0, path)

addPath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../utils'))

from blob_fetcher import BlobFetcher
import glob
import re

base_dir = "/srv/glusterfs/patilv/Datasets/oxford/2014-05-06-12-54-54/stereo/centre/"
store_dir = "/home/kwolters/sp/data/oxford/" # "/home/kwolters/sp/data/vis_lid/"

orig_files = sorted(glob.glob(os.path.join(base_dir, "*")))
fetcher = BlobFetcher(base_dir, transform = True, channel_split = False, jitter = False)
fetcher.start()

total_img = 0
files = []
for idx, fil in enumerate(orig_files):
    if (idx % 5) != 0: continue
    #loc = "{}/{}{:06}.png".format(dirs[i], local_file_dir, j)
    total_img += 1
    #print(idx, fil)
    
    new_loc = "{:06}.png".format(idx)
    if not os.path.exists(os.path.join(store_dir, new_loc)):
        fetcher.req(fil, (1, 3, 227, 227))
        files.append(new_loc)
        
# fetch images 
print(len(files), total_img)
k = 0
for loc in files:
    print(k, '/', total_img, ' ', loc, sep='')
    k = k+1
    
    im = fetcher.get()
    im = np.transpose(im, axes = (1, 2, 0))
    if im.shape[2] == 1:
        im = im[:, :, 0]
    
    #print(im[0].shape)
    #plt.imshow(im[:, :, 0], cmap="gray")
    #plt.show()
    
    im_path = os.path.join(store_dir, loc)
    #print(im_path, im.shape)
    plt.imsave(im_path, im, cmap = 'gray')
    #sys.exit(0)
    #plt.imshow(im[:, :, 1], cmap="gray")
    #plt.show()
    #plt.imshow(im[:, :, 2], cmap="gray")
    #plt.show()
    #plt.imshow(im)
    #plt.show()
        
fetcher.terminate()
