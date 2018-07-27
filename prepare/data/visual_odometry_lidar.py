from __future__ import print_function, absolute_import
import matplotlib.pyplot as plt
import numpy as np
import os, sys
import scipy
#if len(sys.argv) != 2:
    #print('ERROR: pass base location as argument')
    #sys.exit(1)

## ADD PATHS
def addPath(path):
    if path not in sys.path:
        sys.path.insert(0, path)

addPath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../utils'))

from blob_fetcher import BlobFetcher
import glob
import re

base_dir = "/srv/glusterfs/patilv/Datasets/kitti/visual_odometry/dataset/sequences"
local_file_dir = "velodyne_proj_height_2/"
store_dir = "/home/kwolters/sp/data/vis_lid_height_int/"

dirs = sorted(glob.glob(os.path.join(base_dir, "*")))
dir_size = []
fetchers = []

fetcher = BlobFetcher(base_dir, transform = True, channel_split = False, jitter = False)
fetcher.start()
for d in dirs:
    local_dir = os.path.join(d, local_file_dir)
    
    #fetcher = KittiBlobFetcher(base_dir, subdate = date, subdrive = drive, verbose = False)
    #fetcher.start()
    fetchers.append(fetcher)
    
    new_dir = os.path.join(store_dir, os.path.basename(d))
    if not os.path.exists(new_dir): os.makedirs(new_dir)
    
    dir_size.append(len(os.listdir(local_dir)))
    #break
    
#for fetcher in fetchers:
    #fetcher.terminate()
    
# request images
total_img = 0
files = []
for i, ds in enumerate(dir_size):
    files.append([])
    for j in range(ds):
        loc = "{}/{}{:06}.png".format(dirs[i], local_file_dir, j)
        total_img += 1
        #print(loc)
       
        new_loc = "{}/{:06}.png".format(os.path.basename(dirs[i]), j)
        if not os.path.exists(os.path.join(store_dir, new_loc)):
            fetchers[i].req(loc, (1, 3, 227, 227))
            files[i].append(new_loc)
        
# fetch images 
print(len(dirs), total_img)
k = 0
for i, ds in enumerate(dir_size):
    for loc in files[i]:
        print(k, '/', total_img, ' ', loc, sep='')
        k = k+1
        
        im = fetchers[i].get()
        im = np.transpose(im, axes = (1, 2, 0))
        if im.shape[2] == 1:
            im = im[:, :, 0]
            
        print(np.max(im), np.min(im))
        print(im)
        im[im < 1e-3] = 0
        nz = np.nonzero(im)
        
        grid = np.mgrid[0:im.shape[0],0:im.shape[1]]
        
        print(nz[0].shape, im.shape[0]*im.shape[1])
        print(nz, grid)
        nim = scipy.interpolate.griddata(nz, im[nz], tuple(grid), method = 'linear')
       
        #nim = np.zeros(im.shape)
        #temp = np.zeros(im.shape)
        #for idx, val in np.ndenumerate(im):
            #if val <= 1e-9: continue
            
            #tot = 1e-3
            #temp[:, :] = 0
            #for i in range(max(0, idx[0] - 4), min(im.shape[0], idx[0] + 5)):
                #for j in range(max(0, idx[1] - 4), min(im.shape[1], idx[1] + 5)):
                    #rel = (1.0/(np.hypot(i - idx[0], j - idx[1]) + 0.1)) * (np.max(im) - val)
                    #tot = tot + rel
                    #temp[i, j] = rel * val
                  
            ##print(tot)      
            #for i in range(max(0, idx[0] - 4), min(im.shape[0], idx[0] + 5)):
                #for j in range(max(0, idx[1] - 4), min(im.shape[1], idx[1] + 5)):
                    #temp[i, j] /= tot
                    
            #nim += temp
            
       
        #print(im[0].shape)
        plt.figure()
        plt.imshow(im, cmap="gray")
        plt.figure()
        plt.imshow(nim, cmap="gray")
        plt.show()
        sys.exit(0)
        
        #print(nim, im)
        
        
        im_path = os.path.join(store_dir, loc)
        #print(im_path, im.shape)
        plt.imsave(im_path, nim, cmap = 'gray')
        #sys.exit(0)
        #plt.imshow(im[:, :, 1], cmap="gray")
        #plt.show()
        #plt.imshow(im[:, :, 2], cmap="gray")
        #plt.show()
        #plt.imshow(im)
        #plt.show()
        #sys.exit(0)
        
for fetcher in fetchers:
    fetcher.terminate()
