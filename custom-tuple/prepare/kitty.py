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

from blob_fetcher import KittiBlobFetcher
import glob
import re

base_dir = "/srv/glusterfs/patilv/Datasets/kitti/raw/"
local_file_dir = "image_02/data/"
store_dir = "/home/kwolters/logs/data/"

dirs = sorted(glob.glob(os.path.join(base_dir, 'extracted', "*_sync")))
# REMOVE FAULTY DIR
for d in dirs:
    mtch = re.match(".*/([0-9_]+)_drive_([0-9]+)_sync", d)
    if mtch is None: raise Exception("invalid kitti directory")
    date, drive = mtch.groups()
    if date == "2011_09_26" and drive == "0009": 
        dirs.remove(d)
        continue

dir_size = []
sys.stdout.flush()

fetchers = []

fetcher = KittiBlobFetcher(base_dir, verbose = True)
fetcher.start()
for d in dirs:
    local_dir = os.path.join(d, local_file_dir)
    
    mtch = re.match(".*/([0-9_]+)_drive_([0-9]+)_sync", d)
    if mtch is None: raise Exception("invalid kitti directory")
    date, drive = mtch.groups()
    
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
        loc = "{}/{}{:010}.png".format(dirs[i], local_file_dir, j)
        total_img += 1
        #print(loc)
       
        new_loc = "{}/{:010}.png".format(os.path.basename(dirs[i]), j)
        if not os.path.exists(os.path.join(store_dir, new_loc)):
            fetchers[i].req(loc, (1, 3, 227, 227))
            files[i].append(new_loc)
        
# fetch images 
print(len(dirs), total_img)
for i, ds in enumerate(dir_size):
    for loc in files[i]:
        print(loc)
        
        im = fetchers[i].get()
        im = np.transpose(im, axes = (1, 2, 0))
       
        #print(im[0].shape)
        #plt.imshow(im[:, :, 0], cmap="gray")
        #plt.show()
        
        plt.imsave(os.path.join(store_dir, loc), im)
        #plt.imshow(im[:, :, 1], cmap="gray")
        #plt.show()
        #plt.imshow(im[:, :, 2], cmap="gray")
        #plt.show()
        #plt.imshow(im)
        #plt.show()
        
for fetcher in fetchers:
    fetcher.terminate()
