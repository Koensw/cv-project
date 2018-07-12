from __future__ import print_function
from itertools import permutations
import numpy as np
import os
import glob
import cv2
import re
import sys
import matplotlib.pyplot as plt

ROUNDS = 10000
DIFF = 7
DIFF_RANGE = 3

MOV_TRESHOLD = 30 # 65

MODE = 'all'

MIN_DIR = int(sys.argv[1])
MAX_DIR = int(sys.argv[2])

SIZE = 4
SAME = 4

order = range(0, SIZE)
orders = np.array(list(permutations(order)))
#order_idx = np.zeros(orders.shape[0])
#for idx, o in enumerate(orders):
#    print(idx, o)
    
labels_file = open("odo_image_labels.txt", 'w')
keys_file = open("odo_image_keys.txt", 'w')

#base_name = "/srv/glusterfs/patilv/Datasets/kitti/raw/extracted"
##local_file_dir = "2011_09_26_drive_0005_sync/image_02/data/"
#local_file_dir = "image_02/data/"
base_name = "/home/kwolters/sp/data/vis_od_color/"
local_file_dir = ""

dirs = glob.glob(os.path.join(base_name, "*"))
dir_size = []
total_size = 0
print(len(dirs))
for d in dirs[:]:
    base_dir = os.path.basename(d)
    if not (MIN_DIR <= int(base_dir) <= MAX_DIR):
        dirs.remove(d)
    
for d in dirs:
    local_dir = os.path.join(base_name, d, local_file_dir)
    
    dir_size.append(len(os.listdir(local_dir)))
    total_size += dir_size[-1]
    
print(total_size)
size_to_idx = np.zeros(total_size, dtype=np.int32) 
cur_idx = 0
for i in range(len(dirs)):
    #for j in range(dir_size[i]):
    size_to_idx[cur_idx:cur_idx+dir_size[i]] = i
    cur_idx += dir_size[i]

for r in range(ROUNDS):
    # prepare
    while True:
        dir_idx = size_to_idx[np.random.randint(total_size)]
        local_dir = os.path.join(base_name,  dirs[dir_idx], local_file_dir)
        
        min_nr = 0
        max_nr = dir_size[dir_idx]
        
        if MODE == 'upper': min_nr = 2*max_nr//3
        elif MODE == 'lower': max_nr = 2*max_nr//3
        
        max_nr -= SIZE * (DIFF + DIFF_RANGE)
        #max_nr -= DIFF + DIFF_RANGE
        
        idx = np.random.randint(min_nr, max_nr)

        files = np.zeros(SIZE, dtype=np.int32) #np.arange(idx - DIFF, idx + DIFF + 1, DIFF)
        files[0] = idx 
        for i in range(1, SIZE):
            files[i] = files[i-1] + DIFF + np.random.randint(-DIFF_RANGE, DIFF_RANGE + 1)
        #files[0] = idx - DIFF + np.random.randint(-DIFF_RANGE, DIFF_RANGE + 1)
        #files[2] = idx + DIFF + np.random.randint(-DIFF_RANGE, DIFF_RANGE + 1)
        
        # check image suitability
        #print("{}{:06}.png".format(local_dir, out[0]))
        frame1 = cv2.imread("{}{:06}.png".format(local_dir, files[0]))
        prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255

        frame2 = cv2.imread("{}{:06}.png".format(local_dir, files[-1]))
        nxt = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs, nxt, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        magind = np.mean(np.sort(mag.flatten())[:-10:-1])
    
        if magind < MOV_TRESHOLD:
            print("Retry", magind)
            pass
            #wait = 0

            #cv2.imshow('image1',prvs)
            #cv2.moveWindow("image1", 20, 0)
            #cv2.imshow('image2',nxt)
            #cv2.moveWindow("image2", 20, 300)
            #cv2.imshow('frame',rgb)
            #cv2.moveWindow("frame", 20, 600)

            #k = cv2.waitKey(wait) & 0xff

            #cv2.destroyAllWindows()
        else: 
            break
        
    # create pair    
    indices = np.random.choice(np.arange(0, orders.shape[0]), SAME, False)
    for index in indices:
        order = orders[index]
        out = files[order]
        print(r, local_dir, out, min_nr, max_nr)
        
        for idx in out:
            keys_file.write("{}{:06}.png".format(local_dir, idx))
            keys_file.write(" ")
        keys_file.write("\n")
        labels_file.write("{}\n".format(index))
