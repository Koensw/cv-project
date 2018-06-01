from __future__ import print_function
import numpy as np
import os
import glob
import cv2
import re
import sys
import matplotlib.pyplot as plt

ROUNDS = 20000
DIFF = 7
DIFF_RANGE = 3

MOV_TRESHOLD = 65

MODE = 'all'

MIN_DIR = int(sys.argv[1])
MAX_DIR = int(sys.argv[2])

#array = np.arange(min_nr, max_nr+1)
order = np.arange(0, 3)

labels_file = open("odo_image_labels.txt", 'w')
keys_file = open("odo_image_keys.txt", 'w')

#base_name = "/srv/glusterfs/patilv/Datasets/kitti/raw/extracted"
##local_file_dir = "2011_09_26_drive_0005_sync/image_02/data/"
#local_file_dir = "image_02/data/"
base_name = "/srv/glusterfs/patilv/Datasets/kitti/visual_odometry/dataset/sequences"
local_file_dir = "image_0/"

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
        
        min_nr += DIFF + DIFF_RANGE
        max_nr -= DIFF + DIFF_RANGE
        
        idx = np.random.randint(min_nr, max_nr)

        out = np.zeros(3, dtype=np.int32) #np.arange(idx - DIFF, idx + DIFF + 1, DIFF)
        out[1] = idx 
        out[0] = idx - DIFF + np.random.randint(-DIFF_RANGE, DIFF_RANGE + 1)
        out[2] = idx + DIFF + np.random.randint(-DIFF_RANGE, DIFF_RANGE + 1)
        
        # check image suitability
        #print("{}{:06}.png".format(local_dir, out[0]))
        frame1 = cv2.imread("{}{:06}.png".format(local_dir, out[0]))
        #print(frame1.shape)
        #plt.imshow(frame1)
        #plt.show()
        prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255

        frame2 = cv2.imread("{}{:06}.png".format(local_dir, out[2]))
        nxt = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs, nxt, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        magind = np.mean(np.sort(mag.flatten())[:-10:-1])

        #mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        #hsv[...,0] = ang*180/np.pi/2
        #hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        #rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        #cv2.imshow('frame2',rgb)
        #k = cv2.waitKey(30) & 0xff
        #if k == 27:
            #break
        #elif k == ord('s'):
            #cv2.imwrite('opticalfb.png',frame2)
            #cv2.imwrite('opticalhsv.png',rgb)
        #prvs = nxt

        #cap.release()
        #cv2.destroyAllWindows()
    
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
    print(r, local_dir, out, min_nr, max_nr)
    for idx in out:
        keys_file.write("{}{:06}.png".format(local_dir, idx))
        keys_file.write(" ")
    keys_file.write("\n")
    labels_file.write("{}\n".format(1))
    
    opt = ([0, 2, 1], [1, 0, 2])
    rnd = np.random.randint(2)   
    out = out[opt[rnd]]
    
    for idx in out:
        keys_file.write("{}{:06}.png".format(local_dir, idx))
        keys_file.write(" ")
    keys_file.write("\n")
    labels_file.write("{}\n".format(0))
    
    keys_file.flush()
    labels_file.flush()
          
        
    #outo = np.random.choice(order, 3, replace = False)
    #outosor = np.sort(outo)
    #outosorrev = outosor[::-1]

    #out = out[outo]

    #for idx in out:
        #keys_file.write("{}{:010}.png".format(local_dir, idx))
        #keys_file.write(" ")
        #keys_file.write("\n")

    #if np.all(outosor == outo) or np.all(outosorrev == outo):
        #labels_file.write("{}\n".format(1))
        #print(r, out, "sorted")
    #else:
        #labels_file.write("{}\n".format(0))
        #print(r, out, "not sorted")
