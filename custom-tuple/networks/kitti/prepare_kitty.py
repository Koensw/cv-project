from __future__ import print_function
import numpy as np
import os
import glob
import cv2

rounds = 10000

diff = 5

#array = np.arange(min_nr, max_nr+1)
order = np.arange(0, 3)

labels_file = open("kitti_image_labels.txt", 'w')
keys_file = open("kitti_image_keys.txt", 'w')

base_name = "/srv/glusterfs/patilv/Datasets/kitti/raw/extracted"
#local_file_dir = "2011_09_26_drive_0005_sync/image_02/data/"
local_file_dir = "image_02/data/"

dirs = glob.glob(os.path.join(base_name, "*_sync"))
dir_size = []

for d in dirs:
    local_dir = os.path.join(base_name, d, local_file_dir)
    
    dir_size.append(len(os.listdir(local_dir)))

for r in range(rounds):
    # prepare
    while True:
        dir_idx = np.random.randint(len(dirs))
        local_dir = os.path.join(base_name,  dirs[dir_idx], local_file_dir)
        
        min_nr = 0
        max_nr = dir_size[dir_idx]
        
        min_nr += diff
        max_nr -= diff
        
        idx = np.random.randint(min_nr, max_nr)

        out = np.arange(idx - diff, idx + diff + 1, diff)
        
        # check image suitability
        frame1 = cv2.imread("{}{:010}.png".format(local_dir, out[0]))
        prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255

        frame2 = cv2.imread("{}{:010}.png".format(local_dir, out[2]))
        nxt = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs, nxt, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        magind = np.mean(np.sort(mag.flatten())[:-10:-1])

        if magind < 30:
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
    outo = np.random.choice(order, 3, replace = False)
    outosor = np.sort(outo)
    outosorrev = outosor[::-1]

    out = out[outo]

    for idx in out:
        keys_file.write("{}{:010}.png".format(local_dir, idx))
        keys_file.write(" ")
        keys_file.write("\n")

    if np.all(outosor == outo) or np.all(outosorrev == outo):
        labels_file.write("{}\n".format(1))
        print(r, out, "sorted")
    else:
        labels_file.write("{}\n".format(0))
        print(r, out, "not sorted")
