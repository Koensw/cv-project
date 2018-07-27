from __future__ import print_function
from itertools import permutations
import numpy as np
import os
import glob
import cv2
import re
import sys
import matplotlib.pyplot as plt

ROUNDS = 100
DIFF = 5
DIFF_RANGE = 0

MODE = 'all'

MIN_DIR = int(sys.argv[1])
MAX_DIR = int(sys.argv[2])

SIZE = 4
SAME = 4

#array = np.arange(min_nr, max_nr+1)
order = np.arange(0, SIZE)
orders = np.array(list(permutations(order)))

labels_file = open("odo_image_labels.txt", 'w')
keys_file = open("odo_image_keys.txt", 'w')

select_name = "select.txt"
select_file = open(select_name)
select = {}
for s in select_file.readlines():
    s = s.split()
    idx = int(s[0])
    nr = int(s[1])
    if idx not in select: select[idx] = []
    select[idx].append(nr)

#base_name = "/srv/glusterfs/patilv/Datasets/kitti/raw/extracted"
##local_file_dir = "2011_09_26_drive_0005_sync/image_02/data/"
#local_file_dir = "image_02/data/"
base_name = "/srv/glusterfs/patilv/Datasets/kitti/visual_odometry/dataset/sequences"
local_file_dir = "image_0/"

dirs = glob.glob(os.path.join(base_name, "*"))
dir_size = []
total_size = 0
for d in dirs[:]:
    base_dir = os.path.basename(d)
    if not (MIN_DIR <= int(base_dir) <= MAX_DIR):
        dirs.remove(d)
    
for d in dirs:
    int_dir = int(os.path.basename(d))
    if int_dir not in select: select[int_dir] = []  
    local_dir = os.path.join(base_name, d, local_file_dir)
    #len(os.listdir(local_dir))
    dir_size.append(len(select[int_dir]))
    total_size += dir_size[-1]
    
size_to_idx = np.zeros(total_size, dtype=np.int32)
cur_idx = 0
for i in range(len(dirs)):
    #for j in range(dir_size[i]):    
    size_to_idx[cur_idx:cur_idx+dir_size[i]] = i
    cur_idx += dir_size[i]

for r in range(ROUNDS):
    # prepare
    dir_idx = size_to_idx[np.random.randint(total_size)]
    local_dir = os.path.join(base_name,  dirs[dir_idx], local_file_dir)
    
    min_nr = 0
    max_nr = dir_size[dir_idx]
    
    #print(int(os.path.basename(dirs[dir_idx])), len(select[int(os.path.basename(dirs[dir_idx]))]), max_nr)
    idx = select[int(os.path.basename(dirs[dir_idx]))][np.random.randint(min_nr, max_nr)]

    files = np.zeros(SIZE, dtype=np.int32) #np.arange(idx - DIFF, idx + DIFF + 1, DIFF)
    files[0] = idx 
    for i in range(1, SIZE):
        files[i] = files[i-1] + DIFF + np.random.randint(-DIFF_RANGE, DIFF_RANGE + 1)
        
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
 
print(total_size)