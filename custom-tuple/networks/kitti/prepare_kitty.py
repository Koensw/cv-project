from __future__ import print_function
import numpy as np

diff = 10

min_nr = 0
max_nr = 153

min_nr += diff
max_nr -= diff

#array = np.arange(min_nr, max_nr+1)
order = np.arange(0, 3)

labels_file = open("kitti_image_labels.txt", 'w')
keys_file = open("kitti_image_keys.txt", 'w')

base_name = "/srv/glusterfs/patilv/Datasets/kitti/raw/extracted"
local_file_name = "2011_09_26_drive_0005_sync/image_02/data/"

for i in range(min_nr, max_nr+1):
    out = np.arange(i - diff, i + diff + 1, diff)

    outo = np.random.choice(order, 3, replace = False)
    outosor = np.sort(outo)
    outosorrev = outosor[::-1]

    out = out[outo]
    
    for idx in out:
        keys_file.write("{}{:010}.png".format(local_file_name, idx))
        keys_file.write(" ")
    keys_file.write("\n")
    
    if np.all(outosor == outo) or np.all(outosorrev == outo):
        labels_file.write("{}\n".format(1))
        print(out, "sorted")
    else:
        labels_file.write("{}\n".format(0))
        print(out, "not sorted")
