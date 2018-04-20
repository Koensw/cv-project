import numpy as np

rounds = 100

min_nr = 100
max_nr = 153

array = np.arange(min_nr, max_nr+1)

labels_file = open("train02_image_labels.txt", 'w')
keys_file = open("train02_image_keys.txt", 'w')

base_name = "/home/koen/uni/sp/kitti/"
local_file_name = "2011_09_26/2011_09_26_drive_0005_sync/image_02/data/"

for i in range(rounds):
    out = np.random.choice(array, 3, replace = False)
    outsor = np.sort(out)
    
    for idx in out:
        keys_file.write("{}{:010}".format(local_file_name, idx))
        keys_file.write("\t")
    keys_file.write("IGN\n")
    
    if np.any(outsor == out):
        labels_file.write("{}\n".format(1))
        print(outsor, "sorted")
    else:
        labels_file.write("{}\n".format(0))
        print(out, "not sorted")
