from __future__ import print_function
import sys
#if len(sys.argv) != 2:
    #print('ERROR: pass base location as argument')
    #sys.exit(1)

from utils.kitti_blob_fetcher import KittiBlobFetcher

base_name = "/srv/glusterfs/patilv/Datasets/kitti/raw/"

fetcher = KittiBlobFetcher(base_name)
fetcher.start()

local_file_dir = "image_02/data/"

dirs = glob.glob(os.path.join(base_name, "*_sync"))
dir_size = []

for d in dirs:
    local_dir = os.path.join(base_name, d, 'extracted', local_file_dir)
    
    dir_size.append(len(os.listdir(local_dir)))
    
for i, d in enumerate(dirs):
    for j in range(dir_size[i]):
        loc = "{}{:010}.png".format(local_dir, j)
    
        fetcher.req(loc, (3, 227, 227))
        im = fetcher.get()
        
        plt.imshow(im)
        plt.show()
