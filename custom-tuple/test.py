from __future__ import print_function
import sys
if len(sys.argv) != 3:
    print('ERROR: pass network prototxt and caffemodel as argument')
    sys.exit(1)

print("Loading libraries")

import matplotlib
#matplotlib.use("Agg")

import numpy as np
import os
import caffe

## ADD PATHS
def addPath(path):
    if path not in sys.path:
        sys.path.insert(0, path)
	#path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
	#sys.path.append(path)
        print(('added {}'.format(path)))

addPath('.')
addPath('./layers')

caffe.set_mode_cpu()

## BUILD NET
print("Building net")

net = caffe.Net(sys.argv[1], sys.argv[2], caffe.TEST)
net.forward()

expout = np.vstack((net.blobs['label'].data, net.blobs['prob'].data[:,1])).transpose()

np.set_printoptions(precision=2)
print('Loss:', net.blobs['loss'].data)
print('Exp / Out:\n', expout)

maxeo = (expout > 0.5)
print('Accuracy', np.sum(maxeo[:,0] == maxeo[:,1]) / len(expout))

print("Done")

        
