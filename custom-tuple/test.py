from __future__ import print_function, division
import sys
if len(sys.argv) != 3:
    print('ERROR: pass network prototxt and caffemodel as argument')
    sys.exit(1)

print("Loading libraries")

import matplotlib
matplotlib.use("Agg")

ROUNDS = 5000//8

import numpy as np
import os
import caffe

from visualize_net import * 

## ADD PATHS
def addPath(path):
    if path not in sys.path:
        sys.path.insert(0, path)
        print(('added {}'.format(path)))

addPath('.')
addPath('./layers')
addPath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))

## CHECK RUN MODE
if "SGE_GPU" in os.environ and os.environ["SGE_GPU"] != "":
    print("RUNNING IN GPU MODE")
    gpu = os.getenv("SGE_GPU")
    print("GPU: ", int(gpu))
    caffe.set_device(int(gpu))
    caffe.set_mode_gpu()
else:
    print("RUNNING IN CPU MODE")
    caffe.set_mode_cpu()
    
## BUILD NET
print("Building net")

net = caffe.Net(sys.argv[1], 1, weights=sys.argv[2])
net.forward()

visualize_weights(net, 'conv1', color = False, padding = 1)
visualize_weights(net, 'conv5', padding = 1)
visualize_activations(net, 'conv2', 'data')

#expout = np.vstack((net.blobs['label'].data, net.blobs['prob'].data[:,1])).transpose()

np.set_printoptions(precision=2)

total = 0
cnt = 0
for i in range(ROUNDS):
    if i != 0 and (i % 100) == 0: break
    print(i)
    net.forward()
    expout = np.vstack((net.blobs['label'].data, net.blobs['prob'].data[:,1])).transpose()
    maxeo = (expout > 0.5)
    total += np.sum(maxeo[:,0] == maxeo[:,1]) / len(expout)
    cnt = cnt + 1

print('Loss:', net.blobs['loss'].data)
expout[expout < 0.01] = 0
print('Exp / Out:\n', expout)

maxeo = (expout > 0.5)
print('Accuracy', total / cnt)

print("Done")
