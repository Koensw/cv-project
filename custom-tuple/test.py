from __future__ import print_function, division
import sys
if len(sys.argv) != 3:
    print('ERROR: pass network prototxt and caffemodel as argument')
    sys.exit(1)

print("Loading libraries")

import matplotlib
#matplotlib.use("Agg")

ROUNDS = 1 #5000//8

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

print("Calculating score...")
np.set_printoptions(precision=2)

total = 0
cnt = 0
for i in range(ROUNDS):
    #if i != 0 and (i % 100) == 0: break
    #print(i)
    forward_data = net.forward()
    #print(net.blobs['label'].data, np.argmax(net.blobs['prob'].data[:,:], axis = 1))
    expout = np.vstack((net.blobs['label'].data, np.argmax(net.blobs['prob'].data[:,:], axis = 1), np.max(net.blobs['prob'].data[:,:], axis = 1))).transpose()
    #print(expout)
    #maxeo = (expout > 0.5)
    total += np.sum(expout[:,0] == expout[:,1]) / len(expout)
    cnt = cnt + 1

print('Loss:', net.blobs['loss'].data)
expout[expout < 0.01] = 0
print('Exp / Out:\n', expout)

maxeo = (expout > 0.5)
print('Accuracy', total / cnt)

print("Visualizing...")
net.backward() # saliency maps

visualize_weights(net, 'conv1', color = False, layer = 0, padding = 1, filename = "/home/kwolters/temp/conv1_weights.png")

#visualize_saliency(net, 'im_concat', 24, 0, 0)
#visualize_weights(net, 'conv1', color = False, layer = 1, padding = 1)
#visualize_weights(net, 'conv1', color = False, layer = 2, padding = 1)
visualize_weights(net, 'conv2', padding = 1, filename = "/home/kwolters/temp/conv2_weights.png")
visualize_activated_regions(net, 'conv1', 'im_concat', padding = 1, groups = 16, filename = "/home/kwolters/temp/conv1_activated_regions.png")
visualize_activated_regions(net, 'conv5', 'im_concat', padding = 1, groups = 16, filename = "/home/kwolters/temp/conv5_activated_regions.png")

for i in range(3):
    image = net.blobs['im_concat'].data[16*i]
    #print(image.shape)

    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image[0,:, :], cmap='gray', interpolation='nearest')
    plt.savefig("/home/kwolters/temp/image{}.png".format(i+1), bbox_inches='tight', pad_inches=0)
    #plt.imsave(image[0,:,:], "/home/kwolters/temp/image1.png");
    visualize_activations(net, 'conv1', 'im_concat', 16*i, padding = 1, filename = "/home/kwolters/temp/conv1_activations_image{}.png".format(i+1))
    visualize_activations(net, 'conv5', 'im_concat', 16*i, padding = 1, filename = "/home/kwolters/temp/conv5_activations_image{}.png".format(i+1))

#expout = np.vstack((net.blobs['label'].data, net.blobs['prob'].data[:,1])).transpose()

#plt.show()

print("Done")
