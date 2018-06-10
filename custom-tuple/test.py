from __future__ import print_function, division
import sys
if len(sys.argv) != 3:
    print('ERROR: pass network prototxt and caffemodel as argument')
    sys.exit(1)

print("Loading libraries")

import matplotlib
#matplotlib.use("Agg")

ROUNDS = 10000 // 16

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
#net.forward()

print("Calculating score...")
np.set_printoptions(precision=2)

total = 0
cnt = 0
first_image = None
for i in range(ROUNDS):
    if (cnt % 50) == 0: print(cnt)
    forward_data = net.forward()
    #if first_image is None: first_image = np.copy(net.blobs['im1'].data[0])
    #else:
        #stop = False
        #for img in net.blobs['im1'].data:
            #if np.all(img == first_image): 
                #stop = True
                #break
        
        #if stop:
            #print('STOP:', cnt)
            #break
            
        #print(first_image == net.blobs['im1'].data[:])
    
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
net.force_backward = True
net.backward() # saliency maps

vis_dir = os.path.join(os.path.dirname(sys.argv[1]), '../vis')
grp = 0
while True:
    print(grp)
    if 'grp{}_conv1'.format(grp) not in net.blobs:
        break
    
    visualize_weights(net, 'grp{}_conv1'.format(grp), color = False, layer = 0, padding = 1, filename = os.path.join(vis_dir, "grp{}_conv1_weights.png".format(grp)))

    #visualize_saliency(net, 'im_concat', 24, 0, 0)
    #visualize_weights(net, 'conv1', color = False, layer = 1, padding = 1)
    #visualize_weights(net, 'conv1', color = False, layer = 2, padding = 1)
    visualize_weights(net, 'grp{}_conv2'.format(grp), padding = 1, filename = os.path.join(vis_dir, "grp{}_conv2_weights.png".format(grp)))
    visualize_activated_regions(net, 'grp{}_conv1'.format(grp), 'data{}'.format(grp), padding = 1, groups = 16, filename = os.path.join(vis_dir, "grp{}_conv1_activated_regions.png".format(grp)))
    if 'grp{}_conv5'.format(grp) in net.blobs:
        visualize_activated_regions(net, 'grp{}_conv5'.format(grp), 'data{}'.format(grp), padding = 1, groups = 16, filename = os.path.join(vis_dir, "grp{}_conv5_activated_regions.png".format(grp)))        

    for i in range(len(net.blobs['data{}'.format(grp)].data) // len(net.blobs['im1'].data)):
        image = net.blobs['data{}'.format(grp)].data[len(net.blobs['im1'].data) * i]
        #print(image.shape)

        plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.imshow(image[0,:, :], cmap='gray', interpolation='nearest')
        plt.savefig(os.path.join(vis_dir, "grp{}_image{}.png".format(grp, i+1)), bbox_inches='tight', pad_inches=0)
        plt.close()
        #plt.imsave(image[0,:,:], "/home/kwolters/temp/image1.png");
        visualize_activations(net, 'grp{}_conv1'.format(grp), 'data{}'.format(grp), len(net.blobs['im1'].data) * i, padding = 1, filename = os.path.join(vis_dir, "grp{}_conv1_activations_image{}.png".format(grp, i+1)))
        if 'grp{}_conv5'.format(grp) in net.blobs:
            visualize_activations(net, 'grp{}_conv5'.format(grp), 'data{}'.format(grp), len(net.blobs['im1'].data) * i, padding = 1, filename = os.path.join(vis_dir, "grp{}_conv5_activations_image{}.png".format(grp, i+1)))
    
    grp = grp+1

#expout = np.vstack((net.blobs['label'].data, net.blobs['prob'].data[:,1])).transpose()

#plt.show()

print("Done")
