from __future__ import print_function, division
import sys
if len(sys.argv) != 3:
    print('ERROR: pass network prototxt and caffemodel as argument')
    sys.exit(1)

print("Loading libraries")

import matplotlib as mpl
params = {
   'axes.labelsize': 12,
#   'text.fontsize': 8,
   'legend.fontsize': 10,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'text.usetex': False,
   'figure.figsize': [4.5, 4.5]
   }
mpl.rcParams.update(params)
#matplotlib.use("Agg")

ROUNDS = 1 #320//16 #//16 #5000//16 # 20000//16 # 20000 // 16 #1
SAVE_PLOTS = True
SAVE_PERCENTAGES = False
PLOT_FIRST_IMAGE = False
DIFF = 1

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

batch_len = len(net.blobs['label'].data)
labels = np.arange(ROUNDS * batch_len) * DIFF
percentages = np.zeros(ROUNDS * batch_len)
for i in range(ROUNDS):
    if (cnt % 50) == 0: print(cnt)
    forward_data = net.forward()
    
    if first_image is None: 
        first_image = np.copy(net.blobs['im1'].data[0])
        if PLOT_FIRST_IMAGE:
            plt.figure()
            plt.imshow(first_image[0])
    
    percentages[i*batch_len:(i+1)*batch_len] = net.blobs['prob'].data[np.arange(batch_len), net.blobs['label'].data.astype(np.int32)]
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

if SAVE_PERCENTAGES:
    plt.figure(figsize=(10, 10))
    plt.plot(labels, percentages)
    plt.axvline(x=37, color='r', linestyle='-')
    plt.axvline(x=250, color='g', linestyle='-')
    plt.axvline(x=275, color='y', linestyle='-')
    plt.ylim([0, 1])
    plt.xlim([0, ROUNDS*batch_len*DIFF])
    plt.ylabel("estimated probability of correct permutation")
    plt.xlabel("frame index")
    #plt.gca().axes.get_xaxis().set_ticks([])
    plt.savefig("percentages.png")
    plt.show()
    
if not SAVE_PLOTS:
    sys.exit(0)

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
    if 'grp{}_conv2'.format(grp) in net.blobs:
        visualize_weights(net, 'grp{}_conv2'.format(grp), padding = 1, filename = os.path.join(vis_dir, "grp{}_conv2_weights.png".format(grp)))
        
    visualize_activated_regions(net, 'grp{}_conv1'.format(grp), 'data{}'.format(grp), padding = 1, groups = 16, filename = os.path.join(vis_dir, "grp{}_conv1_activated_regions.png".format(grp)))
    if 'grp{}_conv5'.format(grp) in net.blobs:
        visualize_activated_regions(net, 'grp{}_conv5'.format(grp), 'data{}'.format(grp), padding = 1, groups = 16, filename = os.path.join(vis_dir, "grp{}_conv5_activated_regions.png".format(grp)))        

    for i in range(len(net.blobs['data{}'.format(grp)].data) // len(net.blobs['im1'].data)):
        image = net.blobs['data{}'.format(grp)].data[len(net.blobs['im1'].data) * i]
        #print(image.shape)

        plt.figure(figsize=(10, 10))
        plt.axis('off')
        print(image[:, :].shape)
        plt.imshow(image[0,:, :], cmap='gray', interpolation='nearest')
        plt.savefig(os.path.join(vis_dir, "grp{}_image{}.png".format(grp, i+1)), bbox_inches='tight', pad_inches=0)
        plt.close()
        #plt.imsave(image[0,:,:], "/home/kwolters/temp/image1.png");
        visualize_activations(net, 'grp{}_conv1'.format(grp), 'data{}'.format(grp), len(net.blobs['im1'].data) * i, padding = 1, filename = os.path.join(vis_dir, "grp{}_conv1_activations_image{}.png".format(grp, i+1)))
        if 'grp{}_conv5'.format(grp) in net.blobs:
            visualize_activations(net, 'grp{}_conv5'.format(grp), 'data{}'.format(grp), len(net.blobs['im1'].data) * i, padding = 1, filename = os.path.join(vis_dir, "grp{}_conv5_activations_image{}.png".format(grp, i+1)))
    
    grp = grp+1
    
if 'conv2' in net.blobs:
    visualize_weights(net, 'conv2', padding = 1, filename = os.path.join(vis_dir, "conv2_weights.png"))
    
if 'conv5' in net.blobs:
    visualize_activated_regions(net, 'conv5', 'data0', padding = 1, groups = 16, filename = os.path.join(vis_dir, "conv5_activated_regions.png")) 
    
for i in range(len(net.blobs['data0'].data) // len(net.blobs['im1'].data)):
    #plt.imsave(image[0,:,:], "/home/kwolters/temp/image1.png");
    #visualize_activations(net, 'grp{}_conv1'.format(grp), 'data{}'.format(grp), len(net.blobs['im1'].data) * i, padding = 1, filename = os.path.join(vis_dir, "grp{}_conv1_activations_image{}.png".format(grp, i+1)))
    if 'conv5' in net.blobs:
        visualize_activations(net, 'conv5', 'data0', len(net.blobs['im1'].data) * i, padding = 1, filename = os.path.join(vis_dir, "conv5_activations_image{}.png"))

#expout = np.vstack((net.blobs['label'].data, net.blobs['prob'].data[:,1])).transpose()

#plt.show()

print("Done")
