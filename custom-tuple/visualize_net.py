from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
 
def visualize_weights(net, layer_name, padding=4, color=False, layer=-1, filename=''):
    data = np.copy(net.params[layer_name][0].data)
    # N is the total number of convolutions
    N = data.shape[0] #*data.shape[1]

    if color: assert(data.shape[1] == 3)
    elif layer == -1: N *= data.shape[1]
    else: assert(0 <= layer < data.shape[1])
    
    # Ensure the resulting image is square
    filters_per_row = int(np.ceil(np.sqrt(N)))
    # Assume the filters are square
    filter_size = data.shape[2]
    # Size of the result image including padding
    result_size = filters_per_row*(filter_size + padding) - padding
    # Initialize result image to all zeros
    if color:
        result = np.zeros((result_size, result_size, 3))
    else:
        result = np.zeros((result_size, result_size))
 
    # Tile the filters into the result image
    filter_x = 0
    filter_y = 0
    for n in range(data.shape[0]):
        if color or layer != -1:
            if filter_x == filters_per_row:
                filter_y += 1
                filter_x = 0
            for i in range(filter_size):
                for j in range(filter_size):
                    if color: result[filter_y*(filter_size + padding) + i, filter_x*(filter_size + padding) + j, :] = data[n, :, i, j]
                    else: result[filter_y*(filter_size + padding) + i, filter_x*(filter_size + padding) + j] = data[n, layer, i, j]
            
            filter_x += 1
        elif layer == -1:
            for c in range(data.shape[1]):
                if filter_x == filters_per_row:
                    filter_y += 1
                    filter_x = 0
                for i in range(filter_size):
                    for j in range(filter_size):
                        result[filter_y*(filter_size + padding) + i, filter_x*(filter_size + padding) + j] = data[n, c, i, j]
        
                filter_x += 1
 
    # Normalize image to 0-1
    min = result.min()
    max = result.max()
    result = (result - min) / (max - min)
 
    # Plot figure
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    if color:
        plt.imshow(result, interpolation='nearest')
    else:
        plt.imshow(result, cmap="gray", interpolation='nearest')
 
    # Save plot if filename is set
    if filename != '':
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
    plt.close()
    
def visualize_activations(net, layer_name, image_layer_name, image_idx, padding=4, box_size=20, filename='', groups = 1):    
    # The parameters are a list of [weights, biases]
    data = net.blobs[layer_name].data
        
    N = data.shape[1] #*data.shape[1]
    assert image_idx < data.shape[0]
    
    # Ensure the resulting image is square
    filters_per_row = int(np.ceil(np.sqrt(N)))
    # Assume the filters are square
    # NOTE: assumes same size X/Y
    filter_size = data.shape[2]
    # Size of the result image including padding
    result_size = filters_per_row*(filter_size + padding) - padding
    # Initialize result image to all zeros
    result = np.zeros((result_size, result_size, 3))
 
    # Tile the filters into the result image
    filter_x = 0
    filter_y = 0
    
    for n in range(data.shape[1]):    
        if filter_x == filters_per_row:
            filter_y += 1
            filter_x = 0
        for i in range(filter_size):
            for j in range(filter_size):
                result[filter_y*(filter_size + padding) + i, filter_x*(filter_size + padding) + j, :] = data[image_idx, n, i, j]
        
        filter_x += 1
 
    # Normalize image to 0-1
    min = result.min()
    max = result.max()
    result = (result - min) / (max - min)
 
    # Plot figure
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(result[:, :, 0], cmap='gray', interpolation='nearest')
 
    # Save plot if filename is set
    if filename != '':
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
    plt.close()

def visualize_activated_regions(net, layer_name, image_layer_name, padding=4, box_size=20, amount=5, filename='', groups = 1):    
    # The parameters are a list of [weights, biases]
    data = net.blobs[layer_name].data
    image = net.blobs[image_layer_name].data
        
    N = data.shape[0] #*data.shape[1]
    
    assert (N % groups) == 0
    orig = idxs = np.arange(0, N, groups)
    for i in range(0, groups-1):
        idxs = np.concatenate((idxs, orig + i + 1))
     
    data = data[idxs]
    image = image[idxs]
    
    # Ensure the resulting image is square
    filters_per_row = int(np.ceil(np.sqrt(N)))
    # Assume the filters are square
    # NOTE: assumes same size X/Y
    filter_size = image.shape[2]
    # Size of the result image including padding
    result_size = filters_per_row*(filter_size + padding) - padding
    # Initialize result image to all zeros
    result = np.zeros((result_size, result_size, 3))
 
    # Tile the filters into the result image
    filter_x = 0
    filter_y = 0
    
    for n in range(data.shape[0]):
        srt = np.argsort(data[n], axis = None)[::-1]
        idxs = list(np.unravel_index(srt, data[n].shape))
        
        idxs[1] *= (filter_size // data.shape[2])
        idxs[2] *= (filter_size // data.shape[2])
        
        #print(idxs[1][0], idxs[2][0])
        for i in range(amount):
            image[n, 0, np.max([idxs[1][i]-box_size//2, 0]):idxs[1][i]+box_size//2, np.max([0, idxs[2][i]-box_size//2]):idxs[2][i]+box_size//2] = 1 - i/amount
        if image.shape[1] == 3:
            for x in range(1, 2+1):
                image[n, x, np.max([idxs[1][i]-box_size//2, 0]):idxs[1][i]+box_size//2, np.max([0, idxs[2][i]-box_size//2]):idxs[2][i]+box_size//2] = 0
    
        if filter_x == filters_per_row:
            filter_y += 1
            filter_x = 0
        for i in range(filter_size):
            for j in range(filter_size):
                result[filter_y*(filter_size + padding) + i, filter_x*(filter_size + padding) + j, :] = image[n, :, i, j]
        
        filter_x += 1
 
    # Normalize image to 0-1
    min = result.min()
    max = result.max()
    result = (result - min) / (max - min)
 
    # Plot figure
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(result[:, :, 0], cmap='gray', interpolation='nearest')
 
    # Save plot if filename is set
    if filename != '':
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
    plt.close()
    
## FIXME: updates network?
#def visualize_saliency(net, image_layer_name, num_outputs, image_idx, label_idx, filename=''):    
    #labels = np.zeros((1, num_outputs))
    #labels[0, label_idx] = 1
    
    #print(net.outputs[0])
    #backward_data = net.backward()
    #print(backward_data)
    #delta = backward_data[image_layer_name]
    
    #delta = delta - delta.min()           # Subtract min
    #delta = delta / delta.max()           # Normalize by dividing by max 
    #saliency = np.amax(delta,axis=1)      # Find max across RGB channels 

    #plt.figure(figsize=(10, 10))
    #plt.subplot(1,2,1)
    #plt.imshow (net.blobs[image_layer_name].data[0])
    #plt.axis('off')

    #plt.subplot(1,2,2)
    #plt.imshow (saliency[0,:,:],cmap='copper')
    #plt.axis('off')

    #plt.tight_layout()
    #plt.subplots_adjust(wspace=0, hspace=0)
    #if filename != '':
        #plt.savefig(filename)
        
    #plt.show()

            
