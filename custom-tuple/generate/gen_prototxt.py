#!/usr/bin/env python

import sys
import yaml
import os

if len(sys.argv) != 2:
    print("Pass YAML configuration file")
    sys.exit(1)
    
config = {}
with open(sys.argv[1], 'r') as stream:
    try:
        config = yaml.load(stream)
    except yaml.YAMLError as exc:
        print("Failed to read YAML")
        print(exc)
        sys.exit(1)
    
import caffe
from caffe import layers
from caffe import params
from caffe.proto import caffe_pb2
from google.protobuf import text_format

### NETWORK

# Create network file
net = caffe.proto.caffe_pb2.NetParameter()

output_dir = os.path.join(config['output_dir'], config['name'])

# Insert data layers
train_layer = net.layer.add()
train_layer.name = "input"
for i in range(config['num_inputs']):
    train_layer.top.append("im{}".format(i+1))
train_layer.top.append("label")
    
train_layer.type = "Python"
train_layer.python_param.module = "input_image_tuple"
train_layer.python_param.layer = "InputImageTuple"
include_param = caffe_pb2.NetStateRule()
include_param.phase = caffe_pb2.TRAIN
train_layer.include.extend([include_param])

python_params = {'keys_file': config['train_keys'],
                 'label_file': config['train_labels'],
                 'base_dir': config['data_dir'],
                 'direct': True,
                 'shuffle': True,
                 'saliency_dir': os.path.join(output_dir, 'saliency'),
                 'data_shape': [1, 227, 227],
                 'batch_size': config['batch_size']
                 }
train_layer.python_param.param_str = yaml.dump(python_params)

test_layer = net.layer.add()
test_layer.name = "input"
for i in range(config['num_inputs']):
    test_layer.top.append("im{}".format(i+1))
test_layer.top.append("label")
    
test_layer.type = "Python"
test_layer.python_param.module = "input_image_tuple"
test_layer.python_param.layer = "InputImageTuple"
include_param = caffe_pb2.NetStateRule()
include_param.phase = caffe_pb2.TEST
test_layer.include.extend([include_param])

test_layer.python_param.param_str = yaml.dump(python_params)

concat_layer = net.layer.add()
concat_layer.name = "im_concat"
concat_layer.type = "Concat"
concat_layer.top.append("data")
for i in range(config['num_inputs']):
    concat_layer.bottom.append("im{}".format(i+1))
concat_layer.concat_param.axis = 0

# Load backbone
backbone_file = open(config['backbone'], "r")
if not backbone_file:
    raise Exception("Could not open backbone")
text_format.Merge(str(backbone_file.read()), net)
backbone_file.close()
net.layer[3].bottom[0] = "data"

# Load custom backend
network_file = open(config['network'], "r")
if not network_file:
    raise Exception("Could not open backbone")
text_format.Merge(str(network_file.read()), net)
network_file.close()

# Final output layer
network_final_top = net.layer[-1].top[0]

output_layer = net.layer.add()
output_layer.name = "output"
output_layer.type = "InnerProduct"
output_layer.bottom.append(network_final_top)
output_layer.top.append("output")
output_layer.inner_product_param.num_output = config['num_outputs']
output_layer.inner_product_param.weight_filler.type = "gaussian"
output_layer.inner_product_param.weight_filler.std = 0.005
output_layer.inner_product_param.bias_filler.type = "constant"
output_layer.inner_product_param.bias_filler.value = 1

# Metrics
loss_layer = net.layer.add()
loss_layer.name = "loss"
loss_layer.type = "SoftmaxWithLoss"
loss_layer.bottom.append("output")
loss_layer.bottom.append("label")
loss_layer.top.append("loss")

accuracy_layer = net.layer.add()
accuracy_layer.name = "accuracy"
accuracy_layer.type = "Accuracy"
accuracy_layer.bottom.append("output")
accuracy_layer.bottom.append("label")
accuracy_layer.top.append("acc")

prob_layer = net.layer.add()
prob_layer.name = "prob"
prob_layer.type = "Softmax"
prob_layer.bottom.append("output")
prob_layer.top.append("prob")

#test = layers.Accuracy(train_data, ['fc', 'label'], 0)
#net.layer.extend([train_layer])
if config['visualize']:
    vis_layer = net.layer.add()
    vis_layer.name = "visualize"
    vis_layer.type = "Python"
    vis_layer.python_param.module = "visualize"
    vis_layer.python_param.layer = "Visualize"
    vis_params = {'types': ['img'] * config['num_inputs'] + ['txt', 'txt'], 'plot_iter': config['vis_iter'], 'save_path': os.path.join(output_dir, 'vis'), 'gray': True}
    vis_layer.python_param.param_str = yaml.dump(vis_params)
    for i in range(config['num_inputs']):
        vis_layer.bottom.append("im{}".format(i+1))
    vis_layer.bottom.append("label")
    vis_layer.bottom.append("prob")
    include_param = caffe_pb2.NetStateRule()
    include_param.phase = caffe_pb2.TRAIN
    vis_layer.include.extend([include_param])
   
#print(net)

### SOLVER ###
solver = caffe.proto.caffe_pb2.SolverParameter()
solver.net = os.path.join(output_dir, 'networks', 'net.prototxt')
solver.base_lr = config['base_lr']
solver.lr_policy = "multistep"
solver.momentum = 0.9
solver.regularization_type = "L2"
solver.weight_decay = config['weight_decay']
solver.gamma = 0.1
for step in config['step_value']:
    solver.stepvalue.append(step)
solver.display = 10
solver.iter_size = config['iter_size']
solver.test_iter.append(100)
solver.test_interval = 100
solver.snapshot_prefix = os.path.join(output_dir, 'models')

network_dir = os.path.join(output_dir, 'networks')
if not os.path.exists(network_dir): os.makedirs(network_dir)

net_file = open(os.path.join(network_dir, "net.prototxt"), 'w')
net_file.write(str(net))
solver_file = open(os.path.join(network_dir, "solver.prototxt"), 'w')
solver_file.write(str(solver))

# net: "networks/opn_train_full.prototxt"
#snapshot_prefix: "opn_snapshot"
#base_lr: 0.01
#momentum: 0.9 
#regularization_type: "L2"
#weight_decay: 0.005
#lr_policy: "multistep"
#gamma: 0.1
#stepvalue: [4000, 8000, 16000]
#display: 5
#snapshot: 0
#solver_mode: GPU
#iter_size: 8
#test_iter : 100
#test_interval : 100
