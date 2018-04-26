import _init_paths
import caffe
import numpy as np
import code
import traceback as tb
import os
from model_training_utils import WatchTrainer
from caffe import layers as L

if "SGE_GPU" in os.environ and os.environ["SGE_GPU"] != "":
	print("RUNNING IN GPU MODE")
	gpu = os.getenv("SGE_GPU")
	print("GPU: ", int(gpu))
	caffe.set_device(int(gpu))
	caffe.set_mode_gpu()
else:
	print("RUNNING IN CPU MODE")
	caffe.set_mode_cpu()

# set base path
basePath = "ucf"
import sys
if len(sys.argv) > 1:
	basePath = sys.argv[1]

print("PATH: ", basePath)

solverPath = basePath + '/tuple_solver.prototxt'
solver = caffe.SGDSolver(solverPath)

numIter = 100000;
logStep = 20;
snapshotIter = 5000;
trainer = WatchTrainer(solverPath, solver);
trainer.init_logging();
trainer.train_model(numIter, logStep, snapshotIter, track_indiv_loss=True);
