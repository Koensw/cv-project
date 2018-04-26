from __future__ import print_function
import sys
if len(sys.argv) != 2:
    print('ERROR: pass solver prototxt as argument')
    sys.exit(1)

print("Loading caffe")

import os
import caffe

## ADD PATHS
def addPath(path):
    if path not in sys.path:
        sys.path.insert(0, path)
        print(('added {}'.format(path)))

addPath('.')
addPath('./layers')

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

## LOAD SOLVER
print("Preparing solver")

solver_path = sys.argv[1]
solver = caffe.SGDSolver(solver_path)

print("Start training")

#numIter = 100000;
#logStep = 20;
#snapshotIter = 20000;
#trainer = WatchTrainer(solverPath, solver);
#trainer.init_logging();
#trainer.train_model(numIter, logStep, snapshotIter, track_indiv_loss=True);
        
