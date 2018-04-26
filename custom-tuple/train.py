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

max_iters = 10000
snapshot_iter = 100
log_iter = 1

assert (snapshot_iter % log_iter) == 0

def snapshot(solver, snap_path):
    net = solver.net

    if not os.path.exists(snap_path):
        os.makedirs(snap_path);

    filename = 'snapshot_' + 'iter_{:d}'.format(solver.iter) + '.caffemodel';
    filename = os.path.join(snap_path, filename);
    net.save(str(filename))
    return filename;

print('Checking snapshot')
filename = snapshot(solver, "snapshots")

print("Start training")
while solver.iter < max_iters:
    if (solver.iter % snapshot_iter) == 0:
        snapshot(solver, "snapshots")
    
    print("Stepping...")
    solver.step(log_iter)

print("Finished, saving final model")
solver.net.save("final_model.caffemodel")

#numIter = 100000;
#logStep = 20;
#snapshotIter = 20000;
#trainer = WatchTrainer(solverPath, solver);
#trainer.init_logging();
#trainer.train_model(numIter, logStep, snapshotIter, track_indiv_loss=True);
        
