#/usr/bin/env python
import numpy as np
import math
import os

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
 
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
 
    assert(isRotationMatrix(R))
     
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])

base_dir = "/srv/glusterfs/patilv/Datasets/kitti/visual_odometry/dataset/poses/"
ret_file = open("select.txt", 'w')
poses = sorted(os.listdir(base_dir))
print(poses)
for pose_file in poses:
    fl = open(os.path.join(base_dir, pose_file))
    #fl = open("/srv/glusterfs/patilv/Datasets/kitti/visual_odometry/dataset/poses/00.txt")

    idx = 0
    lst = 0
    lines = fl.readlines()
    file_len = len(lines)
    last_corner = False
    for s in lines:
        sl = s.split()
        mat = np.zeros((3, 4))
        mat[0, :] = sl[0:4]
        mat[1, :] = sl[4:8]
        mat[2, :] = sl[8:12]
    
        cur = rotationMatrixToEulerAngles(mat[:, :3])[1]
        if lst-cur > 0.01:
            if not last_corner:
                ret_file.write(bytes(int(pose_file[:2])))
                ret_file.write(' ')
                ret_file.write(bytes(idx))
                ret_file.write('\n')
                last_corner = True
        else: last_corner = False
            #print(idx, lst - cur)
        lst = cur
        idx = idx + 1
        
        #print(idx, file_len)
        if idx + 22 > file_len: 
            print(pose_file, idx, file_len)
            break
        #break
ret_file.close()