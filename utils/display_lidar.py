"""Example of pykitti.raw usage."""
import itertools
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import pykitti

# Change this to the directory where you store KITTI data
basedir = '/srv/glusterfs/patilv/Datasets/kitti/raw/'

# Specify the dataset to load
date = '2011_09_26'
drive = '0005'

# Load the data. Optionally, specify the frame range to load.
# Passing imformat='cv2' will convert images to uint8 and BGR for
# easy use with OpenCV.
# dataset = pykitti.raw(basedir, date, drive)
dataset = pykitti.raw(basedir, date, drive, frames=range(0, 5))

# dataset.calib:      Calibration data are accessible as a named tuple
# dataset.timestamps: Timestamps are parsed into a list of datetime objects
# dataset.oxts:       Generator to load OXTS packets as named tuples
# dataset.camN:       Generator to load individual images from camera N
# dataset.gray:       Generator to load monochrome stereo pairs (cam0, cam1)
# dataset.rgb:        Generator to load RGB stereo pairs (cam2, cam3)
# dataset.velo:       Generator to load velodyne scans as [x,y,z,reflectance]

# Grab some data
first_gray = dataset.get_gray(0)
first_cam1 = np.array(dataset.get_cam1(0))
first_cam2 = np.array(dataset.get_cam2(0))
first_velo = np.array(dataset.get_velo(0))

second_pose = next(iter(itertools.islice(dataset.oxts, 1, None))).T_w_imu
#first_gray = next(iter(dataset.gray))
#first_cam1 = next(iter(dataset.cam1))
#first_rgb = next(iter(dataset.rgb))
#first_cam2 = next(iter(dataset.cam2))
#third_velo = next(iter(itertools.islice(dataset.velo, 2, None)))

#first_velo = next(iter(dataset.velo))

# Display some of the data
np.set_printoptions(precision=4, suppress=True)
print('\nDrive: ' + str(dataset.drive))
print('\nFrame range: ' + str(dataset.frames))

print('\nIMU-to-Velodyne transformation:\n' + str(dataset.calib.T_velo_imu))
print('\nGray stereo pair baseline [m]: ' + str(dataset.calib.b_gray))
print('\nRGB stereo pair baseline [m]: ' + str(dataset.calib.b_rgb))

print('\nFirst timestamp: ' + str(dataset.timestamps[0]))
print('\nSecond IMU pose:\n' + str(second_pose))

conv = np.matmul(dataset.calib.P_rect_00, np.matmul(dataset.calib.R_rect_00, dataset.calib.T_cam0_velo))
data = np.c_[first_velo[:, 0:3], np.ones(first_velo.shape[0])]

res = np.transpose(np.matmul(conv, np.transpose(data)))
res[res[:, 2] < 1e-4, 2] = 1e-4

lid_img = res[:, 0:2] / res[:, 2, None]

f, ax = plt.subplots(2, 2, figsize=(15, 5))
ax[0, 0].imshow(first_gray[0], cmap='gray')
ax[0, 0].set_title('Left Gray Image (cam0)')

ax[0, 1].imshow(first_cam1, cmap='gray')
ax[0, 1].set_title('Right Gray Image (cam1)')

ax[1, 0].imshow(first_cam2)
ax[1, 0].set_title('Left RGB Image (cam2)')

print(np.min(lid_img[:, 0]), np.min(lid_img[:, 0]), np.min(lid_img[:, 1]), np.max(lid_img[:, 1]))
ax[1, 1].scatter(lid_img[:,0], lid_img[:,1], c=res[:, 2]) #first_velo[:, 3]
ax[1, 1].set_xlim([0, first_cam2.shape[1]])
ax[1, 1].set_ylim([first_cam2.shape[0], 0])
ax[1, 1].set_title('Velodyne image (velo)')

plt.show()
