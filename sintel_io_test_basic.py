import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('../')
import sintel_io as sio

# Test and display some real data
folder_name='alley_1'
frame_no = 1 #smaller than 10
DEPTHFILE = '/cluster/scratch/takmaza/CVL/MPI-Sintel-complete/training/depth/'+folder_name+'/frame_000'+str(frame_no)+'.dpt'
CAMFILE = '/cluster/scratch/takmaza/CVL/MPI-Sintel-complete/training/camdata_left/'+folder_name+'/frame_0001.cam'

# Load data
depth = sio.depth_read(DEPTHFILE)
I,E = sio.cam_read(CAMFILE)

print(depth.shape)
# Display data
#plt.figure()
#plt.imshow(depth,cmap='gray')
#plt.title('depth')

print(I)
print(E)

#plt.show()
