#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 07:51:43 2020

@author: aycatakmaz
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 07:59:15 2020

@author: aycatakmaz
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from matplotlib import animation

def normalize(depth):
    return (depth-np.min(depth))/(np.max(depth)-np.min(depth))
    

def visualize(depth):
    plt.imshow(np.repeat(normalize(depth), 3, axis=2))
depth = np.load('/Users/aycatakmaz/Desktop/CVL-Project/CVL-Thesis/deneme_depth.npy')

depthn = normalize(depth)

visualize(depth)

def histeq(im,nbr_bins=256):

   #get image histogram
   imhist,bins = np.histogram(im.flatten(),nbr_bins,normed=True)
   cdf = imhist.cumsum() #cumulative distribution function
   cdf = 255 * cdf / cdf[-1] #normalize

   #use linear interpolation of cdf to find new pixel values
   im2 = np.interp(im.flatten(),bins[:-1],cdf)

   return np.array(im2.reshape(im.shape)), cdf

depth_eq, _ = histeq(depth)
#visualize(depth_eq)

def plot_only_depth(img_list):
    def init():
        img.set_data(img_list[0])
        return (img,)

    def animate(i):
        img.set_data(img_list[i])
        return (img,)

    fig = plt.figure(frameon=False)
    fig.set_size_inches(6,4)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    img = ax.imshow(img_list[0]);
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                 frames=len(img_list), interval=60, blit=True)
    return anim


depth_list = list(np.repeat(np.expand_dims(depth,axis=0),10,axis=0))

    
def save_depth_video(depth_list):
    #self.set_eval()
    root_anim_dir = '' #self.opt.anim_dir + '/' + self.model_name_temp[-24:] + '/'
    seq_depths_plasma = np.zeros((10,depth_list[0].shape[0],depth_list[0].shape[1],3))  #np.zeros((min(100,len(self.train_loader_vid)),self.opt.height,self.opt.width,3)) 
    cm = plt.get_cmap('plasma')

    for idx, el in enumerate(depth_list):
        seq_depths_plasma[idx,:,:,:] = cm((np.squeeze(normalize(el))))[:,:,0:3]
        
    img_list = list(normalize(seq_depths_plasma))
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15,  bitrate=1800)
    plot_only_depth(img_list).save(root_anim_dir+'anim_kitti_.mp4', writer=writer)
    
    
save_depth_video(depth_list)