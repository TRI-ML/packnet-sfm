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
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from matplotlib import animation

def normalize(depth):
    return (depth-np.min(depth))/(np.max(depth)-np.min(depth))
    

def visualize(depth):
    plt.imshow(np.repeat(normalize(depth), 3, axis=2))

def histeq(im,nbr_bins=256):
   imhist,bins = np.histogram(im.flatten(),nbr_bins,normed=True)
   cdf = imhist.cumsum() #cumulative distribution function
   cdf = 255 * cdf / cdf[-1] #normalize
   im2 = np.interp(im.flatten(),bins[:-1],cdf)
   return np.array(im2.reshape(im.shape)), cdf

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


def save_depth_video(depth_list):
    #self.set_eval()
    root_anim_dir = '' #self.opt.anim_dir + '/' + self.model_name_temp[-24:] + '/'
    seq_depths_plasma = np.zeros((len(depth_list),depth_list[0].shape[0],depth_list[0].shape[1],3))  #np.zeros((min(100,len(self.train_loader_vid)),self.opt.height,self.opt.width,3)) 
    cm = plt.get_cmap('plasma')

    for idx, el in enumerate(depth_list):
        seq_depths_plasma[idx,:,:,:] = cm((np.squeeze(normalize(el))))[:,:,0:3]
        
    img_list = list(normalize(seq_depths_plasma))
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15,  bitrate=1800)
    print(root_anim_dir+'anim_kitti_.mp4')
    plot_only_depth(img_list).save(root_anim_dir+'anim_kitti_.mp4', writer=writer)


def run_eval(max_val=72):
    def compute_depth_errors(gt, pred):
        """Computation of error metrics between predicted and ground truth depths
        """
        thresh = np.maximum((gt / pred), (pred / gt))
        a1 = np.mean((thresh < 1.25     ).astype(np.float))
        a2 =  np.mean((thresh < 1.25  ** 2   ).astype(np.float))
        a3 =  np.mean((thresh < 1.25  ** 3   ).astype(np.float))
    
        rmse = (gt - pred) ** 2
        rmse = np.sqrt(np.mean(rmse))
    
        rmse_log = (np.log(gt) - np.log(pred)) ** 2
        rmse_log = np.sqrt(np.mean(rmse_log))
    
        abs_rel = np.mean(np.abs(gt - pred) / gt)
    
        sq_rel = np.mean((gt - pred) ** 2 / gt)
    
        return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
    
    
    dp = np.squeeze(np.load('/Users/aycatakmaz/Desktop/CVL-Project/CVL-Thesis/depth_pred.npy'))
    dgt = np.squeeze(np.load('/Users/aycatakmaz/Desktop/CVL-Project/CVL-Thesis/depth_gt.npy'))
    
    mask = (dgt.astype(np.int) > 0) * (dgt.astype(np.int)<max_val)
    crop_mask = np.zeros(mask.shape, dtype=np.int)
    crop_mask[ :, :] = 1
    #crop_mask[:, :, 50:333, 50:973] = 1
    #crop_mask[:, :, :, :] = 1
    mask = mask * crop_mask
    
    dgt_uns = dgt[mask.nonzero()]
    dp_uns = dp[mask.nonzero()]
            
    depth_med = np.median(dp_uns)
    depth_gt_med = np.median(dgt_uns)
    
    depth_mean = np.mean(dp_uns)
    depth_gt_mean = np.mean(dgt_uns)
    
    
    from scipy import stats
    slope, intercept, _, _, _ = stats.linregress(np.reshape(dp_uns,(-1,)),np.reshape(dgt_uns, (-1,)))
    new_depth = np.clip(slope*dp + intercept, a_min=0.001,a_max=max_val)
    new_depth_med = dp * depth_gt_med/depth_med
    new_depth_mean = dp * depth_gt_mean/depth_mean
    
    #print(compute_depth_errors(new_depth, dgt))
    #print(compute_depth_errors(new_depth_med, dgt))
    #print(compute_depth_errors(new_depth_mean, dgt))
    
    def histeq(im,nbr_bins=256):
    
       #get image histogram
       imhist,bins = np.histogram(im.flatten(),nbr_bins,normed=True)
       cdf = imhist.cumsum() #cumulative distribution function
       cdf = 255 * cdf / cdf[-1] #normalize
    
       #use linear interpolation of cdf to find new pixel values
       im2 = np.interp(im.flatten(),bins[:-1],cdf)
    
       return np.array(im2.reshape(im.shape)), cdf
    
    dp_eq, _ = histeq(dp)
    dgt_eq, _ = histeq(dgt)

#run_eval(max_val=300)        
#run_eval(max_val=72)
#run_eval(max_val=100)
#run_eval(max_val=10)

    
depth = np.load('/Users/aycatakmaz/Desktop/CVL-Project/CVL-Thesis/deneme_depth.npy')
depthn = normalize(depth)
#visualize(depth)
depth_eq, _ = histeq(depth)
#visualize(depth_eq)
#depth_list = list(np.repeat(np.expand_dims(depth,axis=0),10,axis=0))
#save_depth_video(depth_list)


root_dir = '/Users/aycatakmaz/Desktop/CVL-Project/CVL-Thesis/alley_1'
depth_list = np.squeeze(np.asarray([np.load(os.path.join(root_dir,dpt_file))  for dpt_file in sorted(os.listdir(root_dir)) ]))
save_depth_video(depth_list)
#print()
