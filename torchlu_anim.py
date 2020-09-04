#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 08:28:37 2020

@author: aycatakmaz
"""

    
'''
def save_depth_video(self, save_embeddings=False):
    #self.set_eval()
    root_anim_dir = '' #self.opt.anim_dir + '/' + self.model_name_temp[-24:] + '/'
    seq_depths_plasma = np.zeros((100,self.opt.height,self.opt.width,3))  #np.zeros((min(100,len(self.train_loader_vid)),self.opt.height,self.opt.width,3)) 
    cm = plt.get_cmap('plasma')

    '''
    '''
    if not os.path.exists(root_anim_dir+self.opt.seq_list[0]):
                    os.makedirs(root_anim_dir+self.opt.seq_list[0])
    '''
    '''

    res = {}
    for batch_idx, inputs in enumerate(self.train_loader_vid):
        with torch.no_grad():
            if batch_idx<min(100,len(self.train_loader_vid)):
                if self.opt.learn_weights:
                    outputs, emb_outputs, _ = self.process_batch(inputs)
                else:
                    outputs, _ = self.process_batch(inputs)

                '''
                '''
                if ('gt_depth', 0) in inputs.keys():
                    temp_res = self.compute_depth_losses_all(inputs, outputs)
                    #pdb.set_trace()
                    for key in temp_res.keys():
                        if key not in res.keys():
                            res[key] = temp_res[key]
                        else:
                            res[key] += temp_res[key]
                '''
                '''
                seq_depths_plasma[batch_idx,:,:,:] = cm(1-(np.squeeze(normalize_depth(outputs[("depth", 0, 0)][0]).data.cpu().numpy())))[:,:,0:3]
                
            else:
                break
    for key in res.keys():
        res[key] /= min(50,len(self.train_loader_vid))
    print('RES: ', res)
    
    img_list = list(self.normalize(seq_depths_plasma))
    input_images_nm = list(self.normalize(input_images))
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15,  bitrate=1800)
    self.plot_only_depth(img_list).save('/anim_kitti_.mp4', writer=writer)
    self.set_train() 
    
'''
    
'''

def normalize(self, x):
    return (x-x.min())/(x.max()-x.min())

def plot_only_depth(self, img_list):
    def init():
        img.set_data(img_list[0])
        return (img,)

    def animate(i):
        img.set_data(img_list[i])
        return (img,)

    fig = plt.figure(frameon=False)
    fig.set_size_inches(w,h)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    img = ax.imshow(img_list[0]);
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                 frames=len(img_list), interval=60, blit=True)
    return anim
    
def save_depth_video(self, save_embeddings=False):
    self.set_eval()
    root_anim_dir = self.opt.anim_dir + '/' + self.model_name_temp[-24:] + '/'
    input_images = np.zeros((min(100,len(self.train_loader_vid)),self.opt.height,self.opt.width,3))
    seq_depths_plasma = np.zeros((min(100,len(self.train_loader_vid)),self.opt.height,self.opt.width,3)) #np.zeros((len(self.train_loader_vid),384,1024,3))
    if self.opt.save_embeddings:
        seq_embs = np.zeros((min(100,len(self.train_loader_vid)),self.opt.height,self.opt.width,3))

    cm = plt.get_cmap('plasma')

    if not os.path.exists(root_anim_dir+self.opt.seq_list[0]):
                    os.makedirs(root_anim_dir+self.opt.seq_list[0])

    res = {}
    for batch_idx, inputs in enumerate(self.train_loader_vid):
        with torch.no_grad():
            if batch_idx<min(100,len(self.train_loader_vid)):
                if self.opt.learn_weights:
                    outputs, emb_outputs, _ = self.process_batch(inputs)
                else:
                    outputs, _ = self.process_batch(inputs)

                if ('gt_depth', 0) in inputs.keys():
                    temp_res = self.compute_depth_losses_all(inputs, outputs)
                    #pdb.set_trace()
                    for key in temp_res.keys():
                        if key not in res.keys():
                            res[key] = temp_res[key]
                        else:
                            res[key] += temp_res[key]

                if 'kitti_2' in self.opt.seq_list:
                    seq_depths_plasma[batch_idx,:,:,:] = cm(1-(np.tanh(3*np.squeeze(normalize_depth(outputs[("depth", 0, 0)][0]).data.cpu().numpy()))))[:,:,0:3]
                    seq_depths_plasma[batch_idx,:,:,:] = cm((np.squeeze(normalize_depth(outputs[("depth", 0, 0)][0]).data.cpu().numpy())))[:,:,0:3]
            
                elif self.opt.use_gt_distances:
                    seq_depths_plasma[batch_idx,:,:,:] = cm(1-(np.squeeze(normalize_depth(outputs[("depth", 0, 0)][0]).data.cpu().numpy())))[:,:,0:3]
                    #seq_depths_plasma[batch_idx,:,:,:] = cm((np.squeeze(normalize_depth(outputs[("depth", 0, 0)][0]).data.cpu().numpy())))[:,:,0:3]
            
                else:
                    seq_depths_plasma[batch_idx,:,:,:] = cm(1-(np.squeeze(normalize_depth(outputs[("depth", 0, 0)][0]).data.cpu().numpy())))[:,:,0:3]
                
                input_images[batch_idx,:,:,:] = (np.squeeze(normalize_depth(inputs['images',0]).permute(0,2,3,1).data.cpu().numpy())*255).astype(np.int32)
           

                if self.opt.save_embeddings:
                    embs_for_write = emb_outputs['pix_emb',0][0].data.cpu().numpy()
                    if embs_for_write.shape[0]==1:
                        seq_embs[batch_idx,:,:,0] = embs_for_write[0,:,:]
                        seq_embs[batch_idx,:,:,1] = embs_for_write[0,:,:]
                        seq_embs[batch_idx,:,:,2] = embs_for_write[0,:,:]
                    elif embs_for_write.shape[0]==2:
                        seq_embs[batch_idx,:,:,0] = embs_for_write[0,:,:]
                        seq_embs[batch_idx,:,:,1] = embs_for_write[1,:,:]
                        seq_embs[batch_idx,:,:,2] = embs_for_write[0,:,:]*0 + 0.5
                    else:
                        seq_embs[batch_idx,:,:,:] = embs_for_write[0:3,:,:].transpose(1,2,0)
            else:
                break
    for key in res.keys():
        res[key] /= min(50,len(self.train_loader_vid))
    print('RES: ', res)
    
    img_list = list(self.normalize(seq_depths_plasma))
    input_images_nm = list(self.normalize(input_images))
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15,  bitrate=1800)
    if self.opt.save_embeddings:
        emb_list = list(self.normalize(np.exp(np.abs(self.normalize(seq_embs) + 0.000001))))
        self.plot_images_emb(input_images_nm,img_list, emb_list).save(root_anim_dir+self.opt.seq_list[0]+'/anim_embeddings_'+ str(self.opt.seq_list[0]) + '_' + self.model_name_temp[-24:]+'_'+ str(self.epoch).zfill(3)+'.mp4', writer=writer)
    
    else:
        self.plot_images(input_images_nm,img_list).save(root_anim_dir+self.opt.seq_list[0]+'/anim_embeddings_'+ str(self.opt.seq_list[0]) + '_' + self.model_name_temp[-24:]+'_'+ str(self.epoch).zfill(3)+'.mp4', writer=writer)
    self.set_train() 
        
'''