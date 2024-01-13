#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.



# coding: utf-8

# In[ ]:


# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.


# # Fit a simple Neural Radiance Field via raymarching
#
# This tutorial shows how to fit Neural Radiance Field given a set of views of a scene using differentiable implicit function rendering.
#
# More specifically, this tutorial will explain how to:
# 1. Create a differentiable implicit function renderer with either image-grid or Monte Carlo ray sampling.
# 2. Create an Implicit model of a scene.
# 3. Fit the implicit function (Neural Radiance Field) based on input images using the differentiable implicit renderer.
# 4. Visualize the learnt implicit function.
#
# Note that the presented implicit model is a simplified version of NeRF:<br>
# _Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng: NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis, ECCV 2020._
#
# The simplifications include:
# * *Ray sampling*: This notebook does not perform stratified ray sampling but rather ray sampling at equidistant depths.
# * *Rendering*: We do a single rendering pass, as opposed to the original implementation that does a coarse and fine rendering pass.
# * *Architecture*: Our network is shallower which allows for faster optimization possibly at the cost of surface details.
# * *Mask loss*: Since our observations include segmentation masks, we also optimize a silhouette loss that forces rays to either get fully absorbed inside the volume, or to completely pass through it.
#

# ## 0. Install and Import modules
# Ensure `torch` and `torchvision` are installed. If `pytorch3d` is not installed, install it using the following cell:

# In[ ]:

import glob
import albumentations as AB
import os
import sys
import torch,cv2
from typing import Callable, Tuple, Union
import pytorch3d


# In[ ]:


# %matplotlib inline
# %matplotlib notebook
import os
import sys
import time
import json
import glob
import torch
import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from IPython import display
from tqdm.notebook import tqdm
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.renderer.implicit.utils import RayBundle
from pytorch3d.renderer.implicit.raysampling import  RayBundle
from pytorch3d.renderer.implicit.raymarching import  _check_raymarcher_inputs, _check_density_bounds, _shifted_cumprod
# Data structures and functions for rendering
from pytorch3d.structures import Volumes
from pytorch3d.transforms import so3_exp_map
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    NDCMultinomialRaysampler,
    MonteCarloRaysampler,
    EmissionAbsorptionRaymarcher,
    ImplicitRenderer,
    RayBundle,
    ray_bundle_to_ray_points,
)



import re
import numpy as np
import importlib
import warnings
from typing import Optional, Tuple, Union

import torch

def vp(finalV):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    #pcd2 = o3d.geometry.PointCloud()
    ct=finalV.shape[0]
    #ct2=finalV2.shape[0]
    pcd.points = o3d.utility.Vector3dVector(finalV)
    #pcd2.points = o3d.utility.Vector3dVector(finalV2)
    #vec=np.ones((ct,3));vec[:,1]=0.0;vec[:,2]=0.0;vec1=np.ones((ct2,3));vec1[:,0]=0.0;vec1[:,2]=0.0
    # pcd.colors = o3d.utility.Vector3dVector( color)
    #pcd2.colors = o3d.utility.Vector3dVector(color)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd] + [mesh_frame])
def rotfromeuler(euler):
    one = (torch.ones(1, 1))
    zero = (torch.zeros(1, 1))
    x=euler[0]
    y=euler[1]
    z=euler[2]

    rot_x = torch.Tensor([[1.0, 0.0, 0.0],[0.0, x.cos(), -x.sin()],[0.0, x.sin(), x.cos()]])
    rot_y = torch.Tensor([[y.cos(), 0.0, y.sin()],[0.0, 1.0, 0.0],[-y.sin(), 0.0, y.cos()]])
    rot_z = torch.Tensor([[z.cos(), -z.sin(), 0.0],[z.sin(), z.cos(), 0.0],[0.0, 0.0, 1.0]])

    return torch.mm(rot_z, torch.mm(rot_y, rot_x))
def rotfromeulernp(euler):
    import open3d as o3d
    x=euler[0]
    y=euler[1]
    z=euler[2]

    rot_x = np.array([[1.0, 0.0, 0.0],[0.0, np.cos(x), -np.sin(x)],[0.0, np.sin(x), np.cos(x)]])
    rot_y = np.array([[np.cos(y), 0.0, np.sin(y)],[0.0, 1.0, 0.0],[-np.sin(y), 0.0, np.cos(y)]])
    rot_z = np.array([[np.cos(z), -np.sin(z), 0.0],[np.sin(z), np.cos(z), 0.0],[0.0, 0.0, 1.0]])

    return np.dot(rot_z, np.dot(rot_y, rot_x))
    
def extractRT(path, occid):
    head, tail = os.path.split(path)  ## head = NerfEmb\bop\tless\train\000001\rgb
    mainfolder, _ = os.path.split(head)   ## NerfEmb\bop\tless\train\000001\
    a = re.findall(r'\d+', tail) ##return 000001
    id = str(int(a[0])) ## 1
    transformdata = json.load(open(mainfolder + "/scene_gt.json"))
    R = transformdata[id][occid]["cam_R_m2c"]
    R = np.asarray(R).reshape(3, 3)
    T = np.asarray(transformdata[id][occid]["cam_t_m2c"])
    return R, T
    
def mip360loss(ray_bundle,weights):
    w=weights[...,:-1]
    t=ray_bundle.lengths
    t=t-t[...,0].unsqueeze(-1)
    t=t/torch.max(t,dim=-1)[0].unsqueeze(-1)
    ut = (t[..., 1:] + t[..., :-1]) / 2
    dut = torch.abs(ut[..., :, None] - ut[..., None, :])
    loss_inter = torch.sum(w * torch.sum(w[..., None, :] * dut, axis=-1), axis=-1)

    # The loss incurred within each individual interval with itself.
    loss_intra = torch.sum(w ** 2 * (t[..., 1:] - t[..., :-1]), axis=-1) / 3

    return torch.mean(loss_inter + loss_intra)




def huber(x, y, scaling=0.1):
    """
    A helper function for evaluating the smooth L1 (huber) loss
    between the rendered silhouettes and colors.
    """
    diff_sq = (x - y) ** 2
    loss = ((1 + diff_sq / (scaling ** 2)).clamp(1e-4).sqrt() - 1) * float(scaling)
    return loss


def sample_images_at_mc_locs(target_images, sampled_rays_xy):
    """
    Given a set of Monte Carlo pixel locations `sampled_rays_xy`,
    this method samples the tensor `target_images` at the
    respective 2D locations.

    This function is used in order to extract the colors from
    ground truth images that correspond to the colors
    rendered using `MonteCarloRaysampler`.
    """
    ba = target_images.shape[0]
    dim = target_images.shape[-1]
    spatial_size = sampled_rays_xy.shape[1:-1]
    # In order to sample target_images, we utilize
    # the grid_sample function which implements a
    # bilinear image sampler.
    # Note that we have to invert the sign of the
    # sampled ray positions to convert the NDC xy locations
    # of the MonteCarloRaysampler to the coordinate
    # convention of grid_sample.
    #import pdb;pdb.set_trace()
    images_sampled = torch.nn.functional.grid_sample(
        target_images.permute(0, 3, 1, 2),
        -sampled_rays_xy.view(ba, -1, 1, 2),  # note the sign inversion
        align_corners=True,
        mode='nearest'
    )
    return images_sampled.permute(0, 2, 3, 1).view(
        ba, *spatial_size, dim
    )

def get_emb_vis(emb_img: torch.Tensor, mask: torch.Tensor = None, demean: torch.tensor = False):
            lastDim=emb_img.shape[-1]
            if demean is True:
                demean = emb_img[mask].view(-1, lastDim).mean(dim=0)
            if demean is not False:
                emb_img = emb_img - demean
            shape = emb_img.shape[:-1]
            emb_img = emb_img.view(*shape, 3, -1).mean(dim=-1)
            if mask is not None:
                emb_img[~mask] = 0.
            emb_img /= torch.abs(emb_img).max() + 1e-9
            emb_img.mul_(0.5).add_(0.5)
            return emb_img  
def nt(t1):
    t1 = (t1 - 0.5) * 2
    return t1 / (0.01 + torch.norm(t1, dim=2).unsqueeze(2))
def show_full_render1(
        neural_radiance_field, camera,expID, renderer_grid,normalization=False, rand=0, siren=False, savePath="_nerf"
):
    """
    This is a helper function for visualizing the
    intermediate results of the learning.

    Since the `NeuralRadianceField` suffers from
    a large memory footprint, which does not let us
    render the full image grid in a single forward pass,
    we utilize the `NeuralRadianceField.batched_forward`
    function in combination with disabling the gradient caching.
    This chunks the set of emitted rays to batches and
    evaluates the implicit function on one batch at a time
    to prevent GPU memory overflow.
    """

    # Prevent gradient caching.
    import cv2
    with torch.no_grad():
        # Render using the grid renderer and the
        # batched_forward function of neural_radiance_field.
        rendered_image_silhouette, _,_ = renderer_grid(
            cameras=camera,
            volumetric_function=neural_radiance_field.batched_forward
        )
        # Split the rendering result to a silhouette render
        # and the image render.
        lastDim=rendered_image_silhouette.shape[-1]
        #if lastDim==3:
        rendered_image, rendered_silhouette = (
              rendered_image_silhouette[0].split([lastDim-1, 1], dim=-1)
          )
        #else:
         # rendered_image=rendered_image_silhouette

    #import pdb;pdb.set_trace()

    if rand==0:
      rand=str(int(torch.rand(1)*10))
    
    if lastDim>=(12+1):
      if normalization:
        #import pdb;pdb.set_trace()
        l1=(rendered_image[:,:,0:12]-0.5)*2
        rendered_image[:,:,0:12]=l1/(0.01+torch.norm(l1[:,:,0:12],dim=2).unsqueeze(2))
      
      cv2.imwrite(expID+"/"+rand+"_rendfeat.jpg", get_emb_vis(rendered_image[:,:,0:12].cpu().detach(), demean=True).cpu().numpy() * 255)
      if siren:
        #import pdb;pdb.set_trace()
        cv2.imwrite(expID + "/" + "rfeat1.jpg",normImage(rendered_image[:,:,0:3].detach().cpu()).numpy() *255*2)
        cv2.imwrite(expID + "/" + "rfeat2.jpg",normImage(rendered_image[:,:,3:6].detach().cpu()).numpy() * 255*2)
        cv2.imwrite(expID + "/" + "rfeat3.jpg",normImage(rendered_image[:,:,6:9].detach().cpu()).numpy() * 255*2)
        cv2.imwrite(expID + "/" + "rfeat4.jpg",normImage(rendered_image[:,:,9:12].detach().cpu()).numpy() * 255*2)
        if rendered_image.shape[-1]>12:
        
          cv2.imwrite(expID + "/" + "rfeat5.jpg",normImage(rendered_image[:,:,12:15].detach().cpu()).numpy() *255*2)
          cv2.imwrite(expID + "/" + "rfeat6.jpg",normImage(rendered_image[:,:,15:18].detach().cpu()).numpy() * 255*2)
          cv2.imwrite(expID + "/" + "rfeat7.jpg",normImage(rendered_image[:,:,18:21].detach().cpu()).numpy() * 255*2)
          cv2.imwrite(expID + "/" + "rfeat8.jpg",normImage(rendered_image[:,:,21:24].detach().cpu()).numpy() * 255*2)
        
        
      else:
        cv2.imwrite(expID + "/" + "rfeat1.jpg",rendered_image[:,:,0:3].detach().cpu().numpy() * 255)
        cv2.imwrite(expID + "/" + "rfeat2.jpg",rendered_image[:,:,3:6].detach().cpu().numpy() * 255)
        cv2.imwrite(expID + "/" + "rfeat3.jpg",rendered_image[:,:,6:9].detach().cpu().numpy() * 255)
        cv2.imwrite(expID + "/" + "rfeat4.jpg",rendered_image[:,:,9:12].detach().cpu().numpy() * 255)
      
      #cv2.imwrite(expID + "/" + rand + ".jpg",rendered_image[:, :, 12:].cpu().detach().numpy() * 255)
    elif lastDim>4:
      cv2.imwrite(expID + "/" + rand + ".jpg",rendered_image[:, :, 0:3].cpu().detach().numpy() * 255)
      cv2.imwrite(expID + "/" + rand + "_unray.jpg",rendered_image[:, :, 3:6].cpu().detach().numpy() * 255)
    else:
      
      cv2.imwrite(expID + "/" + rand +savePath+".jpg",rendered_image[:, :, :].cpu().detach().numpy() * 255)
    
    # cv2.imwrite(expID + "/" + "d1.jpg",d1[0].detach().cpu().numpy() * 255)
    #return cv2.imwrite(str(int(torch.rand(1)*100))+".jpg",rendered_image.cpu().numpy()*256)
def show_full_render_featuremat(
        neural_radiance_field, camera,expID, renderer_grid,normalization=False, rand=0, siren=False
):
    """
    This is a helper function for visualizing the
    intermediate results of the learning.

    Since the `NeuralRadianceField` suffers from
    a large memory footprint, which does not let us
    render the full image grid in a single forward pass,
    we utilize the `NeuralRadianceField.batched_forward`
    function in combination with disabling the gradient caching.
    This chunks the set of emitted rays to batches and
    evaluates the implicit function on one batch at a time
    to prevent GPU memory overflow.
    """

    # Prevent gradient caching.
    import cv2
    with torch.no_grad():
        # Render using the grid renderer and the
        # batched_forward function of neural_radiance_field.
        rendered_image_silhouette, _,_ = renderer_grid(
            cameras=camera,
            volumetric_function=neural_radiance_field.batched_forward
        )
        # Split the rendering result to a silhouette render
        # and the image render.
        lastDim=rendered_image_silhouette.shape[-1]
        #if lastDim==3:
        rendered_image, rendered_silhouette = (
              rendered_image_silhouette[0].split([lastDim-1, 1], dim=-1)
          )
        #else:
         # rendered_image=rendered_image_silhouette

    #import pdb;pdb.set_trace()

    if rand==0:
      rand=str(int(torch.rand(1)*10))
    
    if lastDim>=(12+1):
      if normalization:
        #import pdb;pdb.set_trace()
        #rendered_image[:,:,0:12]=l1/(0.01+torch.norm(l1[:,:,0:12],dim=2).unsqueeze(2))
        m1,n1,l1=rendered_image.shape
        featurematrix=rendered_image[..., 0:9].view(*rendered_image.shape[:-1],3,3)
        albedo=rendered_image[..., 9:12]
        finalColor=torch.bmm(featurematrix.view(-1,3,3), albedo.view(-1,3,1)).view(*rendered_image.shape[:-1],3)
        
        cv2.imwrite(expID+"/"+rand+"_unray.jpg", albedo.cpu().cpu().numpy() * 255)
        cv2.imwrite(expID+"/"+rand+"ray.jpg", finalColor.cpu().cpu().numpy() * 255)
        
def normImage(emb_img):
            emb_img /= torch.abs(emb_img).max() + 1e-9
            emb_img.mul_(0.5).add_(0.5)
            return emb_img
def returnCrossEntropy(q1, k1, negFrac=1, device=False):
        sim_pos = (k1 * q1).sum(dim=-1, keepdim=True)  # (B, n_pos, 1)

        # compute similarities for negative pairs
        negKeys = torch.randperm(k1.shape[1], dtype=torch.int64)[0: int(k1.shape[1] * negFrac)]

        sim_neg = q1 @ k1[:, negKeys, :].permute(0, 2, 1)  # (B, n_pos, n_neg)

        # loss
        lgts = torch.cat((sim_pos, sim_neg), dim=-1).permute(0, 2, 1)  # (B, 1 + n_neg, n_pos)
        # import pdb;pdb.set_trace()
        if device:
            target = torch.zeros(1, sim_pos.shape[1], dtype=torch.long).to(device)
        else:
            target = torch.zeros(1, sim_pos.shape[1], dtype=torch.long).cuda()

        from torch.nn import functional as F
        return F.cross_entropy(lgts, target) / 1000
        # import pdb;pdb.set_trace()
def returnCrossEntropyWithNeg(q1, k1, k2, negFrac=1, device=False):
            sim_pos = (k1 * q1).sum(dim=-1, keepdim=True)  # (B, n_pos, 1)

            # compute similarities for negative pairs
            negKeys = torch.randperm(k1.shape[1], dtype=torch.int64)[0: int(k1.shape[1] * negFrac)]

            sim_neg = q1 @ k2.permute(0, 2, 1)  # (B, n_pos, n_neg)

            # loss
            lgts = torch.cat((sim_pos, sim_neg), dim=-1).permute(0, 2, 1)  # (B, 1 + n_neg, n_pos)
            # import pdb;pdb.set_trace()
            if device:
                target = torch.zeros(sim_pos.shape[0], sim_pos.shape[1], dtype=torch.long).to(device)
            else:
                target = torch.zeros(sim_pos.shape[0], sim_pos.shape[1], dtype=torch.long).cuda()

            from torch.nn import functional as F
            return F.cross_entropy(lgts, target) / 1000
def returnCrossEntropy2(q1, k1, negFrac=1,negKeys=1, device=False):
        sim_pos = (k1 * q1).sum(dim=-1, keepdim=True)  # (B, n_pos, 1)

        # compute similarities for negative pairs
        #negKeys = torch.randperm(k1.shape[1], dtype=torch.int64)[0: int(k1.shape[1] * negFrac)]

        sim_neg = q1 @ k1[:, negKeys, :].permute(0, 2, 1)  # (B, n_pos, n_neg)

        # loss
        lgts = torch.cat((sim_pos, sim_neg), dim=-1).permute(0, 2, 1)  # (B, 1 + n_neg, n_pos)
        # import pdb;pdb.set_trace()
        if device:
            target = torch.zeros(1, sim_pos.shape[1], dtype=torch.long).to(device)
        else:
            target = torch.zeros(1, sim_pos.shape[1], dtype=torch.long).cuda()

        from torch.nn import functional as F
        return F.cross_entropy(lgts, target) / 1000
def show_full_render(
        neural_radiance_field, camera,
        target_image, target_silhouette,
        loss_history_color, loss_history_sil,
):
    """
    This is a helper function for visualizing the
    intermediate results of the learning.

    Since the `NeuralRadianceField` suffers from
    a large memory footprint, which does not let us
    render the full image grid in a single forward pass,
    we utilize the `NeuralRadianceField.batched_forward`
    function in combination with disabling the gradient caching.
    This chunks the set of emitted rays to batches and
    evaluates the implicit function on one batch at a time
    to prevent GPU memory overflow.
    """

    # Prevent gradient caching.
    with torch.no_grad():
        # Render using the grid renderer and the
        # batched_forward function of neural_radiance_field.
        rendered_image_silhouette, _ = renderer_grid(
            cameras=camera,
            volumetric_function=neural_radiance_field.batched_forward
        )
        # Split the rendering result to a silhouette render
        # and the image render.
        rendered_image, rendered_silhouette = (
            rendered_image_silhouette[0].split([3, 1], dim=-1)
        )

    # Generate plots.
    fig, ax = plt.subplots(2, 3, figsize=(5, 3))
    ax = ax.ravel()
    clamp_and_detach = lambda x: x.clamp(0.0, 1.0).cpu().detach().numpy()
    ax[0].plot(list(range(len(loss_history_color))), loss_history_color, linewidth=1)
    ax[1].imshow(clamp_and_detach(rendered_image))
    ax[2].imshow(clamp_and_detach(rendered_silhouette[..., 0]))
    ax[3].plot(list(range(len(loss_history_sil))), loss_history_sil, linewidth=1)
    ax[4].imshow(clamp_and_detach(target_image))
    ax[5].imshow(clamp_and_detach(target_silhouette))
    for ax_, title_ in zip(
            ax,
            (
                    "loss color", "rendered image", "rendered silhouette",
                    "loss silhouette", "target image", "target silhouette",
            )
    ):
        if not title_.startswith('loss'):
            ax_.grid("off")
            ax_.axis("off")
        ax_.set_title(title_)
    fig.canvas.draw();
    fig.show()
    display.clear_output(wait=True)
    display.display(fig)
    return fig
def generate_rotating_nerf(neural_radiance_field, n_frames=50, device=1, target_cameras=1, renderer=1):
    logRs = torch.zeros(n_frames, 3, device=device)
    logRs[:, 1] = torch.linspace(-3.14, 3.14, n_frames, device=device)
    Rs = so3_exp_map(logRs)
    Ts = torch.zeros(n_frames, 3, device=device)
    Ts[:, 2] = 2.7
    frames = []
    framesEmb = []

    for R, T in zip(tqdm(Rs), Ts):
        camera = FoVPerspectiveCameras(
            R=R[None],
            T=T[None],
            znear=target_cameras.znear[0],
            zfar=target_cameras.zfar[0],
            aspect_ratio=target_cameras.aspect_ratio[0],
            fov=target_cameras.fov[0],
            device=device,
        )
        # Note that we again render with `NDCMultinomialRaysampler`
        # and the batched_forward function of neural_radiance_field.
        frames.append(
            renderer_grid(
                cameras=camera,
                volumetric_function=neural_radiance_field.batched_forward,
            )[0][..., :12]
        )
        framesEmb.append(
            renderer_grid(
                cameras=camera,
                volumetric_function=neural_radiance_field.batched_forward,
            )[0][..., 12:]
        )
    return torch.cat(frames),torch.cat(framesEmb)

def createEncoder():
    from dep.unet import ResNetUNet
    
    encoder_rgb = ResNetUNet(
        n_class=(12),
        n_decoders=1,
    ).cuda()
    
    encoder_rgb = torch.nn.DataParallel(encoder_rgb).cuda()
    return encoder_rgb


def filterBoundsPC(pc, bounds):
    filteridx = np.where((pc[:, 0] > bounds[0, 0]) & (pc[:, 0] < bounds[0, 1]) &
                         (pc[:, 1] > bounds[1, 0]) & (pc[:, 1] < bounds[1, 1]))
    return filteridx[0]

def estimateNormals(v1):
    pcd=o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(v1)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=180))
    return np.asarray(pcd.normals())
def returnOccludedPC(PC):
    centPC = PC
    dmax = np.max(centPC, axis=0)
    dmin = np.min(centPC, axis=0)
    Zc = np.random.uniform(low=dmin, high=dmax)
    drange = np.max(np.abs(dmax - dmin))
    Zc = np.mean(centPC, axis=0) + np.random.uniform(low=-drange / 2, high=drange / 2)
    # Zc=-randrot1.numpy() + np.random.uniform(low=-0.4 , high=0.4)
    int1 = np.random.randint(2, 6)
    Zr = np.random.uniform(low=10, high=(dmax - dmin) / int1)
    Zr[2] = 0

    idx1 = filterBoundsPC(centPC, np.vstack((Zc - Zr, Zc + Zr)).T)
    # import pdb;pdb.set_trace()

    # occIdx=np.where(realX[id][:,2]<occMidZ-occRange) or  np.where(realX[id][:,2]>occMidZ+occRange)
    idx3 = np.arange(centPC.shape[0])
    idx3 = np.delete(idx3, idx1)
    idx_o = np.copy(idx3)
    return idx_o
def rotfromeuler(euler):
    one = (torch.ones(1, 1))
    zero = (torch.zeros(1, 1))
    x=euler[0]
    y=euler[1]
    z=euler[2]

    rot_x = torch.Tensor([[1.0, 0.0, 0.0],[0.0, x.cos(), -x.sin()],[0.0, x.sin(), x.cos()]])
    rot_y = torch.Tensor([[y.cos(), 0.0, y.sin()],[0.0, 1.0, 0.0],[-y.sin(), 0.0, y.cos()]])
    rot_z = torch.Tensor([[z.cos(), -z.sin(), 0.0],[z.sin(), z.cos(), 0.0],[0.0, 0.0, 1.0]])
    return torch.mm(rot_z, torch.mm(rot_y, rot_x))
def readImages(rgb, mask, transformNo):
    import glob
    import albumentations as AB
    maxDim=rgb.shape[0]*rgb.shape[0]
    # rgb.shapergb=cv2.imre
    # if np.max(np.nonzero(mask)[0])>imX or np.max(np.nonzero(mask)[1])>imX:
    #  maxD=np.max(np.nonzero(mask)[0],np.nonzero(mask)[1])
    # import pdb;pdb.set_trace()
    if False:
        w1 = np.ptp(np.nonzero(mask)[0])
        w2 = np.ptp(np.nonzero(mask)[1])
        imX=128
        if w1 > imX or w2 > imX:
            resFac = (0.9 * imX) / np.max((w1, w2))
            rgb = cv2.resize(rgb, (int(resFac * rgb.shape[1]), (int(resFac * rgb.shape[0]))))
            mask = cv2.resize(mask, (int(resFac * mask.shape[1]), (int(resFac * mask.shape[0]))))
            c1 = 1
        fullrgb = rgb
        fullmask = mask
        fi = np.nonzero(fullmask)
        minX = np.min(fi[0]);
        minY = np.min(fi[1])
        maxX = np.max(fi[0]);
        maxY = np.max(fi[1])

        crgb = fullrgb[minX:maxX, minY:maxY]
        cmask = fullmask[minX:maxX, minY:maxY]

        rgb = np.zeros((imX, imX, 3))
        mask = np.zeros((imX, imX, 3))
        sx = int((imX - crgb.shape[0]) / 2);
        ex = int(sx + crgb.shape[0]);
        sy = int((imX - crgb.shape[1]) / 2);
        ey = sy + crgb.shape[1]

        rgb[sx:ex, sy:ey, :] = crgb
        mask[sx:ex, sy:ey, :] = cmask

    # rgb=
    imX=rgb.shape[0]
    indices = np.nonzero(mask)

    ffpoints = np.zeros((indices[0].shape[0], 6))
    ffpoints[:, 3:6] = (rgb[indices[0], indices[1]])

    ffpoints[:, 0] = indices[0]
    ffpoints[:, 1] = indices[1]

    onePC = np.copy(ffpoints[:, 0:3])
    oneCol = np.copy(ffpoints[:, 3:])
    oneIM = torch.zeros((imX, imX, 3))
    #import pdb;pdb.set_trace()
    oneIM.view(-1, 3)[(ffpoints[:, 1].astype(int) + imX * ffpoints[:, 0].astype(int))] = torch.from_numpy(ffpoints[:, 3:6].astype(np.float32))

    rgbtensor = torch.zeros((transformNo, imX, imX, 3))
    pctensor = torch.zeros((transformNo, onePC.shape[0], 3))
    trR = torch.zeros(transformNo, 3, 3)
    trT = torch.zeros(transformNo, 3)
    trS = torch.zeros(transformNo, 1)
    octensor = torch.zeros((transformNo, onePC.shape[0]), dtype=torch.bool)
    # if c1==1:
    # import pdb;pdb.set_trace()
    for id in range(transformNo):
        randScale = 0.8+0.2*torch.rand(1)
        randTra = (torch.rand(3) - 0.5) * 1.3
        randTra[2] = 0
        randRot = (torch.rand(3) - 0.5) * 2 * np.pi
        randRot[1] = 0
        randRot[0] = 0
        trS[id]=randScale

        # if epoch<50:
        #    randRot[2] = 0

        minWidth = (imX - np.max([np.ptp(onePC[:, 0]), np.ptp(onePC[:, 1])])) / 2
        trT[id] = randTra * minWidth

        trR[id] = rotfromeuler(randRot)

        curPC = (np.copy(onePC - np.mean(onePC, axis=0)).dot(trR[id].numpy()) + trT[id].numpy() + np.mean(onePC,
                                                                                                          axis=0)) * trS[id].numpy()

        if np.max(curPC[:, 0:2]) >= (imX - 2):
            off = imX - np.max(curPC) - 2
            if np.max(curPC[:, 0]) > np.max(curPC[:, 1]):
                curPC[:, 0] = curPC[:, 0] + off
                trT[id][0] = trT[id][0] + off / trS[id]
            else:
                curPC[:, 1] = curPC[:, 1] + off
                trT[id][1] = trT[id][1] + off / trS[id]

        # print("max",id)
        if np.min(curPC[:, 0:2]) <= 2:
            off = -np.min(curPC) + 2
            if np.min(curPC[:, 0]) < np.min(curPC[:, 1]):
                curPC[:, 0] = curPC[:, 0] + off
                trT[id][0] = trT[id][0] + off
            else:
                curPC[:, 1] = curPC[:, 1] + off
                trT[id][1] = trT[id][1] + off

        if id==0:
            curPC=np.copy(onePC)
        pctensor[id] = torch.from_numpy(curPC)
        idx_o = returnOccludedPC(np.copy(curPC))
        
        idx_o=np.setdiff1d(idx_o, np.where(curPC>127)[0])
        idx_o=np.setdiff1d(idx_o, np.where(curPC<0)[0])
        
        aug = AB.Compose([
            AB.ColorJitter(),
            # A.HorizontalFlip(p=0.5),
            AB.RandomBrightnessContrast(p=0.2),
        ])
        
        
        if len(idx_o) > 5 and id>0:
            curPC = curPC[idx_o]
            rgbtensor[id].view(-1, 3)[(curPC[:, 1].astype(int) + imX * curPC[:, 0].astype(int))] = torch.from_numpy(
                oneCol[idx_o].astype(np.float32))
            rgbtensor[id] = torch.from_numpy(aug(image=rgbtensor[id].numpy())["image"])
            octensor[id][idx_o] = True
        else:
            rgbtensor[id].view(-1, 3)[(curPC[:, 1].astype(int) + imX * curPC[:, 0].astype(int))] = torch.from_numpy(
                oneCol.astype(np.float32))
            octensor[id] = True
            rgbtensor[id] = torch.from_numpy(aug(image=rgbtensor[id].numpy())["image"])
        #     print("a1")
        # cv2.imwrite(str(id)+".jpg", rgbtensor[id].cpu().numpy()*255)

    return oneIM, onePC, oneCol, rgbtensor, trR, trT, pctensor, octensor
def augmentImage(rgb,mask,  imX ):
    randScale = 0.5 + 0.5 * np.random.uniform(0, 1)
    randTra = ((np.random.uniform(0, 1, 2) - 0.5) * imX /1.2).astype("int64")
    randRot = int(np.random.uniform(0, 1) * 360)

    aug = AB.Compose([
        AB.ColorJitter(),
        # A.HorizontalFlip(p=0.5),
        AB.RandomBrightnessContrast(p=0.2),
    ])

    rows, cols, _ = rgb.shape
    # cols-1 and rows-1 are the coordinate limits.
    M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), randRot, randScale)

    M[:, 2] = M[:, 2] + randTra

    dst = cv2.warpAffine(rgb, M, (cols, rows))
    dst_mask = cv2.warpAffine(mask, M, (cols, rows))

    rows = np.random.randint(40, imX-40)
    cols = np.random.randint(40, imX-40)

    dst=cv2.resize(dst, (cols,rows))
    dst_mask=cv2.resize(dst_mask, (cols,rows))


    colX = np.random.randint(1, imX-cols-1)
    rowX= np.random.randint(1, imX-rows-1)
    return torch.from_numpy(aug(image=dst)["image"].astype("float32"))/255, dst_mask, rowX, colX, rows, cols
def readImagesWithBGClean(rgb, mask, transformNo, cocoids, datasetPath, aug=True, objid=0, scaleFac=0.5, maskErosion=True):

    imX=rgb.shape[1]
    rgbtensor = torch.zeros((transformNo, imX, imX, 3))
    masktensor = torch.zeros((transformNo, imX, imX))
    if aug:
      for a in range(transformNo):
        w=rgbtensor.shape[1]
        w1=0
        h1=0
        try:
            if  np.random.uniform(0,1)>0.3:
              cid=np.random.randint(0, 80000)
              image1=cv2.imread(cocoids[cid])/255
              w1,h1,_=image1.shape
              #import pdb;pdb.set_trace()
              if w1<w+10 or h1<w+10:
                image1=cv2.resize(image1, (w+10,w+10))
                nw=0;nh=0
              else:
                nw=np.random.randint(0, w1-w)
                nh=np.random.randint(0, h1-w)
              #import pdb;pdb.set_trace() 
              #try: 
              rgbtensor[a] = torch.from_numpy(image1[nw:nw + w, nh:nh + w].astype("float32"))
              #except:
              #  import pdb;pdb.set_trace()
            cid4 = np.random.randint(1, 4)
            for n1 in range(cid4):
                cid2 = np.random.randint(2, 28)
                cid3 = np.random.randint(1, 500)
                if cid2==int(objid):
                    cid2=objid+1

                objPath = datasetPath + "/train_primesense/0000" + str(cid2).zfill(2) + "/"
                c1 = cv2.imread(objPath + "rgb/" + str(cid3).zfill(6) + ".png")[100:300,100:300]
                m1 = cv2.imread(objPath + "mask/" + str(cid3).zfill(6) + "_000000.png")[100:300,100:300]/255



                ten1, dst_mask, rowX, colX, rows, cols= augmentImage(c1, m1, imX)
                maskID1 = np.where(dst_mask[:, :, 0] == 1)
                rgbtensor[a][rowX+maskID1[0], colX+maskID1[1]] =ten1[maskID1[0],maskID1[1]]

        except():
            print("bad choice")
            #import pdb;pdb.set_trace()

    # pctensor = torch.zeros((transformNo, onePC.shape[0], 3))
    trR = torch.zeros(transformNo, 1 )
    trT = torch.zeros(transformNo, 2)
    trS = torch.zeros(transformNo,1)
    for id in range(transformNo):
        randScale = scaleFac + (1-scaleFac) * np.random.uniform(0,1)
        randTra = ((np.random.uniform(0,1,2) - 0.5) * imX/2).astype("int64")
        randRot = int(np.random.uniform(0,1) * 360)

        aug = AB.Compose([
            AB.ColorJitter(),
            # A.HorizontalFlip(p=0.5),
            AB.RandomBrightnessContrast(p=0.2),
        ])

        rows, cols,_ = rgb.shape
        # cols-1 and rows-1 are the coordinate limits.
        M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), randRot, randScale)
        trR[id] = (randRot)
        trT[id] = torch.from_numpy(randTra)
        trS[id] = (randScale)

        M[:,2]=M[:,2]+randTra

        dst = cv2.warpAffine(rgb, M, (cols, rows))
        dst_mask = cv2.warpAffine(mask, M, (cols, rows))

        # rgbtensor[id] = torch.from_numpy(dst.astype(np.float32))
        masktensor[id]=torch.from_numpy(dst_mask)
        if np.random.uniform(0,1)>0.5 or not maskErosion:
            maskID=torch.where(masktensor[id].view(-1)==1)
        else:
            try:
                maskID=torch.where(createOcclusionsOnMask(masktensor[id].cpu().numpy()).reshape(-1)==1)
            except:
                print("bad mask")
                maskID = torch.where(masktensor[id].view(-1) == 1)
        #import pdb;pdb.set_trace()
        if len(maskID[0])>100:
          masktensor[id]=masktensor[id]*0
          masktensor[id].view(-1)[maskID]=1
        else:
          maskID = torch.where(masktensor[id].view(-1) == 1)
          
        rgbtensor[id].view(-1,3)[maskID] = torch.from_numpy(aug(image=dst)["image"]).view(-1,3)[maskID]

    return  rgbtensor,masktensor, trR, trT, trS
def readImagesWithBGCleanOld(rgb, mask, transformNo, cocoids, datasetPath, aug=True, objid=0, scaleFac=0.5, maskErosion=True):

    imX=rgb.shape[1]
    rgbtensor = torch.zeros((transformNo, imX, imX, 3))
    masktensor = torch.zeros((transformNo, imX, imX))
    if aug:
      for a in range(transformNo):
        w=rgbtensor.shape[1]
        w1=0
        h1=0
        try:
            if  np.random.uniform(0,1)>0.3:
              cid=np.random.randint(0, 80000)
              image1=cv2.imread(cocoids[cid])/255
              w1,h1,_=image1.shape
              #import pdb;pdb.set_trace()
              if w1<w+10 or h1<w+10:
                image1=cv2.resize(image1, (w+10,w+10))
                nw=0;nh=0
              else:
                nw=np.random.randint(0, w1-w)
                nh=np.random.randint(0, h1-w)
              #import pdb;pdb.set_trace() 
              #try: 
              rgbtensor[a] = torch.from_numpy(image1[nw:nw + w, nh:nh + w].astype("float32"))
              #except:
              #  import pdb;pdb.set_trace()
            cid4 = np.random.randint(1, 4)
            for n1 in range(cid4):
                cid2 = np.random.randint(2, 28)
                cid3 = np.random.randint(1, 500)
                if cid2==int(objid):
                    cid2=objid+1

                objPath = datasetPath + "/train_primesense/0000" + str(cid2).zfill(2) + "/"
                c1 = cv2.imread(objPath + "rgb/" + str(cid3).zfill(6) + ".png")[100:300,100:300]
                m1 = cv2.imread(objPath + "mask/" + str(cid3).zfill(6) + "_000000.png")[100:300,100:300]/255



                ten1, dst_mask, rowX, colX, rows, cols= augmentImage(c1, m1, imX)
                maskID1 = np.where(dst_mask[:, :, 0] == 1)
                rgbtensor[a][rowX+maskID1[0], colX+maskID1[1]] =ten1[maskID1[0],maskID1[1]]

        except():
            print("bad choice")
            #import pdb;pdb.set_trace()

    # pctensor = torch.zeros((transformNo, onePC.shape[0], 3))
    trR = torch.zeros(transformNo, 1 )
    trT = torch.zeros(transformNo, 2)
    trS = torch.zeros(transformNo,1)
    for id in range(transformNo):
        randScale = scaleFac + (1-scaleFac) * np.random.uniform(0,1)
        randTra = ((np.random.uniform(0,1,2) - 0.5) * imX/2).astype("int64")
        randRot = int(np.random.uniform(0,1) * 360)

        aug = AB.Compose([
            AB.ColorJitter(),
            # A.HorizontalFlip(p=0.5),
            AB.RandomBrightnessContrast(p=0.2),
        ])

        rows, cols,_ = rgb.shape
        # cols-1 and rows-1 are the coordinate limits.
        M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), randRot, randScale)
        trR[id] = (randRot)
        trT[id] = torch.from_numpy(randTra)
        trS[id] = (randScale)

        M[:,2]=M[:,2]+randTra

        dst = cv2.warpAffine(rgb, M, (cols, rows))
        dst_mask = cv2.warpAffine(mask, M, (cols, rows))

        # rgbtensor[id] = torch.from_numpy(dst.astype(np.float32))
        masktensor[id]=torch.from_numpy(dst_mask)
        if np.random.uniform(0,1)>0.5 or not maskErosion:
            maskID=torch.where(masktensor[id].view(-1)==1)
        else:
            try:
                maskID=torch.where(createOcclusionsOnMaskOld(masktensor[id].cpu().numpy()).view(-1)==1)
            except:
                print("bad mask")
                maskID = torch.where(masktensor[id].view(-1) == 1)
        if len(maskID[0])>100:
          masktensor[id]=masktensor[id]*0
          masktensor[id].view(-1)[maskID]=1
        else:
          maskID = torch.where(masktensor[id].view(-1) == 1)
        
        rgbtensor[id].view(-1,3)[maskID] = torch.from_numpy(aug(image=dst)["image"]).view(-1,3)[maskID]

    return  rgbtensor,masktensor, trR, trT, trS
def readImagesWithBGClean3(rgb, mask, transformNo, cocoids, datasetPath, aug=True, objid=0, scaleFac=0.5, maskErosion=True):

    imX=rgb.shape[1]
    rgbtensor = torch.zeros((transformNo, imX, imX, 3))
    masktensor = torch.zeros((transformNo, imX, imX))
    if aug:
      for a in range(transformNo):
        w=rgbtensor.shape[1]
        w1=0
        h1=0
        try:
            if  np.random.uniform(0,1)>0.3:
              cid=np.random.randint(0, 80000)
              image1=cv2.imread(cocoids[cid])/255
              w1,h1,_=image1.shape
              #import pdb;pdb.set_trace()
              if w1<w+10 or h1<w+10:
                image1=cv2.resize(image1, (w+10,w+10))
                nw=0;nh=0
              else:
                nw=np.random.randint(0, w1-w)
                nh=np.random.randint(0, h1-w)
              #import pdb;pdb.set_trace() 
              #try: 
              rgbtensor[a] = torch.from_numpy(image1[nw:nw + w, nh:nh + w].astype("float32"))
              #except:
              #  import pdb;pdb.set_trace()
            cid4 = np.random.randint(1, 4)
            for n1 in range(cid4):
                cid2 = np.random.randint(2, 28)
                cid3 = np.random.randint(1, 500)
                if cid2==int(objid):
                    cid2=objid+1

                objPath = datasetPath + "/train_primesense/0000" + str(cid2).zfill(2) + "/"
                c1 = cv2.imread(objPath + "rgb/" + str(cid3).zfill(6) + ".png")[100:300,100:300]
                m1 = cv2.imread(objPath + "mask/" + str(cid3).zfill(6) + "_000000.png")[100:300,100:300]/255



                ten1, dst_mask, rowX, colX, rows, cols= augmentImage(c1, m1, imX)
                maskID1 = np.where(dst_mask[:, :, 0] == 1)
                rgbtensor[a][rowX+maskID1[0], colX+maskID1[1]] =ten1[maskID1[0],maskID1[1]]

        except():
            print("bad choice")
            #import pdb;pdb.set_trace()

    # pctensor = torch.zeros((transformNo, onePC.shape[0], 3))
    trR = torch.zeros(transformNo, 1 )
    trT = torch.zeros(transformNo, 2)
    trS = torch.zeros(transformNo,1)
    for id in range(transformNo):
        randScale = scaleFac + (1-scaleFac) * np.random.uniform(0,1)
        randTra = ((np.random.uniform(0,1,2) - 0.5) * imX/2).astype("int64")
        randRot = int(np.random.uniform(0,1) * 360)

        aug = AB.Compose([
            AB.ColorJitter(),
            # A.HorizontalFlip(p=0.5),
            AB.RandomBrightnessContrast(p=0.2),
        ])

        rows, cols,_ = rgb.shape
        # cols-1 and rows-1 are the coordinate limits.
        M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), randRot, randScale)
        trR[id] = (randRot)
        trT[id] = torch.from_numpy(randTra)
        trS[id] = (randScale)

        M[:,2]=M[:,2]+randTra

        dst = cv2.warpAffine(rgb, M, (cols, rows))
        dst_mask = cv2.warpAffine(mask, M, (cols, rows))

        # rgbtensor[id] = torch.from_numpy(dst.astype(np.float32))
        masktensor[id]=torch.from_numpy(dst_mask)
        if np.random.uniform(0,1)>0.5 or not maskErosion:
            maskID=torch.where(masktensor[id].view(-1)==1)
        else:
            try:
                maskID=torch.where(createOcclusionsOnMaskInSpecficRegion(masktensor[id].cpu().numpy()).view(-1)==1)
            except:
                print("bad mask")
                maskID = torch.where(masktensor[id].view(-1) == 1)

        rgbtensor[id].view(-1,3)[maskID] = torch.from_numpy(aug(image=dst)["image"]).view(-1,3)[maskID]

    return  rgbtensor,masktensor, trR, trT, trS    

def readImagesWithBGCleanOcc(rgb, mask, transformNo, cocoids, datasetPath, aug=True, objid=0, scaleFac=0.5, maskErosion=True):

    imX=rgb.shape[1]
    rgbtensor = torch.zeros((transformNo, imX, imX, 3))
    masktensor = torch.zeros((transformNo, imX, imX))
    if aug:
      for a in range(transformNo):
        w=rgbtensor.shape[1]
        w1=0
        h1=0
        try:
            if  np.random.uniform(0,1)>0.3:
              cid=np.random.randint(0, 80000)
              image1=cv2.imread(cocoids[cid])/255
              w1,h1,_=image1.shape
              #import pdb;pdb.set_trace()
              if w1<w+10 or h1<w+10:
                image1=cv2.resize(image1, (w+10,w+10))
                nw=0;nh=0
              else:
                nw=np.random.randint(0, w1-w)
                nh=np.random.randint(0, h1-w)
              #import pdb;pdb.set_trace() 
              #try: 
              rgbtensor[a] = torch.from_numpy(image1[nw:nw + w, nh:nh + w].astype("float32"))
              #except:
              #  import pdb;pdb.set_trace()
            cid4 = np.random.randint(1, 4)
            for n1 in range(cid4):
                cid2 = np.random.randint(2, 28)
                cid3 = np.random.randint(1, 500)
                if cid2==int(objid):
                    cid2=objid+1

                objPath = datasetPath + "/train_primesense/0000" + str(cid2).zfill(2) + "/"
                c1 = cv2.imread(objPath + "rgb/" + str(cid3).zfill(6) + ".png")[100:300,100:300]
                m1 = cv2.imread(objPath + "mask/" + str(cid3).zfill(6) + "_000000.png")[100:300,100:300]/255



                ten1, dst_mask, rowX, colX, rows, cols= augmentImage(c1, m1, imX)
                maskID1 = np.where(dst_mask[:, :, 0] == 1)
                rgbtensor[a][rowX+maskID1[0], colX+maskID1[1]] =ten1[maskID1[0],maskID1[1]]

        except():
            print("bad choice")
            #import pdb;pdb.set_trace()

    # pctensor = torch.zeros((transformNo, onePC.shape[0], 3))
    trR = torch.zeros(transformNo, 1 )
    trT = torch.zeros(transformNo, 2)
    trS = torch.zeros(transformNo,1)
    for id in range(transformNo):
        randScale = scaleFac + (1-scaleFac) * np.random.uniform(0,1)
        randTra = ((np.random.uniform(0,1,2) - 0.5) * imX/2).astype("int64")
        randRot = int(np.random.uniform(0,1) * 360)

        aug = AB.Compose([
            AB.ColorJitter(),
            # A.HorizontalFlip(p=0.5),
            AB.RandomBrightnessContrast(p=0.2),
        ])

        rows, cols,_ = rgb.shape
        # cols-1 and rows-1 are the coordinate limits.
        M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), randRot, randScale)
        trR[id] = (randRot)
        trT[id] = torch.from_numpy(randTra)
        trS[id] = (randScale)

        M[:,2]=M[:,2]+randTra

        dst = cv2.warpAffine(rgb, M, (cols, rows))
        dst_mask = cv2.warpAffine(mask, M, (cols, rows))

        # rgbtensor[id] = torch.from_numpy(dst.astype(np.float32))
        masktensor[id]=torch.from_numpy(dst_mask)
        if np.random.uniform(0,1)>0.8 or not maskErosion:
            maskID=torch.where(masktensor[id].view(-1)==1)
        else:
            try:
                maskID=torch.where(createOcclusionsOnMaskInSpecficRegionLine(masktensor[id].cpu().numpy()).reshape(-1)==1)
            except:
                print("bad mask")
                maskID = torch.where(masktensor[id].view(-1) == 1)
        
        #import pdb;pdb.set_trace()
        if len(maskID[0])>100:
          masktensor[id]=masktensor[id]*0
          masktensor[id].view(-1)[maskID]=1
        else:
          maskID = torch.where(masktensor[id].view(-1) == 1)
        
        rgbtensor[id].view(-1,3)[maskID] = torch.from_numpy(aug(image=dst)["image"]).view(-1,3)[maskID]

    return  rgbtensor,masktensor, trR, trT, trS        

def lineErosion(mask):
    #print("lineerosion")
    x2, y2, w2, h2 = cv2.boundingRect(mask )
    a1,b1=mask.shape
    ha1=int(a1/2)
    hb1=int(b1/2)
    dX=400
    hdX=int(dX/2)
    dummyMask=np.zeros((dX,dX))
    dummyMask[hdX-hb1:hdX+hb1,hdX-ha1:hdX+ha1]=mask
    randRot=np.random.randint(0,360)
    #import pdb;pdb.set_trace()
    M = cv2.getRotationMatrix2D(((dX - 1) / 2.0, (dX - 1) / 2.0), randRot, 1)
    M_inv = cv2.getRotationMatrix2D(((dX - 1) / 2.0, (dX - 1) / 2.0), -randRot, 1)
    dummyMask = cv2.warpAffine(dummyMask, M, (dX, dX))
    rint1=np.random.randint(hdX-ha1+x2,hdX-ha1+x2+w2)
    rint2=np.random.randint(hdX-hb1+y2,hdX-hb1+y2+h2)
    #import pdb;pdb.set_trace()
    if rint1%2==0:
      dummyMask[rint2:,:]=0
    else:
      dummyMask[:,rint1:]=0
    
    dummyMask = cv2.warpAffine(dummyMask, M_inv, (dX, dX))
    mask=dummyMask[hdX-hb1:hdX+hb1,hdX-ha1:hdX+ha1]
    return mask    
def createOcclusionsOnMask(mask):
    a1=1
    mask=mask.astype(np.uint8)

    kernel = np.ones((10, 10), np.uint8)
    val=np.random.uniform(0, 1)
    if val < 0.2:
        mask1 = cv2.erode(mask, kernel)
        if mask1.sum()>100:
            return torch.from_numpy(mask1)
        else:
            return torch.from_numpy(mask)
    if val <0.4:
        mask1 = cv2.erode(mask, kernel)
        if mask1.sum()>100:
            return torch.from_numpy(mask1)
        else:
            x, y, w, h = cv2.boundingRect((mask1 * 255).astype("uint8"))
            nx = np.random.randint(x, x + w)
            ny = np.random.randint(y, y + h)
            nw = np.random.randint(0, np.min((w, 30)))
            nh = np.random.randint(0, np.min((h, 30)))
            mask1[ny:ny+nh, nx:nx+nw ]=0
            if mask1.sum()<100:
                      return torch.from_numpy(mask)            
        return torch.from_numpy(mask1)
    if val<0.8:
        mask1 = cv2.erode(mask, kernel)
        if mask1.sum()<100:
            return torch.from_numpy(mask)
        else:
            mask1=lineErosion(mask1)
            if mask1.sum()<100:
                      return torch.from_numpy(mask)            
        return torch.from_numpy(mask1)   
    x, y, w, h = cv2.boundingRect((mask * 255).astype("uint8"))
    nx = np.random.randint(x, x + w)
    ny = np.random.randint(y, y + h)
    nw = np.random.randint(0, np.min((w,30)))
    nh = np.random.randint(0, np.min((h,30)))
    mask[ny:ny + nh, nx:nx + nw] = 0
    return torch.from_numpy(mask)

def createOcclusionsOnMaskOld(mask):
    a1=1
    mask=mask.astype(np.uint8)

    kernel = np.ones((10, 10), np.uint8)
    val=np.random.uniform(0, 1)
    if val < 0.3:
        mask1 = cv2.erode(mask, kernel)
        if mask1.sum()>100:
            return torch.from_numpy(mask1)
        else:
            return torch.from_numpy(mask)
    if val <0.7:
        mask1 = cv2.erode(mask, kernel)
        if mask1.sum()>100:
            return torch.from_numpy(mask1)
        else:
            x, y, w, h = cv2.boundingRect((mask1 * 255).astype("uint8"))
            nx = np.random.randint(x, x + w)
            ny = np.random.randint(y, y + h)
            nw = np.random.randint(0, np.min((w, 30)))
            nh = np.random.randint(0, np.min((h, 30)))
            mask1[ny:ny+nh, nx:nx+nw ]=0
        return torch.from_numpy(mask1)
    
    x, y, w, h = cv2.boundingRect((mask * 255).astype("uint8"))
    nx = np.random.randint(x, x + w)
    ny = np.random.randint(y, y + h)
    nw = np.random.randint(0, np.min((w,30)))
    nh = np.random.randint(0, np.min((h,30)))
    mask[ny:ny + nh, nx:nx + nw] = 0
    return torch.from_numpy(mask)


def createOcclusionsOnMaskInSpecficRegionLine(mask):
    a1=1
    kernel = np.ones((10, 10), np.uint8)
    mask=mask.astype(np.uint8)
    val=np.random.uniform(0, 1)
    
    #import pdb;pdb.set_trace()
    if val < 0.2:
        int1 = np.random.randint(0, mask.shape[0])
        int2 = np.random.randint(0, mask.shape[1])
        int3 = np.random.randint(5, 20)
        int4 = np.random.randint(5, 20)
        
        kernel = np.ones((int3, int4), np.uint8)
        
        mask1=mask.copy()
        x1, y1, w1, h1 = cv2.boundingRect((mask1 * 255).astype("uint8"))
        int1 = np.random.randint(x1, x1+int(w1/1.5))
        int2 = np.random.randint(y1, y1+int(h1/1.5))
        mask1[int2:,int1:] = cv2.erode(mask[int2: ,int1:], kernel)
        #import pdb;pdb.set_trace()
        if mask1.sum()>100:
            return torch.from_numpy(mask1)
        else:
            return torch.from_numpy(mask)
    if val <0.4:
        mask1 = cv2.erode(mask, kernel)
        if mask1.sum()<100:
            return torch.from_numpy(mask)
        else:
            mask1=lineErosion(mask1)
            if mask1.sum()<100:
                      return torch.from_numpy(mask)   
        return torch.from_numpy(mask1)
    if val<0.8:
            mask1=lineErosion(mask)
            if mask1.sum()<100:
                      return torch.from_numpy(mask)   
            return torch.from_numpy(mask1)
        
    x, y, w, h = cv2.boundingRect((mask * 255).astype("uint8"))
    nx = np.random.randint(x, x + w)
    ny = np.random.randint(y, y + h)
    nw = np.random.randint(0, np.min((w,60)))
    nh = np.random.randint(0, np.min((h,60)))
    mask[ny:ny + nh, nx:nx + nw] = 0
    return torch.from_numpy(mask)
    
def createOcclusionsOnMaskInSpecficRegion(mask):
    a1=1
    kernel = np.ones((10, 10), np.uint8)
    mask=mask.astype(np.uint8)
    val=np.random.uniform(0, 1)
    #import pdb;pdb.set_trace()
    if val < 0.4:
        int1 = np.random.randint(0, mask.shape[0])
        int2 = np.random.randint(0, mask.shape[1])
        int3 = np.random.randint(5, 20)
        int4 = np.random.randint(5, 20)
        
        kernel = np.ones((int3, int4), np.uint8)
        
        mask1=mask.copy()
        x1, y1, w1, h1 = cv2.boundingRect((mask1 * 255).astype("uint8"))
        int1 = np.random.randint(x1, x1+int(w1/1.5))
        int2 = np.random.randint(y1, y1+int(h1/1.5))
        mask1[int2:,int1:] = cv2.erode(mask[int2: ,int1:], kernel)
        #import pdb;pdb.set_trace()
        if mask1.sum()>100:
            return torch.from_numpy(mask1)
        else:
            return torch.from_numpy(mask)
    if val <0.7:
        mask1 = cv2.erode(mask, kernel)
        if mask1.sum()>100:
            return torch.from_numpy(mask1)
        else:
            x, y, w, h = cv2.boundingRect((mask1 * 255).astype("uint8"))
            nx = np.random.randint(x, x + w)
            ny = np.random.randint(y, y + h)
            nw = np.random.randint(0, np.min((w, 30)))
            nh = np.random.randint(0, np.min((h, 30)))
            mask1[ny:ny+nh, nx:nx+nw ]=0
        return torch.from_numpy(mask1)
    x, y, w, h = cv2.boundingRect((mask * 255).astype("uint8"))
    nx = np.random.randint(x, x + w)
    ny = np.random.randint(y, y + h)
    nw = np.random.randint(0, np.min((w,60)))
    nh = np.random.randint(0, np.min((h,60)))
    mask[ny:ny + nh, nx:nx + nw] = 0
    return torch.from_numpy(mask)

def readImagesWithBGClean2(rgb, mask, transformNo, cocoids, datasetPath, aug=True, objid=0, scaleFac=0.5, maskErosion=True):

    imX=rgb.shape[1]
    rgbtensor = torch.zeros((transformNo, imX, imX, 3))
    masktensor = torch.zeros((transformNo, imX, imX))
    if aug:
      for a in range(transformNo):
        w=rgbtensor.shape[1]
        w1=0
        h1=0
        try:
            if  np.random.uniform(0,1)>0.3:
              cid=np.random.randint(0, 80000)
              image1=cv2.imread(cocoids[cid])/255
              w1,h1,_=image1.shape
              #import pdb;pdb.set_trace()
              if w1<w+10 or h1<w+10:
                image1=cv2.resize(image1, (w+10,w+10))
                nw=0;nh=0
              else:
                nw=np.random.randint(0, w1-w)
                nh=np.random.randint(0, h1-w)
              #import pdb;pdb.set_trace() 
              #try: 
              rgbtensor[a] = torch.from_numpy(image1[nw:nw + w, nh:nh + w].astype("float32"))
              #except:
              #  import pdb;pdb.set_trace()
            cid4 = np.random.randint(1, 4)
            for n1 in range(cid4):
                cid2 = np.random.randint(2, 28)
                cid3 = np.random.randint(1, 500)
                if cid2==int(objid):
                    cid2=objid+1

                objPath = datasetPath + "/train_primesense/0000" + str(cid2).zfill(2) + "/"
                c1 = cv2.imread(objPath + "rgb/" + str(cid3).zfill(6) + ".png")[100:300,100:300]
                m1 = cv2.imread(objPath + "mask/" + str(cid3).zfill(6) + "_000000.png")[100:300,100:300]/255



                ten1, dst_mask, rowX, colX, rows, cols= augmentImage(c1, m1, imX)
                maskID1 = np.where(dst_mask[:, :, 0] == 1)
                rgbtensor[a][rowX+maskID1[0], colX+maskID1[1]] =ten1[maskID1[0],maskID1[1]]

        except():
            print("bad choice")
            #import pdb;pdb.set_trace()

    # pctensor = torch.zeros((transformNo, onePC.shape[0], 3))
    trR = torch.zeros(transformNo, 1 )
    trT = torch.zeros(transformNo, 2)
    trS = torch.zeros(transformNo,1)
    for id in range(transformNo):
        randScale = scaleFac + (1-scaleFac) * np.random.uniform(0,1)
        randTra = ((np.random.uniform(0,1,2) - 0.5) * imX/2).astype("int64")
        randRot = int(np.random.uniform(0,1) * 360)

        aug = AB.Compose([
            AB.ColorJitter(),
            # A.HorizontalFlip(p=0.5),
            AB.RandomBrightnessContrast(p=0.6),
        ])

        rows, cols,_ = rgb.shape
        # cols-1 and rows-1 are the coordinate limits.
        M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), randRot, randScale)
        trR[id] = (randRot)
        trT[id] = torch.from_numpy(randTra)
        trS[id] = (randScale)

        M[:,2]=M[:,2]+randTra

        dst = cv2.warpAffine(rgb, M, (cols, rows))
        dst_mask = cv2.warpAffine(mask, M, (cols, rows))

        # rgbtensor[id] = torch.from_numpy(dst.astype(np.float32))
        masktensor[id]=torch.from_numpy(dst_mask)
        if np.random.uniform(0,1)>0.5 or not maskErosion:
            maskID=torch.where(masktensor[id].view(-1)==1)
        else:
            try:
                maskID=torch.where(createOcclusionsOnMask(masktensor[id].cpu().numpy()).view(-1)==1)
            except:
                print("bad mask")
                maskID = torch.where(masktensor[id].view(-1) == 1)

        rgbtensor[id].view(-1,3)[maskID] = torch.from_numpy(aug(image=dst)["image"]).view(-1,3)[maskID]

    return  rgbtensor,masktensor, trR, trT, trS    
def readImagesWithBG(rgb, mask, transformNo, cocoids):
    import glob
    import albumentations as AB
    maxDim = rgb.shape[0] * rgb.shape[0]

    # rgb=
    imX = rgb.shape[0]
    indices = np.nonzero(mask)

    ffpoints = np.zeros((indices[0].shape[0], 6))
    ffpoints[:, 3:6] = (rgb[indices[0], indices[1]])

    ffpoints[:, 0] = indices[0]
    ffpoints[:, 1] = indices[1]

    onePC = np.copy(ffpoints[:, 0:3])
    oneCol = np.copy(ffpoints[:, 3:])
    oneIM = torch.zeros((imX, imX, 3))
    # import pdb;pdb.set_trace()
    oneIM.view(-1, 3)[(ffpoints[:, 1].astype(int) + imX * ffpoints[:, 0].astype(int))] = torch.from_numpy(
        ffpoints[:, 3:6].astype(np.float32))

    rgbtensor = torch.zeros((transformNo, imX, imX, 3))
    for a in range(transformNo):
        w=rgbtensor.shape[2]
        w1=0
        h1=0
        try:
            while(w1<w or h1<w):
                cid=np.random.randint(0, 80000)
                image1=cv2.imread(cocoids[cid])/255
                w1,h1,_=image1.shape
            # print("w is",w1)
            # print("w is", h1)

            nw=np.random.randint(0, w1-w)
            nh=np.random.randint(0, h1-w)

            rgbtensor[a]=torch.from_numpy(image1[nw:nw+w,nh:nh+w].astype("float32"))
        except:
            print("bad choice")
    pctensor = torch.zeros((transformNo, onePC.shape[0], 3))
    trR = torch.zeros(transformNo, 3, 3)
    trT = torch.zeros(transformNo, 3)
    trS = torch.zeros(transformNo, 1)
    octensor = torch.zeros((transformNo, onePC.shape[0]), dtype=torch.bool)
    # if c1==1:
    # import pdb;pdb.set_trace()
    for id in range(transformNo):
        randScale = 0.8 + 0.2 * torch.rand(1)
        randTra = (torch.rand(3) - 0.5) * 1.3
        randTra[2] = 0
        randRot = (torch.rand(3) - 0.5) * 2 * np.pi
        randRot[1] = 0
        randRot[0] = 0
        trS[id] = randScale

        # if epoch<50:
        #    randRot[2] = 0

        minWidth = (imX - np.max([np.ptp(onePC[:, 0]), np.ptp(onePC[:, 1])])) / 2
        trT[id] = randTra * minWidth

        trR[id] = rotfromeuler(randRot)
        imcent=np.zeros((3))
        imcent[0:2]=imX/2
        curPC = (np.copy(onePC - imcent).dot(trR[id].numpy()) + trT[id].numpy() + imcent) * trS[id].numpy()

        if np.max(curPC[:, 0:2]) >= (imX - 2):
            off = imX - np.max(curPC) - 2
            if np.max(curPC[:, 0]) > np.max(curPC[:, 1]):
                curPC[:, 0] = curPC[:, 0] + off
                trT[id][0] = trT[id][0] + off / trS[id]
            else:
                curPC[:, 1] = curPC[:, 1] + off
                trT[id][1] = trT[id][1] + off / trS[id]

        # print("max",id)
        if np.min(curPC[:, 0:2]) <= 2:
            off = -np.min(curPC) + 2
            if np.min(curPC[:, 0]) < np.min(curPC[:, 1]):
                curPC[:, 0] = curPC[:, 0] + off
                trT[id][0] = trT[id][0] + off
            else:
                curPC[:, 1] = curPC[:, 1] + off
                trT[id][1] = trT[id][1] + off

        if id == 0:
            curPC = np.copy(onePC)
        pctensor[id] = torch.from_numpy(curPC)
        idx_o = returnOccludedPC(np.copy(curPC))

        idx_o = np.setdiff1d(idx_o, np.where(curPC > 127)[0])
        idx_o = np.setdiff1d(idx_o, np.where(curPC < 0)[0])

        aug = AB.Compose([
            AB.ColorJitter(),
            # A.HorizontalFlip(p=0.5),
            AB.RandomBrightnessContrast(p=0.2),
        ])

        if len(idx_o) > 5 and id > 0:
            curPC = curPC[idx_o]
            rgbtensor[id].view(-1, 3)[(curPC[:, 1].astype(int) + imX * curPC[:, 0].astype(int))] = torch.from_numpy(
                oneCol[idx_o].astype(np.float32))
            rgbtensor[id] = torch.from_numpy(aug(image=rgbtensor[id].numpy())["image"])
            octensor[id][idx_o] = True
        else:
            rgbtensor[id].view(-1, 3)[(curPC[:, 1].astype(int) + imX * curPC[:, 0].astype(int))] = torch.from_numpy(
                oneCol.astype(np.float32))
            octensor[id] = True
            rgbtensor[id] = torch.from_numpy(aug(image=rgbtensor[id].numpy())["image"])
        #     print("a1")
        # cv2.imwrite(str(id)+".jpg", rgbtensor[id].cpu().numpy()*255)

    return oneIM, onePC, oneCol, rgbtensor, trR, trT, trS, pctensor, octensor


def readImagesWithOnlyOcclusion(rgb, mask, transformNo):
    import glob
    import albumentations as AB
    maxDim=rgb.shape[0]*rgb.shape[0]
    # rgb.shapergb=cv2.imre
    # if np.max(np.nonzero(mask)[0])>imX or np.max(np.nonzero(mask)[1])>imX:
    #  maxD=np.max(np.nonzero(mask)[0],np.nonzero(mask)[1])
    # import pdb;pdb.set_trace()
   

    # rgb=
    imX=rgb.shape[0]
    indices = np.nonzero(mask)

    ffpoints = np.zeros((indices[0].shape[0], 6))
    ffpoints[:, 3:6] = (rgb[indices[0], indices[1]])

    ffpoints[:, 0] = indices[0]
    ffpoints[:, 1] = indices[1]

    onePC = np.copy(ffpoints[:, 0:3])
    oneCol = np.copy(ffpoints[:, 3:])
    oneIM = torch.zeros((imX, imX, 3))
    #import pdb;pdb.set_trace()
    oneIM.view(-1, 3)[(ffpoints[:, 1].astype(int) + imX * ffpoints[:, 0].astype(int))] = torch.from_numpy(ffpoints[:, 3:6].astype(np.float32))

    rgbtensor = torch.zeros((transformNo, imX, imX, 3))
    pctensor = torch.zeros((transformNo, onePC.shape[0], 3))
    trR = torch.zeros(transformNo, 3, 3)
    trT = torch.zeros(transformNo, 3)
    trS = torch.zeros(transformNo, 1)
    octensor = torch.zeros((transformNo, onePC.shape[0]), dtype=torch.bool)
    # if c1==1:
    # import pdb;pdb.set_trace()
    for id in range(transformNo):
        randScale = 0
        randTra = (torch.rand(3) - 0.5) * 0
        randTra[2] = 0
        randRot = (torch.rand(3) - 0.5) * 0
        randRot[1] = 0
        randRot[0] = 0
        trS[id]=1

        # if epoch<50:
        #    randRot[2] = 0

        minWidth = (imX - np.max([np.ptp(onePC[:, 0]), np.ptp(onePC[:, 1])])) / 2
        trT[id] = randTra * minWidth

        trR[id] = rotfromeuler(randRot)

        curPC = (np.copy(onePC - np.mean(onePC, axis=0)).dot(trR[id].numpy()) + trT[id].numpy() + np.mean(onePC,
                                                                                                          axis=0))


        if id==0:
            curPC=np.copy(onePC)
        pctensor[id] = torch.from_numpy(curPC)
        idx_o = returnOccludedPC(np.copy(curPC))
        
        idx_o=np.setdiff1d(idx_o, np.where(curPC>127)[0])
        idx_o=np.setdiff1d(idx_o, np.where(curPC<0)[0])
        
        aug = AB.Compose([
            AB.ColorJitter(),
            # A.HorizontalFlip(p=0.5),
            AB.RandomBrightnessContrast(p=0.2),
        ])
        
        
        if len(idx_o) > 5 and id>0:
            curPC = curPC[idx_o]
            rgbtensor[id].view(-1, 3)[(curPC[:, 1].astype(int) + imX * curPC[:, 0].astype(int))] = torch.from_numpy(
                oneCol[idx_o].astype(np.float32))
            rgbtensor[id] = torch.from_numpy(aug(image=rgbtensor[id].numpy())["image"])
            octensor[id][idx_o] = True
        else:
            rgbtensor[id].view(-1, 3)[(curPC[:, 1].astype(int) + imX * curPC[:, 0].astype(int))] = torch.from_numpy(
                oneCol.astype(np.float32))
            octensor[id] = True
            rgbtensor[id] = torch.from_numpy(aug(image=rgbtensor[id].numpy())["image"])
        #     print("a1")
        # cv2.imwrite(str(id)+".jpg", rgbtensor[id].cpu().numpy()*255)

    return oneIM, onePC, oneCol, rgbtensor, trR, trT, pctensor, octensor


def readImagesWithOnlyOcclusionBG(rgb, mask, transformNo):
    import glob
    import albumentations as AB
    maxDim = rgb.shape[0] * rgb.shape[0]
    # rgb.shapergb=cv2.imre
    # if np.max(np.nonzero(mask)[0])>imX or np.max(np.nonzero(mask)[1])>imX:
    #  maxD=np.max(np.nonzero(mask)[0],np.nonzero(mask)[1])
    # import pdb;pdb.set_trace()

    # rgb=
    imX = rgb.shape[0]
    indices = np.nonzero(mask)

    ffpoints = np.zeros((indices[0].shape[0], 6))
    ffpoints[:, 3:6] = (rgb[indices[0], indices[1]])

    ffpoints[:, 0] = indices[0]
    ffpoints[:, 1] = indices[1]

    onePC = np.copy(ffpoints[:, 0:3])
    oneCol = np.copy(ffpoints[:, 3:])
    oneIM = torch.zeros((imX, imX, 3))
    # import pdb;pdb.set_trace()
    oneIM.view(-1, 3)[(ffpoints[:, 1].astype(int) + imX * ffpoints[:, 0].astype(int))] = torch.from_numpy(
        ffpoints[:, 3:6].astype(np.float32))

    rgbtensor = torch.zeros((transformNo, imX, imX, 3))
    pctensor = torch.zeros((transformNo, onePC.shape[0], 3))
    trR = torch.zeros(transformNo, 3, 3)
    trT = torch.zeros(transformNo, 3)
    trS = torch.zeros(transformNo, 1)
    octensor = torch.zeros((transformNo, onePC.shape[0]), dtype=torch.bool)
    # if c1==1:
    # import pdb;pdb.set_trace()
    for id in range(transformNo):
        randScale = 0
        randTra = (torch.rand(3) - 0.5) * 0
        randTra[2] = 0
        randRot = (torch.rand(3) - 0.5) * 0
        randRot[1] = 0
        randRot[0] = 0
        trS[id] = 1

        # if epoch<50:
        #    randRot[2] = 0

        minWidth = (imX - np.max([np.ptp(onePC[:, 0]), np.ptp(onePC[:, 1])])) / 2
        trT[id] = randTra * minWidth

        trR[id] = rotfromeuler(randRot)

        curPC = (np.copy(onePC - np.mean(onePC, axis=0)).dot(trR[id].numpy()) + trT[id].numpy() + np.mean(onePC,
                                                                                                          axis=0))

        if id == 0:
            curPC = np.copy(onePC)
        pctensor[id] = torch.from_numpy(curPC)
        idx_o = returnOccludedPC(np.copy(curPC))

        idx_o = np.setdiff1d(idx_o, np.where(curPC > 127)[0])
        idx_o = np.setdiff1d(idx_o, np.where(curPC < 0)[0])

        aug = AB.Compose([
            AB.ColorJitter(),
            # A.HorizontalFlip(p=0.5),
            AB.RandomBrightnessContrast(p=0.2),
        ])

        if len(idx_o) > 5 and id > 0:
            curPC = curPC[idx_o]
            rgbtensor[id].view(-1, 3)[(curPC[:, 1].astype(int) + imX * curPC[:, 0].astype(int))] = torch.from_numpy(
                oneCol[idx_o].astype(np.float32))
            rgbtensor[id] = torch.from_numpy(aug(image=rgbtensor[id].numpy())["image"])
            octensor[id][idx_o] = True
        else:
            rgbtensor[id].view(-1, 3)[(curPC[:, 1].astype(int) + imX * curPC[:, 0].astype(int))] = torch.from_numpy(
                oneCol.astype(np.float32))
            octensor[id] = True
            rgbtensor[id] = torch.from_numpy(aug(image=rgbtensor[id].numpy())["image"])
        #     print("a1")
        # cv2.imwrite(str(id)+".jpg", rgbtensor[id].cpu().numpy()*255)

    return oneIM, onePC, oneCol, rgbtensor, trR, trT, pctensor, octensor
def generate_rendered_nerf(neural_radiance_field, RFull, TFull, KFull, device, renderer_grid):
    frames = []
    framesEmb = []

    print('Rendering rotating NeRF ...')
    for R, T, K in zip(RFull, TFull, KFull):
        camera = FoVPerspectiveCameras(
            R=R[None],
            T=T[None],
            K=K[None],
            device=device,
        )
        # Note that we again render with `NDCMultinomialRaysampler`
        # and the batched_forward function of neural_radiance_field.
        frames.append(
            renderer_grid(
                cameras=camera,
                volumetric_function=neural_radiance_field.batched_forward,
            )[0][..., :12].cpu()
        )
        framesEmb.append(
            renderer_grid(
                cameras=camera,
                volumetric_function=neural_radiance_field.batched_forward,
            )[0][..., 12:].cpu()
        )
    return torch.cat(frames), torch.cat(framesEmb)

import math
from typing import List

import torch
from pytorch3d.renderer import MonteCarloRaysampler, NDCMultinomialRaysampler, RayBundle
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.renderer.implicit.sample_pdf import sample_pdf
