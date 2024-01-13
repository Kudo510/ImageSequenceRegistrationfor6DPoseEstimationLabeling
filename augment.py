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
def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)
need_pytorch3d = False
try:
    import pytorch3d
except ModuleNotFoundError:
    need_pytorch3d = True
if need_pytorch3d:
    if torch.__version__.startswith("1.12.") and sys.platform.startswith("linux"):
        # We try to install PyTorch3D via a released wheel.
        pyt_version_str = torch.__version__.split("+")[0].replace(".", "")
        version_str = "".join([
            f"py3{sys.version_info.minor}_cu",
            torch.version.cuda.replace(".", ""),
            f"_pyt{pyt_version_str}"
        ])
        get_ipython().system('pip install fvcore iopath')
        get_ipython().system(
            'pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html')
    else:
        # We try to install PyTorch3D from source.
        get_ipython().system('curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz')
        get_ipython().system('tar xzf 1.10.0.tar.gz')
        os.environ["CUB_HOME"] = os.getcwd() + "/cub-1.10.0"
        get_ipython().system("pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'")

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

from typing import Optional, Tuple, Union


def nt(t1):
    t1 = (t1 - 0.5) * 2
    return t1 / (0.01 + torch.norm(t1, dim=2).unsqueeze(2))

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
def augmentImage(rgb,mask,  imX, augMax=40 ):
    randScale = 0.75 + 0.25 * np.random.uniform(0, 1)
    randTra = ((np.random.uniform(0, 1, 2) - 0.5) * imX /1.2).astype("int64")
    randRot = int(np.random.uniform(0, 1) * 360)
    
    #randTra=randTra*0
    aug = AB.Compose([
            AB.ColorJitter(),
            AB.RandomBrightnessContrast(p=0.2),
            AB.GaussianBlur(blur_limit=(1, 3)),
            AB.ISONoise(),
            AB.CLAHE(),  
            AB.GaussianBlur(blur_limit=(1, 3))
    ])

    rows, cols, _ = rgb.shape
    # cols-1 and rows-1 are the coordinate limits.
    M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), randRot, randScale)

    M[:, 2] = M[:, 2] + randTra
    
    dst = cv2.warpAffine(rgb, M, (cols, rows))
    dst_mask = cv2.warpAffine(mask, M, (cols, rows))

    rows = np.random.randint(augMax, imX-40)
    cols = np.random.randint(augMax, imX-40)
    
    dst=cv2.resize(dst, (cols,rows))
    dst_mask=cv2.resize(dst_mask, (cols,rows))


    colX = np.random.randint(1, imX-cols-1)
    rowX= np.random.randint(1, imX-rows-1)
    return torch.from_numpy(dst.astype("float32"))/255, dst_mask, rowX, colX, rows, cols
def getTransformedBackground(background,rows, M_clone, randTra):

    bgrows, bgcols, _ = background.shape
    hbg = int(bgrows / 2)
    hw = int(rows / 2)

    dst_bg = cv2.warpAffine(background.cpu().numpy(), M_clone, (background.shape[1], background.shape[0]))
    dst_bg_crop = dst_bg[hbg - hw - randTra[1]:hbg + hw - randTra[1], hbg - hw - randTra[0]:hbg + hw - randTra[0]]

    return dst_bg_crop
def generateImages(rgb, mask, background, transformNo, cocoids, datasetPath, coco_aug=True,tless_aug= True, objid=0, scaleFac=0.5,maskErosion=True, maxScale=1.0, transScale=1.0,surfEmbScaling=False, surfEmbScaleFac=1,lineErosion=False,augMax=40, maskMax=7000, transScale2=0.2, borderADD=False):
    imX = rgb.shape[1]
    mask=mask.astype("uint8")
    if mask.max()<2:
        _, mask = cv2.threshold(mask, 0.9, 1, cv2.THRESH_BINARY)
    else:
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

    rgbtensor = torch.zeros((transformNo, imX, imX, 3))
    masktensor = torch.zeros((transformNo, imX, imX))
    masktensor_orig = torch.zeros((transformNo, imX, imX))

    orig_mask=mask.copy()
    if coco_aug:
        for a in range(transformNo):

            if np.random.uniform(0, 1) > 0.1:
                    cid = np.random.randint(0, len(cocoids)-1)
                    image1 = cv2.imread(cocoids[cid]) / 255
                    #import pdb;pdb.set_trace()
                    image1 = cv2.resize(image1, (image1.shape[1], image1.shape[0]))
                    w1, h1, _ = image1.shape
                    # import pdb;pdb.set_trace()
                    if w1 < imX + 10 or h1 < imX + 10:
                        image1 = cv2.resize(image1, (imX + 10, imX + 10))
                        nw = 0;nh = 0
                    else:
                    
                        nw = np.random.randint(0, w1 - imX)
                        nh = np.random.randint(0, h1 - imX)
                    rgbtensor[a] = torch.from_numpy(image1[nw:nw + imX, nh:nh + imX].astype("float32"))
    if maskErosion and np.random.uniform(0, 1) > 0.3:
            mask1=createOcclusionsWithoutErosion(mask)
            if mask1.sum()>maskMax:
                mask=mask1
            if lineErosion and np.random.uniform(0, 1) > 0.3:
                mask1=lineErode(mask)

                
            if mask1.sum()>maskMax:
                mask=mask1
                            
    trR = torch.zeros(transformNo, 1)
    trT = torch.zeros(transformNo, 2)
    trS = torch.zeros(transformNo, 1)
    for id in range(transformNo):
        #import pdb;pdb.set_trace()
        randScale = (scaleFac + (1 - scaleFac) * np.random.uniform(0, 1))*maxScale
        if surfEmbScaling:
          x3, y3, w3, h3 = cv2.boundingRect(mask.astype(np.uint8))
          scale = 224 / max(w3,h3) / 1.2
          scale = scale * np.random.uniform(1 - 0.05*surfEmbScaleFac, 1 + 0.05*surfEmbScaleFac)
          randScale=scale  
          #print("surfembscaling", surfEmbScaleFac)
        randTra1 = ((np.random.uniform(0, 1, 2) - 0.5) * transScale2*imX / 2).astype("int64")
        #randTra1=randTra1*0
        randTra = ((np.random.uniform(0, 1, 2) - 0.5) * transScale*imX / 2).astype("int64")

        randRot = int(np.random.uniform(0, 1) * 360)

        aug = AB.Compose([
            AB.ColorJitter(),
            AB.RandomBrightnessContrast(p=0.2),
            AB.GaussianBlur(blur_limit=(1, 3)),
            AB.ISONoise(),
            AB.CLAHE(),  
        ])

        rows, cols, _ = rgb.shape
        # cols-1 and rows-1 are the coordinate limits.
        M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), randRot, randScale)
        M_identity=cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 0, 1)
        trR[id] = (randRot)
        trT[id] = torch.from_numpy(randTra)
        trS[id] = (randScale)
        M_mask=M.copy()
        M_mask[:, 2] = M_mask[:, 2] + randTra

        M[:, 2] = M[:, 2] + randTra1
        M_identity[:, 2] = M_identity[:, 2] - randTra1+randTra

        dst = cv2.warpAffine(rgb, M, (cols, rows))
        dst_mask = cv2.warpAffine(mask, M, (cols, rows))
        dst_mask_orig = cv2.warpAffine(orig_mask, M_mask, (cols, rows))

        dst = cv2.warpAffine(dst, M_identity, (cols, rows))
        if background is not None:
           M_clone = cv2.getRotationMatrix2D(((background.shape[1] - 1) / 2.0, (background.shape[0] - 1) / 2.0),randRot, randScale)
           dst_bg= getTransformedBackground(background, rows, M_clone, randTra)
           if np.random.uniform(0, 1) > 0.5:
                rgbtensor[id] = torch.from_numpy(dst_bg.astype("float32"))
        dst_mask = cv2.warpAffine(dst_mask, M_identity, (cols, rows))

        masktensor[id] = torch.from_numpy(dst_mask)
        masktensor_orig[id] = torch.from_numpy(dst_mask_orig)

        maskID = torch.where(masktensor[id].view(-1) == 1)
        #import pdb;pdb.set_trace()

        #if np.random.uniform(0, 1)>1.3:
         #   rgbtensor[id].view(-1, 3)[maskID] = torch.from_numpy(aug(image=(dst*255).astype(np.uint8))["image"].astype("float32")/255).view(-1, 3)[maskID]
        #else:
         #   rgbtensor[id].view(-1, 3)[maskID] =torch.from_numpy(dst.astype("float32")).view(-1, 3)[maskID]
        
        if np.random.uniform(0, 1)>1.3:
            rgbtensor[id].view(-1, 3)[maskID] = torch.from_numpy(aug(image=(dst*255).astype(np.uint8))["image"].astype("float32")/255).view(-1, 3)[maskID]
        else:
            if background is not None:
                rgbtensor[id].view(-1, 3)[maskID] = torch.from_numpy(dst.astype("float32")).view(-1, 3)[maskID]
            else:
                if np.random.uniform(0, 1)>0.3:
                  rgbtensor[id].view(-1, 3)[maskID] =torch.from_numpy(aug(image=(dst*255).astype(np.uint8))["image"].astype("float32")/255).view(-1, 3)[maskID]
                else:
                  rgbtensor[id].view(-1, 3)[maskID] =torch.from_numpy(dst.astype("float32")).view(-1, 3)[maskID]
            



        cid4 = np.random.randint(1, 4)
        if tless_aug and np.random.uniform(0, 1)>0.3:
            for n1 in range(cid4):
                cid2 = np.random.randint(2, 28)
                cid3 = np.random.randint(1, 500)
                if cid2 == int(objid):
                    cid2 = int(objid) + 1

                objPath = datasetPath + "/train_primesense/0000" + str(cid2).zfill(2) + "/"
                c1 = cv2.imread(objPath + "rgb/" + str(cid3).zfill(6) + ".png")[100:300, 100:300]
                m1 = cv2.imread(objPath + "mask/" + str(cid3).zfill(6) + "_000000.png")[100:300, 100:300] / 255

                ten1, dst_mask, rowX, colX, rows, cols = augmentImage(c1, m1, imX,augMax)
                maskID1 = np.where(dst_mask[:, :, 0] == 1)
                if torch.sum(masktensor[0])-torch.sum(masktensor[0][rowX + maskID1[0], colX + maskID1[1]])>5000:
                    rgbtensor[0][rowX + maskID1[0], colX + maskID1[1]] = ten1[maskID1[0], maskID1[1]]
                    masktensor[0][rowX + maskID1[0], colX + maskID1[1]] = 0        
       
      
       
        if np.random.uniform(0, 1)>0.3:
             rgbtensor[id]= torch.from_numpy(aug(image=(rgbtensor[id].numpy()*255).astype(np.uint8))["image"].astype("float32")/255)

        if borderADD and np.random.uniform(0, 1)>0.6:
            borderMask = cv2.dilate(masktensor[id].cpu().numpy(), np.ones(( np.random.randint(1, 20),  np.random.randint(1, 20)), np.uint8), iterations=1)
            borderID = np.where(borderMask.reshape(-1) <= 0.9)
            rgbtensor[id].view(-1, 3)[borderID] =0

        if masktensor[id].max() > 1 or masktensor[id].max() == 255 or masktensor[id].max() > 200:
            print("bad masking")
    return rgbtensor, masktensor_orig,masktensor, trR, trT, trS

def lineErode(mask):
    #print("lineerosion")
    x2, y2, w2, h2 = cv2.boundingRect((mask * 255).astype("uint8"))
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
    

    #if np.sum(dummyMask[rint2:,:])<np.sum(dummyMask[:,rint1:]):
    if rint1%2==0:
      dummyMask[rint2:,:]=0
    else:
      dummyMask[:,rint1:]=0
    
    dummyMask = cv2.warpAffine(dummyMask, M_inv, (dX, dX))
    mask=dummyMask[hdX-hb1:hdX+hb1,hdX-ha1:hdX+ha1]
    return mask


def createOcclusionsWithoutErosion(mask):
    mask = mask.astype(np.uint8)

    val = np.random.uniform(0, 1)
    if val < 0.3:
            return mask
    if val < 0.99:

        x, y, w, h = cv2.boundingRect((mask * 255).astype("uint8"))
        nx = np.random.randint(x, x + w)
        ny = np.random.randint(y, y + h)
        rint1=np.random.randint(30, 70)
        rint2=np.random.randint(30, 70)
        nw = np.random.randint(0, np.min((w, rint1)))
        nh = np.random.randint(0, np.min((h, rint2)))
        mask[ny:ny + nh, nx:nx + nw] = 0
        return mask

    x, y, w, h = cv2.boundingRect((mask * 255).astype("uint8"))
    nx = np.random.randint(x, x + w)
    ny = np.random.randint(y, y + h)
    nw = np.random.randint(0, np.min((w, 30)))
    nh = np.random.randint(0, np.min((h, 30)))
    mask[ny:ny + nh, nx:nx + nw] = 0
    return mask


def createOcclusions(mask):

    mask=mask.astype(np.uint8)

    kernel = np.ones((10, 10), np.uint8)
    val=np.random.uniform(0, 1)
    if val < 0.3:
        mask1 = cv2.erode(mask, kernel)
        if mask1.sum()>100:
            return mask1
        else:
            return mask
    if val <0.7:
        mask1 = cv2.erode(mask, kernel)
        if mask1.sum()>100:
            return mask1
        else:
            x, y, w, h = cv2.boundingRect((mask1 * 255).astype("uint8"))
            nx = np.random.randint(x, x + w)
            ny = np.random.randint(y, y + h)
            nw = np.random.randint(0, np.min((w, 30)))
            nh = np.random.randint(0, np.min((h, 30)))
            mask1[ny:ny+nh, nx:nx+nw ]=0
        return mask1
    
    x, y, w, h = cv2.boundingRect((mask * 255).astype("uint8"))
    nx = np.random.randint(x, x + w)
    ny = np.random.randint(y, y + h)
    nw = np.random.randint(0, np.min((w,30)))
    nh = np.random.randint(0, np.min((h,30)))
    mask[ny:ny + nh, nx:nx + nw] = 0
    return mask

def denormalize(img: Union[np.ndarray, torch.Tensor]):
    imagenet_stats = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    mu, std = imagenet_stats
    if isinstance(img, torch.Tensor):
        mu, std = [torch.Tensor(v).type(img.dtype).to(img.device)[:, None, None] for v in (mu, std)]
    return img * std + mu
def applyRandomMask(mask,objid, datasetPath, bsize=100):
    cid2 = np.random.randint(2, 28)
    cid3 = np.random.randint(1, 500)
    if cid2 == int(objid):
        cid2 = objid + 1

    objPath = datasetPath + "/train_primesense/0000" + str(cid2).zfill(2) + "/"
    m1 = cv2.imread(objPath + "mask/" + str(cid3).zfill(6) + "_000000.png")[100:300, 100:300,0]
    col1 = cv2.imread(objPath + "rgb/" + str(cid3).zfill(6) + ".png")[100:300, 100:300]
    x, y, w, h = cv2.boundingRect(m1)
    m1=m1[y:y+h, x:x+h]
    col1 = col1[y:y + h, x:x + h]




    m1=cv2.resize(m1, (bsize,bsize))
    idx1 = np.where(m1)
    mas1=m1.copy()
    m1[:]=255
    m1[idx1]=0
    m1=m1/255
    col1=cv2.resize(col1, (bsize,bsize))

    x, y, w, h = cv2.boundingRect((mask * 255).astype("uint8"))
    h1=0;w1=0
    if np.random.uniform(0, 1)>0.5:
        h1 = np.random.randint(0, mask.shape[0]-bsize-10)
    else:
        w1 = np.random.randint(0, mask.shape[0]-bsize-10)

    mask[h1:h1+bsize,w1:w1+bsize]=mask[h1:h1+bsize,w1:w1+bsize]*m1

    return mask, col1,mas1, h1,w1
def createBlobsCV(mask):
    from numpy.random import default_rng
    import skimage
    height, width = mask.shape[:2]

    # define random seed to change the pattern
    seedval = 75
    rng = default_rng(seed=seedval)

    # create random noise image
    noise = rng.integers(0, 255, (height, width), np.uint8, True)

    # blur the noise image to control the size
    blur = cv2.GaussianBlur(noise, (0, 0), sigmaX=15, sigmaY=15, borderType=cv2.BORDER_DEFAULT)

    # stretch the blurred image to full dynamic range
    stretch = skimage.exposure.rescale_intensity(blur, in_range='image', out_range=(0, 255)).astype(np.uint8)

    # threshold stretched image to control the size
    thresh = cv2.threshold(stretch, 175, 255, cv2.THRESH_BINARY)[1]

    # apply morphology open and close to smooth out and make 3 channels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.merge([mask, mask, mask])

    # add mask to input
    result1 = cv2.add(img, mask)

def getNerfSamples(bid,nerfID, render_size,sampleSize,imD, trR, trT, trS):

        if render_size==448 or render_size==224:
          backRaysxys = torch.load(nerfID + "/" + str(render_size) + "_sampledRayBackxys/" + str(bid) + ".pt")
          s2 = backRaysxys[0]
        else:
          backRays = torch.load(nerfID + "/" + str(render_size) + "_sampledRayBack/" + str(bid) + ".pt")
          s2 = backRays.xys[0]
        backVec1 = torch.load(nerfID + "/" + str(render_size) + "_posVecBack/" + str(bid) + ".pt")
        M = cv2.getRotationMatrix2D((0, 0), trR, float(trS))
        s2 = (torch.mm(s2, torch.from_numpy(M.astype("float32"))[0:2, 0:2].T) - (trT / (imD / 2)))
        if s2.max() >= 1 or s2.min() <= -1:
            idx_b1 = torch.where((s2[:, 0] > -1) & (s2[:, 0] < 1) & (s2[:, 1] > -1) & (s2[:, 1] < 1))[0]
            samps = idx_b1[torch.randperm(idx_b1.shape[0], dtype=torch.int64)[0:sampleSize]]
        else:
            samps = torch.randperm(backVec1.shape[1], dtype=torch.int64)[0:sampleSize]

        backraysxys1 = s2[samps].unsqueeze(0)

        backVec1 = backVec1[:, samps]
        if render_size==448 or render_size==224:
          sampledRaysxys = torch.load(nerfID + "/" + str(render_size) + "_sampledRayxys/" + str(bid) + ".pt")
          s1 = sampledRaysxys[0]
          
        else:
          sampledRays = torch.load(nerfID + "/" + str(render_size) + "_sampledRay/" + str(bid) + ".pt")
          s1 = sampledRays.xys[0]
          
        posVec1 = torch.load(nerfID + "/" + str(render_size) + "_posVec/" + str(bid) + ".pt")
        s1 = (torch.mm(s1, torch.from_numpy(M.astype("float32"))[0:2, 0:2].T) - (trT / (imD / 2)))
        if s1.max() >= 1 or s1.min() <= -1:
            idx_b2 = torch.where((s1[:, 0] > -1) & (s1[:, 0] < 1) & (s1[:, 1] > -1) & (s1[:, 1] < 1))[0]

            samps = idx_b2[torch.randperm(idx_b2.shape[0], dtype=torch.int64)[0:sampleSize]]
        else:
            samps = torch.randperm(s1.shape[0], dtype=torch.int64)[0:sampleSize]

        sampled_raysxys1 = s1[ samps].unsqueeze(0)

        posVec1 = posVec1[:, samps]

        return posVec1, sampled_raysxys1, backVec1, backraysxys1


def getNerfSamples(bid, nerfID, render_size, sampleSize, imD, trR, trT, trS, NOCS):
    if render_size == 448 or render_size == 224:
        backRaysxys = torch.load(nerfID + "/" + str(render_size) + "_sampledRayBackxys/" + str(bid) + ".pt")
        s2 = backRaysxys[0]
    else:
        backRays = torch.load(nerfID + "/" + str(render_size) + "_sampledRayBack/" + str(bid) + ".pt")
        s2 = backRays.xys[0]
    backVec1 = torch.load(nerfID + "/" + str(render_size) + "_posVecBack/" + str(bid) + ".pt")
    M = cv2.getRotationMatrix2D((0, 0), trR, float(trS))
    s2 = (torch.mm(s2, torch.from_numpy(M.astype("float32"))[0:2, 0:2].T) - (trT / (imD / 2)))
    if s2.max() >= 1 or s2.min() <= -1:
        idx_b1 = torch.where((s2[:, 0] > -1) & (s2[:, 0] < 1) & (s2[:, 1] > -1) & (s2[:, 1] < 1))[0]
        samps = idx_b1[torch.randperm(idx_b1.shape[0], dtype=torch.int64)[0:sampleSize]]
    else:
        samps = torch.randperm(backVec1.shape[1], dtype=torch.int64)[0:sampleSize]

    backraysxys1 = s2[samps].unsqueeze(0)

    backVec1 = backVec1[:, samps]
    if render_size == 448 or render_size == 224:
        sampledRaysxys = torch.load(nerfID + "/" + str(render_size) + "_sampledRayxys/" + str(bid) + ".pt")
        s1 = sampledRaysxys[0]

    else:
        sampledRays = torch.load(nerfID + "/" + str(render_size) + "_sampledRay/" + str(bid) + ".pt")
        s1 = sampledRays.xys[0]

    posVec1 = torch.load(nerfID + "/" + str(render_size) + "_posVec/" + str(bid) + ".pt")
    s1 = (torch.mm(s1, torch.from_numpy(M.astype("float32"))[0:2, 0:2].T) - (trT / (imD / 2)))
    if s1.max() >= 1 or s1.min() <= -1:
        idx_b2 = torch.where((s1[:, 0] > -1) & (s1[:, 0] < 1) & (s1[:, 1] > -1) & (s1[:, 1] < 1))[0]

        samps = idx_b2[torch.randperm(idx_b2.shape[0], dtype=torch.int64)[0:sampleSize]]
        if NOCS:
            nocssamps = idx_b2[torch.randperm(idx_b2.shape[0], dtype=torch.int64)]

            nocs2d = s1.clone()[nocssamps]
            nocs3d= posVec1.clone()[0,nocssamps]
    else:
        samps = torch.randperm(s1.shape[0], dtype=torch.int64)[0:sampleSize]
        if NOCS:
            nocs2d = s1.clone()
            nocs3d= posVec1.clone()[0]

    sampled_raysxys1 = s1[samps].unsqueeze(0)

    posVec1 = posVec1[:, samps]
    if NOCS:
        res=224
        nocsMap = torch.zeros((res, res, 3))
        #nocs2d=1-nocs2d
        nocs2d[:,0]=-nocs2d[:,0]
        nocs2d1 = (((nocs2d + 1) / 2) * res).to(torch.int64)
        nocs2d2 = (((nocs2d + 1) / 2) * (res-1)).to(torch.int64)
        nocs2d3 = (((nocs2d + 1) / 2) * (res-2)).to(torch.int64)
        nocsMap[nocs2d2[:, 0], nocs2d2[:, 1]] = nocs3d
        nocsMap[nocs2d3[:, 0], nocs2d3[:, 1]] = nocs3d
        # nocsMap[nocs2d3[:, 0], nocs2d3[:, 1]] = nocs3d
        nocsMap=torch.rot90(nocsMap)
        nocsMap=nocsMap.unsqueeze(0)
        # nocsMap=torch.movedim(torch.nn.functional.interpolate(torch.movedim(nocsMap.unsqueeze(0), 3,1),size=torch.Size([224,224]), mode='bicubic'),1,3)
        return posVec1, sampled_raysxys1, backVec1, backraysxys1, nocsMap
    else:
        return posVec1, sampled_raysxys1, backVec1, backraysxys1
def getNerfSamplesWithoutBack(bid, nerfID, render_size, sampleSize, imD, trR, trT, trS, NOCS):
    #if render_size == 448 or render_size == 224:
        #backRaysxys = torch.load(nerfID + "/" + str(render_size) + "_sampledRayBackxys/" + str(bid) + ".pt")
        #s2 = backRaysxys[0]
    #else:
        #backRays = torch.load(nerfID + "/" + str(render_size) + "_sampledRayBack/" + str(bid) + ".pt")
        #s2 = backRays.xys[0]
    #backVec1 = torch.load(nerfID + "/" + str(render_size) + "_posVecBack/" + str(bid) + ".pt")
    M = cv2.getRotationMatrix2D((0, 0), trR, float(trS))
    #s2 = (torch.mm(s2, torch.from_numpy(M.astype("float32"))[0:2, 0:2].T) - (trT / (imD / 2)))
    #if s2.max() >= 1 or s2.min() <= -1:
     #   idx_b1 = torch.where((s2[:, 0] > -1) & (s2[:, 0] < 1) & (s2[:, 1] > -1) & (s2[:, 1] < 1))[0]
      #  samps = idx_b1[torch.randperm(idx_b1.shape[0], dtype=torch.int64)[0:sampleSize]]
    #else:
     #   samps = torch.randperm(backVec1.shape[1], dtype=torch.int64)[0:sampleSize]

    #backraysxys1 = s2[samps].unsqueeze(0)

    #backVec1 = backVec1[:, samps]
    if render_size == 448 or render_size == 224 or render_size==160:
        sampledRaysxys = torch.load(nerfID + "/" + str(render_size) + "_sampledRayxys/" + str(bid) + ".pt")
        s1 = sampledRaysxys[0]

    else:
        sampledRays = torch.load(nerfID + "/" + str(render_size) + "_sampledRay/" + str(bid) + ".pt")
        s1 = sampledRays.xys[0]

    posVec1 = torch.load(nerfID + "/" + str(render_size) + "_posVec/" + str(bid) + ".pt")
    s1 = (torch.mm(s1, torch.from_numpy(M.astype("float32"))[0:2, 0:2].T) - (trT / (imD / 2)))
    if s1.max() >= 1 or s1.min() <= -1:
        idx_b2 = torch.where((s1[:, 0] > -1) & (s1[:, 0] < 1) & (s1[:, 1] > -1) & (s1[:, 1] < 1))[0]

        samps = idx_b2[torch.randperm(idx_b2.shape[0], dtype=torch.int64)[0:sampleSize]]
        if NOCS:
            nocssamps = idx_b2[torch.randperm(idx_b2.shape[0], dtype=torch.int64)]

            nocs2d = s1.clone()[nocssamps]
            nocs3d= posVec1.clone()[0,nocssamps]
    else:
        samps = torch.randperm(s1.shape[0], dtype=torch.int64)[0:sampleSize]
        if NOCS:
            nocs2d = s1.clone()
            nocs3d= posVec1.clone()[0]

    sampled_raysxys1 = s1[samps].unsqueeze(0)

    posVec1 = posVec1[:, samps]
    if NOCS:
        res=224
        nocsMap = torch.zeros((res, res, 3))
        #nocs2d=1-nocs2d
        nocs2d[:,0]=-nocs2d[:,0]
        nocs2d1 = (((nocs2d + 1) / 2) * res).to(torch.int64)
        nocs2d2 = (((nocs2d + 1) / 2) * (res-1)).to(torch.int64)
        nocs2d3 = (((nocs2d + 1) / 2) * (res-2)).to(torch.int64)
        nocsMap[nocs2d2[:, 0], nocs2d2[:, 1]] = nocs3d
        nocsMap[nocs2d3[:, 0], nocs2d3[:, 1]] = nocs3d
        # nocsMap[nocs2d3[:, 0], nocs2d3[:, 1]] = nocs3d
        nocsMap=torch.rot90(nocsMap)
        nocsMap=nocsMap.unsqueeze(0)
        # nocsMap=torch.movedim(torch.nn.functional.interpolate(torch.movedim(nocsMap.unsqueeze(0), 3,1),size=torch.Size([224,224]), mode='bicubic'),1,3)
        return posVec1, sampled_raysxys1, nocsMap
    else:
        return posVec1, sampled_raysxys1