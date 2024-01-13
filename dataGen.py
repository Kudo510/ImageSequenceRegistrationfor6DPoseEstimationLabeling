import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data
from augment import generateImages,getNerfSamples
from typing import Union
imagenet_stats = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
mu, std = imagenet_stats
mu=torch.from_numpy(np.asarray(mu).astype("float32"))
std=torch.from_numpy(np.asarray(std).astype("float32"))


def normalize(img: np.ndarray):  # (h, w, 3) -> (3, h, w)
    # if img.dtype == np.uint8:
    #     img = img / 255
    img = (img - mu) / std
    return img


def denormalize(img: Union[np.ndarray, torch.Tensor]):
    mu, std = imagenet_stats
    if isinstance(img, torch.Tensor):
        mu, std = [torch.Tensor(v).type(img.dtype).to(img.device)[:, None, None] for v in (mu, std)]
    return img * std + mu


class AugmentedSamples(torch.utils.data.Dataset):
    def __init__(self,cocopath, datasetPath, target_images,target_silhouettes, nerfID,render_size=224, sampleSize=-1,imD=-1,coco_aug=True,tless_aug=True, objid=-1, scaleFac=0.8, maskErosion=True, batch_size=16, totalSamples=-1,maxScale=1.0,transScale=1, surfEmbScaling=False,surfEmbScaleFac=1,lineErosion=False,augMax=40, cocoList=[], maskMax=15000,transScale2=0.2, target_backgrounds=None, borderADD=False, NOCS=False):

        self.nerfID=nerfID
        self.sampleSize=sampleSize
        self.render_size=render_size
        self.datasetPath=datasetPath
        self.imD=imD
        self.coco_aug=coco_aug
        self.tless_aug=tless_aug
        self.objid=objid
        self.scaleFac=scaleFac
        self.maskErosion=maskErosion
        #self.cocoList=glob.glob(cocopath + "*.jpg")
        self.cocoList=cocoList
        
        self.totalSamples=totalSamples
        self.batch_size=batch_size
        self.target_images=target_images
        self.target_silhouettes=target_silhouettes
        self.maxScale=maxScale
        self.transScale=transScale
        self.surfEmbScaling=surfEmbScaling
        self.surfEmbScaleFac=surfEmbScaleFac
        self.lineErosion=lineErosion
        self.augMax=augMax
        self.maskMax=maskMax
        self.transScale2=transScale2
        self.target_backgrounds=target_backgrounds
        self.borderADD=borderADD
        self.NOCS=NOCS
    def __len__(self):
        return self.totalSamples

    def __getitem__(self, idx):
        #batch_idx = torch.randperm(len(self.totalSamples))[:self.batch_size]

            rgb1=self.target_images[idx]
            mask1 = self.target_silhouettes[idx]
            background=None
            if self.target_backgrounds is not None:
                background=self.target_backgrounds[idx]
            rgbtensor1, masktensor1, masktensorcrop1, trR, trT, trS = generateImages(rgb1.numpy().astype("float32"),mask1.numpy().astype("float32"),transformNo= 1,
                                                                                 cocoids=self.cocoList,
                                                                                 datasetPath=self.datasetPath, coco_aug=self.coco_aug,
                                                                                 tless_aug=self.tless_aug, objid=self.objid,
                                                                                 scaleFac=self.scaleFac, maskErosion=self.maskErosion,maxScale=self.maxScale,transScale=self.transScale,surfEmbScaling=self.surfEmbScaling,surfEmbScaleFac=self.surfEmbScaleFac,lineErosion=self.lineErosion, augMax=self.augMax, maskMax=self.maskMax,transScale2=self.transScale2, background=background, borderADD=self.borderADD)
            rgbtensor1=normalize(rgbtensor1)
            if self.NOCS:
                posVec1, sampled_raysxys1, backVec1, backraysxys1, nocsMap=getNerfSamples(bid=idx,nerfID=self.nerfID, render_size=self.render_size,sampleSize=self.sampleSize,
                                                                             imD=self.imD, trR=int(trR),trT=trT, trS=trS, NOCS=self.NOCS)
                return [rgbtensor1[0], masktensor1[0], masktensorcrop1[0], trR[0], trT[0], trS[0], posVec1[0],
                        sampled_raysxys1[0], backVec1[0], backraysxys1[0], nocsMap[0]], idx
            else:
                posVec1, sampled_raysxys1, backVec1, backraysxys1 = getNerfSamples(bid=idx, nerfID=self.nerfID,render_size=self.render_size,sampleSize=self.sampleSize,
                                                                               imD=self.imD, trR=int(trR), trT=trT,trS=trS, NOCS=self.NOCS)

                return [rgbtensor1[0], masktensor1[0], masktensorcrop1[0], trR[0], trT[0], trS[0], posVec1[0],
                        sampled_raysxys1[0], backVec1[0], backraysxys1[0]], idx

