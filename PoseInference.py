import os
import sys
import time
import json
import glob
import torch
import math
import matplotlib.pyplot as plt
import numpy as np
from nutil import normImage
import torch.utils.data as data_utils
from dep.unet import ResNetUNetNew as ResNetUNet
from nutil import show_full_render1
from augment import generateImages
from nutil import get_emb_vis, huber, sample_images_at_mc_locs, returnCrossEntropy,returnCrossEntropyWithNeg
from nerf import NeuralRadianceFieldFeat
from pren import ImplicitRendererStratified, EmissionAbsorptionRaymarcherStratified
from PIL import Image
from IPython import display
from tqdm.notebook import tqdm
from cowrendersynth import generate_cow_rendersWithRT, generate_bop_realsamples
from torch.utils.tensorboard import SummaryWriter
from nutil import mip360loss
from pytorch3d.structures import Volumes
from pytorch3d.transforms import so3_exp_map

from sklearn.neighbors import KDTree
from dataGen import AugmentedSamples
import pytorch3d,cv2
from nutil import rotfromeulernp
import argparse


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
objid="1"
expID = "ruapc/"+objid+"poseEst"
batch_size = 16
sampleSize = 1024
key_noise = 1e-3
Siren=True
datasetPath = "bop/ruapc/"
imD = 224
in_ndc = False

torch.manual_seed(1)
# Instantiate the radiance field model.
neural_radiance_field = NeuralRadianceFieldFeat(siren=Siren)
encoder_rgb = ResNetUNet(n_class=(13),n_decoders=1,)  ##?n_class =13 - dm cos here is the query image with embedding size =13 ddo

# load nerf model as key model
data_nerf = torch.load(expID+"/nerflatest.pth")	
neural_radiance_field.load_state_dict(data_nerf["model_state_dict"])
# load CNN as query model	
data_rgb = torch.load(expID+"/encoderRGBlatest.pth")	
encoder_rgb.load_state_dict(data_rgb["model_state_dict"])	
neural_radiance_field=neural_radiance_field
encoder_rgb=encoder_rgb

#set models to eval
neural_radiance_field.eval()
encoder_rgb.eval()
print("continue")

# load images from second sequence
fsamps = int(2561 * 0.5)
fewIds = np.arange(int(2561 * 0.5)) +1280
target_images, target_silhouettes, RObj, TObj, KObj, fewIds = generate_bop_realsamples(datasetPath,objid=objid, crop=True,maskStr="mask",offset=5, synth=False,makeNDC=in_ndc,dataset="tless", maxB=imD, background=False, fewSamps=True, fewCT=fsamps, fewids=fewIds)

## load fullNegVec
if os.path.exists(expID+"/negVec.npy"):
    fullNegVec= torch.from_numpy(np.load(expID+"/negVec.npy").astype("float32")).unsqueeze(0)
negSamps = torch.randperm(fullNegVec.shape[1], dtype=torch.int64)[0:batch_size*sampleSize]
negVec=fullNegVec[:,negSamps].clone()
# negVec+= torch.randn_like(negVec) * key_noise # negvec =negvec +negvec*keynoise
print('fullNegVec shape', fullNegVec.shape)
print('negVec shape', negVec.shape) #only take 16384 points from the all the 3D points- to reduce computation at the end


input =torch.movedim(target_images, 3, 1)
with torch.no_grad():
    queries = [encoder_rgb.to(device)(input[i].unsqueeze(0).to(device)) for i in range(input.shape[0])]
    feats = neural_radiance_field.batched_customForward(negVec)

del encoder_rgb

queries = torch.stack([tensor.to('cpu') for tensor in queries]).squeeze()
print('queries.shape', queries.shape)
print('feats.shape', queries.shape)

resized_queries = queries.reshape(1280,13,-1)
resized_feats = feats.squeeze() 
print('resized_queries.shape', resized_queries.shape)
print('resized_feats.shape', resized_feats.shape)


def getCors(queries, feats, leaves=1):
    cMat = torch.log_softmax(torch.matmul(queries, feats).permute(0, 2, 1), dim=-1)
    ids=torch.argmax(cMat, dim=-1)
    #vals, idx = torch.topk(cMat, k=leaves, dim=-1)
    # if leaves == 1:
    #     return idx[..., 0].cpu(), vals
    # else:
    #     return idx.cpu(), vals
    return ids















































