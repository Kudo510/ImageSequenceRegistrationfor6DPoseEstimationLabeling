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
# from nutil import NeuralRadianceFieldOrig,HarmonicEmbedding, readImages, createEncoder, filterBoundsPC, returnOccludedPC, generate_rotating_nerf, show_full_render, show_full_render1
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
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PerspectiveCameras,
    NDCMultinomialRaysampler,
    MonteCarloRaysampler,
    EmissionAbsorptionRaymarcher,
    ImplicitRenderer,
    RayBundle,
    ray_bundle_to_ray_points,
)
from sklearn.neighbors import KDTree
from dataGen import AugmentedSamples
import pytorch3d,cv2
from nutil import rotfromeulernp
import argparse

arg_parser = argparse.ArgumentParser(description="Train a Linemod")
arg_parser.add_argument("--objid", dest="objid", default="15", )
arg_parser.add_argument("--cont", dest="cont", default=False, )
arg_parser.add_argument("--UH", dest="UH", default=0)
args = arg_parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cocoList=[]
cocoPath="coco/coco_set/"
cocoList=glob.glob(cocoPath + "*.jpg")
cocoLen=len(cocoList)
objid="1"
objid=str(args.objid)
rayID ="ruapc/"+ objid+"Cors/"
nerfID ="ruapc/"+ objid+"TLESSObj_Fine/"
expID = "ruapc/"+objid+"poseEst"
Siren=True
onlyNerf=False
useCroppedMask=False
sampleNegativeOnce=True
sampleSize=1024
epochChange=0000
sampleIds=torch.arange(1295)
imDOrig=200
imD=224
key_noise=1e-3
datasetPath = "bop/ruapc/"
imD = 224
render_size=224
batch_size = 16
lr = 1e-3
lr_cnn=3e-4
lr_mlp=3e-5


in_ndc = False

target_backgrounds=None
fsamps = int(2561 * 0.5)
fewIds = np.arange(int(2561 * 0.5))
fsamps=30
fewIds = np.arange(int(30))

if not int(args.UH):
  fewIds = fewIds + 1280

target_images, target_silhouettes, RObj, TObj, KObj, fewIds = generate_bop_realsamples(datasetPath,objid=objid, crop=True,maskStr="mask",offset=5, synth=False,makeNDC=in_ndc,dataset="tless", maxB=imD, background=False, fewSamps=True, fewCT=fsamps, fewids=fewIds)


rot180 = rotfromeulernp(np.array([0, 0, np.pi]))
for a in range(target_images.shape[0]):
  RObj[a] = (RObj[a].T).dot(rot180)
  TObj[a, 0:2] = -TObj[a, 0:2]
meshdetails = json.load(open(datasetPath + "/models" + "/models_info.json"))
diam = meshdetails[objid]['diameter']
diamScaling=1.8
offset=0
scale=diam/diamScaling
TObj = TObj/scale


stratified=False
enable3D=False
enableSSP=False
negativeSampling=True
maskRays=True
rayFreeze=False

target_cameras = PerspectiveCameras(device=device, R=torch.from_numpy(RObj.astype("float32")),
                                    T=torch.from_numpy(TObj.astype("float32")),
                                    K=torch.from_numpy(KObj.astype("float32")))
min_depth = np.min(np.abs(TObj[:,2])) - 2
volume_extent_world = np.max(np.abs(TObj[:,2])) + 2

print(f'Generated {len(target_images)} images/silhouettes/cameras.')


augmented_dataset = AugmentedSamples(cocopath=cocoPath, datasetPath=datasetPath,nerfID=rayID, render_size=render_size, sampleSize=sampleSize,imD=imD,
                                     target_images=target_images, target_silhouettes=target_silhouettes, coco_aug=True,tless_aug=False, objid=objid, scaleFac=0.8,
                                     maskErosion=True, batch_size=batch_size, totalSamples=target_images.shape[0],maxScale=1.2,transScale=0.2,surfEmbScaling=True,surfEmbScaleFac=1.5,lineErosion=True,cocoList=cocoList, augMax=20, 
                                     maskMax=10000,transScale2=0.4, borderADD=True, NOCS=True)
import torch.utils.data as data_utils

sdf_loader = data_utils.DataLoader(
        augmented_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )
  
volume_extent_world = offset+np.max(np.abs(TObj))

raysampler_grid = NDCMultinomialRaysampler(image_height=render_size, image_width=render_size, n_pts_per_ray=128, min_depth=min_depth, max_depth=volume_extent_world,)
raysampler_grid_eval = NDCMultinomialRaysampler(image_height=int(render_size/2), image_width=int(render_size/2), n_pts_per_ray=128, min_depth=min_depth, max_depth=volume_extent_world,)

raysampler_mc = MonteCarloRaysampler(min_x=-1.0,max_x=1.0,min_y=-1.0,max_y=1.0,n_rays_per_image=50, n_pts_per_ray=128,
                                     min_depth=min_depth,max_depth=volume_extent_world,stratified_sampling=True)

raymarcher = EmissionAbsorptionRaymarcherStratified()

renderer_grid = ImplicitRendererStratified(raysampler=raysampler_grid, raymarcher=raymarcher,device=device)
renderer_mc = ImplicitRendererStratified(raysampler=raysampler_mc, raymarcher=raymarcher,device=device, rayFreeze=rayFreeze)
renderer_grid_eval = ImplicitRendererStratified(raysampler=raysampler_grid_eval, raymarcher=raymarcher,device=device)

# First move all relevant variables to the correct device.
renderer_grid = renderer_grid.to(device)
renderer_grid_eval = renderer_grid_eval.to(device)

renderer_mc = renderer_mc.to(device)
target_cameras = target_cameras.to(device)


# Set the seed for reproducibility
torch.manual_seed(1)
# Instantiate the radiance field model.
neural_radiance_field = NeuralRadianceFieldFeat(siren=Siren)


neural_radiance_field.mlp.requires_grad_(False)
neural_radiance_field.harmonic_embedding.requires_grad_(False)
neural_radiance_field.density_layer.requires_grad_(False)
neural_radiance_field.color_layer.requires_grad_(False)
neural_radiance_field.feature_layer.requires_grad_(True)
neural_radiance_field.mode="feature"
raymarcher.thresholdMode=True
raymarcher.threshold=0.2

from dep.unet import ResNetUNetNew as ResNetUNet
encoder_rgb = ResNetUNet(n_class=(13),n_decoders=1,)
#encoder_rgb2 = ResNetUNet(n_class=(3),n_decoders=1,)



if not os.path.exists(expID):
    os.makedirs(expID)
np.save(expID+"/few.npy", fewIds)


data_nerf = torch.load(nerfID+"/nerflatestFine.pth")
neural_radiance_field.load_state_dict(data_nerf["model_state_dict"])

if args.cont:	
    data_nerf = torch.load(expID+"/nerflatest.pth")	
    neural_radiance_field.load_state_dict(data_nerf["model_state_dict"])	
    data_rgb = torch.load(expID+"/encoderRGBlatest.pth")	
    encoder_rgb.load_state_dict(data_rgb["model_state_dict"])	
    data_rgb2 = torch.load(expID+"/encoderRGB2latest.pth")	
    encoder_rgb2.load_state_dict(data_rgb2["model_state_dict"])	
    print("continuing")

neural_radiance_field=neural_radiance_field.to(device)
encoder_rgb=encoder_rgb.to(device)

v1=torch.from_numpy(np.load(rayID+"/subvert1.npy").astype("float32"))
n1=torch.from_numpy(np.load(rayID+"/subnormal1.npy").astype("float32"))
tree = KDTree(v1.cpu().numpy(), leaf_size=2)

# Instantiate the Adam optimizer. We set its master learning rate to 1e-3.


intraE2=0;intraE3=0;intraE4=0;color_err=0;asiler=0
if onlyNerf:
    optimizer = torch.optim.Adam([{"params": neural_radiance_field.parameters(), "lr": lr}])
else:
    optimizer = torch.optim.Adam([{"params": neural_radiance_field.parameters(), "lr": lr_mlp},
                                  {"params": encoder_rgb.parameters(), "lr": lr_cnn},
                                  #{"params": encoder_rgb2.parameters(), "lr": lr_cnn},

                                  ])

n_iter = int(60000*batch_size/target_images.shape[0])

# Init the loss history buffers.
loss_history_color, loss_history_sil = [], []
err0=0
batch_idx1=[]
iteration=0
fullNegVec=torch.Tensor([])
for iterate in range(n_iter):
  for train_data, indices in sdf_loader:
    a1 = 1
    # In case we reached the last 75% of iterations,
    # decrease the learning rate of the optimizer 10-fold.
    iteration+=1

    for i, param_group in enumerate(optimizer.param_groups):
            frac=(iteration+1)/2000
            if frac>=1:
                frac=1
            if i==0:
              param_group["lr"] = lr_mlp*frac
            else:
              param_group["lr"] = lr_cnn*frac


    if iteration%100==0 and iteration>1:
        torch.save({"epoch": iteration, "model_state_dict": neural_radiance_field.state_dict()},expID+"/nerflatest.pth",)
        if not onlyNerf:
            torch.save({"epoch": iteration, "model_state_dict": encoder_rgb.state_dict()},expID+"/encoderRGBlatest.pth",)


    # Zero the optimizer gradient.
    optimizer.zero_grad()

    batch_idx=indices

    if not onlyNerf:
        if True:
            g1=1
            # randIDX=torch.arange(ct)[torch.randperm(ct)][0:batch_size]

        rgbtensor, masktensor, masktensorcrop, trRt, trTt, trSt,posVec, sampled_raysxys, backVec, backraysxys,nocsMap =train_data
        #nocsMap=nocsMap*masktensorcrop.unsqueeze(-1)
        #import pdb;pdb.set_trace()
        RGBfeatFull1 = torch.movedim(encoder_rgb(input =torch.movedim(rgbtensor.to(device), 3, 1)), 1, 3)
        #RGBfeatFull2 = torch.movedim(encoder_rgb2(input=torch.movedim(rgbtensor.to(device), 3, 1)), 1, 3)

        maskFeat1 = RGBfeatFull1[..., -1:]
        # maskFeat2 = RGBfeatFull2[..., -1:]
        RGBfeatFull1 = RGBfeatFull1[..., 0:12]
        #RGBfeatFull2 = torch.sigmoid(RGBfeatFull2[..., 0:3])
        #RGBfeatFull2 = 2*(RGBfeatFull2-0.5)

    
    
    intraE1 = 0;
    # Sample the minibatch of cameras.
    focal = target_cameras.K[batch_idx][:, 0, 0:2]
    focal[:, 1] = target_cameras.K[batch_idx][:, 1, 1]
    principal_point = target_cameras.K[batch_idx][:, 0:2, 2]
    imageSize = principal_point * 0
    imageSize += imD
    batch_cameras = PerspectiveCameras(
        R=target_cameras.R[batch_idx],
        T=target_cameras.T[batch_idx],
        focal_length=focal,
        principal_point=principal_point,
        image_size=imageSize,
        # K=target_cameras.K[batch_idx],
        in_ndc=in_ndc,
        device=device,
    )

    add_input_samples=False
    if negativeSampling:
        if not rayFreeze or len(batch_idx1)==0:
          batch_idx1 = torch.randperm(len(target_cameras))
          focal = target_cameras.K[batch_idx1][:, 0, 0:2]
          focal[:, 1] = target_cameras.K[batch_idx1][:, 1, 1]
          principal_point = target_cameras.K[batch_idx1][:, 0:2, 2]
          imageSize = principal_point * 0
          imageSize += imD
          batch_cameras1 = PerspectiveCameras(
              R=target_cameras.R[batch_idx1],
              T=target_cameras.T[batch_idx1],
              focal_length=focal,
              principal_point=principal_point,
              image_size=imageSize,
              # K=target_cameras.K[batch_idx],
              in_ndc=in_ndc,
              device=device,
          )
          if sampleNegativeOnce:
              loopCT=20
          else:
              loopCT=1
              
          if fullNegVec.shape[0]==0:
            if os.path.exists(expID+"/negVec.npy"):
                fullNegVec= torch.from_numpy(np.load(expID+"/negVec.npy").astype("float32")).cuda().unsqueeze(0)
            else:
              for a in range(loopCT):
                  with torch.no_grad():
                    _, sampled_rays1, weightsNeg = renderer_mc(
                        cameras=batch_cameras1,
                        volumetric_function=neural_radiance_field.batched_forward, add_input_samples=add_input_samples,
                        stratified=stratified,
                        maskRays=maskRays, mask=target_silhouettes[batch_idx1, ..., None])

                  sampled_raysxys1=sampled_rays1.xys
                  negVec1 = sampled_rays1.origins + sampled_rays1.directions * torch.max(sampled_rays1.lengths * weightsNeg, dim=-1)[0].unsqueeze(-1)

                  #import pdb;pdb.set_trace()
                  idx2 = torch.where(torch.norm(negVec1 - sampled_rays1.origins, dim=-1)[0])[0]
                  negVec1=negVec1[:,idx2]
                  negVec1=negVec1
                  fullNegVec = torch.cat([fullNegVec, negVec1.cpu()], dim=1)
              from pytorch3d.ops import sample_farthest_points as fps
              #import pdb;pdb.set_trace()
              idx3 = fps(fullNegVec, K=80000)[1][0].cpu()
              fullNegVec = fullNegVec[:,idx3]
              fullNegVec = fullNegVec[0, torch.where(torch.max(torch.abs(fullNegVec[0, :, :]), dim=-1)[0] < 1.2)[0]].unsqueeze(0)
              with torch.no_grad():
                  mverts= neural_radiance_field.batched_forward_forPC(threshold=0.07)[0]
              import open3d as o3d
              pcd = o3d.geometry.PointCloud()
              pcd.points = o3d.utility.Vector3dVector(mverts)
              cl, ind=pcd.remove_radius_outlier(nb_points=16, radius=0.05)
              mverts=mverts[ind]
              
              tree2 = KDTree(np.asarray(mverts), leaf_size=2)
              pdist1,pind1 = tree2.query(fullNegVec[0].cpu().numpy(), k=1)
              fullNegVec=fullNegVec[:,np.where(pdist1[:,0]<0.05)[0]]
              np.save(expID+"/negVec.npy", fullNegVec[0].cpu().numpy())
              quit()
              #import pdb;pdb.set_trace()
              # sampled_raysxys1=sampled_raysxys1[:,idx2]
        #import pdb;pdb.set_trace(  
        negSamps = torch.randperm(fullNegVec.shape[1], dtype=torch.int64)[0:batch_size*sampleSize]
        negVec=fullNegVec[:,negSamps].clone()
        negVec+= torch.randn_like(negVec) * key_noise


        rendered_images_silhouettes1 = neural_radiance_field.batched_customForward(negVec.to(device))

        lastDim1=rendered_images_silhouettes1.shape[-1]
        rendered_images1, rendered_silhouettes1 = (
                rendered_images_silhouettes1.split([lastDim1-1, 1], dim=-1)
            )

        if not maskRays:
            silhouettes_at_rays1 = sample_images_at_mc_locs(
                  target_silhouettes[batch_idx1, ..., None],
                  sampled_raysxys1.cpu()
              )
            negKeys = rendered_images1.reshape(-1,12)[torch.where(silhouettes_at_rays1.view(-1).long())].unsqueeze(0)
        else:
            negKeys = rendered_images1.reshape(-1, 12).unsqueeze(0)

    
    rendered_images_silhouettes = neural_radiance_field.batched_customForward(posVec.to(device))
    rendered_images_silhouettesBack = neural_radiance_field.batched_customForward(backVec.to(device))

    lastDim=rendered_images_silhouettes.shape[-1]
    if onlyNerf:
        rendered_images, rendered_silhouettes = (
            rendered_images_silhouettes.split([lastDim-1, 1], dim=-1)
        )
    else:
        rendered_images=rendered_images_silhouettes
        rendered_imagesBack=rendered_images_silhouettesBack


    if not onlyNerf:

        key = rendered_images[:, :, 0:12]
        keyBack = rendered_imagesBack[:, :, 0:12]
 
        queries = sample_images_at_mc_locs(
            RGBfeatFull1,
            sampled_raysxys.to(device)
        )


        if not maskRays:
          key=key[:,maskIDX,:]

    err5=0
    intraLoss=0
    intraE1=0
    if iteration >=epochChange :

        if iteration<0000:
            err5=torch.mean(torch.abs(queries-key.detach()))
        else:                
            if negKeys.shape[1]<batch_size*sampleSize:
                negSamps = torch.randperm(negKeys.shape[1], dtype=torch.int64)[0:sampleSize]
                err5=returnCrossEntropyWithNeg(queries,key, torch.cat(batch_size*[negKeys[:,negSamps]],dim=0))
                #err5 = err5 + torch.mean(torch.abs((RGBfeatFull2-nocsMap)*masktensorcrop.unsqueeze(-1)))
                       # returnCrossEntropyWithNeg(queriesBack, keyBack.detach(),torch.cat(batch_size * [negKeys[:,negSamps].detach()], dim=0))
            else:
                #import pdb;pdb.set_trace()
                #try:
                  err5 = returnCrossEntropyWithNeg(queries, key, negKeys.view(batch_size,sampleSize,-1))
                  #err5 = err5 + torch.mean(torch.abs((RGBfeatFull2-nocsMap.cuda())*masktensorcrop.unsqueeze(-1).cuda()))
                         # returnCrossEntropyWithNeg(queriesBack, keyBack.detach(),negKeys.view(batch_size,sampleSize,-1).detach())

            mask_loss=0
            from torch.nn import functional as F
            # mask_loss =mask_loss+(F.binary_cross_entropy(torch.sigmoid(maskFeat2[...,0]), masktensorcrop.to(device))/1000)
            mask_loss = mask_loss + (F.binary_cross_entropy(torch.sigmoid(maskFeat1[...,0]), masktensor.to(device)) / 1000)

    loss=err5+mask_loss+intraE1


    loss_history_color.append(float(color_err))
    loss_history_sil.append(float(asiler))

    # Every 10 iterations, print the current values of the losses.
    if iteration % 10 == 0:
        print(
            f'Iteration {iteration:05d}:'
            + f' feature loss= {float(err5):1.2e}'
            + f' mask_loss = {float(mask_loss):1.2e}'

        )

    loss.backward()
    optimizer.step()
    
    # Visualize the full renders every 100 iterations.

    if iteration % 100 == 1:
        show_idx = torch.randperm(len(target_cameras))[:1]
        import cv2

        mu, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
       # import pdb;pdb.set_trace()
        r11=str(int(torch.rand(1)*10))
        if not onlyNerf:
            nIM=normImage(RGBfeatFull1[0:1].detach().cpu())
            # cv2.imwrite(expID + "/" + r11 + "_encoder.jpg",nIM[0,:,:,0:3].cpu().numpy() * 255*2)

            m1=torch.sigmoid(maskFeat1[0]).detach().cpu().numpy()
            # m2 = torch.sigmoid(maskFeat2[0]).detach().cpu().numpy()

            cv2.imwrite(expID + "/" + r11 + "_mask.jpg",(m1 / (0.01 + m1.max())) * 255)
            # cv2.imwrite(expID + "/" + r11 + "_mask2.jpg",(m2 / (0.01 + m2.max())) * 255)
            cv2.imwrite(expID + "/" + r11 + "_feat.jpg",
                    get_emb_vis(RGBfeatFull1[0].detach().cpu(), demean=True).cpu().numpy() * 255)


        cv2.imwrite(expID + "/" + r11 + "_target.jpg", ((rgbtensor[0].cpu().numpy()*std)+mu) * 255)


        c1= neural_radiance_field
        focal = target_cameras.K[batch_idx[0:1]][:, 0, 0:2]
        focal[:, 1] = target_cameras.K[batch_idx[0:1]][:, 1, 1]
        principal_point = target_cameras.K[batch_idx[0:1]][:, 0:2, 2]
        imageSize = principal_point * 0
        imageSize += imD
        show_full_render1(
            neural_radiance_field,
            PerspectiveCameras(
                R=target_cameras.R[batch_idx[0:1]],
                T=target_cameras.T[batch_idx[0:1]],
                focal_length=focal[0:1],
                principal_point=principal_point,
                image_size=imageSize,
                # K=target_cameras.K[batch_idx[0:1]],
                in_ndc=in_ndc,
                device=device
            ),

            expID=expID, renderer_grid=renderer_grid_eval, normalization=True
        )
    
