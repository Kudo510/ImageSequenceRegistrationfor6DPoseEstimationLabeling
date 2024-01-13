import os
import sys
import time
import json
import glob
import torch
import math
import matplotlib.pyplot as plt
import numpy as np
from nutil import show_full_render1
from nutil import get_emb_vis, huber, sample_images_at_mc_locs, rotfromeulernp
from nerf import NeuralRadianceFieldFeat
from prenBack import ImplicitRendererStratified as IRSBack
from prenBack import  EmissionAbsorptionRaymarcherStratified as MarchBack
from pren import ImplicitRendererStratified, EmissionAbsorptionRaymarcherStratified
from PIL import Image
from IPython import display
from tqdm.notebook import tqdm
from cowrendersynth import image_grid, generate_cow_renders, generate_cow_rendersWithRT, generate_tless_realsamples,generate_lm_realsamplesWithoutLMTrains, generate_bop_realsamples

from pytorch3d.renderer import (
    PerspectiveCameras,
    NDCMultinomialRaysampler,
    MonteCarloRaysampler,
    EmissionAbsorptionRaymarcher,
    ImplicitRenderer,
    RayBundle,
    ray_bundle_to_ray_points,
)
from sklearn.neighbors import KDTree

dataParallel=False
if dataParallel:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import argparse

arg_parser = argparse.ArgumentParser(description="Train a Linemod")
arg_parser.add_argument("--objid",dest="objid",default="15",)
arg_parser.add_argument("--strat",dest="strat",default="",)
arg_parser.add_argument("--UH", dest="UH", default=0)
arg_parser.add_argument("--viz", dest="viz", default=0)
arg_parser.add_argument("--dataset",dest="dataset",default="tless",)

args = arg_parser.parse_args()

objid="31"
objid=str(args.objid)
#nerfID = objid+"mip3Nerf"
nerfID = str(args.UH) +'_'+ args.dataset+'_obj_' + str(objid)+ "/"+objid+"Cors"

if not os.path.exists(nerfID):
    os.makedirs(nerfID)
# expID = "ruapc/"+objid+"TLESSObj_Fine"
# if not os.path.exists(str(args.UH) +'_'+ args.dataset+'_obj_' + str(objid)):
#     os.makedirs(str(args.UH) +'_'+ args.dataset+'_obj_' + str(objid))

if args.dataset == "tless":
    expID = str(args.UH) +'_'+ args.dataset+'_obj_' + str(objid)+"/"+objid+"TLESSObj_Fine"
    datasetPath = "bop/tless"
else:    
    expID = str(args.UH) +'_'+ args.dataset+'_obj_' + str(objid)+"/"+objid+"TLESSObj_Fine"
    datasetPath = "bop/ruapc"

siren=True
# datasetPath = "bop/ruapc/"

print("objid", objid)

imD=200
maxB=200

in_ndc=False

gridSampler=True
if gridSampler:
  render_size=224

# fsamps=int(2561*0.5)
# fewIds=np.arange(int(2561*0.5))
# if not int(args.UH):
#     fewIds=fewIds+1280

if args.dataset == 'tless': # dataset only has 1000 images
    fsamps=int(1001*0.5)      ## number of samples 
    fewIds=np.arange(int(1001*0.5))  ## list of ID of samples (from 0 to 1280)
    if not int(args.UH):
        fewIds=fewIds+500
else:
    fsamps=int(2561*0.5)      ## number of samples 
    fewIds=np.arange(int(2561*0.5))  ## list of ID of samples (from 0 to 1279)
    if not int(args.UH):
        fewIds=fewIds+1280  


target_images, target_silhouettes,RObj,TObj, KObj, fewIds = generate_bop_realsamples(datasetPath,objid=objid,crop=True,maskStr="mask", offset=5,synth=False, makeNDC=in_ndc, dataset=args.dataset, fewSamps=True, fewCT=fsamps, fewids=fewIds, maxB=render_size)
from nutil import rotfromeulernp
rot180=rotfromeulernp(np.array([0,0,np.pi]))
for a in range(target_images.shape[0]):
  RObj[a]=(RObj[a].T).dot(rot180)
  TObj[a,0:2] = -TObj[a,0:2]
meshdetails = json.load(open(datasetPath+"/models" + "/models_info.json"))
diam = meshdetails[objid]['diameter']

diamScaling=1.8
scale=diam/diamScaling
TObj = TObj/scale

onlyNerf=True
globalStratified=True

estimateNormals=True
customForward=True
onlyNerf=False
stratified=False
enable3D=False
enableSSP=False
negativeSampling=False
maskRays=True
rayFreeze=True
oldScaling=False


target_cameras = PerspectiveCameras(device=device, R=torch.from_numpy(RObj.astype("float32")),
                                    T=torch.from_numpy(TObj.astype("float32")),
                                    K=torch.from_numpy(KObj.astype("float32")))

min_depth = np.min(np.abs(TObj[:, 2])) - 2
volume_extent_world = np.max(np.abs(TObj[:, 2])) + 2

print(f'Generated {len(target_images)} images/silhouettes/cameras.')


rayCT=256
raysampler_grid = NDCMultinomialRaysampler(image_height=render_size, image_width=render_size, n_pts_per_ray=rayCT, min_depth=min_depth, max_depth=volume_extent_world,)

raysampler_mc = MonteCarloRaysampler(min_x=-1.0,max_x=1.0,min_y=-1.0,max_y=1.0,n_rays_per_image=4, n_pts_per_ray=128,
                                     min_depth=min_depth,max_depth=volume_extent_world,stratified_sampling=True)

raymarcher = EmissionAbsorptionRaymarcherStratified()

renderer_grid = ImplicitRendererStratified(raysampler=raysampler_grid, raymarcher=raymarcher,device=device)
renderer_mc = ImplicitRendererStratified(raysampler=raysampler_mc, raymarcher=raymarcher,device=device, rayFreeze=rayFreeze)

raymarcherBack = MarchBack()

renderer_gridBack = IRSBack(raysampler=raysampler_grid, raymarcher=raymarcher,device=device)
renderer_mcBack = IRSBack(raysampler=raysampler_mc, raymarcher=raymarcher,device=device, rayFreeze=rayFreeze)


# First move all relevant variables to the correct device.
renderer_grid = renderer_grid.to(device)
renderer_mc = renderer_mc.to(device)
renderer_grid = renderer_gridBack.to(device)
renderer_mc = renderer_mcBack.to(device)
target_cameras = target_cameras.to(device)

# Set the seed for reproducibility
torch.manual_seed(1)
# Instantiate the radiance field model.
neural_radiance_field = NeuralRadianceFieldFeat(siren=siren)

if onlyNerf:
    neural_radiance_field.mlp.requires_grad_(True)
    neural_radiance_field.harmonic_embedding.requires_grad_(True)
    neural_radiance_field.density_layer.requires_grad_(True)
    neural_radiance_field.color_layer.requires_grad_(True)
    neural_radiance_field.feature_layer.requires_grad_(False)
    neural_radiance_field.mode="color"
else:
    neural_radiance_field.mlp.requires_grad_(False)
    neural_radiance_field.harmonic_embedding.requires_grad_(False)
    neural_radiance_field.density_layer.requires_grad_(False)
    neural_radiance_field.color_layer.requires_grad_(False)
    neural_radiance_field.feature_layer.requires_grad_(True)
    neural_radiance_field.mode="feature"
    raymarcher.thresholdMode=True
    raymarcherBack.thresholdMode=True
    thresholdFac=0.2
    raymarcher.threshold=thresholdFac
    raymarcherBack.threshold=thresholdFac
    

if not os.path.exists(expID):
    os.makedirs(expID)
    os.makedirs(expID + "/RotRes")

continue_from=True
if continue_from and not onlyNerf:
    data_nerf = torch.load(expID+"/nerflatestFine.pth")

    neural_radiance_field.load_state_dict(data_nerf["model_state_dict"])



neural_radiance_field=neural_radiance_field.to(device)


if estimateNormals:
    if not os.path.exists(nerfID+"/vert1.npy"):
      with torch.no_grad():
        np.save(nerfID+"/vert1.npy",  ( neural_radiance_field.batched_forward_forPC(threshold=thresholdFac)[0]))
    if not os.path.exists(nerfID+"/subnormal1.npy"):
        v1=np.load(nerfID+"/vert1.npy")
        v1T=torch.from_numpy(v1.astype("float32")).cuda()    
        from pytorch3d.ops import sample_farthest_points as fps
        idx3 =fps(v1T.unsqueeze(0),K=1000)[1][0].cpu()
        fv=v1T[idx3]
        import pytorch3d
        n1=-pytorch3d.ops.estimate_pointcloud_normals(fv.unsqueeze(0), neighborhood_size=400)[0]
        v1=fv.cpu().numpy()
        n1=n1.cpu().numpy()
        np.save(nerfID+"/subvert1.npy", v1)
        np.save(nerfID+"/subnormal1.npy", n1)
        tree = KDTree(v1, leaf_size=2)
    else:
        v1=torch.from_numpy(np.load(nerfID+"/subvert1.npy").astype("float32"))
        n1=torch.from_numpy(np.load(nerfID+"/subnormal1.npy").astype("float32"))
        tree = KDTree(v1.cpu().numpy(), leaf_size=2)

batch_size = 1
n_iter=20000

batch_idx1=[]
weightsList=[]
sampledRayList=[]
posVecList=[]

wpath=nerfID + "/"+str(render_size)+"_weights"
srpath=nerfID + "/"+str(render_size)+"_sampledRay"
srxyspath=nerfID + "/"+str(render_size)+"_sampledRayxys"
pvpath=nerfID + "/"+str(render_size)+"_posVec"
pvpathBack=nerfID + "/"+str(render_size)+"_posVecBack"
backsrpath=nerfID + "/"+str(render_size)+"_sampledRayBack"
backsrxyspath=nerfID + "/"+str(render_size)+"_sampledRayBackxys"


if not os.path.exists(srxyspath):
  os.makedirs(srxyspath)

if not os.path.exists(backsrxyspath):
  os.makedirs(backsrxyspath)

if not os.path.exists(pvpath):
  os.makedirs(pvpath)
if not os.path.exists(pvpathBack):
  os.makedirs(pvpathBack)        

with torch.no_grad():
    mverts= neural_radiance_field.batched_forward_forPC(threshold=thresholdFac)[0]
    tree2 = KDTree(np.asarray(mverts), leaf_size=2)

import open3d as o3d
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(mverts)
cl, ind=pcd.remove_radius_outlier(nb_points=20, radius=0.05)
mverts=mverts[ind]
tree2 = KDTree(np.asarray(mverts), leaf_size=2)


from nutil import vp
if int(args.viz) !=0:
  vp(mverts)
np.save(nerfID+"/a1.npy", mverts)


for iteration in range(len(target_cameras)):
      
    # Sample random batch indices.
    batch_idx = torch.randperm(len(target_cameras))[:1]
    batch_idx = batch_idx*0+iteration

    sampledRayList=[]
    posVecList=[]
    print("batch_idx", batch_idx)
    if os.path.exists( pvpath+"/"+str(iteration)+".pt"):
        continue
    focal = target_cameras.K[batch_idx][:, 0, 0:2]
    focal[:, 1] = target_cameras.K[batch_idx][:, 1, 1]
    principal_point = target_cameras.K[batch_idx][:, 0:2, 2]
    imageSize = principal_point * 0
    imageSize += render_size
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


    if gridSampler:
      
      with torch.no_grad():
        _, sampled_rays, weights = renderer_grid(
            cameras=batch_cameras,
            volumetric_function=neural_radiance_field.batched_forward,add_input_samples=add_input_samples, stratified=stratified,
            maskRays=maskRays,mask=target_silhouettes[batch_idx, ..., None]
        )

        posVec=(sampled_rays.origins+sampled_rays.directions*torch.max(sampled_rays.lengths * weights[:,:,0:rayCT],dim=-1)[0].unsqueeze(-1)).cpu()
        
        pdist1,pind1 = tree2.query(posVec[0].cpu().numpy(), k=1)
        idx1= np.where(pdist1[:,0]<0.1)[0]
          
        weights=weights.cpu()
        
        posVec=posVec[:,idx1,:].cpu()
        

        sampled_rays=RayBundle(
                    origins=sampled_rays.origins[:,idx1].cpu(),
                    directions=sampled_rays.directions[:,idx1].cpu(),
                    lengths=sampled_rays.lengths[:,idx1].cpu(),
                    xys=sampled_rays.xys[:,idx1].cpu(),
                )                
        
        backRaysLengths=(sampled_rays.lengths-sampled_rays.lengths[:,:,0].unsqueeze(-1))/3
        backRays=RayBundle(
                    origins=posVec.cuda(),
                    directions=-(sampled_rays.origins/torch.norm(sampled_rays.origins,dim=-1).unsqueeze(-1)).cuda(),
                    lengths=backRaysLengths.cuda(),
                    xys=sampled_rays.xys.cuda(),
                )
        posVec=posVec
        back_rays_densities, _ = neural_radiance_field.batched_forward_fordensity(ray_bundle=backRays)
        _,backWeights = raymarcherBack(rays_densities=back_rays_densities, rays_features=torch.zeros((back_rays_densities.shape)).cuda())
        del back_rays_densities
        posVecBack=(backRays.origins+backRays.directions*torch.max(backRays.lengths * backWeights[:,:,rayCT:],dim=-1)[0].unsqueeze(-1)).cpu()
 
        posVecBack=posVecBack
        
        pdist2,pind2 = tree2.query(posVecBack[0].cpu().numpy(), k=1)
        idx2= np.where(pdist2[:,0]<0.1)[0]
        
        posVecBack=posVecBack[:,idx2].cpu()
        

        cpuray_bundle_backRays= RayBundle(
                    origins=backRays.origins.cpu()[:,idx2],
                    directions=backRays.directions.cpu()[:,idx2],
                    lengths=backRays.lengths.cpu()[:,idx2],
                    xys=backRays.xys.cpu()[:,idx2],
                )

        cpuray_bundle= RayBundle(
                    origins=sampled_rays.origins.cpu(),
                    directions=sampled_rays.directions.cpu(),
                    lengths=sampled_rays.lengths.cpu(),
                    xys=sampled_rays.xys.cpu(),
                )

        torch.save(cpuray_bundle.xys, srxyspath+"/"+str(iteration)+".pt")
        torch.save(posVec.cpu(), pvpath+"/"+str(iteration)+".pt")
        torch.save(posVecBack.cpu(), pvpathBack+"/"+str(iteration)+".pt")
        torch.save(cpuray_bundle_backRays.xys, backsrxyspath+"/"+str(iteration)+".pt")
        del weights, cpuray_bundle,cpuray_bundle_backRays, posVec,posVecBack,idx2,backWeights,backRaysLengths,sampled_rays,idx1
        torch.cuda.empty_cache()
