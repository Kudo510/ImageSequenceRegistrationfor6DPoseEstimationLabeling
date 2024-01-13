import os, json, torch
import numpy as np
from nerf import NeuralRadianceFieldFeat
from pren import ImplicitRendererStratified, EmissionAbsorptionRaymarcherStratified

from cowrendersynth import  generate_cow_rendersWithRT, generate_bop_realsamples

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
arg_parser = argparse.ArgumentParser(description="Train a BOP NeRF")
arg_parser.add_argument("--objid",dest="objid",default="35",)
arg_parser.add_argument("--cont", dest="cont", default=False, )
arg_parser.add_argument("--UH", dest="UH", default=0)
arg_parser.add_argument("--dataset",dest="dataset",default="tless",)
args = arg_parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

objid=str(args.objid)
# rayID ="ruapc/"+ objid+"Cors/"
# nerfID ="ruapc/"+ objid+"TLESSObj_Fine/"
# expID = "ruapc/"+objid+"poseEst"
rayID =str(args.UH) +'_'+ args.dataset+'_obj_' + str(objid)+ "/"+objid+"Cors"
nerfID =str(args.UH) +'_'+ args.dataset+'_obj_' + str(objid)+ "/" + objid+"TLESSObj_Fine/"
expID = str(args.UH) +'_'+ args.dataset+'_obj_' + str(objid)+ "/" +objid+"poseEst"
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
# datasetPath = "bop/ruapc/"
datasetPath = "bop/" + args.dataset
print("objid", objid)
imD = 224
render_size=224
batch_size = 16

in_ndc = False

target_backgrounds=None
fsamps = int(2561 * 0.5)
fewIds = np.arange(int(2561 * 0.5))
if not int(args.UH):
  fewIds = fewIds + 1280

target_images, target_silhouettes, RObj, TObj, KObj, fewIds = generate_bop_realsamples(datasetPath,objid=objid, crop=True,maskStr="mask",offset=5, synth=False,makeNDC=in_ndc,dataset=args.dataset, maxB=imD, background=False, fewSamps=True, fewCT=fsamps, fewids=fewIds)


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
estimateNormals=False
target_cameras = PerspectiveCameras(device=device, R=torch.from_numpy(RObj.astype("float32")),
                                    T=torch.from_numpy(TObj.astype("float32")),
                                    K=torch.from_numpy(KObj.astype("float32")))
min_depth = np.min(np.abs(TObj[:,2])) - 2
volume_extent_world = np.max(np.abs(TObj[:,2])) + 2

render_size = target_images.shape[1]
#import pdb;pdb.set_trace()
gridSampler=True
if gridSampler:
  if not onlyNerf:  ## our case onlyNerf = F
    render_size=224

batch_size = 16
if onlyNerf:
    batch_size=16

raysampler_grid = NDCMultinomialRaysampler(image_height=render_size, image_width=render_size, n_pts_per_ray=128, min_depth=min_depth, max_depth=volume_extent_world,)
raysampler_grid_eval = NDCMultinomialRaysampler(image_height=int(render_size/2), image_width=int(render_size/2), n_pts_per_ray=128, min_depth=min_depth, max_depth=volume_extent_world,)
#import pdb;pdb.set_trace()
raysampler_mc = MonteCarloRaysampler(min_x=-1.0,max_x=1.0,min_y=-1.0,max_y=1.0,n_rays_per_image=30, n_pts_per_ray=128,
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



if not os.path.exists(expID):
    os.makedirs(expID)
np.save(expID+"/few.npy", fewIds)




data_nerf = torch.load(expID+"/nerflatest.pth") ## after training trainPose.py
#data_nerf = torch.load("1_ruapc_obj_2/2TLESSObj_Fine/nerflatestFine.pth") ## use to create surface points only - after training trainNerffine.py
neural_radiance_field.load_state_dict(data_nerf["model_state_dict"])
neural_radiance_field=neural_radiance_field.to(device)






fullNegVec=torch.Tensor([])
batch_idx1=[]
add_input_samples = False
if negativeSampling:
    if not rayFreeze or len(batch_idx1) == 0:
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
            loopCT = 19
        else:
            loopCT = 1
        if fullNegVec.shape[0] == 0:
            for a in range(loopCT):
                with torch.no_grad():
                    _, sampled_rays1, weightsNeg = renderer_mc(
                        cameras=batch_cameras1,
                        volumetric_function=neural_radiance_field.batched_forward, add_input_samples=add_input_samples,
                        stratified=stratified,
                        maskRays=maskRays, mask=target_silhouettes[batch_idx1, ..., None])

                sampled_raysxys1 = sampled_rays1.xys
                negVec1 = sampled_rays1.origins + sampled_rays1.directions * \
                          torch.max(sampled_rays1.lengths * weightsNeg, dim=-1)[0].unsqueeze(-1)

                # import pdb;pdb.set_trace()
                idx2 = torch.where(torch.norm(negVec1 - sampled_rays1.origins, dim=-1)[0])[0]
                negVec1 = negVec1[:, idx2]
                fullNegVec = torch.cat([fullNegVec, negVec1.cpu()], dim=1)
            from pytorch3d.ops import sample_farthest_points as fps

            idx3 = fps(fullNegVec, K=80000)[1][0].cpu()
            fullNegVec = fullNegVec[:, idx3]
            # sampled_raysxys1=sampled_raysxys1[:,idx2]
    fnVec = fullNegVec[0, torch.where(torch.max(torch.abs(fullNegVec[0, :, :]), dim=-1)[0] < 1.2)[0]].unsqueeze(0)
    with torch.no_grad():
        mverts, mtriangles= neural_radiance_field.batched_forward_forPC(threshold=0.05)
    #import pdb;pdb.set_trace()
    import trimesh
    mmesh=trimesh.Trimesh(mverts, mtriangles)
    mnormals=np.asarray(mmesh.vertex_normals)

    tree2 = KDTree(np.asarray(mverts), leaf_size=2)

    pdist1,pind1 = tree2.query(fnVec[0].cpu().numpy(), k=1)
    closeidx=np.where(pdist1[:,0]<0.05)[0]
    fnVec=fnVec[:,closeidx]
    fnormalsVec=mnormals[pind1[:,0]][closeidx]
    with torch.no_grad():
        rendered_images_silhouettes1 = neural_radiance_field.batched_customForward(fnVec.to(device))
        lastDim1 = rendered_images_silhouettes1.shape[-1]
        rendered_images1, rendered_silhouettes1 = (
            rendered_images_silhouettes1.split([lastDim1 - 1, 1], dim=-1))
        surfacePointsScaled = fnVec[0].cpu().numpy() * (diam / diamScaling)
        surfaceFeatures = rendered_images1[0].cpu().numpy()

        np.save(expID + "/vert1_scaled.npy", surfacePointsScaled)
        np.save(expID + "/feat1_scaled.npy", surfaceFeatures)
        np.save(expID + "/normals_scaled.npy", fnormalsVec)
