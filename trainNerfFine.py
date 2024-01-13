import os, json, torch, cv2
import numpy as np
from nutil import show_full_render1
from nutil import get_emb_vis, huber, sample_images_at_mc_locs, rotfromeulernp
  
from nerf import NeuralRadianceFieldFeat
from pren import ImplicitRendererStratified, EmissionAbsorptionRaymarcherStratified
from pren2 import ImplicitRendererStratified as ImplicitRendererStratified2
from pren2 import EmissionAbsorptionRaymarcherStratified as EmissionAbsorptionRaymarcherStratified2

from cowrendersynth import generate_cow_rendersWithRT, generate_bop_realsamples, generate_tless_realsamples
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
#from vispc import v2p,vp
from pytorch3d.renderer import ray_bundle_to_ray_points

dataParallel=False
if dataParallel:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import argparse

arg_parser = argparse.ArgumentParser(description="Train a BOP NeRF")
arg_parser.add_argument("--objid",dest="objid",default="1",)
arg_parser.add_argument("--cont", dest="cont", default=False, )
arg_parser.add_argument("--UH", dest="UH", default=0)
arg_parser.add_argument("--dataset",dest="dataset",default="tless",)

args = arg_parser.parse_args()

siren=True
# objid="1"
objid=str(args.objid)

if not os.path.exists(str(args.UH) +'_'+ args.dataset+'_obj_' + str(objid)):
    os.makedirs(str(args.UH) +'_'+ args.dataset+'_obj_' + str(objid))

if args.dataset == "tless":
    expID = str(args.UH) +'_'+ args.dataset+'_obj_' + str(objid)+"/"+objid+"TLESSObj_Fine"
    datasetPath = "bop/tless"
else:    
    expID = str(args.UH) +'_'+ args.dataset+'_obj_' + str(objid)+"/"+objid+"TLESSObj_Fine"
    datasetPath = "bop/ruapc"
print("objid", objid)

lr = 1e-3
fineEpoch=0000
epochChange=300000
batch_size=3
n_iter = 500
imD=200
maxB=200



pytorchRendered=False
if not pytorchRendered:
  in_ndc=False
  #fsamps=int(0.5*2861)
  #fewIds=np.load("TL75/few.npy")
  #if int(objid)==19 or int(objid)==20:
  if args.dataset == 'tless': # dataset only has 1000 images
    fsamps=int(1001*0.5)      ## number of samples 
    fewIds=np.arange(int(1001*0.5))  ## list of ID of samples (from 0 to 1280)
    if not int(args.UH):
        fewIds=fewIds+500
  else:
    fsamps=int(2561*0.5)      ## number of samples 
    fewIds=np.arange(int(2561*0.5))  ## list of ID of samples (from 0 to 1280)
    if not int(args.UH):
        fewIds=fewIds+1280  
  #import pdb;pdb.set_trace()
  print("Image Ids", fewIds)
  if args.dataset == 'tless':
    target_images, target_silhouettes,RObj,TObj, KObj, fewIds = generate_bop_realsamples(datasetPath,objid=objid,crop=True,maskStr="mask", offset=5,synth=False, makeNDC=in_ndc, dataset=args.dataset, fewSamps=True, fewCT=fsamps, fewids=fewIds)
    #target_images, target_silhouettes,RObj,TObj, KObj= generate_tless_realsamples(datasetPath=datasetPath, sampleIds=fewIds, objid=objid, imD=128, crop=True, maskStr="mask_visib", cropDim=70,justScaleImages=False, ScaledSize=128, makeNDC=in_ndc)
  else:
      target_images, target_silhouettes,RObj,TObj, KObj, fewIds = generate_bop_realsamples(datasetPath,objid=objid,crop=True,maskStr="mask", offset=5,synth=False, makeNDC=in_ndc, dataset=args.dataset, fewSamps=True, fewCT=fsamps, fewids=fewIds)
  cv2.imwrite("nhap/" +"target0.jpg", target_images[5].numpy())
  cv2.imwrite("nhap/" + "mask0.jpg", target_silhouettes[5].numpy())
  #transforming rotation matrices and transltion components to obey pytorch's convention
  rot180=rotfromeulernp(np.array([0,0,np.pi]))
  for a in range(target_images.shape[0]):
      RObj[a]=(RObj[a].T).dot(rot180)
      TObj[a,0:2] = -TObj[a,0:2]
  meshdetails = json.load(open(datasetPath+"/models" + "/models_info.json"))
  diam = meshdetails[objid]['diameter']
  diamScaling=1.8
  offset=0
    
  scale=diam/diamScaling
  TObj = TObj/scale
    
else:
  target_cameras, target_images, target_silhouettes,RObj,TObj=generate_cow_rendersWithRT(num_views=150, azimuth_range=360,objid="obj_000005.obj")


print("target_cameras", target_images.shape[0])
onlyNerf=True


if not pytorchRendered:
  min_depth=np.max(np.abs(TObj))-offset

  target_cameras = PerspectiveCameras(device=device, R=torch.from_numpy(RObj.astype("float32")), T=torch.from_numpy(TObj.astype("float32")),
                                       K=torch.from_numpy(KObj.astype("float32")))
  volume_extent_world = offset+np.max(np.abs(TObj))
  
  min_depth=np.min(np.abs(TObj[:,2]))-2
  volume_extent_world = np.max(np.abs(TObj[:,2]))+2
  
else:
  min_depth=0
  volume_extent_world = 3
print(f'Generated {len(target_images)} images/silhouettes/cameras.')


render_size = int(target_images.shape[1]*2)


raysampler_grid = NDCMultinomialRaysampler(image_height=render_size, image_width=render_size, n_pts_per_ray=256,
                                           min_depth=min_depth, max_depth=volume_extent_world,)

raysampler_mc = MonteCarloRaysampler(min_x=-1.0,max_x=1.0,min_y=-1.0,max_y=1.0,n_rays_per_image=400, n_pts_per_ray=64,
                                     min_depth=min_depth,max_depth=volume_extent_world,stratified_sampling=True)
raysampler_mc2 = MonteCarloRaysampler(min_x=-1.0,max_x=1.0,min_y=-1.0,max_y=1.0,n_rays_per_image=400, n_pts_per_ray=256,
                                     min_depth=min_depth,max_depth=volume_extent_world,stratified_sampling=True)
raysampler_grid_eval = NDCMultinomialRaysampler(image_height=int(render_size/2), image_width=int(render_size/2), n_pts_per_ray=128,
                                           min_depth=min_depth, max_depth=volume_extent_world,)
raymarcher = EmissionAbsorptionRaymarcherStratified()
raymarcher.thresholdMode=False
raymarcher2 = EmissionAbsorptionRaymarcherStratified2()
raymarcher2.thresholdMode=False

renderer_grid = ImplicitRendererStratified(raysampler=raysampler_grid, raymarcher=raymarcher,device=device)
renderer_mc = ImplicitRendererStratified(raysampler=raysampler_mc, raymarcher=raymarcher,device=device)
renderer_mc2 = ImplicitRendererStratified2(raysampler=raysampler_mc2, raymarcher=raymarcher2,device=device)

renderer_grid_eval = ImplicitRendererStratified(raysampler=raysampler_grid_eval, raymarcher=raymarcher,device=device)

# First move all relevant variables to the correct device.
renderer_grid = renderer_grid.to(device)
renderer_mc = renderer_mc.to(device)
renderer_mc2 = renderer_mc2.to(device)

target_cameras = target_cameras.to(device)


# Set the seed for reproducibility
torch.manual_seed(1)

# Instantiate the radiance field model.
neural_radiance_field = NeuralRadianceFieldFeat(siren=siren)
neural_radiance_field_FINE = NeuralRadianceFieldFeat(siren=siren)

if args.cont:	
    data_nerf = torch.load(expID+"/nerflatestFine.pth")	
    neural_radiance_field.load_state_dict(data_nerf["model_state_dict"])	
    # data_rgb = torch.load(expID+"/encoderRGBlatest.pth")	
    # encoder_rgb.load_state_dict(data_rgb["model_state_dict"])	
    #data_rgb2 = torch.load(expID+"/encoderRGBlatest.pth")	
    #encoder_rgb2.load_state_dict(data_rgb2["model_state_dict"])	
    print("continuing")

neural_radiance_field.mlp.requires_grad_(True)
neural_radiance_field.harmonic_embedding.requires_grad_(True)
neural_radiance_field.density_layer.requires_grad_(True)
neural_radiance_field.color_layer.requires_grad_(True)
neural_radiance_field.feature_layer.requires_grad_(False)
neural_radiance_field.mode="color"

neural_radiance_field_FINE.mlp.requires_grad_(True)
neural_radiance_field_FINE.harmonic_embedding.requires_grad_(True)
neural_radiance_field_FINE.density_layer.requires_grad_(True)
neural_radiance_field_FINE.color_layer.requires_grad_(True)
neural_radiance_field_FINE.feature_layer.requires_grad_(False)
neural_radiance_field_FINE.mode="color"



if not os.path.exists(expID):
    os.makedirs(expID)
    os.makedirs(expID + "/RotRes")



      
if dataParallel:
    neural_radiance_field = torch.nn.DataParallel(neural_radiance_field)
    neural_radiance_field=neural_radiance_field.to(device)
else:
    neural_radiance_field=neural_radiance_field.to(device)
    neural_radiance_field_FINE=neural_radiance_field_FINE.to(device)
    




# Instantiate the Adam optimizer. We set its master learning rate to 1e-3.


color_err=0;asiler=0 # silhouette error


optimizer = torch.optim.Adam([{"params": neural_radiance_field.parameters(), "lr": lr},{"params": neural_radiance_field_FINE.parameters(), "lr": lr}])


# Init the loss history buffers.
loss_history_color, loss_history_sil = [], []
err0=0
iteration=0
for iteration3 in range(n_iter):
    idx11=torch.randperm(len(target_cameras))
    for iteration2 in range(int(len(idx11)/batch_size)):
            iteration+=1


            if iteration%100==0 and iteration>1:
                torch.save(
                    {"epoch": iteration, "model_state_dict": neural_radiance_field.state_dict()},
                    expID+"/nerflatest.pth",
                )
                torch.save(
                    {"epoch": iteration, "model_state_dict": neural_radiance_field_FINE.state_dict()},
                    expID+"/nerflatestFine.pth",
                )

            # Zero the optimizer gradient.
            optimizer.zero_grad()

            # Sample random batch indices.
            batch_idx = idx11[iteration2:iteration2+5][:batch_size]

            # Sample the minibatch of cameras.
            if pytorchRendered:
              batch_cameras = FoVPerspectiveCameras(
                R = target_cameras.R[batch_idx],
                T = target_cameras.T[batch_idx],
                znear = target_cameras.znear[batch_idx],
                zfar = target_cameras.zfar[batch_idx],
                aspect_ratio = target_cameras.aspect_ratio[batch_idx],
                fov = target_cameras.fov[batch_idx],
                device = device,
            )
            else:
              focal=target_cameras.K[batch_idx][:,0,0:2]
              focal[:,1] = target_cameras.K[batch_idx][:,1,1]
              principal_point= target_cameras.K[batch_idx][:,0:2,2]
              imageSize=principal_point*0
              imageSize+=imD
              if in_ndc:
                  batch_cameras = PerspectiveCameras(
                      R=target_cameras.R[batch_idx],
                      T=target_cameras.T[batch_idx],
                      # focal_length=focal,
                      # principal_point=principal_point,
                      # image_size=imageSize,
                      K=target_cameras.K[batch_idx],
                      in_ndc=in_ndc,
                      device=device,
                  )
              else:
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
            stratified=False




            rendered_images_silhouettes, sampled_rays, weights = renderer_mc(
                    cameras=batch_cameras,
                    volumetric_function=neural_radiance_field, add_input_samples=False,stratified=False
                )

            if iteration>fineEpoch:
                renderer_mc2.coarseR=sampled_rays
                renderer_mc2.coarseW=weights.detach()

                rendered_images_silhouettes_FINE, sampled_rays2, weights2 = renderer_mc2(
                      cameras=batch_cameras,
                      volumetric_function=neural_radiance_field_FINE, add_input_samples=True,stratified=True
                  )
            sampled_raysxys=sampled_rays2.xys


            lastDim=rendered_images_silhouettes.shape[-1]

            rendered_images, rendered_silhouettes = (
                rendered_images_silhouettes.split([lastDim-1, 1], dim=-1)
            )
            if iteration>fineEpoch:
              rendered_images_FINE, rendered_silhouettes_FINE = (
                  rendered_images_silhouettes_FINE.split([lastDim-1, 1], dim=-1)
              )

            silhouettes_at_rays = sample_images_at_mc_locs(
                    target_silhouettes[batch_idx, ..., None].to(device),
                    sampled_raysxys
                )


            colors_at_rays = sample_images_at_mc_locs(
                    target_images[batch_idx].to(device),
                    sampled_raysxys
                )
            asiler = huber(rendered_silhouettes,silhouettes_at_rays,).abs().mean()

            color_err = huber(rendered_images[:, :, :], colors_at_rays, ).abs().mean()
            if iteration>fineEpoch:
              asiler = asiler + huber(rendered_silhouettes_FINE,silhouettes_at_rays,).abs().mean()
              color_err = color_err + huber(rendered_images_FINE[:, :, :], colors_at_rays, ).abs().mean()


            ct=len(target_images)

            asiler=500*asiler
            color_err=500*color_err
            loss=color_err+asiler

            loss_history_color.append(float(color_err))
            loss_history_sil.append(float(asiler))

            # Every 10 iterations, print the current values of the losses.
            if iteration % 10 == 5:
                print(
                    f'Iteration {iteration:05d}:'
                    + f' loss color = {float(color_err):1.2e}'
                    + f' loss Sil = {float(asiler):1.2e}'
                )

            if iteration==50000:
                 torch.save({"epoch": iteration, "model_state_dict": neural_radiance_field.state_dict()},expID+"/nerf50k.pth",)
                 torch.save({"epoch": iteration, "model_state_dict": neural_radiance_field_FINE.state_dict()},expID+"/nerffine50k.pth",)

            loss.backward()
            optimizer.step()

            # Visualize the full renders every 100 iterations.
            if iteration %1000 ==0:
                with torch.no_grad():
                    # neural_radiance_field.scale=1
                    np.save(expID+"/v1.npy", neural_radiance_field.batched_forward_forPC(threshold=0.03)[0])
                    if iteration>fineEpoch:
                      np.save(expID+"/v1fine.npy", neural_radiance_field_FINE.batched_forward_forPC(threshold=0.03)[0])
                    
                    # neural_radiance_field.scale=scale
            if iteration % 100 == 0:
                show_idx = torch.randperm(len(target_cameras))[:1]
                import cv2

               # import pdb;pdb.set_trace()
                r11=str(int(torch.rand(1)*100)+1)

                cv2.imwrite(expID + "/" + r11 + "_target.jpg", target_images[batch_idx][0].cpu().numpy() * 255)
                if pytorchRendered:
                   show_full_render1(
                        neural_radiance_field,
                        FoVPerspectiveCameras(
                            R=target_cameras.R[batch_idx[0:1]],
                            T=target_cameras.T[batch_idx[0:1]],

                            znear=target_cameras.znear[batch_idx[0:1]],
                            zfar=target_cameras.zfar[batch_idx[0:1]],
                            aspect_ratio=target_cameras.aspect_ratio[batch_idx[0:1]],
                            fov=target_cameras.fov[batch_idx[0:1]],
                            device=device
                        ),

                        expID=expID,renderer_grid=renderer_grid_eval, normalization=True
                    )

                else:
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

                        expID=expID, renderer_grid=renderer_grid_eval, normalization=True, rand=r11, 
                    )
                    if iteration>fineEpoch:
                      show_full_render1(
                          neural_radiance_field_FINE,
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

                          expID=expID, renderer_grid=renderer_grid_eval, normalization=True, rand=r11, savePath="_nerfFine"
                      )