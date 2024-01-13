# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import cv2, json, trimesh
from nutil import extractRT
import numpy as np
import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    BlendParams,
    FoVPerspectiveCameras,
    look_at_view_transform,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    SoftSilhouetteShader,
)


# create the default data directory
current_dir = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(current_dir, "..", "data", "cow_mesh")


def generate_cow_renders(
    num_views: int = 40, data_dir: str = DATA_DIR, azimuth_range: float = 180
):
    """
    This function generates `num_views` renders of a cow mesh.
    The renders are generated from viewpoints sampled at uniformly distributed
    azimuth intervals. The elevation is kept constant so that the camera's
    vertical position coincides with the equator.

    For a more detailed explanation of this code, please refer to the
    docs/tutorials/fit_textured_mesh.ipynb notebook.

    Args:
        num_views: The number of generated renders.
        data_dir: The folder that contains the cow mesh files. If the cow mesh
            files do not exist in the folder, this function will automatically
            download them.
        azimuth_range: number of degrees on each side of the start position to
            take samples

    Returns:
        cameras: A batch of `num_views` `FoVPerspectiveCameras` from which the
            images are rendered.
        images: A tensor of shape `(num_views, height, width, 3)` containing
            the rendered images.
        silhouettes: A tensor of shape `(num_views, height, width)` containing
            the rendered silhouettes.
    """

    # set the paths

    # download the cow mesh if not done before
    cow_mesh_files = [
        os.path.join(data_dir, fl) for fl in ("cow.obj", "cow.mtl", "cow_texture.png")
    ]
    if any(not os.path.isfile(f) for f in cow_mesh_files):
        os.makedirs(data_dir, exist_ok=True)
        os.system(
            f"wget -P {data_dir} "
            + "https://dl.fbaipublicfiles.com/pytorch3d/data/cow_mesh/cow.obj"
        )
        os.system(
            f"wget -P {data_dir} "
            + "https://dl.fbaipublicfiles.com/pytorch3d/data/cow_mesh/cow.mtl"
        )
        os.system(
            f"wget -P {data_dir} "
            + "https://dl.fbaipublicfiles.com/pytorch3d/data/cow_mesh/cow_texture.png"
        )

    # Setup
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # Load obj file
    obj_filename = os.path.join(data_dir, "cow.obj")
    mesh = load_objs_as_meshes([obj_filename], device=device)

    # We scale normalize and center the target mesh to fit in a sphere of radius 1
    # centered at (0,0,0). (scale, center) will be used to bring the predicted mesh
    # to its original center and scale.  Note that normalizing the target mesh,
    # speeds up the optimization but is not necessary!
    verts = mesh.verts_packed()
    N = verts.shape[0]
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])
    mesh.offset_verts_(-(center.expand(N, 3)))
    mesh.scale_verts_((1.0 / float(scale)))

    # Get a batch of viewing angles.
    elev = torch.linspace(0, 0, num_views)  # keep constant
    azim = torch.linspace(-azimuth_range, azimuth_range, num_views) + 180.0

    # Place a point light in front of the object. As mentioned above, the front of
    # the cow is facing the -z direction.
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    # Initialize an OpenGL perspective camera that represents a batch of different
    # viewing angles. All the cameras helper methods support mixed type inputs and
    # broadcasting. So we can view the camera from the a distance of dist=2.7, and
    # then specify elevation and azimuth angles for each viewpoint as tensors.
    def m1(R1,T1,R2,T2):
      F1=np.eye(4)
      F2=np.eye(4)
      F1[0:3,0:3]=R1
      F1[3,0:3]=T1
      F2[0:3,0:3]=R2
      F2[3,0:3]=T2
      return F1,F2
    R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
    #import pdb;pdb.set_trace()
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    # Define the settings for rasterization and shading. Here we set the output
    # image to be of size 128X128. As we are rendering images for visualization
    # purposes only we will set faces_per_pixel=1 and blur_radius=0.0. Refer to
    # rasterize_meshes.py for explanations of these parameters.  We also leave
    # bin_size and max_faces_per_bin to their default values of None, which sets
    # their values using heuristics and ensures that the faster coarse-to-fine
    # rasterization method is used.  Refer to docs/notes/renderer.md for an
    # explanation of the difference between naive and coarse-to-fine rasterization.
    raster_settings = RasterizationSettings(
        image_size=128, blur_radius=0.0, faces_per_pixel=1
    )

    # Create a Phong renderer by composing a rasterizer and a shader. The textured
    # Phong shader will interpolate the texture uv coordinates for each vertex,
    # sample from a texture image and apply the Phong lighting model
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0.0, 0.0, 0.0))
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(
            device=device, cameras=cameras, lights=lights, blend_params=blend_params
        ),
    )

    # Create a batch of meshes by repeating the cow mesh and associated textures.
    # Meshes has a useful `extend` method which allows us do this very easily.
    # This also extends the textures.
    meshes = mesh.extend(num_views)

    # Render the cow mesh from each viewing angle
    target_images = renderer(meshes, cameras=cameras, lights=lights)

    # Rasterization settings for silhouette rendering
    sigma = 1e-4
    raster_settings_silhouette = RasterizationSettings(
        image_size=128, blur_radius=np.log(1.0 / 1e-4 - 1.0) * sigma, faces_per_pixel=50
    )

    # Silhouette renderer
    renderer_silhouette = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, raster_settings=raster_settings_silhouette
        ),
        shader=SoftSilhouetteShader(),
    )

    # Render silhouette images.  The 3rd channel of the rendering output is
    # the alpha/silhouette channel
    silhouette_images = renderer_silhouette(meshes, cameras=cameras, lights=lights)

    # binary silhouettes
    silhouette_binary = (silhouette_images[..., 3] > 1e-4).float()

    return cameras, target_images[..., :3], silhouette_binary

def generate_tless_realsamples(datasetPath, sampleIds, objid="22", imD=128, crop=True, maskStr="mask_visibl", cropDim=70,justScaleImages=False, ScaledSize=128, makeNDC=True):
    
    
    imCT = len(sampleIds)
    target_images = torch.zeros((imCT, imD, imD, 3))
    target_silhouettes = torch.zeros((imCT, imD, imD))
    if justScaleImages:
      target_images = torch.zeros((imCT, ScaledSize, ScaledSize, 3))
      target_silhouettes = torch.zeros((imCT, ScaledSize,ScaledSize))
     
    RObj = np.zeros((imCT, 3, 3))
    TObj = np.zeros((imCT, 3))
    KObj = np.zeros((imCT, 4, 4))

    imIds = sampleIds

    objPath = datasetPath+"/train_primesense/0000" + str(objid).zfill(2) + "/"
    meshdetails = json.load(open(datasetPath+"/models_cad" + "/models_info.json"))
    camParams = json.load(open(objPath+"/scene_camera.json"))
    scale = meshdetails[objid]['diameter']

    mesh = trimesh.load(datasetPath+"/models_cad/obj_0000" + objid.zfill(2) + ".ply")
    cx1=0
    cx2=400
    if crop:
        cx1=cx1+cropDim
        cx2=cx2-cropDim
    for a in range(imCT):
        imId = int(imIds[a])
        rgbPath = objPath + "rgb/" + str(imId).zfill(6) + ".png"
        c1 = cv2.imread(objPath + "rgb/" + str(imId).zfill(6) + ".png")[cx1:cx2,cx1:cx2]
        m1 = cv2.imread(objPath + maskStr + str(imId).zfill(6) + "_000000.png")[cx1:cx2,cx1:cx2][:, :, 0]
        if imD < c1.shape[0]:
            target_images[a] = torch.from_numpy(cv2.resize(c1, (imD, imD),interpolation=cv2.INTER_LANCZOS4).astype("float32") / 255)
            target_silhouettes[a] = torch.from_numpy(cv2.resize(m1, (imD, imD),interpolation=cv2.INTER_LINEAR).astype("float32") / 255)

        
        else:
            if justScaleImages:
              target_images[a] = torch.from_numpy(cv2.resize(c1, (ScaledSize, ScaledSize),interpolation=cv2.INTER_LANCZOS4).astype("float32") / 255)
              
              target_silhouettes[a] = torch.from_numpy(cv2.resize(m1, (ScaledSize, ScaledSize), interpolation=cv2.INTER_LINEAR).astype("float32") / 255)
            else:
              target_images[a] = torch.from_numpy(c1.astype("float32") / 255)
              target_silhouettes[a] = torch.from_numpy(m1.astype("float32") / 255)
            
        gtR, gtT = extractRT(objPath + "rgb/" + str(imId).zfill(6) + ".png", occid=0)
        RObj[a] = gtR
        TObj[a] = gtT
        camparams=np.asarray(camParams[str(imId)]["cam_K"]).reshape(3,3)
        if crop:
            camparams[0,2]-=cx1
            camparams[1,2]-=cx1
            camparams = camparams * imD / c1.shape[0]
            camparams[2,2]=1
            
        if makeNDC:
            camparams = camparams * 2 / imD
            camparams[2, 2] = 0
            # camparams[0, 2] -= 1
            # camparams[1, 2] -= 1

            camparams[0, 2] = -(camparams[0, 2]-1)
            camparams[1, 2] = -(camparams[1, 2]-1)

        KObj[a][0:3, 0:3] = camparams
        KObj[a][3, 2] = 1
        KObj[a][2, 3] = 1


        a11=0
    return target_images, target_silhouettes, RObj, TObj, KObj

def generate_lm_realsamples(datasetPath, objid="22", imD=128, crop=True, maskStr="mask",cropDim=70, justScaleImages=False, ScaledSize=128, maxB=200, offset=10,
                            synth=True, makeNDC=True):
    lmDir="lm"
    if synth:
      lmDir="lm_synth"
    # bboxDets = json.load(open(datasetPath + "/"+lmDir+"/" + str(objid).zfill(6) + "/scene_gt_info.json"))

    with open(datasetPath+"lmTrains/"+str(objid)+".txt") as f:
        lines = f.readlines()
    
    # imCT = len(bboxDets)
    imCT= len(lines)
    target_images = torch.zeros((imCT, maxB, maxB, 3))
    target_silhouettes = torch.zeros((imCT, maxB, maxB))
    RObj = np.zeros((imCT, 3, 3))
    TObj = np.zeros((imCT, 3))
    KObj = np.zeros((imCT, 4, 4))



    objPath = datasetPath + "/"+lmDir+"/0000" + str(objid).zfill(2) + "/"
    camParams = json.load(open(objPath + "/scene_camera.json"))


    for a11 in range(imCT):
        imId = int(lines[a11])
        rgbPath = objPath + "rgb/" + str(imId).zfill(6) + ".png"
        c1 = cv2.imread(objPath + "rgb/" + str(imId).zfill(6) + ".png")
        m1 = cv2.imread(objPath + maskStr+"/" + str(imId).zfill(6) + "_000000.png")
        c1[np.where(m1==0)]=0
        x2, y2, w2, h2 = cv2.boundingRect(m1[:,:,0])
        hd1=int(np.max((w2,h2))/2)

        centerX=x2+int(w2/2)
        centerY=y2+int(h2/2)

        cropRGB  = c1[centerY-offset-hd1:centerY+hd1+offset, centerX-offset-hd1:centerX+offset+hd1]
        cropMask = m1[centerY-offset-hd1:centerY+hd1+offset, centerX-offset-hd1:centerX+offset+hd1,0]

        # print("im1")
        # if cropRGB.shape[0]!=cropRGB.shape[1]:
        #   import pdb;pdb.set_trace()
        if justScaleImages:
                target_images[a11] = torch.from_numpy(cv2.resize(cropRGB, (ScaledSize, ScaledSize)).astype("float32") / 255)
                target_silhouettes[a11] = torch.from_numpy(
                    cv2.resize(cropMask, (ScaledSize, ScaledSize), cv2.INTER_NEAREST).astype("float32") / 255)
        else:
            target_images[a11] = torch.from_numpy(cv2.resize(cropRGB, (maxB, maxB), cv2.INTER_CUBIC).astype("float32") / 255)
            target_silhouettes[a11] = torch.from_numpy(cv2.resize(cropMask, (maxB, maxB), cv2.INTER_NEAREST).astype("float32") / 255)

        gtR, gtT = extractRT(objPath + "rgb/" + str(imId).zfill(6) + ".png", occid=0)
        RObj[a11] = gtR
        TObj[a11] = gtT
        camparams = np.asarray(camParams[str(imId)]["cam_K"]).reshape(3, 3)

        camparams[0,2]=camparams[0,2]-(centerX - offset-hd1)
        camparams[1,2]=camparams[1,2]-(centerY - offset-hd1)
        camparams = camparams*maxB/cropRGB.shape[0]
        camparams[2,2] = 1


        if makeNDC:
            camparams = camparams * 2 / maxB
            camparams[2, 2] = 0
            # camparams[0, 2] -= 1
            # camparams[1, 2] -= 1

            camparams[0, 2] = -(camparams[0, 2]-1)
            camparams[1, 2] = -(camparams[1, 2]-1)

        KObj[a11][0:3, 0:3] = camparams
        KObj[a11][3, 2] = 1
        KObj[a11][2, 3] = 1

    return target_images, target_silhouettes, RObj, TObj, KObj

def generate_lm_realsamplesWithoutLMTrainsBG(datasetPath, objid="22", imD=128, crop=True, maskStr="mask", cropDim=70,
                                           justScaleImages=False, ScaledSize=128, maxB=200, offset=10,
                                           synth=True, makeNDC=True, dataset="tless", fewSamps=False, fewCT=20,
                                           fewids=[0], background=False):
    lmDir = "train_primesense"

    if dataset == "lm":
        lmDir = "lm"
        # background=True
        if synth:
            lmDir = "lm_synth"
    objPath = datasetPath + "/" + lmDir + "/0000" + str(objid).zfill(2) + "/"
    camParams = json.load(open(objPath + "/scene_camera.json"))

    bboxDets = json.load(open(datasetPath + "/" + lmDir + "/" + str(objid).zfill(6) + "/scene_gt_info.json"))
    if dataset == "tless":
        sampleIds = torch.arange(len(bboxDets))
        imCT = len(camParams)
    else:
        with open(datasetPath + "lmTrains/" + str(objid) + ".txt") as f:
            lines = f.readlines()
        imCT = len(lines)
        if fewSamps:
            if len(fewids) == 1:
                fewids = np.random.random_integers(0, imCT - 1, fewCT)
                lines = np.asarray(lines)[fewids]
            else:
                lines = fewids
            imCT = len(fewids)

    target_images = torch.zeros((imCT, maxB, maxB, 3))
    target_silhouettes = torch.zeros((imCT, maxB, maxB))
    if background:
        target_backgrounds = []
    RObj = np.zeros((imCT, 3, 3))
    TObj = np.zeros((imCT, 3))
    KObj = np.zeros((imCT, 4, 4))

    for a11 in range(imCT):
        if dataset == "tless":
            imId = int(sampleIds[a11])
        else:
            imId = int(lines[a11])
        rgbPath = objPath + "rgb/" + str(imId).zfill(6) + ".png"
        c1 = cv2.imread(objPath + "rgb/" + str(imId).zfill(6) + ".png")
        m1 = cv2.imread(objPath + maskStr + "/" + str(imId).zfill(6) + "_000000.png")
        if background:
            bgFull = np.zeros((c1.shape[0] + 200, c1.shape[1] + 200, 3))
            bgFull[100:-100, 100:-100] = c1.copy()
        # c1[np.where(m1 == 0)] = 0
        x2, y2, w2, h2 = cv2.boundingRect(m1[:, :, 0])
        if w2 % 2 != 0:
            w2 = w2 - 1
        if h2 % 2 != 0:
            h2 = h2 - 1
        hw = int(w2 / 2)
        hh = int(h2 / 2)
        hd1 = int(np.max((w2, h2)) / 2)
        maxd = int(np.max((w2, h2)))

        squareRGB1 = np.zeros((maxd + 2 * offset, maxd + 2 * offset, 3), np.uint8)
        if background:
            squareBG1 = np.zeros((maxd + 2 * offset + 200, maxd + 2 * offset + 200, 3), np.uint8)

        squareMask1 = np.zeros((maxd + 2 * offset, maxd + 2 * offset), np.uint8)
        hs1 = int(squareRGB1.shape[0] / 2)
        squareRGB1[hs1 - hh:hs1 + hh, hs1 - hw:hs1 + hw] = c1[y2:y2 + h2, x2:x2 + w2]
        squareMask1[hs1 - hh:hs1 + hh, hs1 - hw:hs1 + hw] = m1[y2:y2 + h2, x2:x2 + w2, 0]
        if background:
            squareBG1[hs1 - hh:hs1 + hh + 200, hs1 - hw:hs1 + hw + 200] = bgFull[y2:y2 + h2 + 200, x2:x2 + w2 + 200]

        centerX = x2 + int(w2 / 2)
        centerY = y2 + int(h2 / 2)

        # print("im1")
        # if cropRGB.shape[0]!=cropRGB.shape[1]:
        #   import pdb;pdb.set_trace()
        # if justScaleImages:
        #         target_images[a11] = torch.from_numpy(cv2.resize(cropRGB, (ScaledSize, ScaledSize)).astype("float32") / 255)
        #         target_silhouettes[a11] = torch.from_numpy(
        #             cv2.resize(cropMask, (ScaledSize, ScaledSize), cv2.INTER_NEAREST).astype("float32") / 255)
        # else:
        #     try:
        target_images[a11] = torch.from_numpy(
            cv2.resize(squareRGB1, (maxB, maxB), cv2.INTER_CUBIC).astype("float32") / 255)
        target_silhouettes[a11] = torch.from_numpy(
            cv2.resize(squareMask1, (maxB, maxB), cv2.INTER_NEAREST).astype("float32") / 255)
        if background:
            bgScale = int(maxB * squareBG1.shape[0] / squareRGB1.shape[0])
            if bgScale % 2 == 0:
                bgScale = bgScale - 1
            target_backgrounds = target_backgrounds + [
                torch.from_numpy(cv2.resize(squareBG1, (bgScale, bgScale), cv2.INTER_CUBIC).astype("float32") / 255)]
        # except:
        #     print("a")
        # import pdb;pdb.set_trace()
        gtR, gtT = extractRT(objPath + "rgb/" + str(imId).zfill(6) + ".png", occid=0)
        RObj[a11] = gtR
        TObj[a11] = gtT
        camparams = np.asarray(camParams[str(imId)]["cam_K"]).reshape(3, 3)

        camparams[0, 2] = camparams[0, 2] + (-x2 + hs1 - hw)
        camparams[1, 2] = camparams[1, 2] + (-y2 + hs1 - hh)
        # (centerY - offset-hd1)
        camparams = camparams * maxB / squareRGB1.shape[0]
        camparams[2, 2] = 1

        if makeNDC:
            camparams = camparams * 2 / maxB
            camparams[2, 2] = 0
            # camparams[0, 2] -= 1
            # camparams[1, 2] -= 1

            camparams[0, 2] = -(camparams[0, 2] - 1)
            camparams[1, 2] = -(camparams[1, 2] - 1)

        KObj[a11][0:3, 0:3] = camparams
        KObj[a11][3, 2] = 1
        KObj[a11][2, 3] = 1
    if fewSamps:
        if background:
            return target_images, target_silhouettes, RObj, TObj, KObj, lines, target_backgrounds
        else:
            return target_images, target_silhouettes, RObj, TObj, KObj, lines

    else:
        if background:
            return target_images, target_silhouettes, RObj, TObj, KObj, target_backgrounds
        else:
            return target_images, target_silhouettes, RObj, TObj, KObj

def generate_lm_realsamplesWithoutLMTrains(datasetPath, objid="22", imD=128, crop=True, maskStr="mask", cropDim=70,
                                           justScaleImages=False, ScaledSize=128, maxB=200, offset=10,
                                           synth=True, makeNDC=True, dataset="tless", fewSamps=False, fewCT=20,
                                           fewids=[0], background=False):
    lmDir = "train_primesense"

    if dataset == "lm":
        lmDir = "lm"
        # background=True
        if synth:
            lmDir = "lm_synth"
    objPath = datasetPath + "/" + lmDir + "/0000" + str(objid).zfill(2) + "/"
    camParams = json.load(open(objPath + "/scene_camera.json"))

    bboxDets = json.load(open(datasetPath + "/" + lmDir + "/" + str(objid).zfill(6) + "/scene_gt_info.json"))
    #if dataset == "tless":
     #   sampleIds = torch.arange(len(bboxDets))
      #  imCT = len(camParams)
    if dataset=="tless" and not fewSamps:
      sampleIds=torch.arange(len(bboxDets))
      imCT= len(camParams)
    elif dataset=="tless" and fewSamps:
      sampleIds=torch.arange(len(bboxDets))
      imCT= len(camParams) 
      #import pdb;pdb.set_trace()   
      if fewSamps:
            if len(fewids)==1:
              fewids=np.random.random_integers(0,imCT-1, fewCT)
              lines=np.arange(imCT)[fewids]
            else:
              lines=fewids
            imCT=len(fewids)
    else:
        with open(datasetPath + "lmTrains/" + str(objid) + ".txt") as f:
            lines = f.readlines()
        imCT = len(lines)
        if fewSamps:
            if len(fewids) == 1:
                fewids = np.random.random_integers(0, imCT - 1, fewCT)
                lines = np.asarray(lines)[fewids]
            else:
                lines = fewids
            imCT = len(fewids)

    target_images = torch.zeros((imCT, maxB, maxB, 3))
    target_silhouettes = torch.zeros((imCT, maxB, maxB))
    if background:
        target_backgrounds = []
    RObj = np.zeros((imCT, 3, 3))
    TObj = np.zeros((imCT, 3))
    KObj = np.zeros((imCT, 4, 4))

    for a11 in range(imCT):
        if dataset == "tless":
            if fewSamps:
              imId = int(lines[a11])
            else:
              imId = int(sampleIds[a11])
            
        else:
            imId = int(lines[a11])
        rgbPath = objPath + "rgb/" + str(imId).zfill(6) + ".png"
        c1 = cv2.imread(objPath + "rgb/" + str(imId).zfill(6) + ".png")
        m1 = cv2.imread(objPath + maskStr + "/" + str(imId).zfill(6) + "_000000.png")
        if background:
            bgFull = np.zeros((c1.shape[0] + 200, c1.shape[1] + 200, 3))
            bgFull[100:-100, 100:-100] = c1.copy()
        c1[np.where(m1 == 0)] = 0
        x2, y2, w2, h2 = cv2.boundingRect(m1[:, :, 0])
        if w2 % 2 != 0:
            w2 = w2 - 1
        if h2 % 2 != 0:
            h2 = h2 - 1
        hw = int(w2 / 2)
        hh = int(h2 / 2)
        hd1 = int(np.max((w2, h2)) / 2)
        maxd = int(np.max((w2, h2)))

        squareRGB1 = np.zeros((maxd + 2 * offset, maxd + 2 * offset, 3), np.uint8)
        if background:
            squareBG1 = np.zeros((maxd + 2 * offset + 200, maxd + 2 * offset + 200, 3), np.uint8)

        squareMask1 = np.zeros((maxd + 2 * offset, maxd + 2 * offset), np.uint8)
        hs1 = int(squareRGB1.shape[0] / 2)
        squareRGB1[hs1 - hh:hs1 + hh, hs1 - hw:hs1 + hw] = c1[y2:y2 + h2, x2:x2 + w2]
        squareMask1[hs1 - hh:hs1 + hh, hs1 - hw:hs1 + hw] = m1[y2:y2 + h2, x2:x2 + w2, 0]
        if background:
            squareBG1[hs1 - hh:hs1 + hh + 200, hs1 - hw:hs1 + hw + 200] = bgFull[y2:y2 + h2 + 200, x2:x2 + w2 + 200]

        centerX = x2 + int(w2 / 2)
        centerY = y2 + int(h2 / 2)

        # print("im1")
        # if cropRGB.shape[0]!=cropRGB.shape[1]:
        #   import pdb;pdb.set_trace()
        # if justScaleImages:
        #         target_images[a11] = torch.from_numpy(cv2.resize(cropRGB, (ScaledSize, ScaledSize)).astype("float32") / 255)
        #         target_silhouettes[a11] = torch.from_numpy(
        #             cv2.resize(cropMask, (ScaledSize, ScaledSize), cv2.INTER_NEAREST).astype("float32") / 255)
        # else:
        #     try:
        target_images[a11] = torch.from_numpy(
            cv2.resize(squareRGB1, (maxB, maxB), cv2.INTER_CUBIC).astype("float32") / 255)
        target_silhouettes[a11] = torch.from_numpy(
            cv2.resize(squareMask1, (maxB, maxB), cv2.INTER_NEAREST).astype("float32") / 255)
        if background:
            bgScale = int(maxB * squareBG1.shape[0] / squareRGB1.shape[0])
            if bgScale % 2 == 0:
                bgScale = bgScale - 1
            target_backgrounds = target_backgrounds + [
                torch.from_numpy(cv2.resize(squareBG1, (bgScale, bgScale), cv2.INTER_CUBIC).astype("float32") / 255)]
        # except:
        #     print("a")
        # import pdb;pdb.set_trace()
        gtR, gtT = extractRT(objPath + "rgb/" + str(imId).zfill(6) + ".png", occid=0)
        RObj[a11] = gtR
        TObj[a11] = gtT
        camparams = np.asarray(camParams[str(imId)]["cam_K"]).reshape(3, 3)

        camparams[0, 2] = camparams[0, 2] + (-x2 + hs1 - hw)
        camparams[1, 2] = camparams[1, 2] + (-y2 + hs1 - hh)
        # (centerY - offset-hd1)
        camparams = camparams * maxB / squareRGB1.shape[0]
        camparams[2, 2] = 1

        if makeNDC:
            camparams = camparams * 2 / maxB
            camparams[2, 2] = 0
            # camparams[0, 2] -= 1
            # camparams[1, 2] -= 1

            camparams[0, 2] = -(camparams[0, 2] - 1)
            camparams[1, 2] = -(camparams[1, 2] - 1)

        KObj[a11][0:3, 0:3] = camparams
        KObj[a11][3, 2] = 1
        KObj[a11][2, 3] = 1
    if fewSamps:
        if background:
            return target_images, target_silhouettes, RObj, TObj, KObj, lines, target_backgrounds
        else:
            return target_images, target_silhouettes, RObj, TObj, KObj, lines

    else:
        if background:
            return target_images, target_silhouettes, RObj, TObj, KObj, target_backgrounds
        else:
            return target_images, target_silhouettes, RObj, TObj, KObj

def generate_bop_realsamples(datasetPath, objid="22", imD=128, crop=True, maskStr="mask", cropDim=70,
                                           justScaleImages=False, ScaledSize=128, maxB=200, offset=10,
                                           synth=True, makeNDC=True, dataset="tless", fewSamps=False, fewCT=20,
                                           fewids=[0], background=False, lmDir="train"):
    #lmDir = "train"

    if dataset == "lm":
        lmDir = "lm"
        # background=True
        if synth:
            lmDir = "lm_synth"
    objPath = datasetPath + "/" + lmDir + "/0000" + str(objid).zfill(2) + "/"
    camParams = json.load(open(objPath + "/scene_camera.json"))

    bboxDets = json.load(open(datasetPath + "/" + lmDir + "/" + str(objid).zfill(6) + "/scene_gt_info.json"))
    #if dataset == "tless":
     #   sampleIds = torch.arange(len(bboxDets))
      #  imCT = len(camParams)
    if not fewSamps:
      sampleIds=torch.arange(len(bboxDets))
      imCT= len(camParams)
    else:
      sampleIds=torch.arange(len(bboxDets))
      imCT= len(camParams) 
      #import pdb;pdb.set_trace()   
     
      if len(fewids)==1:
        fewids=np.random.random_integers(0,imCT-1, fewCT)
        lines=np.arange(imCT)[fewids]
      else:
        lines=fewids
      imCT=len(fewids)


    target_images = torch.zeros((imCT, maxB, maxB, 3))
    target_silhouettes = torch.zeros((imCT, maxB, maxB))
    if background:
        target_backgrounds = []
    RObj = np.zeros((imCT, 3, 3))
    TObj = np.zeros((imCT, 3))
    KObj = np.zeros((imCT, 4, 4))

    for a11 in range(imCT):
        
        if fewSamps:
          imId = int(lines[a11])
        else:
          imId = int(sampleIds[a11])
            
      
        rgbPath = objPath + "rgb/" + str(imId).zfill(6) + ".png"
        c1 = cv2.imread(objPath + "rgb/" + str(imId).zfill(6) + ".png")
        m1 = cv2.imread(objPath + maskStr + "/" + str(imId).zfill(6) + "_000000.png")

        if background:
            bgFull = np.zeros((c1.shape[0] + 200, c1.shape[1] + 200, 3))
            bgFull[100:-100, 100:-100] = c1.copy()
        c1[np.where(m1 == 0)] = 0
        x2, y2, w2, h2 = cv2.boundingRect(m1[:, :, 0])
        if w2 % 2 != 0:
            w2 = w2 - 1
        if h2 % 2 != 0:
            h2 = h2 - 1
        hw = int(w2 / 2)
        hh = int(h2 / 2)
        hd1 = int(np.max((w2, h2)) / 2)
        maxd = int(np.max((w2, h2)))

        squareRGB1 = np.zeros((maxd + 2 * offset, maxd + 2 * offset, 3), np.uint8)
        if background:
            squareBG1 = np.zeros((maxd + 2 * offset + 200, maxd + 2 * offset + 200, 3), np.uint8)

        squareMask1 = np.zeros((maxd + 2 * offset, maxd + 2 * offset), np.uint8)
        hs1 = int(squareRGB1.shape[0] / 2)
        squareRGB1[hs1 - hh:hs1 + hh, hs1 - hw:hs1 + hw] = c1[y2:y2 + h2, x2:x2 + w2]
        squareMask1[hs1 - hh:hs1 + hh, hs1 - hw:hs1 + hw] = m1[y2:y2 + h2, x2:x2 + w2, 0]
        if background:
            squareBG1[hs1 - hh:hs1 + hh + 200, hs1 - hw:hs1 + hw + 200] = bgFull[y2:y2 + h2 + 200, x2:x2 + w2 + 200]

        centerX = x2 + int(w2 / 2)
        centerY = y2 + int(h2 / 2)

        # print("im1")
        # if cropRGB.shape[0]!=cropRGB.shape[1]:
        #   import pdb;pdb.set_trace()
        # if justScaleImages:
        #         target_images[a11] = torch.from_numpy(cv2.resize(cropRGB, (ScaledSize, ScaledSize)).astype("float32") / 255)
        #         target_silhouettes[a11] = torch.from_numpy(
        #             cv2.resize(cropMask, (ScaledSize, ScaledSize), cv2.INTER_NEAREST).astype("float32") / 255)
        # else:
        #     try:
        target_images[a11] = torch.from_numpy(
            cv2.resize(squareRGB1, (maxB, maxB), cv2.INTER_CUBIC).astype("float32") / 255)
        target_silhouettes[a11] = torch.from_numpy(
            cv2.resize(squareMask1, (maxB, maxB), cv2.INTER_NEAREST).astype("float32") / 255)
        if background:
            bgScale = int(maxB * squareBG1.shape[0] / squareRGB1.shape[0])
            if bgScale % 2 == 0:
                bgScale = bgScale - 1
            target_backgrounds = target_backgrounds + [
                torch.from_numpy(cv2.resize(squareBG1, (bgScale, bgScale), cv2.INTER_CUBIC).astype("float32") / 255)]
        # except:
        #     print("a")
        # import pdb;pdb.set_trace()
        gtR, gtT = extractRT(objPath + "rgb/" + str(imId).zfill(6) + ".png", occid=0)
        RObj[a11] = gtR
        TObj[a11] = gtT
        camparams = np.asarray(camParams[str(imId)]["cam_K"]).reshape(3, 3)

        camparams[0, 2] = camparams[0, 2] + (-x2 + hs1 - hw)
        camparams[1, 2] = camparams[1, 2] + (-y2 + hs1 - hh)
        # (centerY - offset-hd1)
        camparams = camparams * maxB / squareRGB1.shape[0]
        camparams[2, 2] = 1

        if makeNDC:
            camparams = camparams * 2 / maxB
            camparams[2, 2] = 0
            # camparams[0, 2] -= 1
            # camparams[1, 2] -= 1

            camparams[0, 2] = -(camparams[0, 2] - 1)
            camparams[1, 2] = -(camparams[1, 2] - 1)

        KObj[a11][0:3, 0:3] = camparams
        KObj[a11][3, 2] = 1
        KObj[a11][2, 3] = 1
    if fewSamps:
        if background:
            return target_images, target_silhouettes, RObj, TObj, KObj, lines, target_backgrounds
        else:
            return target_images, target_silhouettes, RObj, TObj, KObj, lines

    else:
        if background:
            return target_images, target_silhouettes, RObj, TObj, KObj, target_backgrounds
        else:
            return target_images, target_silhouettes, RObj, TObj, KObj


def generate_lm_realsamplesWithoutLMTrainsSAM(datasetPath, objid="22", imD=128, crop=True, maskStr="mask", cropDim=70,
                                           justScaleImages=False, ScaledSize=128, maxB=200, offset=10,
                                           synth=True, makeNDC=True, dataset="tless", fewSamps=False, fewCT=20,
                                           fewids=[0], background=False, masking=True):
    lmDir = "train_primesense"

    if dataset == "lm":
        lmDir = "lm"
        # background=True
        if synth:
            lmDir = "lm_synth"
    objPath = datasetPath + "/" + lmDir + "/0000" + str(objid).zfill(2) + "/"
    camParams = json.load(open(objPath + "/scene_camera.json"))

    bboxDets = json.load(open(datasetPath + "/" + lmDir + "/" + str(objid).zfill(6) + "/scene_gt_info.json"))
    if dataset == "tless":
        sampleIds = torch.arange(len(bboxDets))
        imCT = len(camParams)
    else:
        with open(datasetPath + "lmTrains/" + str(objid) + ".txt") as f:
            lines = f.readlines()
        imCT = len(lines)
        if fewSamps:
            if len(fewids) == 1:
                fewids = np.random.random_integers(0, imCT - 1, fewCT)
                lines = np.asarray(lines)[fewids]
            else:
                lines = fewids
            imCT = len(fewids)

    target_images = torch.zeros((imCT, maxB, maxB, 3))
    target_silhouettes = torch.zeros((imCT, maxB, maxB))
    if background:
        target_backgrounds = []
    RObj = np.zeros((imCT, 3, 3))
    TObj = np.zeros((imCT, 3))
    KObj = np.zeros((imCT, 4, 4))

    for a11 in range(imCT):
        if dataset == "tless":
            imId = int(sampleIds[a11])
        else:
            imId = int(lines[a11])
        import cv2
        rgbPath = objPath + "rgb/" + str(imId).zfill(6) + ".png"
        c1 = cv2.imread(objPath + "rgb/" + str(imId).zfill(6) + ".png")
        m1 = cv2.imread(objPath.replace("lm//lm", "lm/customMasks") + maskStr + "/" + str(imId).zfill(6) + "_000000.png")

        kernel = np.ones((5, 5), np.uint8)
        erode1 = cv2.erode(m1, kernel, iterations=1)
        m1 = cv2.dilate(erode1, kernel, iterations=1)

        if background:
            bgFull = np.zeros((c1.shape[0] + 200, c1.shape[1] + 200, 3))
            bgFull[100:-100, 100:-100] = c1.copy()
        if masking:
            c1[np.where(m1 == 0)] = 0
        x2, y2, w2, h2 = cv2.boundingRect(m1[:, :, 0])

        if w2 % 2 != 0:
            w2 = w2 - 1
        if h2 % 2 != 0:
            h2 = h2 - 1
        hw = int(w2 / 2)
        hh = int(h2 / 2)
        hd1 = int(np.max((w2, h2)) / 2)
        maxd = int(np.max((w2, h2)))

        squareRGB1 = np.zeros((maxd + 2 * offset, maxd + 2 * offset, 3), np.uint8)
        if background:
            squareBG1 = np.zeros((maxd + 2 * offset + 200, maxd + 2 * offset + 200, 3), np.uint8)

        squareMask1 = np.zeros((maxd + 2 * offset, maxd + 2 * offset), np.uint8)
        hs1 = int(squareRGB1.shape[0] / 2)
        squareRGB1[hs1 - hh:hs1 + hh, hs1 - hw:hs1 + hw] = c1[y2:y2 + h2, x2:x2 + w2]
        squareMask1[hs1 - hh:hs1 + hh, hs1 - hw:hs1 + hw] = m1[y2:y2 + h2, x2:x2 + w2, 0]
        if background:
            squareBG1[hs1 - hh:hs1 + hh + 200, hs1 - hw:hs1 + hw + 200] = bgFull[y2:y2 + h2 + 200, x2:x2 + w2 + 200]

        centerX = x2 + int(w2 / 2)
        centerY = y2 + int(h2 / 2)

        # print("im1")
        # if cropRGB.shape[0]!=cropRGB.shape[1]:
        #   import pdb;pdb.set_trace()
        # if justScaleImages:
        #         target_images[a11] = torch.from_numpy(cv2.resize(cropRGB, (ScaledSize, ScaledSize)).astype("float32") / 255)
        #         target_silhouettes[a11] = torch.from_numpy(
        #             cv2.resize(cropMask, (ScaledSize, ScaledSize), cv2.INTER_NEAREST).astype("float32") / 255)
        # else:
        #     try:
        target_images[a11] = torch.from_numpy(
            cv2.resize(squareRGB1, (maxB, maxB), cv2.INTER_CUBIC).astype("float32") / 255)
        target_silhouettes[a11] = torch.from_numpy(
            cv2.resize(squareMask1, (maxB, maxB), cv2.INTER_NEAREST).astype("float32") / 255)
        if background:
            bgScale = int(maxB * squareBG1.shape[0] / squareRGB1.shape[0])
            if bgScale % 2 == 0:
                bgScale = bgScale - 1
            target_backgrounds = target_backgrounds + [
                torch.from_numpy(cv2.resize(squareBG1, (bgScale, bgScale), cv2.INTER_CUBIC).astype("float32") / 255)]
        # except:
        #     print("a")
        # import pdb;pdb.set_trace()
        gtR, gtT = extractRT(objPath + "rgb/" + str(imId).zfill(6) + ".png", occid=0)
        RObj[a11] = gtR
        TObj[a11] = gtT
        camparams = np.asarray(camParams[str(imId)]["cam_K"]).reshape(3, 3)

        camparams[0, 2] = camparams[0, 2] + (-x2 + hs1 - hw)
        camparams[1, 2] = camparams[1, 2] + (-y2 + hs1 - hh)
        # (centerY - offset-hd1)
        camparams = camparams * maxB / squareRGB1.shape[0]
        camparams[2, 2] = 1

        if makeNDC:
            camparams = camparams * 2 / maxB
            camparams[2, 2] = 0
            # camparams[0, 2] -= 1
            # camparams[1, 2] -= 1

            camparams[0, 2] = -(camparams[0, 2] - 1)
            camparams[1, 2] = -(camparams[1, 2] - 1)

        KObj[a11][0:3, 0:3] = camparams
        KObj[a11][3, 2] = 1
        KObj[a11][2, 3] = 1
    if fewSamps:
        if background:
            return target_images, target_silhouettes, RObj, TObj, KObj, lines, target_backgrounds
        else:
            return target_images, target_silhouettes, RObj, TObj, KObj, lines

    else:
        if background:
            return target_images, target_silhouettes, RObj, TObj, KObj, target_backgrounds
        else:
            return target_images, target_silhouettes, RObj, TObj, KObj


def generate_lm_realsamplesWithoutLMTrainsOld(datasetPath, objid="22", imD=128, crop=True, maskStr="mask",cropDim=70, justScaleImages=False, ScaledSize=128, maxB=200, offset=10,
                            synth=True, makeNDC=True, dataset="tless", fewSamps=False, fewCT=20, fewids=[0]):
    lmDir="train_primesense"
    
    if dataset=="lm":
      lmDir="lm"          
      if synth:
        lmDir="lm_synth"
    objPath = datasetPath + "/"+lmDir+"/0000" + str(objid).zfill(2) + "/"
    camParams = json.load(open(objPath + "/scene_camera.json"))


    bboxDets=json.load(open(datasetPath+"/"+lmDir+"/"+str(objid).zfill(6)+"/scene_gt_info.json"))
    if dataset=="tless":
      sampleIds=torch.arange(len(bboxDets))
      imCT= len(camParams)
    else:
        with open(datasetPath+"lmTrains/"+str(objid)+".txt") as f:
           lines = f.readlines()   
        imCT= len(lines)    
        if fewSamps:
            if len(fewids)==1:
              fewids=np.random.random_integers(0,imCT-1, fewCT)
              lines=np.asarray(lines)[fewids]
            else:
              lines=fewids
            imCT=len(fewids)

    

    
    target_images = torch.zeros((imCT, maxB, maxB, 3))
    target_silhouettes = torch.zeros((imCT, maxB, maxB))
    RObj = np.zeros((imCT, 3, 3))
    TObj = np.zeros((imCT, 3))
    KObj = np.zeros((imCT, 4, 4))





    for a11 in range(imCT):
        if dataset=="tless":
          imId = int(sampleIds[a11])
        else:
          imId = int(lines[a11])
        rgbPath = objPath + "rgb/" + str(imId).zfill(6) + ".png"
        c1 = cv2.imread(objPath + "rgb/" + str(imId).zfill(6) + ".png")
        m1 = cv2.imread(objPath + maskStr+"/" + str(imId).zfill(6) + "_000000.png")
        c1[np.where(m1==0)]=0
        x2, y2, w2, h2 = cv2.boundingRect(m1[:,:,0])
        if w2 % 2 != 0:
            w2 = w2 - 1
        if h2 % 2 != 0:
            h2 = h2 - 1
        hw=int(w2/2)
        hh=int(h2/2)
        hd1 = int(np.max((w2,h2))/2)
        maxd = int(np.max((w2, h2)))



        squareRGB1=np.zeros((maxd+2*offset,maxd+ 2*offset,3), np.uint8)
        squareMask1=np.zeros((maxd+2*offset,maxd + 2*offset), np.uint8)
        hs1=int(squareRGB1.shape[0]/2)
        squareRGB1[hs1-hh:hs1+hh, hs1-hw:hs1+hw] = c1[y2:y2+h2, x2:x2+w2]
        squareMask1[hs1-hh:hs1+hh, hs1-hw:hs1+hw] = m1[y2:y2+h2, x2:x2+w2, 0]

        centerX=x2+int(w2/2)
        centerY=y2+int(h2/2)


        # print("im1")
        # if cropRGB.shape[0]!=cropRGB.shape[1]:
        #   import pdb;pdb.set_trace()
        if justScaleImages:
                target_images[a11] = torch.from_numpy(cv2.resize(cropRGB, (ScaledSize, ScaledSize)).astype("float32") / 255)
                target_silhouettes[a11] = torch.from_numpy(
                    cv2.resize(cropMask, (ScaledSize, ScaledSize), cv2.INTER_NEAREST).astype("float32") / 255)
        else:
            try:
              target_images[a11] = torch.from_numpy(cv2.resize(squareRGB1, (maxB, maxB), cv2.INTER_CUBIC).astype("float32") / 255)
              target_silhouettes[a11] = torch.from_numpy(cv2.resize(squareMask1, (maxB, maxB), cv2.INTER_NEAREST).astype("float32") / 255)
            except:
                print("a")
              #import pdb;pdb.set_trace()
        gtR, gtT = extractRT(objPath + "rgb/" + str(imId).zfill(6) + ".png", occid=0)
        RObj[a11] = gtR
        TObj[a11] = gtT
        camparams = np.asarray(camParams[str(imId)]["cam_K"]).reshape(3, 3)

        camparams[0,2]=camparams[0,2]+(-x2+hs1-hw)
        camparams[1,2]=camparams[1,2]+(-y2+hs1-hh)
                       # (centerY - offset-hd1)
        camparams = camparams*maxB/squareRGB1.shape[0]
        camparams[2,2] = 1


        if makeNDC:
            camparams = camparams * 2 / maxB
            camparams[2, 2] = 0
            # camparams[0, 2] -= 1
            # camparams[1, 2] -= 1

            camparams[0, 2] = -(camparams[0, 2]-1)
            camparams[1, 2] = -(camparams[1, 2]-1)

        KObj[a11][0:3, 0:3] = camparams
        KObj[a11][3, 2] = 1
        KObj[a11][2, 3] = 1
    if fewSamps:
      return target_images, target_silhouettes, RObj, TObj, KObj, lines
    else:
      return target_images, target_silhouettes, RObj, TObj, KObj

def generate_lm_realsamplesWithoutLMTrainsOld(datasetPath, objid="22", imD=128, crop=True, maskStr="mask",cropDim=70, justScaleImages=False, ScaledSize=128, maxB=200, offset=10,
                            synth=True, makeNDC=True, dataset="tless"):
    lmDir="train_primesense"
    #4.27.2023
    if dataset=="lm":
      lmDir="lm"          
      if synth:
        lmDir="lm_synth"
    objPath = datasetPath + "/"+lmDir+"/0000" + str(objid).zfill(2) + "/"
    camParams = json.load(open(objPath + "/scene_camera.json"))


    bboxDets=json.load(open(datasetPath+"/"+lmDir+"/"+str(objid).zfill(6)+"/scene_gt_info.json"))
    if dataset=="tless":
      sampleIds=torch.arange(len(bboxDets))
      imCT= len(camParams)
    else:
        with open(datasetPath+"lmTrains/"+str(objid)+".txt") as f:
           lines = f.readlines()
        imCT= len(lines)    
        
    

    

    
    target_images = torch.zeros((imCT, maxB, maxB, 3))
    target_silhouettes = torch.zeros((imCT, maxB, maxB))
    RObj = np.zeros((imCT, 3, 3))
    TObj = np.zeros((imCT, 3))
    KObj = np.zeros((imCT, 4, 4))





    for a11 in range(imCT):
        if dataset=="tless":
          imId = int(sampleIds[a11])
        else:
          imId = int(lines[a11])
        rgbPath = objPath + "rgb/" + str(imId).zfill(6) + ".png"
        c1 = cv2.imread(objPath + "rgb/" + str(imId).zfill(6) + ".png")
        m1 = cv2.imread(objPath + maskStr+"/" + str(imId).zfill(6) + "_000000.png")
        c1[np.where(m1==0)]=0
        x2, y2, w2, h2 = cv2.boundingRect(m1[:,:,0])
        hd1=int(np.max((w2,h2))/2)

        centerX=x2+int(w2/2)
        centerY=y2+int(h2/2)
        offset1=offset
        
        cropRGB  = c1[centerY-offset-hd1:centerY+hd1+offset, centerX-offset-hd1:centerX+offset+hd1]
        cropMask = m1[centerY-offset-hd1:centerY+hd1+offset, centerX-offset-hd1:centerX+offset+hd1,0]
        if cropRGB.shape[0]==0 or cropRGB.shape[1] ==0:
          offset1=0
          cropRGB  = c1[centerY-hd1:centerY+hd1, centerX-hd1:centerX+hd1]
          cropMask = m1[centerY-hd1:centerY+hd1, centerX-hd1:centerX+hd1,0]
          
        # print("im1")
        # if cropRGB.shape[0]!=cropRGB.shape[1]:
        #   import pdb;pdb.set_trace()
        if justScaleImages:
                target_images[a11] = torch.from_numpy(cv2.resize(cropRGB, (ScaledSize, ScaledSize)).astype("float32") / 255)
                target_silhouettes[a11] = torch.from_numpy(
                    cv2.resize(cropMask, (ScaledSize, ScaledSize), cv2.INTER_NEAREST).astype("float32") / 255)
        else:
            #try:
              target_images[a11] = torch.from_numpy(cv2.resize(cropRGB, (maxB, maxB), cv2.INTER_CUBIC).astype("float32") / 255)
              target_silhouettes[a11] = torch.from_numpy(cv2.resize(cropMask, (maxB, maxB), cv2.INTER_NEAREST).astype("float32") / 255)
            #except:
              #import pdb;pdb.set_trace()
        gtR, gtT = extractRT(objPath + "rgb/" + str(imId).zfill(6) + ".png", occid=0)
        RObj[a11] = gtR
        TObj[a11] = gtT
        camparams = np.asarray(camParams[str(imId)]["cam_K"]).reshape(3, 3)

        camparams[0,2]=camparams[0,2]-(centerX - offset1-hd1)
        camparams[1,2]=camparams[1,2]-(centerY - offset1-hd1)
        camparams = camparams*maxB/cropRGB.shape[0]
        camparams[2,2] = 1


        if makeNDC:
            camparams = camparams * 2 / maxB
            camparams[2, 2] = 0
            # camparams[0, 2] -= 1
            # camparams[1, 2] -= 1

            camparams[0, 2] = -(camparams[0, 2]-1)
            camparams[1, 2] = -(camparams[1, 2]-1)

        KObj[a11][0:3, 0:3] = camparams
        KObj[a11][3, 2] = 1
        KObj[a11][2, 3] = 1

    return target_images, target_silhouettes, RObj, TObj, KObj


def generate_lm_realsamplesWorkingForSynth(datasetPath, objid="22", imD=128, crop=True, maskStr="mask", cropDim=70,
                            justScaleImages=False, ScaledSize=128, maxB=200, offset=10,
                            synth=True, makeNDC=True):
    lmDir = "lm"
    if synth:
        lmDir = "lm_synth"
    bboxDets = json.load(open(datasetPath + "/" + lmDir + "/" + str(objid).zfill(6) + "/scene_gt_info.json"))

    with open(datasetPath + "lmTrains/" + str(objid) + ".txt") as f:
        lines = f.readlines()

    # imCT = len(bboxDets)
    imCT = len(lines)
    target_images = torch.zeros((imCT, maxB, maxB, 3))
    target_silhouettes = torch.zeros((imCT, maxB, maxB))
    RObj = np.zeros((imCT, 3, 3))
    TObj = np.zeros((imCT, 3))
    KObj = np.zeros((imCT, 4, 4))

    objPath = datasetPath + "/" + lmDir + "/0000" + str(objid).zfill(2) + "/"
    camParams = json.load(open(objPath + "/scene_camera.json"))

    for a11 in range(imCT):
        imId = int(lines[a11])
        rgbPath = objPath + "rgb/" + str(imId).zfill(6) + ".png"
        c1 = cv2.imread(objPath + "rgb/" + str(imId).zfill(6) + ".png")
        m1 = cv2.imread(objPath + maskStr + "/" + str(imId).zfill(6) + "_000000.png")
        c1[np.where(m1 == 0)] = 0
        x2, y2, w2, h2 = cv2.boundingRect(m1[:, :, 0])
        hd1 = int(np.max((w2, h2)) / 2)

        centerX = x2 + int(w2 / 2)
        centerY = y2 + int(h2 / 2)

        cropRGB = c1[centerY - offset - hd1:centerY + hd1 + offset, centerX - offset - hd1:centerX + offset + hd1]
        cropMask = m1[centerY - offset - hd1:centerY + hd1 + offset, centerX - offset - hd1:centerX + offset + hd1, 0]

        # print("im1")
        # if cropRGB.shape[0]!=cropRGB.shape[1]:
        #   import pdb;pdb.set_trace()
        if justScaleImages:
            target_images[a11] = torch.from_numpy(cv2.resize(cropRGB, (ScaledSize, ScaledSize)).astype("float32") / 255)
            target_silhouettes[a11] = torch.from_numpy(
                cv2.resize(cropMask, (ScaledSize, ScaledSize), cv2.INTER_NEAREST).astype("float32") / 255)
        else:
            target_images[a11] = torch.from_numpy(
                cv2.resize(cropRGB, (maxB, maxB), cv2.INTER_CUBIC).astype("float32") / 255)
            target_silhouettes[a11] = torch.from_numpy(
                cv2.resize(cropMask, (maxB, maxB), cv2.INTER_NEAREST).astype("float32") / 255)

        gtR, gtT = extractRT(objPath + "rgb/" + str(imId).zfill(6) + ".png", occid=0)
        RObj[a11] = gtR
        TObj[a11] = gtT
        camparams = np.asarray(camParams[str(imId)]["cam_K"]).reshape(3, 3)

        camparams[0, 2] = camparams[0, 2] - (centerX - offset - hd1)
        camparams[1, 2] = camparams[1, 2] - (centerY - offset - hd1)
        camparams = camparams * maxB / cropRGB.shape[0]
        camparams[2, 2] = 1

        if makeNDC:
            camparams = camparams * 2 / maxB
            camparams[2, 2] = 0
            # camparams[0, 2] -= 1
            # camparams[1, 2] -= 1

            camparams[0, 2] = -(camparams[0, 2] - 1)
            camparams[1, 2] = -(camparams[1, 2] - 1)

        KObj[a11][0:3, 0:3] = camparams
        KObj[a11][3, 2] = 1
        KObj[a11][2, 3] = 1

    return target_images, target_silhouettes, RObj, TObj, KObj


def renderWithCam(rgb,LMCano, cx, cy, fx, fy, gtR, gtT):
    tMesh = LMCano.dot(gtR.T)+gtT
    xid = ((tMesh[:, 0] * fx / tMesh[:, 2]) + cx).astype(int)
    yid = ((tMesh[:, 1] * fy / tMesh[:, 2]) + cy).astype(int)
    rgb[yid, xid, 2] = 255
    rgb[yid, xid, 0] = 0
    rgb[yid, xid, 1] = 0

    cv2.imwrite("a1.jpg", rgb)
def generate_cow_rendersWithRT(
    num_views: int = 40, data_dir: str = DATA_DIR, azimuth_range: float = 180, objid = "obj_00005.obj"):
    """
    This function generates `num_views` renders of a cow mesh.
    The renders are generated from viewpoints sampled at uniformly distributed
    azimuth intervals. The elevation is kept constant so that the camera's
    vertical position coincides with the equator.

    For a more detailed explanation of this code, please refer to the
    docs/tutorials/fit_textured_mesh.ipynb notebook.

    Args:
        num_views: The number of generated renders.
        data_dir: The folder that contains the cow mesh files. If the cow mesh
            files do not exist in the folder, this function will automatically
            download them.
        azimuth_range: number of degrees on each side of the start position to
            take samples

    Returns:
        cameras: A batch of `num_views` `FoVPerspectiveCameras` from which the
            images are rendered.
        images: A tensor of shape `(num_views, height, width, 3)` containing
            the rendered images.
        silhouettes: A tensor of shape `(num_views, height, width)` containing
            the rendered silhouettes.
    """

    # set the paths

    # download the cow mesh if not done before
    cow_mesh_files = [
        os.path.join(data_dir, fl) for fl in ("cow.obj", "cow.mtl", "cow_texture.png")
    ]
    if any(not os.path.isfile(f) for f in cow_mesh_files):
        os.makedirs(data_dir, exist_ok=True)
        os.system(
            f"wget -P {data_dir} "
            + "https://dl.fbaipublicfiles.com/pytorch3d/data/cow_mesh/cow.obj"
        )
        os.system(
            f"wget -P {data_dir} "
            + "https://dl.fbaipublicfiles.com/pytorch3d/data/cow_mesh/cow.mtl"
        )
        os.system(
            f"wget -P {data_dir} "
            + "https://dl.fbaipublicfiles.com/pytorch3d/data/cow_mesh/cow_texture.png"
        )

    # Setup
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # Load obj file 
    obj_filename = os.path.join(data_dir, "cow.obj")
    if objid!=-1:
      mesh = load_objs_as_meshes([objid], device=device)
    else:
      mesh = load_objs_as_meshes([obj_filename], device=device)
    
    # We scale normalize and center the target mesh to fit in a sphere of radius 1
    # centered at (0,0,0). (scale, center) will be used to bring the predicted mesh
    # to its original center and scale.  Note that normalizing the target mesh,
    # speeds up the optimization but is not necessary!
    verts = mesh.verts_packed()
    N = verts.shape[0]
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])
    mesh.offset_verts_(-(center.expand(N, 3)))
    mesh.scale_verts_((1.0 / float(scale)))

    # Get a batch of viewing angles.
    nv=int(np.sqrt(num_views))+2
    elev = torch.linspace(-azimuth_range, azimuth_range, nv)  # keep constant
    azim = torch.linspace(-azimuth_range, azimuth_range, nv) + 180.0

    elev=elev.repeat(nv)[0:num_views]
    azim=azim.repeat_interleave(nv)[0:num_views]
    # Place a point light in front of the object. As mentioned above, the front of
    # the cow is facing the -z direction.
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    # Initialize an OpenGL perspective camera that represents a batch of different
    # viewing angles. All the cameras helper methods support mixed type inputs and
    # broadcasting. So we can view the camera from the a distance of dist=2.7, and
    # then specify elevation and azimuth angles for each viewpoint as tensors.


    R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
    #import pdb;pdb.set_trace()
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    # Define the settings for rasterization and shading. Here we set the output
    # image to be of size 128X128. As we are rendering images for visualization
    # purposes only we will set faces_per_pixel=1 and blur_radius=0.0. Refer to
    # rasterize_meshes.py for explanations of these parameters.  We also leave
    # bin_size and max_faces_per_bin to their default values of None, which sets
    # their values using heuristics and ensures that the faster coarse-to-fine
    # rasterization method is used.  Refer to docs/notes/renderer.md for an
    # explanation of the difference between naive and coarse-to-fine rasterization.
    raster_settings = RasterizationSettings(
        image_size=128, blur_radius=0.0, faces_per_pixel=1
    )

    # Create a Phong renderer by composing a rasterizer and a shader. The textured
    # Phong shader will interpolate the texture uv coordinates for each vertex,
    # sample from a texture image and apply the Phong lighting model
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0.0, 0.0, 0.0))
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(
            device=device, cameras=cameras, lights=lights, blend_params=blend_params
        ),
    )

    # Create a batch of meshes by repeating the cow mesh and associated textures.
    # Meshes has a useful `extend` method which allows us do this very easily.
    # This also extends the textures.
    meshes = mesh.extend(num_views)

    # Render the cow mesh from each viewing angle
    target_images = renderer(meshes, cameras=cameras, lights=lights)

    # Rasterization settings for silhouette rendering
    sigma = 1e-4
    raster_settings_silhouette = RasterizationSettings(
        image_size=128, blur_radius=np.log(1.0 / 1e-4 - 1.0) * sigma, faces_per_pixel=50
    )

    # Silhouette renderer
    renderer_silhouette = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, raster_settings=raster_settings_silhouette
        ),
        shader=SoftSilhouetteShader(),
    )

    # Render silhouette images.  The 3rd channel of the rendering output is
    # the alpha/silhouette channel
    silhouette_images = renderer_silhouette(meshes, cameras=cameras, lights=lights)

    # binary silhouettes
    silhouette_binary = (silhouette_images[..., 3] > 1e-4).float()

    return cameras, target_images[..., :3], silhouette_binary, R, T

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt


def image_grid(
    images,
    rows=None,
    cols=None,
    fill: bool = True,
    show_axes: bool = False,
    rgb: bool = True,
):
    """
    A util function for plotting a grid of images.

    Args:
        images: (N, H, W, 4) array of RGBA images
        rows: number of rows in the grid
        cols: number of columns in the grid
        fill: boolean indicating if the space between images should be filled
        show_axes: boolean indicating if the axes of the plots should be visible
        rgb: boolean, If True, only RGB channels are plotted.
            If False, only the alpha channel is plotted.

    Returns:
        None
    """
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")

    if rows is None:
        rows = len(images)
        cols = 1

    gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
    fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=(15, 9))
    bleed = 0
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for ax, im in zip(axarr.ravel(), images):
        if rgb:
            # only render RGB channels
            ax.imshow(im[..., :3])
        else:
            # only render Alpha channel
            ax.imshow(im[..., 3])
        if not show_axes:
            ax.set_axis_off()
