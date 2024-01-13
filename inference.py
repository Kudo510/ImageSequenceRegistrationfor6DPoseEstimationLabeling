import os
import json
import glob
import torch
import numpy as np
from sklearn.neighbors import KDTree
from nerf import NeuralRadianceFieldFeat
import cv2, trimesh
import re, argparse
from nutil import get_emb_vis
arg_parser = argparse.ArgumentParser(description="Train a Linemod")
arg_parser.add_argument("--objid", dest="objid", default="-1", )
arg_parser.add_argument("--id", dest="id", default=-1)
arg_parser.add_argument("--dataset",dest="dataset",default="tless",)
arg_parser.add_argument("--UH", dest="UH", default=0)
args = arg_parser.parse_args()

device = torch.device("cuda:0")
objid=str(args.objid)
print("objid", objid)

# expID = "ruapc/"+objid+"poseEst"
expID = str(args.UH) +'_'+ args.dataset+'_obj_' + str(objid)+ "/"+objid+"poseEst"
Siren = True
camMatScaling=True
useMask=True
useSurfEval = False

maskBeforeEncoder=True

imD = 224
datasetPath = "bop/" + args.dataset

meshdetails = json.load(open(datasetPath + "/models" + "/models_info.json"))
diam = meshdetails[objid]['diameter']

torch.manual_seed(1)


from dep.unet import ResNetUNetNew as ResNetUNet


encoder_rgb = ResNetUNet(n_class=(13), n_decoders=1, )

data_rgb = torch.load(expID + "/encoderRGBlatest.pth")

encoder_rgb.load_state_dict(data_rgb["model_state_dict"])

encoder_rgb.to(device).eval()


##? what this function for?
def renderWithCam1(rgb, LMCano, K, R, T, unitScale=True, colorVec=np.array(())):
        fx = K[0, 0];
        fy = K[1, 1];
        cx = K[0, 2];
        cy = K[1, 2];
        tMesh = LMCano.dot(R.T) + T
        xid = ((tMesh[:, 0] * fx / tMesh[:, 2]) + cx)
        yid = ((tMesh[:, 1] * fy / tMesh[:, 2]) + cy)

        xid = xid.astype("int")
        yid = yid.astype("int")
        if colorVec.shape[0]:
            rgb[yid, xid, :] = colorVec
        else:
            rgb[yid, xid, 2] = 255
            rgb[yid, xid, 0] = 0
            rgb[yid, xid, 1] = 0
        import cv2; cv2.imwrite("a1.jpg", rgb)
        return xid, yid



# add_score = 0
# add_scoreT = 0
# best_score = 0
# best_scoreT = 0
add = 0
addT = 0
bestadd = 0
bestaddT = 0
fin = 0
mesh1 = trimesh.load_mesh(datasetPath + "/models/obj_0000" + str(objid).zfill(2) + ".ply")
meshdetails = json.load(open(datasetPath + "/models" + "/models_info.json"))
diameter = meshdetails[str(objid)]['diameter']

surfacePointsScaled = np.load(expID + "/vert1_scaled.npy")
surfaceFeatures = np.load(expID + "/feat1_scaled.npy")
sfeats = torch.from_numpy(surfaceFeatures).cuda()
n2Scaled = np.load(expID + "/normals_scaled.npy")
tree = KDTree(surfaceFeatures, leaf_size=2)

workCT = 0
totct = 0
rotWorkCT = 0
refCT=0

if useSurfEval: ## not our case
  data_nerf = torch.load(expID + "/nerflatest.pth")
  neural_radiance_field = NeuralRadianceFieldFeat(siren=Siren)
  neural_radiance_field.mode = "feature"
  neural_radiance_field.load_state_dict(data_nerf["model_state_dict"])
  
  neural_radiance_field.to(device).eval()

  from pose_refine import refine_pose
  from renderer import ObjCoordRenderer
  from pathlib import Path
  from obj import load_obj

  objs = load_obj(Path(datasetPath + "/models/"), int(objid))
  renderer = ObjCoordRenderer([objs], w=224, h=224)

## to cal new point =R*old_point +t
def ADD(verts, gtR1, gtT1, R1, T1):  
    return np.linalg.norm(verts.dot(gtR1.T) + gtT1- verts.dot(R1.T) - T1, axis=-1).mean()
def ADDS(verts, gtR1, gtT1, R1, T1):
    treeTr = KDTree(surfacePointsScaled.dot(R1.T) + T1, leaf_size=2)
    return treeTr.query(verts.dot(gtR1.T) + gtT1, k=1)[0].mean()


def pnp(h3d, h2d, cam, itr=100, reperr=2, flag=cv2.SOLVEPNP_P3P, gtR=None, gtT=None, spts=None,
        ret=None):
    status, rvec1, tvec1, in1 = cv2.solvePnPRansac(h3d, h2d, cam, distCoeffs=None,
                                                   iterationsCount=itr, reprojectionError=reperr,
                                                   flags=flag)
    rotmat1 = cv2.Rodrigues(rvec1)[0]
    # import pdb;pdb.set_trace()
    if status:
        return rotmat1, tvec1[:, 0], in1[:, 0]
    else:
        print("pose could not be estimated with these correspondences")
        return 1, 1, 1
def normalize(img: np.ndarray):  # (h, w, 3) -> (3, h, w)
            imagenet_stats = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            mu, std = imagenet_stats
            if img.dtype == np.uint8:
                img = img / 255
            img = (img - mu) / std
            return img
def getCors(queries, feats, leaves=1):
    cMat = torch.log_softmax(queries @ feats.T, dim=-1) 
    # idx=torch.argmax(cMat, dim=-1)
    vals, idx = torch.topk(cMat, k=leaves, dim=-1)
    if leaves == 1:
        return idx[..., 0].cpu(), vals
    else:
        return idx.cpu(), vals
        
refinement=False
subSurfaccePointsScaled=surfacePointsScaled[np.random.choice(surfacePointsScaled.shape[0], 5000)].copy()
modelVerts=np.asarray(mesh1.vertices)

file_list= sorted(glob.glob(datasetPath+"/train/"+str(objid.zfill(6))+"/depth/*"))

if args.id!=-1:
  file_list=[file_list[int(args.id)]]

#print("file_list", file_list)

correct_predicted_ids = []
for path in (file_list[:1280] if args.id!=-1 else file_list): ## inference for second sequence 
# for path in (file_list): ## inference for single image only
    print("workCT", workCT)
    print("rotworkCT", rotWorkCT)
    print("totCT", totct)
    totct += 1
    
    sceneGT = json.load(open(os.path.split(os.path.split(path)[0])[0] + "/scene_gt.json"))
    sceneGTInfo = json.load(open(os.path.split(os.path.split(path)[0])[0] + "/scene_gt_info.json"))
    imID = int(re.findall(r'\d+', os.path.split(path)[1])[0])
    sceneDet = sceneGT[str(imID)]
    sceneDetInfo = sceneGTInfo[str(imID)]
    sceneObjId = -1
    sceneid1 = int(os.path.split(os.path.split(os.path.split(path)[0])[0])[1])
    prevVisib = 0
    gtRList = []
    gtTList = []
    
    for a in range(len(sceneDet)):
        if sceneDet[a]["obj_id"] == int(objid):
            sceneObjId = a
            gtR = np.asarray(sceneDet[a]["cam_R_m2c"]).reshape(3, 3)
            gtT = np.asarray(sceneDet[a]["cam_t_m2c"])
            gtRList = gtRList + [gtR]
            gtTList = gtTList + [gtT]

        
    head, tail = os.path.split(path)
    camppath = head.replace("depth", "scene_camera.json")
    cam = json.load(open(camppath))
    camparams = np.array(cam[str(int(tail[0:6]))]["cam_K"]).reshape(3, 3)
    fx = camparams[0, 0];fy = camparams[1, 1];cx = camparams[0, 2];cy = camparams[1, 2];

    if True:
        rgb = cv2.imread(path.replace("depth", "rgb"))
        maskpath = (path.replace("depth", "mask_visib").replace(".png", "_000000.png"))
        
        mask = cv2.imread(maskpath)

        x, y, w, h = cv2.boundingRect(mask[:, :, 0])
        if w % 2 != 0:
            w = w - 1
        if h % 2 != 0:
            h = h - 1

        maskFullrgb = rgb.copy()
        if maskBeforeEncoder:
           maskFullrgb[np.where(mask[:, :, 0] == 0)[0], np.where(mask[:, :, 0] == 0)[1], :] = 0

        centerX = x + (w / 2)
        centerY = y + (h / 2)

        size = 224 / max(w, h) / 1.2
        r = 224
        R = np.array(((1, -0),(0, 1),))
        M = np.concatenate((R, [[-centerX], [-centerY]]), axis=1) * size
        M[:, 2] += r / 2

        Ms = np.concatenate((M, [[0, 0, 1]]))
        camMat  = Ms @ camparams

        cropRGB=cv2.warpAffine(rgb, M, (r, r))
        cropMask = cv2.warpAffine(mask, M, (r, r))
        if useMask:
            cropRGB[np.where(cropMask[:, :, 0] == 0)[0], np.where(cropMask[:, :, 0] == 0)[1], :] = 0

        inputMask = torch.from_numpy(cropMask[:,:,0])


        inputIM = torch.movedim(torch.from_numpy(normalize(cropRGB).astype("float32")).cuda().unsqueeze(0), 3, 1)


        ## see here featuers after cnn has 13 dimension 12 for imfeaet, the last one for the mask
        with torch.no_grad():
            imfeatsfull = torch.movedim(encoder_rgb(inputIM), 1, 3)
        imfeatsmask = imfeatsfull[..., -1]  ## mask for the image feature
        imfeats = imfeatsfull[..., 0:12]  ##  real image feature as the first 12 dimensions
        # print('imfeats.shape', imfeats.shape) # 1,224,224,12
        # featimage = imfeats.squeeze()# 224,224,12
        # print("featimage.shape", featimage.shape)
        
        ## visualize feat image
        # cv2.imwrite("a1.jpg", get_emb_vis(imfeats[0]).cpu().numpy()*255)





        if not useSurfEval: ## our case
            newmask = cropMask[:,:,0]

            down_sampling = True
            if down_sampling:
                down_sample = 3
                imfeats = imfeats[:, ::down_sample, ::down_sample]

                inputMask = inputMask[::down_sample, ::down_sample]
                if camMatScaling:
                    camMat[:2, 2] += 0.5  # change origin to corner
                    camMat[:2] /= down_sample
                    camMat[:2, 2] -= 0.5  # change origin back

            maskIds = torch.where(inputMask)
            maskedfeats = imfeats[0][maskIds]


            X1 = maskIds[0].cpu().numpy()
            Y1 = maskIds[1].cpu().numpy()


            idx1, in1 = getCors(maskedfeats, sfeats, leaves=1,)
            ep3d = surfacePointsScaled[idx1.cpu()]

            ep2d = np.zeros((ep3d.shape))
            n3d = n2Scaled[idx1]
            ep2d[:, 0] = maskIds[1]
            ep2d[:, 1] = maskIds[0]
            ep2d = ep2d[:, 0:2]

            if len(in1) > 500:
                perc=int(0.8*len(in1))
                threshval = torch.sort(in1[:, 0])[0][-perc + 1]
            else:
                threshval = torch.sort(in1[:, 0])[0][-len(in1) + 1]

            nidx=torch.where(in1[:,0]>threshval)[0].cpu().numpy()
            ep3d = ep3d[nidx,:]
            ep2d = ep2d[nidx, :]


            R2, T2, in2 = pnp(ep3d, ep2d, camMat, itr=500, reperr=2, flag=cv2.SOLVEPNP_P3P, gtR=gtR,
                              gtT=gtT, spts=surfacePointsScaled, ret=True)
           
            # print("R2, T2", R2,T2)     
            ### R2, T2 are our poses
            print("ground truth", gtR, gtT)
            print("predicted poses", R2,T2)
            if args.dataset=="tless":
                final_error = ADDS(modelVerts, gtR, gtT, R2, T2)
                final_errorR = ADDS(modelVerts, gtR, np.zeros(3), R2, np.zeros(3))
            else:
                final_error = ADD(modelVerts, gtR, gtT, R2, T2)
                final_errorR = ADD(modelVerts, gtR, np.zeros(3), R2, np.zeros(3))
            #finADD = ADD(modelVerts, gtR, gtT, R2, T2)
            # finADDR = ADD(modelVerts, gtR, np.zeros(3), R2, np.zeros(3))
            #print("addError", finADD)
            print("rotationOnlyADDError", final_errorR )

            ## finADD is Error for both Rotation and Translation
            if final_error < 0.1 * diameter:
                print("working")
                workCT += 1

                correct_predicted_ids.append(path.split("/")[-1])
            #### finADDR is Error for  Rotation
            if final_errorR < 0.1 * diameter:
                print("rotation working", final_errorR)
                rotWorkCT += 1

            print("sampleID", imID)

        else:
            from poseEstSurf import estimate_pose
            down_sample_scale = 3
            R, t, pose_scores, mask_scores, coord_scores, dist_2d, size_mask, normals_mask = estimate_pose(
                mask_lgts=imfeatsmask[0].cuda(), query_img=imfeats[0].cuda(),
                obj_pts=torch.from_numpy(surfacePointsScaled).cuda(),
                obj_normals=(n2Scaled), obj_keys=torch.from_numpy(surfaceFeatures.astype("float32")).cuda(),
                obj_diameter=diameter, K=camMat.copy(), returnPoints=False)
            # , max_poses=10000, max_pose_evaluations=1000, down_sample_scale=down_sample_scale,
            # alpha=1.5,
            # dist_2d_min=0.1, pnp_method=cv2.SOLVEPNP_AP3P, pose_batch_size=500, max_pool=False,
            # avg_queries=False, do_prune=True, visualize=False, poses=None, debug=False)
            print("samplID", imID)
            if len(mask_scores) > 0:
                bestId = torch.argsort(pose_scores)[-1]
                R2 = R[bestId].cpu().numpy()
                T2 = t[bestId].cpu().numpy()
                finT2 = T2

                finADD= ADD(modelVerts, gtR, gtT, R2, finT2)

                R_est_r, t_est_r, score_r = refine_pose(
                    R=R2, t=T2, query_img=imfeats[0], K_crop=camMat,
                    renderer=renderer, obj_idx=0, obj_=objs, neural_radiance_field=neural_radiance_field,
                    keys_verts=sfeats,
                )

                refADD = ADD(modelVerts, gtR, gtT, R2, t_est_r)
                finADDR = ADD(modelVerts, gtR, np.zeros((3)), R2, np.zeros((3)))

                if refADD < 0.1 * diameter:
                    print("ref", np.min((refADD, refADD)))
                    refCT += 1

                if finADD < 0.1 * diameter:
                    print("working", np.min((finADD, finADD)))
                    workCT += 1

                print("TADD", finADD)

                if finADDR < 0.1 * diameter:
                    rotWorkCT += 1
                print("RADD",finADDR)


file_name = str(args.UH) +'_'+ args.dataset+'_obj_' + str(objid)+ "/" + str(objid) + "correctly_predicted_list.txt"

# uncomment to save file
with open(file_name, 'w') as file:
    for item in correct_predicted_ids:
        file.write(str(item) + '\n')



