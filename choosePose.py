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
def normalize(img: np.ndarray):  # (h, w, 3) -> (3, h, w)
            imagenet_stats = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            mu, std = imagenet_stats
            if img.dtype == np.uint8:
                img = img / 255
            img = (img - mu) / std
            return img
def ADD(verts, gtR1, gtT1, R1, T1):  
    return np.linalg.norm(verts.dot(gtR1.T) + gtT1- verts.dot(R1.T) - T1, axis=-1).mean()
def ADDS(verts, gtR1, gtT1, R1, T1):
    treeTr = KDTree(surfacePointsScaled.dot(R1.T) + T1, leaf_size=2)
    return treeTr.query(verts.dot(gtR1.T) + gtT1, k=1)[0].mean()
def pnp(h3d, h2d, cam, itr=100, reperr=2, flag=cv2.SOLVEPNP_P3P, gtR=None, gtT=None, spts=None, ret=None):
    status, rvec1, tvec1, in1 = cv2.solvePnPRansac(h3d, h2d, cam, distCoeffs=None,
                                                   iterationsCount=itr, reprojectionError=reperr,
                                                   flags=flag)
    rotmat1 = cv2.Rodrigues(rvec1)[0]  ## to convert r from (3,1) to (3,3)
    # import pdb;pdb.set_trace()
    if status: ## check sucess or not
        return rotmat1, tvec1[:, 0], in1[:, 0]
    else:
        print("pose could not be estimated with these correspondences")
        return 1, 1, 1
    
def getCors(queries, feats, leaves=1):
    cMat = torch.log_softmax(queries @ feats.T, dim=-1) 
    # idx=torch.argmax(cMat, dim=-1)
    vals, idx = torch.topk(cMat, k=leaves, dim=-1)
    if leaves == 1:
        return idx[..., 0].cpu(), vals
    else:
        return idx.cpu(), vals
def compute_rel_poses(R1,t1,R2,t2):
    # Compute relative transformation
    R1_inv = R1.T
    #t1_inv = -np.dot(R1_inv, t1)
    relative_rotation = np.dot(R1_inv, R2)
    relative_translation = t2 - t1

    # Relative pose of image 2 w.r.t image 1
    return relative_rotation, relative_translation 

arg_parser = argparse.ArgumentParser(description="Train a Linemod")
arg_parser.add_argument("--objid", dest="objid", default="2", )
arg_parser.add_argument("--id", dest="id", default=-1)
arg_parser.add_argument("--posesEst", dest="posesEst", default=0) ## estimate 6D poses of second sequence
arg_parser.add_argument("--cal_GT", dest="cal_GT", default=0) ## calculate gt rel_poses
arg_parser.add_argument("--cal_pred", dest="cal_pred", default=0) ## cal predicted rel poses
arg_parser.add_argument("--rel_poses", dest="rel_poses", default=0) ## 1 to calculate rel_poses 
arg_parser.add_argument("--choose_image", dest="choose_image", default=0) ## choose best image 
arg_parser.add_argument("--dataset",dest="dataset",default="tless",)
arg_parser.add_argument("--UH", dest="UH", default=0)  ## 0 for UH1
args = arg_parser.parse_args()

device = torch.device("cuda:0")
objid=str(args.objid)
# datasetPath = "bop/ruapc/" 
datasetPath = "bop/" + str(args.dataset) + "/"
meshdetails = json.load(open(datasetPath + "/models" + "/models_info.json"))
diameter = meshdetails[str(objid)]['diameter']
mesh1 = trimesh.load_mesh(datasetPath + "/models/obj_0000" + str(objid).zfill(2) + ".ply")
modelVerts=np.asarray(mesh1.vertices)
RList = []
TList = []
transformation_matrix_list = []

file_list= sorted(glob.glob(datasetPath+"/train/"+str(objid.zfill(6))+"/depth/*"))

if args.rel_poses:
    if args.cal_GT:
        for path in (file_list[:1280]):
            sceneGT = json.load(open(os.path.split(os.path.split(path)[0])[0] + "/scene_gt.json")) ## R, t 
            imID = int(re.findall(r'\d+', os.path.split(path)[1])[0]) ## return 0, 1, 2...
            sceneDet = sceneGT[str(imID)]

            print("saving here R, t of image ", path.split("/")[-1])
            gtR = np.asarray(sceneDet[0]["cam_R_m2c"]).reshape(3, 3)
            RList.append(gtR)
            gtT = np.asarray(sceneDet[0]["cam_t_m2c"])
            TList.append(gtT)
    if args.cal_pred:
        print("loading predicted R, t")
        # RList = np.load(f"ruapc/{objid}pred_R.npy")
        # TList = np.load(f"ruapc/{objid}pred_t.npy")
        RList = np.load(str(args.UH) +'_'+ args.dataset+'_obj_' + str(objid)+"/" +objid +"pred_R.npy")
        TList = np.load(str(args.UH) +'_'+ args.dataset+'_obj_' + str(objid)+"/" +objid +"pred_t.npy")

    relative_poses = np.zeros((len(TList),len(TList),4,4))
    for i in range (len(TList)):
        for j in range (len(TList)):
            # pose from i to j
            rel_R,rel_t = compute_rel_poses(RList[i],TList[i], RList[j], TList[j])
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rel_R
            transformation_matrix[:3, 3:4] = rel_t.reshape(3,1)
            print("cal relative pose of ", i, j)
            relative_poses[i][j] = transformation_matrix

    if args.cal_GT:
        #np.save(f'ruapc/{objid}gt_relative_poses.npy', relative_poses)
        np.save(str(args.UH) +'_'+ args.dataset+'_obj_' + str(objid)+"/"+ objid +"gt_relative_poses.npy", relative_poses)
    if args.cal_pred:
        #np.save(f'ruapc/{objid}pred_relative_poses.npy', relative_poses)
        np.save(str(args.UH) +'_'+ args.dataset+'_obj_' + str(objid)+"/"+ objid +"pred_relative_poses.npy", relative_poses)

if args.choose_image:
    # pred_rel_poses = np.load(f"ruapc/{objid}pred_relative_poses.npy")
    # gt_rel_poses = np.load(f"ruapc/{objid}gt_relative_poses.npy")
    pred_rel_poses = np.load(str(args.UH) +'_'+ args.dataset+'_obj_' + str(objid)+"/"+objid+"pred_relative_poses.npy")
    gt_rel_poses = np.load(str(args.UH) +'_'+ args.dataset+'_obj_' + str(objid)+"/" + objid +"gt_relative_poses.npy")
    error = np.zeros((pred_rel_poses.shape[0],pred_rel_poses.shape[1]))

    agreed_poses = []
    for i in range(pred_rel_poses.shape[0]):
        for j in range(pred_rel_poses.shape[1]):
            print("calculating error for relative pose ", i,j)
            gtR = gt_rel_poses[i][j][:3, :3]
            gtT =  np.squeeze(gt_rel_poses[i][j][:3, 3:4])
            pred_R = pred_rel_poses[i][j][:3, :3]
            pred_t =  np.squeeze(pred_rel_poses[i][j][:3, 3:4])
            if args.dataset == "tless":
                final_error = ADDS(modelVerts, gtR, gtT, pred_R, pred_t)
            else:
                final_error = ADDS(modelVerts, gtR, gtT, pred_R, pred_t)
            if final_error < 0.1 * diameter: # if the predicted relative pose is close to gt 
                error[i][j] = 1 
                print(f"agreed poses of images {i} and {j}")
                agreed_poses.append((i,j))
    # np.save(f"ruapc/{objid}agreedposes.npy", agreed_poses)
    # np.save(f"ruapc/{objid}error.npy",error)
    np.save(str(args.UH) +'_'+ args.dataset+'_obj_' + str(objid)+"/" +objid +"agreedposes.npy", agreed_poses)
    np.save(str(args.UH) +'_'+ args.dataset+'_obj_' + str(objid)+"/" +objid + "error.npy",error)
    
    image_id = np.argmax(np.sum(error, axis=1))
    top_indices = np.argsort(-np.sum(error, axis=1))[:50]
    # file_name = "ruapc/" + str(objid) + "top_50_choices.txt"
    file_name = str(args.UH) +'_'+ args.dataset+'_obj_' + str(objid)+"/" + str(objid) + "top_50_choices.txt"
    with open(file_name, 'w') as file:
        for item in top_indices:
            file.write(str(item) + '\n')
    print("image which should be chosen is ", image_id)


if args.posesEst:
    Siren = True
    camMatScaling=True
    useMask=True
    useSurfEval = False
    expID = str(args.UH) +'_'+ args.dataset+'_obj_' + str(objid)+"/"+objid+"poseEst"
    surfacePointsScaled = np.load(expID + "/vert1_scaled.npy")  ## 3D sampled points of object
    surfaceFeatures = np.load(expID + "/feat1_scaled.npy")  
    sfeats = torch.from_numpy(surfaceFeatures).cuda()
    n2Scaled = np.load(expID + "/normals_scaled.npy")
    tree = KDTree(surfaceFeatures, leaf_size=2)

    torch.manual_seed(1)

    ## load query model - cnn
    from dep.unet import ResNetUNetNew as ResNetUNet
    encoder_rgb = ResNetUNet(n_class=(13), n_decoders=1, )
    data_rgb = torch.load(expID + "/encoderRGBlatest.pth")
    encoder_rgb.load_state_dict(data_rgb["model_state_dict"])
    encoder_rgb.to(device).eval()

    maskBeforeEncoder=True
    pred_R = []
    pred_t = []
    # correct_predicted_ids = []
    for path in (file_list[:1280]):
      
        sceneGT = json.load(open(os.path.split(os.path.split(path)[0])[0] + "/scene_gt.json")) ## R, t 
        sceneGTInfo = json.load(open(os.path.split(os.path.split(path)[0])[0] + "/scene_gt_info.json")) ## bbox of object coordinates
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
                print("saving R, t of image ", path.split("/")[-1])
                pred_R.append(R2)
                pred_t.append(T2)

    # np.save(f"ruapc/{objid}pred_R.npy", pred_R)
    # np.save(f"ruapc/{objid}pred_t.npy", pred_t)
    np.save(str(args.UH) +'_'+ args.dataset+'_obj_' + str(objid)+"/" + objid + "pred_R.npy", pred_R)
    np.save(str(args.UH) +'_'+ args.dataset+'_obj_' + str(objid)+"/" + objid + "pred_t.npy", pred_t)

print("finish")
                 