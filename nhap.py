# import numpy as np

# list_of_tuples = np.load(r"ruapc/2agreedposes.npy")

# from collections import Counter

# # # Example list of tuples
# # list_of_tuples = [(1, 2), (2, 3), (1, 2), (3, 4), (2, 3), (1, 2)]

# # Flatten the list of tuples into a single list of numbers
# flattened_list = [number for tup in list_of_tuples for number in tup]

# # Count occurrences of each number
# counter = Counter(flattened_list)

# # Find the top 50 most common numbers
# most_common_50 = [num for num, _ in counter.most_common(50)]

# print("Top 50 appearing images:", most_common_50)


# import numpy as np

# from nutil import vp


# upper = np.load(r"ruapc_1_upper/2TLESSObj_Fine/v1fine.npy") ##(N,3)
# lower = np.load(r'ruapc_0_lower/2TLESSObj_Fine/v1fine.npy') ##(M,3)
# R_pose6D_chosen = np.load("ruapc/2pred_R.npy")[869] ##(3,3)
# t_pose6D_chosen = np.load("ruapc/2pred_t.npy")[869] ##(3)

# transformed_upper = upper@R_pose6D_chosen.T +  t_pose6D_chosen
# obj_full = np.stack((lower, transformed_upper), axis=0)

# vp(obj_full)

import open3d as o3d
from nutil import vp
import numpy as np

# # Load the .ply file
# cad_model = o3d.io.read_point_cloud(r'bop/ruapc/models/obj_000002.ply')
# pointclouds = np.array([point for point in cad_model.points])
# vp(pointclouds)
# # Visualize the loaded model (optional)
# o3d.visualization.draw_geometries([cad_model])
# 


R = np.load(r'0_ruapc_obj_1/1pred_R.npy')
t = np.load(r'0_ruapc_obj_1/1pred_t.npy')
print(len(R))
print('pred R of id 3', R[5], t[5])

print('finish')


# def generate_bop_realsamples(datasetPath, objid="22", imD=128, crop=True, maskStr="mask", cropDim=70,
#                                            justScaleImages=False, ScaledSize=128, maxB=200, offset=10,
#                                            synth=True, makeNDC=True, dataset="tless", fewSamps=False, fewCT=20,
#                                            fewids=[0], background=False, lmDir="train"):
#     #lmDir = "train"

#     if dataset == "lm":
#         lmDir = "lm"
#         # background=True
#         if synth:
#             lmDir = "lm_synth"
#     ## our case lmDir = train
#     objPath = datasetPath + "/" + lmDir + "/0000" + str(objid).zfill(2) + "/"
#     camParams = json.load(open(objPath + "/scene_camera.json"))

#     bboxDets = json.load(open(datasetPath + "/" + lmDir + "/" + str(objid).zfill(6) + "/scene_gt_info.json"))
#     #if dataset == "tless":
#      #   sampleIds = torch.arange(len(bboxDets))
#       #  imCT = len(camParams)
#     if not fewSamps:
#       sampleIds=torch.arange(len(bboxDets))
#       imCT= len(camParams)
#     else:  ## our case fewsamples=True
#       sampleIds=torch.arange(len(bboxDets))
#       imCT= len(camParams) 
#       #import pdb;pdb.set_trace()   
     
#       if len(fewids)==1:
#         fewids=np.random.random_integers(0,imCT-1, fewCT)
#         lines=np.arange(imCT)[fewids]
#       else: # our case len(fewids) = 1281
#         lines=fewids  ## list of number from 0 to 1280
#       imCT=len(fewids)  ## =1281


#     target_images = torch.zeros((imCT, maxB, maxB, 3))  ## (1281,200,200,3)- so 1281 RGB images size 200
#     target_silhouettes = torch.zeros((imCT, maxB, maxB)) ##(1281,200,200)
#     if background: # background =False do not our case
#         target_backgrounds = []
#     RObj = np.zeros((imCT, 3, 3)) #(1281,3,3)
#     TObj = np.zeros((imCT, 3)) # (1281,3)
#     KObj = np.zeros((imCT, 4, 4)) # (1281,4,4) -why 4*4 i thought it should be 3+3 for intrinsics ##?

#     for a11 in range(imCT):
        
#         if fewSamps: #our case
#           imId = int(lines[a11]) # a number in range of 0 to 1280 - no need int actually cos lines is int already
#         else:
#           imId = int(sampleIds[a11])
            
#         if dataset == 'tless':  ## tless is special cause in each image we have 19 objects stead of just 1 object as ruapc
#             rgbPath = objPath + "rgb/" + str(imId).zfill(6) + ".jpg"
#             c1 = cv2.imread(objPath + "rgb/" + str(imId).zfill(6) + ".jpg")  # image shape (128,128,3)
#             m1 = cv2.imread(objPath + maskStr + "/" + str(imId).zfill(6) + "_000000.png") ##mask shape (128,128,1)
            
#         else:
#             rgbPath = objPath + "rgb/" + str(imId).zfill(6) + ".png"
#             c1 = cv2.imread(objPath + "rgb/" + str(imId).zfill(6) + ".png")  # image shape (128,128,3)
#             m1 = cv2.imread(objPath + maskStr + "/" + str(imId).zfill(6) + "_000000.png") ##mask shape (128,128,1)
#         if background: ## background = F not out case - it is the background we use from Imagenet to make it as back ground ddos
#             bgFull = np.zeros((c1.shape[0] + 200, c1.shape[1] + 200, 3)) ## then backgrounf will have shape (328,328,3)
#             bgFull[100:-100, 100:-100] = c1.copy() # then background have shpe(200,200,3)
#         c1[np.where(m1 == 0)] = 0  ## means for the pixels that not inside object set it as background/black in the RGB image- so we get RGB image of only object- cos the background is black only
#         x2, y2, w2, h2 = cv2.boundingRect(m1[:, :, 0])

#         ## lb just to make sure w2,h2 are even
#         if w2 % 2 != 0:
#             w2 = w2 - 1
#         if h2 % 2 != 0:
#             h2 = h2 - 1

#         hw = int(w2 / 2)
#         hh = int(h2 / 2)
#         hd1 = int(np.max((w2, h2)) / 2) # max h2,w2/2
#         maxd = int(np.max((w2, h2))) # max h2, w2

#         squareRGB1 = np.zeros((maxd + 2 * offset, maxd + 2 * offset, 3), np.uint8) ##square cos it the bbox ddos -10*
#         if background: #background = False -not our case
#             squareBG1 = np.zeros((maxd + 2 * offset + 200, maxd + 2 * offset + 200, 3), np.uint8)

#         squareMask1 = np.zeros((maxd + 2 * offset, maxd + 2 * offset), np.uint8) ## offset here =10 - so mask for the bbox
#         hs1 = int(squareRGB1.shape[0] / 2)
#         squareRGB1[hs1 - hh:hs1 + hh, hs1 - hw:hs1 + hw] = c1[y2:y2 + h2, x2:x2 + w2] ## so we got it as RGB image with RGB value for the bbox for object (the background = 0 as black) 
#         squareMask1[hs1 - hh:hs1 + hh, hs1 - hw:hs1 + hw] = m1[y2:y2 + h2, x2:x2 + w2, 0]
#         if background:
#             squareBG1[hs1 - hh:hs1 + hh + 200, hs1 - hw:hs1 + hw + 200] = bgFull[y2:y2 + h2 + 200, x2:x2 + w2 + 200]

#         centerX = x2 + int(w2 / 2)
#         centerY = y2 + int(h2 / 2)

#         ##lb just convert numpy to torch - see resize squareRGB1 to (200,200) 
#         target_images[a11] = torch.from_numpy(
#             cv2.resize(squareRGB1, (maxB, maxB), cv2.INTER_CUBIC).astype("float32") / 255)  ## maxB =200 here
#         target_silhouettes[a11] = torch.from_numpy(
#             cv2.resize(squareMask1, (maxB, maxB), cv2.INTER_NEAREST).astype("float32") / 255)
#         if background:
#             bgScale = int(maxB * squareBG1.shape[0] / squareRGB1.shape[0])
#             if bgScale % 2 == 0:
#                 bgScale = bgScale - 1
#             target_backgrounds = target_backgrounds + [
#                 torch.from_numpy(cv2.resize(squareBG1, (bgScale, bgScale), cv2.INTER_CUBIC).astype("float32") / 255)]
#         # except:
#         #     print("a")
#         # import pdb;pdb.set_trace()
#         if dataset == "tless":
#             gtR, gtT = extractRT(objPath + "rgb/" + str(imId).zfill(6) + ".jpg", occid=0)
#         else:
#             gtR, gtT = extractRT(objPath + "rgb/" + str(imId).zfill(6) + ".png", occid=0)
#         RObj[a11] = gtR
#         TObj[a11] = gtT
#         camparams = np.asarray(camParams[str(imId)]["cam_K"]).reshape(3, 3) ## intricsics 

#         camparams[0, 2] = camparams[0, 2] + (-x2 + hs1 - hw)  ##here we set the x axis of principle points again as the center of the bbox in the squareRGB1
#         camparams[1, 2] = camparams[1, 2] + (-y2 + hs1 - hh) ##here we set the y axis of principle points again
#         # (centerY - offset-hd1)
#         camparams = camparams * maxB / squareRGB1.shape[0]  ##? why need to rescale like this to maxB / squareRGB1.shape[0]
#         camparams[2, 2] = 1 # just set it =1 again other wise from above line it is = maxB / squareRGB1.shape[0] , which is not correct for an intrinsics

#         if makeNDC:  # false in our case -so not our case here
#             ## here to rescale the intrinsics again
#             camparams = camparams * 2 / maxB
#             camparams[2, 2] = 0
#             # camparams[0, 2] -= 1
#             # camparams[1, 2] -= 1

#             camparams[0, 2] = -(camparams[0, 2] - 1)
#             camparams[1, 2] = -(camparams[1, 2] - 1)

#         KObj[a11][0:3, 0:3] = camparams
#         KObj[a11][3, 2] = 1 
#         KObj[a11][2, 3] = 1
#     if fewSamps: ## our case
#         if background:
#             return target_images, target_silhouettes, RObj, TObj, KObj, lines, target_backgrounds  ## lines is just the fewIDs
#         else:
#             return target_images, target_silhouettes, RObj, TObj, KObj, lines ## our case here dont care about background

#     else:
#         if background:
#             return target_images, target_silhouettes, RObj, TObj, KObj, target_backgrounds
#         else:
#             return target_images, target_silhouettes, RObj, TObj, KObj
