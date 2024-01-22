
import json
import numpy as np
import open3d as o3d
import re, argparse
import copy

# 定义一个函数来计算相对姿态
def calculate_relative_pose(R1, T1, R2, T2):
    T1_column = T1.reshape(-1, 1)
    RT1 = np.hstack((R1, T1_column))
    RT1_homogeneous = np.vstack([RT1, [0, 0, 0, 1]])
    T2_column = T2.reshape(-1, 1)
    RT2 = np.hstack((R2, T2_column))
    RT2_homogeneous = np.vstack([RT2, [0, 0, 0, 1]])
    Rel = np.dot(RT2_homogeneous,np.linalg.inv(RT1_homogeneous))
    R_relative = Rel[:3,:3]
    T_relative = Rel[:3,-1]
    return R_relative, T_relative

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])

arg_parser = argparse.ArgumentParser(description="Train a Linemod")
arg_parser.add_argument("--objid", dest="objid", default="15", )
args = arg_parser.parse_args()

objid=str(args.objid)
UH="0"

datapath = "bop/Tless"
scenegt = datapath +"/train/"+str(objid.zfill(6))+"/scene_gt.json"

with open(scenegt, 'r') as file:
    data = json.load(file)
keysgt = sorted(data.keys(), key=lambda x: int(x))

# read gt and pred
pred6dpath = "Tless/"+objid +"poseEst_UH" + UH +"/pred6d.json"

with open(pred6dpath, 'r') as file:
    pred6d = json.load(file)
keyspred = sorted(pred6d.keys(), key=lambda x: int(x))

# pointcloud
pc1path = "Tless/"+objid +"poseEst_UH" + UH +"/vert1_scaled.npy"
pc1 = np.load(pc1path)


all_relative_poses = {}
chamferdis = []
for i in range(0,len(keyspred) - 1):
    key1 = keysgt[i]
    key2 = keysgt[i + 1]
    key3 = keyspred[i]
    key4 = keyspred[i+1]
    # print(key1,key2,key3,key4)
    print("id",i)

    # if data[key1] and data[key2]:
    R1 = np.array(data[key1][0]["cam_R_m2c"]).reshape(3, 3)
    T1 = np.array(data[key1][0]["cam_t_m2c"])
    R2 = np.array(data[key2][0]["cam_R_m2c"]).reshape(3, 3)
    T2 = np.array(data[key2][0]["cam_t_m2c"])

    R_relative, T_relative = calculate_relative_pose(R1, T1, R2, T2)

    R1pred = np.array(pred6d[key3][0]["R"]).reshape(3, 3)
    T1pred = np.array(pred6d[key3][0]["T"])
    R2pred = np.array(pred6d[key4][0]["R"]).reshape(3, 3)
    T2pred = np.array(pred6d[key4][0]["T"])
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pc1)
    pc1p1 = pc1.dot(R1pred.T) #+T1pred
    pcgt = pc1p1.dot(R_relative) #+T_relative
    pcpred = pc1.dot(R2pred) #+ T2pred
    pcdgt = o3d.geometry.PointCloud()
    pcdgt.points = o3d.utility.Vector3dVector(pcgt)
    pcdpred = o3d.geometry.PointCloud()
    pcdpred.points = o3d.utility.Vector3dVector(pcpred)
    trans_init = np.asarray([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0], [0.0, 0.0, 0.0, 1.0]])
    # draw_registration_result(pcdgt, pcdpred, trans_init)


    ##chamfer distance
    dists_pcdpred_to_pcdgt = np.asarray(pcdpred.compute_point_cloud_distance(pcdgt))
    mean_dists_pcdpred_to_pcdgt = np.mean(dists_pcdpred_to_pcdgt)
    dists_pcdgt_to_pcdpred = np.asarray(pcdgt.compute_point_cloud_distance(pcdpred))
    mean_dists_pcdgt_to_pcdpred = np.mean(dists_pcdgt_to_pcdpred)
    chamfer_distance = (mean_dists_pcdpred_to_pcdgt + mean_dists_pcdgt_to_pcdpred) / 2
    chamferdis.append(chamfer_distance)


min_chamfer = min(chamferdis)
min_index = chamferdis.index(min_chamfer)
print("best image",min_index)
print("min chamfer distance",min_chamfer)
















