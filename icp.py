import numpy as np
import json
import open3d as o3d
import copy 

def vp(finalV): ##vp only the point clouds wihtout colors
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    ct=finalV.shape[0]
    pcd.points = o3d.utility.Vector3dVector(finalV)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd] + [mesh_frame])

#point clouds visualization 
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
arg_parser.add_argument("--objid", dest="objid", default="1", )
# arg_parser.add_argument("--UH", dest="UH", default=0)
arg_parser.add_argument("--dataset",dest="dataset",default="tless",)
args = arg_parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("0_" + str(args.dataset)+"_obj_" +str(args.objid) +"/" + str(args.objid) "top_50_choices.txt", 'r') as file:
    # Load the JSON data
    id_chosen = json.load(file)[0]
# id_chosen = 820

meshdetails = json.load(open('bop/'+ args.dataset +'/models/models_info.json'))
diam = meshdetails[str(args.objid)]['diameter']
diamScaling=1.8
scale=diam/diamScaling


##upper - second sequence - UH = 1 - id from 0-1279
upper = np.load("1_" + args.dataset + "_obj_" + str(args.objid) +"/" +str(args.objid)+ "poseEst/vert1_scaled.npy").astype('float32') ##(N,3)
##lower - first sequence - UH = 0 - id from 1280 -2559
lower = np.load("0_" + args.dataset + "_obj_" + str(args.objid) +"/" +str(args.objid)+ "poseEst/vert1_scaled.npy").astype('float32') ##(N,3)

## rescale only for Nerffine.npy
# upper = upper * scale
# lower = lower * scale


R_pose6D_chosen = np.load("0_" + str(args.dataset) +"_obj_" + str(args.objid) +"/" + str(args.objid + "pred_R.npy")[id_chose] ##(3,3)
t_pose6D_chosen = np.load("0_" + str(args.dataset) +"_obj_" + str(args.objid) +"/" + str(args.objid + "pred_1.npy")[id_chose] ##(3)

with open("bop/" + str(args.dataset) +"/train/" + str(args.objid).zfill(6) +"/scene_gt.json", 'r') as file:
    # Load the JSON data
    data = json.load(file)

R_GT = np.array(data[str(id_chosen)][0]["cam_R_m2c"]).reshape(3,3) # not id_chose +1281
t_GT = np.array(data[str(id_chosen)][0]["cam_t_m2c"])

# Convert Nerf with reference frame of CAD model(first sequence) to the frame of chosen image (in second sequence) 
actual_upper = upper.dot(R_GT.T) +  t_GT ##? why dot? I though = 3D point* R + t

# using the predicted pose to convert second sequence to same frame as first sequence (same as CAD model)
inverse_rot = R_pose6D_chosen.T  # Compute the inverse of the rotation matrix
inverse_trans = -np.dot(inverse_rot, t_pose6D_chosen)
transformed_upper = np.dot(actual_upper - t_pose6D_chosen, inverse_rot.T)

#full object
pred_obj_full = np.concatenate((lower, transformed_upper), axis=0)
# obj_full = np.concatenate((lower, upper), axis=0)
#vp(pred_obj_full)


# icp here    
# Create Open3D point clouds from NumPy arrays
source = o3d.geometry.PointCloud()
source.points = o3d.utility.Vector3dVector(actual_upper)
target = o3d.geometry.PointCloud()
target.points = o3d.utility.Vector3dVector(lower)

threshold = 1
transformation_matrix = np.eye(4)
transformation_matrix[:3, :3] = R_pose6D_chosen
transformation_matrix[:3, 3] = t_pose6D_chosen
inverse_transform_matrix = np.linalg.inv(transformation_matrix)

draw_registration_result(source, target, inverse_transform_matrix)
print("Initial alignment")
threshold = 20  ##* so increase this to get more correspondence to be able to rotate or translate point clouds more ( not just alittle)
evaluation = o3d.pipelines.registration.evaluate_registration(
    source, target, threshold, inverse_transform_matrix)
print(evaluation)
print("Apply point-to-point ICP")
reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target, threshold, inverse_transform_matrix,
    o3d.pipelines.registration.TransformationEstimationPointToPoint())
print(reg_p2p)
print("Transformation is:")
print(reg_p2p.transformation)
draw_registration_result(source, target, reg_p2p.transformation)

# calculating chamfer distance between predicted 3D model and CAD model
transformed_source = source.transform(reg_p2p.transformation)
pred_obj_full = transformed_source + target
cad_model = o3d.io.read_point_cloud("bop/" + str(args.dataset) +"/models/obj_" +str(args.objid).zfill(6) +".ply")
dists_pcd1_to_cad_model = np.asarray(pred_obj_full.compute_point_cloud_distance(cad_model))
mean_dist_pcd1_to_cad_model = np.mean(dists_pcd1_to_cad_model)
dists_cad_model_to_pcd1 = np.asarray(cad_model.compute_point_cloud_distance(pred_obj_full))
mean_dist_cad_model_to_pcd1 = np.mean(dists_cad_model_to_pcd1)
chamfer_distance = (mean_dist_pcd1_to_cad_model + mean_dist_cad_model_to_pcd1) / 2
trans_init = np.asarray([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0], [0.0, 0.0, 0.0, 1.0]])
draw_registration_result(pred_obj_full, cad_model, trans_init)

# print final result
print('diameter', diam)
print("Chamfer Distance(final):", chamfer_distance) ## should be < dimater * 0.1
print("final transformation matrix between first and second sequence is:", reg_p2p.transformation)


