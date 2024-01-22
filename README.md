# ImageSequenceRegistrationfor6DPoseEstimationLabeling
Image Sequence Registration for 6D Pose Estimation Labeling
## Methodology
Given two image sequences of a textureless object from Tless dataset in BOP benchmark, we utilize the idea from Surfemb architecture to register the two sequences by estimating the 6D relative pose between them. Specifically, we applied Suremb to find 3D-2D correspondences
between the NeRF reconstructed from first sequence and 2D images from the second sequence.
The relative pose is then calculated based on the correspondences via PnP with RANSAC.
In order to choose the best predicted 6D pose apply a verification scheme that we compare all predicted relative poses between images with the ground truth then choose the one with the smallest Chamfer distance loss. Eventually we obtain the most accurate 6D pose between the first and second sequence out of all predictions. Nevertheless, the prediction cannot be 100% precise comparing with the ground truth pose. In order to refine the 6D pose prediction, we first recontruct Nerf for both sequences. Then we transform the second sequence to the same canonical frame as first sequence using the predicted 6D pose. Subsequently, we obtain the correct relative pose through the refinement step applying ICP.
After obtaining the refined pose, we stack the 2 NeRF together to obtain the final full predicted 3D model. We evaluate the quantitative results using Chamfer distance metric. The pose prediction is correct when the error is much smaller than the threshold of 0.1*diameter
## Results
During the first phrase of our project we started with the simple textured ruapc dataset from
## Install packages:
pip install -r requirements.txt
## Training Nerf:
1. You have to create a folder with this structure bop/ruapc/
2. unzip the synthetic training images(https://bop.felk.cvut.cz/media/data/bop_datasets/ruapc_train.zip) and models(https://bop.felk.cvut.cz/media/data/bop_datasets/ruapc_models.zip)
3. Then, change the datasetPath variable in the trainNerf.py file to the location of "bop/ruapc"
4. This is the command to run the Nerf 
( python trainNerfFine.py --objid 1 --dataset tless --UH 1 )

You can mention object id using objid and (--UH 0) means lower half of the object and (--UH 1) means upper half of the objects. After training, you can see the generated Nerf images along with the point cloud of the reconstruciton
v1.npy contains pointcloud as 3D numpy array
v1Fine contains the same but reconstructed with finer Nerf model
train two different models for upper half and bottom half of the object by changing the UH variable. you also need to change the folder. It will overwrite it self. I didn't code that

## Generating Correspondences:
we generate 3D correposnding coordinates for the set of training image using the command below
python generateCors.py --objid 2 --dataset ruapc --UH 1 --viz 0

set viz as 1 to visualize the denoised pointcloud to see if it doesn't have any noise. ideally the visualization should contain the object pointcloud which covers our viewpoints

## Train NeRFEmb: Our pose estimator

python trainPose.py --objid 2 --cont True ## # to obtain the few.npy and negVec.npy (negative 3D point clouds) first then reun the second time to train the Pose

You need to downlaod coco dataset for backgrounds. Set the path to coco dataset in the trainPose File. You can also use a subset of coco. It doesn't haver to have so many images.
The more backgrounds we have, the more generalized our pose estimator becomes.
However, since our target is not general pose estimation and we only want to do pose estimation for another nerf sequence which is already segmented and put on black backgorund,
we can train with fewer backgrounds also.

## Inference
## # Firstly generating scaled features
To perform inference, we need to first estimate features for pointcloud from NeRF Feature MLP and scale them to actual cad model scale. We learn nerf in a normalized space[0-1]. We perform inference on normalized pointcloud to extract features
and save the feature. We then scale the point cloud and save the real world scale point cloud along with per-point features for visualization.
The command to generate per-point features is 

python genFeat.py --objid 1

You should see vert1_scaled.npy, feat1_scaled.npy, normal1_scaled.npy saved in the "7poseEst" folder after executing this statement

### Then we run inference on desired image with image ID.

python inference.py --objid 2 --id 1285

"id" is the number of the image in training image.

## Verification scheme, choosing best image for registration 
python inference.py  --objid 1 
You can get pred6d.json after running this command, which is the predicted 6d poses of all images in dataset.
python verification.py --objid1
You can get the id of best image for the following ICP
python ICP.py --objid --bestimage (the best image id from verification.py)
It visualize the two pointcloud before icp and two pointcloud after icp and point clouds with Cad model. print the chamfer distance between point cloud and Cad model.

## refine pose with ICP and get final transformation result
python icp.py --dataset ruapc --objid 1 


