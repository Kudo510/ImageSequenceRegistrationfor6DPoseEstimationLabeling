# Image Sequence Registration for 6D Pose Estimation Labeling

## Motivation
In 6D pose estimation tasks, CAD models are always needed for training. However, in reality, what we can easily obtain for each object are only the images of the object, not its CAD model. So we need to find a way to reconstruct the object, and in our case, we want to use NeRF to replace the CAD model. Using NeRF to reconstruct the whole object requires images that cover the entire object. However, in reality, when we capture images of an object, one sequence cannot cover the full object (only, for example, the upper or lower part of objects). As a consequence, we need at least 2 sequences, e.g., for the upper and lower part of objects, to get a full image sequence for the entire object. Nevertheless, one problem will occur: we need to convert the 2 sequences to the same reference frame (to be able to use NeRF on that), or in other words, we need to register the 2 sequences. By doing so, we obtain a dataset covering full objects that NeRF could be applied to generate a 3D model for objects. Our project solves this registration problem.

## Related Works
Generally, there are three approaches for this task:
- 1st approach: Finding poses between images using 2D-2D correspondences via Essential Matrix
- 2nd approach: Finding poses between an image and a 3D model using 2D-3D correspondences via Pnp + RANSAC (This approach is used in our solution)
- 3rd approach: Finding poses between 2 3D models using 3D-3D correspondences - via ICP (Dreg-Nerf and Nerf2Nerf apply this method)

## Methodology
Given two image sequences of a textureless object from the Tless dataset in the BOP benchmark, we utilize the idea from Surfemb architecture to register the two sequences by estimating the 6D relative pose between them. Specifically, we applied Surfemb to find 3D-2D correspondences between the NeRF reconstructed from the first sequence and 2D images from the second sequence.

The relative pose is then calculated based on the correspondences via PnP with RANSAC.

In order to choose the best predicted 6D pose, we apply a verification scheme where we compare all predicted relative poses between images with the ground truth, then choose the one with the smallest Chamfer distance loss. Eventually, we obtain the most accurate 6D pose between the first and second sequence out of all predictions.

Nevertheless, the prediction cannot be 100% precise compared with the ground truth pose. To refine the 6D pose prediction, we first reconstruct NeRF for both sequences. Then we transform the second sequence to the same canonical frame as the first sequence using the predicted 6D pose. Subsequently, we obtain the correct relative pose through the refinement step applying ICP.

After obtaining the refined pose, we stack the 2 NeRF together to obtain the final full predicted 3D model. We evaluate the quantitative results using the Chamfer distance metric. The pose prediction is correct when the error is much smaller than the threshold of 0.1 * diameter.

## Results

### Ruapc dataset

During the first phase of our project, we started with the simple textured Ruapc dataset from the BOP Benchmark. We tested our model on object id 000001. Multiplying the NeRF from the second sequence with the predicted 6D pose, we obtain the NeRF of both sequences in the same canonical frame:
![image](https://github.com/Kudo510/ImageSequenceRegistrationfor6DPoseEstimationLabeling/assets/68633914/ada8e112-6bd2-43e7-85f9-007fd3681569)

Afterward, ICP is employed to refine the prediction. It results in correct registration:
![image](https://github.com/Kudo510/ImageSequenceRegistrationfor6DPoseEstimationLabeling/assets/68633914/c58c12c7-b26a-482b-8690-8913259d8286)

We also compare the predicted object with the CAD model using the Chamfer distance metric (in our case here the error is 1.26, much smaller than the threshold of 0.1 * diameter):
![image](https://github.com/Kudo510/ImageSequenceRegistrationfor6DPoseEstimationLabeling/assets/68633914/0dff6c47-a77f-4850-9546-ed41fe0aa084)

### Tless dataset

Since our methodology worked on textured objects, we want to test it on a much more challenging dataset which is the textureless symmetric dataset Tless (also from BOP benchmark).

We chose the T-LESS dataset because it is still challenging not only for pure RGB detectors but also for RGBD detectors. Several factors contribute to the complexity of the dataset. First, all objects are textureless in a sense that they do not have distinctive colors. All of them are colored in more or less the same shade of gray, except for certain structural parts. Second, the T-LESS objects exhibit symmetries leading to pose ambiguity.

#### Registration for continuous symmetric object
![image](https://github.com/Kudo510/ImageSequenceRegistrationfor6DPoseEstimationLabeling/assets/68633914/36e1fc8e-b774-4097-b22d-dc188f6c7889)

#### Registration for discrete symmetric object
![image](https://github.com/Kudo510/ImageSequenceRegistrationfor6DPoseEstimationLabeling/assets/68633914/efd7dd8b-bce3-4f0a-81b3-c719dc943441)

## Install packages:
``` pip install -r requirements.txt```

## Training NeRF:
1. You have to create a folder with this structure bop/ruapc/
2. Unzip the synthetic training images ([ruapc_train.zip](https://bop.felk.cvut.cz/media/data/bop_datasets/ruapc_train.zip)) and models ([ruapc_models.zip](https://bop.felk.cvut.cz/media/data/bop_datasets/ruapc_models.zip)).
3. Then, change the `datasetPath` variable in the `trainNeRF.py` file to the location of "bop/ruapc".
4. This is the command to run the NeRF: 





## Results
### Ruapc dataset
During the first phrase of our project, we started with the simple textured ruapc dataset from BOP Benchmark. 
We tested our model on object id 000001. Multiply the NeRF from second sequence with predicted 6D pose, we obtain the NeRF of both sequences in a same canonical frame:
![image](https://github.com/Kudo510/ImageSequenceRegistrationfor6DPoseEstimationLabeling/assets/68633914/ada8e112-6bd2-43e7-85f9-007fd3681569)

Afterwards, ICP is employed to refine the prediction. It results in a corect registration
![image](https://github.com/Kudo510/ImageSequenceRegistrationfor6DPoseEstimationLabeling/assets/68633914/c58c12c7-b26a-482b-8690-8913259d8286)

We also compare the predicted object with the CAD model using Chamfer distance metric (in our case here the error is 1.26 much smaller than the threshold of 0.1*diamter)

![image](https://github.com/Kudo510/ImageSequenceRegistrationfor6DPoseEstimationLabeling/assets/68633914/0dff6c47-a77f-4850-9546-ed41fe0aa084)

### Tless dataset
Since our methodology worked on textured objects, we want to test it in a much more challeging dataset which is the textureless symmetric dataset Tless (also from BOP benchmark).

We chose the T-LESS dataset
because it is still challenging not only for pure RGB detectors but also for RGBD
detectors. Several factors contribute to the complexity of the dataset. First, all
objects are textureless in a sense that they do not have distinctive colors. All
of them are colored in more or less the same shade of gray, except for certain
structural parts. Second, the T-LESS
objects exhibit symmetries leading to pose ambiguity.
#### Registration for continuous symmetric object
![image](https://github.com/Kudo510/ImageSequenceRegistrationfor6DPoseEstimationLabeling/assets/68633914/36e1fc8e-b774-4097-b22d-dc188f6c7889)
#### Registration for discrete symmetric object
![image](https://github.com/Kudo510/ImageSequenceRegistrationfor6DPoseEstimationLabeling/assets/68633914/efd7dd8b-bce3-4f0a-81b3-c719dc943441)

## Install packages:
``` pip install -r requirements.txt```
## Training NeRF:
1. You have to create a folder with this structure bop/ruapc/
2. unzip the synthetic training images(https://bop.felk.cvut.cz/media/data/bop_datasets/ruapc_train.zip) and models(https://bop.felk.cvut.cz/media/data/bop_datasets/ruapc_models.zip)
3. Then, change the datasetPath variable in the trainNeRF.py file to the location of "bop/ruapc"
4. This is the command to run the NeRF 
``` python trainNeRFFine.py --objid 1 --dataset tless --UH 1 ```

You can mention object id using objid and (--UH 0) means lower half of the object and (--UH 1) means upper half of the objects. After training, you can see the generated NeRF images along with the point cloud of the reconstruciton
v1.npy contains pointcloud as 3D numpy array
v1Fine contains the same but reconstructed with finer NeRF model
train two different models for upper half and bottom half of the object by changing the UH variable.

## Generating Correspondences:
we generate 3D correposnding coordinates for the set of training image using the command below

``` python generateCors.py --objid 2 --dataset ruapc --UH 1 --viz 0 ```
set viz as 1 to visualize the denoised pointcloud to see if it doesn't have any noise. ideally the visualization should contain the object pointcloud which covers our viewpoints

## Train NeRFEmb: Our pose estimator

To obtain the few.npy and negVec.npy (negative 3D point clouds) first then reun the second time to train the Pose

``` python trainPose.py --objid 2 --cont True ```
You need to downlaod coco dataset for backgrounds. Set the path to coco dataset in the trainPose File. You can also use a subset of coco. It doesn't haver to have so many images.
The more backgrounds we have, the more generalized our pose estimator becomes.
However, since our target is not general pose estimation and we only want to do pose estimation for another NeRF sequence which is already segmented and put on black backgorund,
we can train with fewer backgrounds also.

## Inference
### Firstly generating scaled features
To perform inference, we need to first estimate features for pointcloud from NeRF Feature MLP and scale them to actual cad model scale. We learn NeRF in a normalized space[0-1]. We perform inference on normalized pointcloud to extract features and save the feature. We then scale the point cloud and save the real world scale point cloud along with per-point features for visualization.
The command to generate per-point features is:

``` python genFeat.py --objid 1 ```
You should see vert1_scaled.npy, feat1_scaled.npy, normal1_scaled.npy saved in the "7poseEst" folder after executing this statement

### Then we run inference on desired image with image ID.

``` python inference.py --objid 2 --id 1285 ```
"id" is the number of the image in training image.

## Verification scheme
``` python inference.py  --objid 1 ```
You can get pred6d.json after running this command, which is the predicted 6d poses of all images in dataset.

``` python verification.py --objid1 ```
You can get the id of best image for the following ICP

``` python ICP.py --objid --bestimage ``` (the best image id from verification.py)
It visualize the two pointcloud before icp and two pointcloud after icp and point clouds with Cad model. print the chamfer distance between point cloud and Cad model.

## Refine pose with ICP and get final transformation result
``` python icp.py --dataset ruapc --objid 1  ```


