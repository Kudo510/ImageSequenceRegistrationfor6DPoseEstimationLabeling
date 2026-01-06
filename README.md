# Image Sequence Registration for 6D Pose Estimation Labeling

## Motivation

In 6D pose estimation tasks, CAD models are traditionally required for training. However, obtaining CAD models for real-world objects is often impractical—what we can easily capture are images of the object instead. Our solution uses NeRF (Neural Radiance Fields) to reconstruct objects and replace traditional CAD models.

To reconstruct a complete object with NeRF, we need images covering the entire object from all angles. In practice, a single image sequence typically captures only one portion of an object (e.g., the upper or lower half). This necessitates capturing at least two sequences to achieve full coverage.

**The Challenge:** These separate sequences exist in different reference frames. To apply NeRF reconstruction, we must register these sequences—transforming them into a unified coordinate system. Our project provides a robust solution to this registration problem, enabling the creation of complete 3D object models from multiple partial image sequences.

## Related Works

There are three primary approaches for image sequence registration:

1. **2D-2D Correspondences**: Finding poses between images using the Essential Matrix
2. **2D-3D Correspondences**: Finding poses between images and 3D models using PnP + RANSAC *(our approach)*
3. **3D-3D Correspondences**: Finding poses between 3D models using ICP (employed by Dreg-NeRF and NeRF2NeRF)

## Methodology

Given two image sequences of a textureless object from the T-LESS dataset in the BOP benchmark, we utilize the Surfemb architecture to register the sequences by estimating the 6D relative pose between them.

### Pipeline Overview

1. **Initial Correspondence Finding**: Apply Surfemb to establish 3D-2D correspondences between:
   - The NeRF model reconstructed from the first sequence (3D)
   - 2D images from the second sequence

2. **Pose Estimation**: Calculate the relative pose using PnP with RANSAC based on the correspondences

3. **Verification Scheme**: Select the best predicted 6D pose by:
   - Comparing all predicted relative poses against ground truth
   - Choosing the prediction with the smallest Chamfer distance loss

4. **Pose Refinement**: Since initial predictions aren't perfect, we refine them:
   - Reconstruct NeRF models for both sequences
   - Transform the second sequence to the first sequence's canonical frame using the predicted pose
   - Apply ICP (Iterative Closest Point) to obtain the refined relative pose

5. **Final Reconstruction**: Merge both NeRF models using the refined pose to create a complete 3D object model

### Evaluation

We evaluate results using the Chamfer distance metric. A pose prediction is considered correct when the error is significantly smaller than the threshold of **0.1 × object diameter**.

## Results

### Ruapc Dataset (Textured Objects)

During the initial phase, we validated our approach on the textured Ruapc dataset from the BOP Benchmark, testing on object ID 000001.

**Initial Registration**: Transforming the second sequence NeRF with the predicted 6D pose:

![Initial Registration](https://github.com/Kudo510/ImageSequenceRegistrationfor6DPoseEstimationLabeling/assets/68633914/ada8e112-6bd2-43e7-85f9-007fd3681569)

**After ICP Refinement**: Achieving correct registration:

![ICP Refinement](https://github.com/Kudo510/ImageSequenceRegistrationfor6DPoseEstimationLabeling/assets/68633914/c58c12c7-b26a-482b-8690-8913259d8286)

**Comparison with CAD Model**: Chamfer distance error of 1.26 (well below the threshold of 0.1 × diameter):

![CAD Comparison](https://github.com/Kudo510/ImageSequenceRegistrationfor6DPoseEstimationLabeling/assets/68633914/0dff6c47-a77f-4850-9546-ed41fe0aa084)

### T-LESS Dataset (Textureless Objects)

After validating on textured objects, we tested our methodology on the more challenging T-LESS dataset—a textureless, symmetric object dataset from the BOP benchmark.

**Why T-LESS is Challenging:**
- All objects are textureless with uniform gray coloring (except structural parts)
- Objects exhibit symmetries leading to pose ambiguity
- Remains challenging for both RGB and RGBD detectors

**Registration for Continuous Symmetric Object:**

![Continuous Symmetric](https://github.com/Kudo510/ImageSequenceRegistrationfor6DPoseEstimationLabeling/assets/68633914/36e1fc8e-b774-4097-b22d-dc188f6c7889)

**Registration for Discrete Symmetric Object:**

![Discrete Symmetric](https://github.com/Kudo510/ImageSequenceRegistrationfor6DPoseEstimationLabeling/assets/68633914/efd7dd8b-bce3-4f0a-81b3-c719dc943441)

## Installation

Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Training NeRF

**Setup:**
1. Create a folder structure: `bop/ruapc/`
2. Download and unzip:
   - Synthetic training images: [ruapc_train.zip](https://bop.felk.cvut.cz/media/data/bop_datasets/ruapc_train.zip)
   - Models: [ruapc_models.zip](https://bop.felk.cvut.cz/media/data/bop_datasets/ruapc_models.zip)
3. Update the `datasetPath` variable in `trainNeRF.py` to point to `bop/ruapc`

**Training Command:**
```bash
python trainNeRFFine.py --objid 1 --dataset tless --UH 1
```

**Parameters:**
- `--objid`: Object ID
- `--UH`: Upper/lower half selection
  - `0`: Lower half
  - `1`: Upper half

**Output:**
- Generated NeRF images
- Point cloud reconstruction
- `v1.npy`: Point cloud as 3D numpy array
- `v1Fine.npy`: Finer NeRF model reconstruction

*Note: Train separate models for upper and lower halves by changing the `UH` parameter.*

### 2. Generating Correspondences

Generate 3D corresponding coordinates for training images:
```bash
python generateCors.py --objid 2 --dataset ruapc --UH 1 --viz 0
```

**Parameters:**
- `--viz`: Visualization flag
  - `0`: No visualization
  - `1`: Visualize denoised point cloud (verify no noise present)

### 3. Training NeRFEmb (Pose Estimator)

**First Run** (generates `few.npy` and `negVec.npy`):
```bash
python trainPose.py --objid 2 --cont False
```

**Second Run** (trains the pose estimator):
```bash
python trainPose.py --objid 2 --cont True
```

**Background Dataset:**
- Download the COCO dataset for background augmentation
- Set the COCO dataset path in the `trainPose.py` file
- A subset of COCO is sufficient for this specific use case
- More backgrounds improve generalization, but fewer backgrounds work well for segmented sequences on black backgrounds

### 4. Inference

#### Generate Scaled Features

Extract and scale features from the NeRF Feature MLP:
```bash
python genFeat.py --objid 1
```

This generates features for the normalized NeRF point cloud, then scales them to match the actual CAD model scale.

**Output files** (saved in `7poseEst` folder):
- `vert1_scaled.npy`: Scaled point cloud vertices
- `feat1_scaled.npy`: Per-point features
- `normal1_scaled.npy`: Point normals

#### Run Inference on Specific Image
```bash
python inference.py --objid 2 --id 1285
```

**Parameters:**
- `--id`: Image ID from the training dataset

### 5. Verification Scheme

**Generate pose predictions for all images:**
```bash
python inference.py --objid 1
```

**Output:** `pred6d.json` containing predicted 6D poses for all images

**Select best prediction:**
```bash
python verification.py --objid 1
```

**Output:** ID of the best image for ICP refinement

### 6. Pose Refinement with ICP

**Visualize and refine:**
```bash
python ICP.py --objid 1 --bestimage <best_image_id>
```

Replace `<best_image_id>` with the ID from the verification step.

**Output:**
- Visualization of point clouds before ICP
- Visualization of point clouds after ICP
- Comparison with CAD model
- Chamfer distance between predicted and CAD model

**Final transformation:**
```bash
python icp.py --dataset ruapc --objid 1
```

This generates the final refined transformation between the two sequences.

## Citation

If you use this code in your research, please cite our work:
```bibtex
@misc{imagesequenceregistration,
  title={Image Sequence Registration for 6D Pose Estimation Labeling},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/Kudo510/ImageSequenceRegistrationfor6DPoseEstimationLabeling}}
}
```
