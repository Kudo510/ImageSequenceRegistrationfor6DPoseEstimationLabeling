#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.



# coding: utf-8

# In[ ]:


# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.


# # Fit a simple Neural Radiance Field via raymarching
#
# This tutorial shows how to fit Neural Radiance Field given a set of views of a scene using differentiable implicit function rendering.
#
# More specifically, this tutorial will explain how to:
# 1. Create a differentiable implicit function renderer with either image-grid or Monte Carlo ray sampling.
# 2. Create an Implicit model of a scene.
# 3. Fit the implicit function (Neural Radiance Field) based on input images using the differentiable implicit renderer.
# 4. Visualize the learnt implicit function.
#
# Note that the presented implicit model is a simplified version of NeRF:<br>
# _Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng: NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis, ECCV 2020._
#
# The simplifications include:
# * *Ray sampling*: This notebook does not perform stratified ray sampling but rather ray sampling at equidistant depths.
# * *Rendering*: We do a single rendering pass, as opposed to the original implementation that does a coarse and fine rendering pass.
# * *Architecture*: Our network is shallower which allows for faster optimization possibly at the cost of surface details.
# * *Mask loss*: Since our observations include segmentation masks, we also optimize a silhouette loss that forces rays to either get fully absorbed inside the volume, or to completely pass through it.
#

# ## 0. Install and Import modules
# Ensure `torch` and `torchvision` are installed. If `pytorch3d` is not installed, install it using the following cell:

# In[ ]:


import os
import sys
import torch
from typing import Callable, Tuple, Union
need_pytorch3d = False
try:
    import pytorch3d
except ModuleNotFoundError:
    need_pytorch3d = True
if need_pytorch3d:
    if torch.__version__.startswith("1.12.") and sys.platform.startswith("linux"):
        # We try to install PyTorch3D via a released wheel.
        pyt_version_str = torch.__version__.split("+")[0].replace(".", "")
        version_str = "".join([
            f"py3{sys.version_info.minor}_cu",
            torch.version.cuda.replace(".", ""),
            f"_pyt{pyt_version_str}"
        ])
        get_ipython().system('pip install fvcore iopath')
        get_ipython().system(
            'pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html')
    else:
        # We try to install PyTorch3D from source.
        get_ipython().system('curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz')
        get_ipython().system('tar xzf 1.10.0.tar.gz')
        os.environ["CUB_HOME"] = os.getcwd() + "/cub-1.10.0"
        get_ipython().system("pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'")

# In[ ]:


# %matplotlib inline
# %matplotlib notebook
import os
import sys
import time
import json
import glob
import torch
import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from IPython import display
from tqdm.notebook import tqdm
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.renderer.implicit.utils import RayBundle
from pytorch3d.renderer.implicit.raysampling import  RayBundle
from pytorch3d.renderer.implicit.raymarching import  _check_raymarcher_inputs, _check_density_bounds, _shifted_cumprod
# Data structures and functions for rendering
from pytorch3d.structures import Volumes
from pytorch3d.transforms import so3_exp_map
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    NDCMultinomialRaysampler,
    MonteCarloRaysampler,
    EmissionAbsorptionRaymarcher,
    ImplicitRenderer,
    RayBundle,
    ray_bundle_to_ray_points,
)


#from cowrender import image_grid, generate_cow_renders, generate_cow_rendersWithRT
from dep.siren import Siren

class HarmonicEmbedding(torch.nn.Module):
    def __init__(self, n_harmonic_functions=60, omega0=0.1):
        """
        Given an input tensor `x` of shape [minibatch, ... , dim],
        the harmonic embedding layer converts each feature
        in `x` into a series of harmonic features `embedding`
        as follows:
            embedding[..., i*dim:(i+1)*dim] = [
                sin(x[..., i]),
                sin(2*x[..., i]),
                sin(4*x[..., i]),
                ...
                sin(2**(self.n_harmonic_functions-1) * x[..., i]),
                cos(x[..., i]),
                cos(2*x[..., i]),
                cos(4*x[..., i]),
                ...
                cos(2**(self.n_harmonic_functions-1) * x[..., i])
            ]

        Note that `x` is also premultiplied by `omega0` before
        evaluating the harmonic functions.
        """
        super().__init__()
        self.register_buffer(
            'frequencies',
            omega0 * (2.0 ** torch.arange(n_harmonic_functions)),
        )

    def forward(self, x):
        """
        Args:
            x: tensor of shape [..., dim]
        Returns:
            embedding: a harmonic embedding of `x`
                of shape [..., n_harmonic_functions * dim * 2]
        """
        embed = (x[..., None] * self.frequencies).view(*x.shape[:-1], -1)
        return torch.cat((embed.sin(), embed.cos()), dim=-1)



class NeuralRadianceFieldFeat(torch.nn.Module):
    def __init__(self, n_harmonic_functions=60, n_hidden_neurons=256, color_embedding_dims=12,siren=False, mode="color"):
        super().__init__()
        """
        Args:
            n_harmonic_functions: The number of harmonic functions
                used to form the harmonic embedding of each point.
            n_hidden_neurons: The number of hidden units in the
                fully connected layers of the MLPs of the model.
        """
        self.mode=mode
        self.siren=siren
        # The harmonic embedding layer converts input 3D coordinates
        # to a representation that is more suitable for
        # processing with a deep neural network.
        self.harmonic_embedding = HarmonicEmbedding(n_harmonic_functions)

        # The dimension of the harmonic embedding.
        embedding_dim = n_harmonic_functions * 2 * 3

        # self.mlp is a simple 2-layer multi-layer perceptron
        # which converts the input per-point harmonic embeddings
        # to a latent representation.
        # Not that we use Softplus activations instead of ReLU.
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, n_hidden_neurons),
            torch.nn.Softplus(beta=10.0),
            torch.nn.Linear(n_hidden_neurons, n_hidden_neurons),
            torch.nn.Softplus(beta=10.0),
        )

        # Given features predicted by self.mlp, self.color_layer
        # is responsible for predicting a 3-D per-point vector
        # that represents the RGB color of the point.
        self.color_layer = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_neurons +embedding_dim , n_hidden_neurons),
            torch.nn.Softplus(beta=10.0),
            torch.nn.Linear(n_hidden_neurons, 3),
            torch.nn.Sigmoid(),
            # To ensure that the colors correctly range between [0-1],
            # the layer is terminated with a sigmoid layer.
        )
        
        if not siren:
          self.feature_layer = torch.nn.Sequential(
              torch.nn.Linear(embedding_dim , n_hidden_neurons),
              torch.nn.Softplus(beta=10.0),
              torch.nn.Linear(n_hidden_neurons, color_embedding_dims),
              torch.nn.Sigmoid(),
              # To ensure that the colors correctly range between [0-1],
              # the layer is terminated with a sigmoid layer.
          )
        else:
          self.feature_layer=Siren(in_features=3,out_features=color_embedding_dims,
              hidden_features=n_hidden_neurons, hidden_layers=2)
        # The density layer converts the features of self.mlp
        # to a 1D density value representing the raw opacity
        # of each point.
        self.density_layer = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_neurons, 1),
            torch.nn.Softplus(beta=10.0),
            # Sofplus activation ensures that the raw opacity
            # is a non-negative number.
        )

        # We set the bias of the density layer to -1.5
        # in order to initialize the opacities of the
        # ray points to values close to 0.
        # This is a crucial detail for ensuring convergence
        # of the model.
        self.density_layer[0].bias.data[0] = -1.5

    def _get_densities(self, features):
        """
        This function takes `features` predicted by `self.mlp`
        and converts them to `raw_densities` with `self.density_layer`.
        `raw_densities` are later mapped to [0-1] range with
        1 - inverse exponential of `raw_densities`.
        """
        raw_densities = self.density_layer(features)
        return 1 - (-raw_densities).exp()

    def _get_colors(self, features, rays_directions):
        """
        This function takes per-point `features` predicted by `self.mlp`
        and evaluates the color model in order to attach to each
        point a 3D vector of its RGB color.

        In order to represent viewpoint dependent effects,
        before evaluating `self.color_layer`, `NeuralRadianceField`
        concatenates to the `features` a harmonic embedding
        of `ray_directions`, which are per-point directions
        of point rays expressed as 3D l2-normalized vectors
        in world coordinates.
        """
        spatial_size = features.shape[:-1]

        # Normalize the ray_directions to unit l2 norm.
        rays_directions_normed = torch.nn.functional.normalize(
            rays_directions, dim=-1
        )

        # Obtain the harmonic embedding of the normalized ray directions.
        rays_embedding = self.harmonic_embedding(
            rays_directions_normed
        )

        # Expand the ray directions tensor so that its spatial size
        # is equal to the size of features.
        rays_embedding_expand = rays_embedding[..., None, :].expand(
            *spatial_size, rays_embedding.shape[-1]
        )

        # Concatenate ray direction embeddings with
        # features and evaluate the color model.
        color_layer_input = torch.cat(
            (features, rays_embedding_expand),
            dim=-1
        )
        #color_layer_input = (features)
        return self.color_layer(color_layer_input)
    def _get_features_colors(self, features, rays_directions,embeds):
        """
        This function takes per-point `features` predicted by `self.mlp`
        and evaluates the color model in order to attach to each
        point a 3D vector of its RGB color.

        In order to represent viewpoint dependent effects,
        before evaluating `self.color_layer`, `NeuralRadianceField`
        concatenates to the `features` a harmonic embedding
        of `ray_directions`, which are per-point directions
        of point rays expressed as 3D l2-normalized vectors
        in world coordinates.
        """
        spatial_size = features.shape[:-1]

        # Normalize the ray_directions to unit l2 norm.
        rays_directions_normed = torch.nn.functional.normalize(
            rays_directions, dim=-1
        )

        # Obtain the harmonic embedding of the normalized ray directions.
        rays_embedding = self.harmonic_embedding(
            rays_directions_normed
        )

        # Expand the ray directions tensor so that its spatial size
        # is equal to the size of features.
        rays_embedding_expand = rays_embedding[..., None, :].expand(
            *spatial_size, rays_embedding.shape[-1]
        )

        # Concatenate ray direction embeddings with
        # features and evaluate the color model.
        color_layer_input = torch.cat(
            (features, rays_embedding_expand),
            dim=-1
        )
        color_out=self.color_layer(color_layer_input)
        if self.siren:
          feature_out=self.feature_layer(embeds)
        else:
          feature_out=self.feature_layer(embeds)
        
        #color_layer_input = (features)
        return torch.cat([color_out,feature_out],dim=-1)
    def _get_features(self, features, rays_directions):
        """
        This function takes per-point `features` predicted by `self.mlp`
        and evaluates the color model in order to attach to each
        point a 3D vector of its RGB color.

        In order to represent viewpoint dependent effects,
        before evaluating `self.color_layer`, `NeuralRadianceField`
        concatenates to the `features` a harmonic embedding
        of `ray_directions`, which are per-point directions
        of point rays expressed as 3D l2-normalized vectors
        in world coordinates.
        """
        spatial_size = features.shape[:-1]

        # Normalize the ray_directions to unit l2 norm.
        

        # Concatenate ray direction embeddings with
        # features and evaluate the color model.
        #color_layer_input = torch.cat(
         #   (features),
          #  dim=-1
        #)
        #color_layer_input = (features)
        return self.feature_layer(features)
    def forward(
            self,
            ray_bundle: RayBundle,featRend=1,
            **kwargs,
    ):
        """
        The forward function accepts the parametrizations of
        3D points sampled along projection rays. The forward
        pass is responsible for attaching a 3D vector
        and a 1D scalar representing the point's
        RGB color and opacity respectively.

        Args:
            ray_bundle: A RayBundle object containing the following variables:
                origins: A tensor of shape `(minibatch, ..., 3)` denoting the
                    origins of the sampling rays in world coords.
                directions: A tensor of shape `(minibatch, ..., 3)`
                    containing the direction vectors of sampling rays in world coords.
                lengths: A tensor of shape `(minibatch, ..., num_points_per_ray)`
                    containing the lengths at which the rays are sampled.

        Returns:
            rays_densities: A tensor of shape `(minibatch, ..., num_points_per_ray, 1)`
                denoting the opacity of each ray point.
            rays_colors: A tensor of shape `(minibatch, ..., num_points_per_ray, 3)`
                denoting the color of each ray point.
        """
        # We first convert the ray parametrizations to world
        # coordinates with `ray_bundle_to_ray_points`.
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)
        # rays_points_world.shape = [minibatch x ... x 3]

        # For each 3D world coordinate, we obtain its harmonic embedding.
        embeds = self.harmonic_embedding(
            rays_points_world
        )
        # embeds.shape = [minibatch x ... x self.n_harmonic_functions*6]

        # self.mlp maps each harmonic embedding to a latent feature space.
        features = self.mlp(embeds)
        # features.shape = [minibatch x ... x n_hidden_neurons]

        # Finally, given the per-point features,
        # execute the density and color branches.

        rays_densities = self._get_densities(features)
        # rays_densities.shape = [minibatch x ... x 1]
        #import pdb;pdb.set_trace()
        if self.mode=="feature":
          if self.siren:
            rays_colors = self._get_features(rays_points_world, ray_bundle.directions)
          else:
            rays_colors = self._get_features(embeds, ray_bundle.directions)
          
          
        elif self.mode=="color":
          rays_colors = self._get_colors(features, ray_bundle.directions)
        else:
          rays_colors = self._get_features_colors(features, ray_bundle.directions)
        
        # rays_colors.shape = [minibatch x ... x 3]

        return rays_densities, rays_colors
    
    def customForward(self,rays_points_world):
          
          if self.siren:
            rays_colors = self._get_features(rays_points_world, 1)
          else:
            embeds = self.harmonic_embedding(
              rays_points_world
            )
            rays_colors = self._get_features(embeds, 1)
          
          #import pdb;pdb.set_trace()
          return torch.cat([rays_colors,rays_colors[...,0:1]*0], dim=-1)

    def customForwardForDensity(self, rays_points_world):
        embeds = self.harmonic_embedding(
            rays_points_world
        )
        # embeds.shape = [minibatch x ... x self.n_harmonic_functions*6]

        # self.mlp maps each harmonic embedding to a latent feature space.
        features = self.mlp(embeds)
        # features.shape = [minibatch x ... x n_hidden_neurons]

        # Finally, given the per-point features,
        # execute the density and color branches.

        rays_densities = self._get_densities(features)
              # import pdb;pdb.set_trace()
        return rays_densities
          #batch_outputs = [
           #       self.forwardWithPoints(
      
            #          gridCoords.view(-1, 3)[batch_idx]
             #     ) for batch_idx in batches2
              #]
      
              #rays_densities, rays_color = [torch.cat([batch_output[output_i] for batch_output in batch_outputs], dim=0).view(
               #   *torch.Size((gridRes, gridRes, gridRes)), -1) for output_i in (0, 1)]
    def batched_customForward(self,rays_points_world, n_batches: int = 16,):
        tot_samples = rays_points_world.shape[:-1].numel()
        batches = torch.chunk(torch.arange(tot_samples), n_batches)
        spatial_size = [*rays_points_world.shape[:-1]]
        # For each batch, execute the standard forward pass.
        batch_outputs = [
            self.customForward(
                rays_points_world.view(-1,3)[batch_idx]
            ) for batch_idx in batches
        ]
        #import pdb;pdb.set_trace()
        # Concatenate the per-batch rays_densities and rays_colors
        # and reshape according to the sizes of the inputs.
        rays_feats= torch.cat([batch_output for batch_output in batch_outputs], dim=0).view(*spatial_size,-1)
        
        return rays_feats
    def batched_forward(
            self,
            ray_bundle: RayBundle,
            n_batches: int = 16,
            **kwargs,
    ):
        """
        This function is used to allow for memory efficient processing
        of input rays. The input rays are first split to `n_batches`
        chunks and passed through the `self.forward` function one at a time
        in a for loop. Combined with disabling PyTorch gradient caching
        (`torch.no_grad()`), this allows for rendering large batches
        of rays that do not all fit into GPU memory in a single forward pass.
        In our case, batched_forward is used to export a fully-sized render
        of the radiance field for visualization purposes.

        Args:
            ray_bundle: A RayBundle object containing the following variables:
                origins: A tensor of shape `(minibatch, ..., 3)` denoting the
                    origins of the sampling rays in world coords.
                directions: A tensor of shape `(minibatch, ..., 3)`
                    containing the direction vectors of sampling rays in world coords.
                lengths: A tensor of shape `(minibatch, ..., num_points_per_ray)`
                    containing the lengths at which the rays are sampled.
            n_batches: Specifies the number of batches the input rays are split into.
                The larger the number of batches, the smaller the memory footprint
                and the lower the processing speed.

        Returns:
            rays_densities: A tensor of shape `(minibatch, ..., num_points_per_ray, 1)`
                denoting the opacity of each ray point.
            rays_colors: A tensor of shape `(minibatch, ..., num_points_per_ray, 3)`
                denoting the color of each ray point.

        """

        # Parse out shapes needed for tensor reshaping in this function.
        n_pts_per_ray = ray_bundle.lengths.shape[-1]
        spatial_size = [*ray_bundle.origins.shape[:-1], n_pts_per_ray]

        # Split the rays to `n_batches` batches.
        tot_samples = ray_bundle.origins.shape[:-1].numel()
        batches = torch.chunk(torch.arange(tot_samples), n_batches)

        # For each batch, execute the standard forward pass.
        batch_outputs = [
            self.forward(
                RayBundle(
                    origins=ray_bundle.origins.view(-1, 3)[batch_idx],
                    directions=ray_bundle.directions.view(-1, 3)[batch_idx],
                    lengths=ray_bundle.lengths.view(-1, n_pts_per_ray)[batch_idx],
                    xys=None,
                )
            ) for batch_idx in batches
        ]

        # Concatenate the per-batch rays_densities and rays_colors
        # and reshape according to the sizes of the inputs.
        rays_densities, rays_colors = [
            torch.cat(
                [batch_output[output_i] for batch_output in batch_outputs], dim=0
            ).view(*spatial_size, -1) for output_i in (0, 1)
        ]
        return rays_densities, rays_colors

    def batched_forward_fordensity(
            self,
            ray_bundle: RayBundle,
            n_batches: int = 16,
            **kwargs,
    ):
        """
        This function is used to allow for memory efficient processing
        of input rays. The input rays are first split to `n_batches`
        chunks and passed through the `self.forward` function one at a time
        in a for loop. Combined with disabling PyTorch gradient caching
        (`torch.no_grad()`), this allows for rendering large batches
        of rays that do not all fit into GPU memory in a single forward pass.
        In our case, batched_forward is used to export a fully-sized render
        of the radiance field for visualization purposes.

        Args:
            ray_bundle: A RayBundle object containing the following variables:
                origins: A tensor of shape `(minibatch, ..., 3)` denoting the
                    origins of the sampling rays in world coords.
                directions: A tensor of shape `(minibatch, ..., 3)`
                    containing the direction vectors of sampling rays in world coords.
                lengths: A tensor of shape `(minibatch, ..., num_points_per_ray)`
                    containing the lengths at which the rays are sampled.
            n_batches: Specifies the number of batches the input rays are split into.
                The larger the number of batches, the smaller the memory footprint
                and the lower the processing speed.

        Returns:
            rays_densities: A tensor of shape `(minibatch, ..., num_points_per_ray, 1)`
                denoting the opacity of each ray point.
            rays_colors: A tensor of shape `(minibatch, ..., num_points_per_ray, 3)`
                denoting the color of each ray point.

        """

        # Parse out shapes needed for tensor reshaping in this function.
        n_pts_per_ray = ray_bundle.lengths.shape[-1]
        spatial_size = [*ray_bundle.origins.shape[:-1], n_pts_per_ray]

        # Split the rays to `n_batches` batches.
        tot_samples = ray_bundle.origins.shape[:-1].numel()
        batches = torch.chunk(torch.arange(tot_samples), n_batches)

        # For each batch, execute the standard forward pass.
        batch_outputs = [
            self.forward_fordensity(
                RayBundle(
                    origins=ray_bundle.origins.view(-1, 3)[batch_idx],
                    directions=ray_bundle.directions.view(-1, 3)[batch_idx],
                    lengths=ray_bundle.lengths.view(-1, n_pts_per_ray)[batch_idx],
                    xys=None,
                )
            ) for batch_idx in batches
        ]

        # Concatenate the per-batch rays_densities and rays_colors
        # and reshape according to the sizes of the inputs.
        rays_densities = torch.cat([batch_output for batch_output in batch_outputs], dim=0).view(*spatial_size, -1)
        
        a1=torch.zeros((*rays_densities.shape[:-1],3)).cuda()
        
        
        return rays_densities, a1
    def forward_fordensity(
            self,
            ray_bundle: RayBundle,featRend=1,
            **kwargs,
    ):
        """
        The forward function accepts the parametrizations of
        3D points sampled along projection rays. The forward
        pass is responsible for attaching a 3D vector
        and a 1D scalar representing the point's
        RGB color and opacity respectively.

        Args:
            ray_bundle: A RayBundle object containing the following variables:
                origins: A tensor of shape `(minibatch, ..., 3)` denoting the
                    origins of the sampling rays in world coords.
                directions: A tensor of shape `(minibatch, ..., 3)`
                    containing the direction vectors of sampling rays in world coords.
                lengths: A tensor of shape `(minibatch, ..., num_points_per_ray)`
                    containing the lengths at which the rays are sampled.

        Returns:
            rays_densities: A tensor of shape `(minibatch, ..., num_points_per_ray, 1)`
                denoting the opacity of each ray point.
            rays_colors: A tensor of shape `(minibatch, ..., num_points_per_ray, 3)`
                denoting the color of each ray point.
        """
        # We first convert the ray parametrizations to world
        # coordinates with `ray_bundle_to_ray_points`.
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)
        # rays_points_world.shape = [minibatch x ... x 3]

        # For each 3D world coordinate, we obtain its harmonic embedding.
        embeds = self.harmonic_embedding(
            rays_points_world
        )
        # embeds.shape = [minibatch x ... x self.n_harmonic_functions*6]

        # self.mlp maps each harmonic embedding to a latent feature space.
        features = self.mlp(embeds)
        # features.shape = [minibatch x ... x n_hidden_neurons]

        # Finally, given the per-point features,
        # execute the density and color branches.

        rays_densities = self._get_densities(features)
        # rays_densities.shape = [minibatch x ... x 1]
        #import pdb;pdb.set_trace()
       
        # rays_colors.shape = [minibatch x ... x 3]

        return rays_densities
        
    def batched_forward_forPC(
            self,
            threshold=0.1,

            **kwargs,
    ):
        """
         This function is used to allow for memory efficient processing
         of input rays. The input rays are first split to `n_batches`
         chunks and passed through the `self.forward` function one at a time
         in a for loop. Combined with disabling PyTorch gradient caching
         (`torch.no_grad()`), this allows for rendering large batches
         of rays that do not all fit into GPU memory in a single forward pass.
         In our case, batched_forward is used to export a fully-sized render
         of the radiance field for visualization purposes.

         Args:
             ray_bundle: A RayBundle object containing the following variables:
                 origins: A tensor of shape `(minibatch, ..., 3)` denoting the
                     origins of the sampling rays in world coords.
                 directions: A tensor of shape `(minibatch, ..., 3)`
                     containing the direction vectors of sampling rays in world coords.
                 lengths: A tensor of shape `(minibatch, ..., num_points_per_ray)`
                     containing the lengths at which the rays are sampled.
             n_batches: Specifies the number of batches the input rays are split into.
                 The larger the number of batches, the smaller the memory footprint
                 and the lower the processing speed.

         Returns:
             rays_densities: A tensor of shape `(minibatch, ..., num_points_per_ray, 1)`
                 denoting the opacity of each ray point.
             rays_colors: A tensor of shape `(minibatch, ..., num_points_per_ray, 3)`
                 denoting the color of each ray point.

         """

        n_batches = 16
        gridRes = 128
        t = np.linspace(-1, 1, gridRes)

        def cartesianproduct(arr):
            np_tr = np.asarray([[z0, y0, x0] for x0 in arr for y0 in arr for z0 in arr])
            return np_tr

        gridCoords = torch.from_numpy(
            cartesianproduct(t).reshape(gridRes, gridRes, gridRes, 3).astype("float32")).cuda()

        batches2 = torch.chunk(torch.arange(gridCoords.view(-1, 3).shape[0]), n_batches)

        batch_outputs = [
            self.forwardWithPoints(

                gridCoords.view(-1, 3)[batch_idx]
            ) for batch_idx in batches2
        ]

        rays_densities, rays_color = [torch.cat([batch_output[output_i] for batch_output in batch_outputs], dim=0).view(
            *torch.Size((gridRes, gridRes, gridRes)), -1) for output_i in (0, 1)]

        import mcubes
        mvertices, mtriangles = mcubes.marching_cubes(rays_densities[:, :, :, 0].movedim(0, 2).movedim(1, 0).cpu().numpy(), threshold);
        mvertices = (mvertices - 64) / 64
        #import pdb;pdb.set_trace()
        return mvertices,mtriangles
        if True:
            import trimesh

            sampledPoints= trimesh.sample.sample_surface(trimesh.PointCloud(mvertices).convex_hull, 10000)[0]
            from sklearn.neighbors import KDTree
            tree = KDTree(mvertices, leaf_size=2)
            idx1 = tree.query(sampledPoints)[1]
            closeVerts2=mvertices[idx1[:,0]]


            ptIds = torch.where(rays_densities.view(-1, 1) > threshold)[0]

            vertices = gridCoords.view(-1, 3)[ptIds].cpu().numpy()
            from vispc import visualizepoints as vp
            import pdb;pdb.set_trace()
            vfeat = rays_color.view(-1, 12)[ptIds].cpu().numpy()
            tree = KDTree(vertices, leaf_size=1)
            idx1 = tree.query(sampledPoints)[1][:,0]
            closeVerts=vertices[idx1]
            closeFeat=vfeat[idx1]
            
            
            return closeVerts, closeFeat
        #        visualizepoints((vertices - 64) / 64)
        # from sklearn.neighbors import KDTree
        # tree = KDTree(gridCoords.view(-1, 3).cpu().numpy(), leaf_size=2)
        # pind = tree.query(mvertices, k=1)[1][:, 0]
        # mvfeat = rays_color.view(-1, 15)[pind]
        if True:
            a1 = 2*(np.random.rand(10000, 3)-0.5)
            a1 = a1 / np.expand_dims(np.linalg.norm(a1, axis=1), axis=1)
            #a1.dot(np.swapaxes(mvertices, 0, 1)).shape
            normMvert=mvertices / np.expand_dims(np.linalg.norm(mvertices, axis=1), axis=1)
            maxPts=np.argmax(a1.dot(np.swapaxes(10*mvertices, 0, 1))+10*a1.dot(np.swapaxes(normMvert, 0, 1)), axis=1)

        threshold = 0.5

        ptIds = torch.where(rays_densities.view(-1, 1) > threshold)[0]
        vertices = gridCoords.view(-1, 3)[ptIds]
        vfeat = rays_color.view(-1, 15)[ptIds]
        np.save("v1.npy", vertices.cpu().numpy())
        np.save("f1.npy", vfeat.cpu().numpy())
        from vispc import vcol
        vcol(vertices.cpu().numpy(), vfeat[:, 12:].cpu().numpy())
        # import pdb;pdb.set_trace()
        return vertices, vfeat
    def forwardWithPoints(
            self,
            rays_points_world,
            **kwargs,
    ):

        embeds = self.harmonic_embedding(
            rays_points_world
        )
        features = self.mlp(embeds)
        rays_densities = self._get_densities(features)
        #rays_colors = self._get_colors(features, None)
        if self.siren:
          rays_features=self._get_features(rays_points_world, None)
        else:
          rays_features=self._get_features(embeds, None)
        
        return rays_densities, rays_features