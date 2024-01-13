
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
import torch
from pytorch3d.renderer import MonteCarloRaysampler, NDCMultinomialRaysampler, RayBundle
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.renderer.implicit.sample_pdf import sample_pdf


from typing import Callable, Tuple, Union

#from cowrender import image_grid, generate_cow_renders, generate_cow_rendersWithRT
from nutil import sample_images_at_mc_locs

class ImplicitRendererStratified(torch.nn.Module):
    """
    A class for rendering a batch of implicit surfaces. The class should
    be initialized with a raysampler and raymarcher class which both have
    to be a `Callable`.
    VOLUMETRIC_FUNCTION
    The `forward` function of the renderer accepts as input the rendering cameras
    as well as the `volumetric_function` `Callable`, which defines a field of opacity
    and feature vectors over the 3D domain of the scene.
    A standard `volumetric_function` has the following signature::
        def volumetric_function(
            ray_bundle: Union[RayBundle, HeterogeneousRayBundle],
            **kwargs,
        ) -> Tuple[torch.Tensor, torch.Tensor]
    With the following arguments:
        `ray_bundle`: A RayBundle or HeterogeneousRayBundle object
            containing the following variables:
            `origins`: A tensor of shape `(minibatch, ..., 3)` denoting
                the origins of the rendering rays.
            `directions`: A tensor of shape `(minibatch, ..., 3)`
                containing the direction vectors of rendering rays.
            `lengths`: A tensor of shape
                `(minibatch, ..., num_points_per_ray)`containing the
                lengths at which the ray points are sampled.
            `xys`: A tensor of shape
                `(minibatch, ..., 2)` containing the
                xy locations of each ray's pixel in the screen space.
    Calling `volumetric_function` then returns the following:
        `rays_densities`: A tensor of shape
            `(minibatch, ..., num_points_per_ray, opacity_dim)` containing
            the an opacity vector for each ray point.
        `rays_features`: A tensor of shape
            `(minibatch, ..., num_points_per_ray, feature_dim)` containing
            the an feature vector for each ray point.
    Note that, in order to increase flexibility of the API, we allow multiple
    other arguments to enter the volumetric function via additional
    (optional) keyword arguments `**kwargs`.
    A typical use-case is passing a `CamerasBase` object as an additional
    keyword argument, which can allow the volumetric function to adjust its
    outputs based on the directions of the projection rays.
    Example:
        A simple volumetric function of a 0-centered
        RGB sphere with a unit diameter is defined as follows::
            def volumetric_function(
                ray_bundle: Union[RayBundle, HeterogeneousRayBundle],
                **kwargs,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                # first convert the ray origins, directions and lengths
                # to 3D ray point locations in world coords
                rays_points_world = ray_bundle_to_ray_points(ray_bundle)
                # set the densities as an inverse sigmoid of the
                # ray point distance from the sphere centroid
                rays_densities = torch.sigmoid(
                    -100.0 * rays_points_world.norm(dim=-1, keepdim=True)
                )
                # set the ray features to RGB colors proportional
                # to the 3D location of the projection of ray points
                # on the sphere surface
                rays_features = torch.nn.functional.normalize(
                    rays_points_world, dim=-1
                ) * 0.5 + 0.5
                return rays_densities, rays_features
    """

    def __init__(self, raysampler: Callable, raymarcher: Callable,device, rayFreeze=False) -> None:
        """
        Args:
            raysampler: A `Callable` that takes as input scene cameras
                (an instance of `CamerasBase`) and returns a
                RayBundle or HeterogeneousRayBundle, that
                describes the rays emitted from the cameras.
            raymarcher: A `Callable` that receives the response of the
                `volumetric_function` (an input to `self.forward`) evaluated
                along the sampled rays, and renders the rays with a
                ray-marching algorithm.
        """
        super().__init__()

        if not callable(raysampler):
            raise ValueError('"raysampler" has to be a "Callable" object.')
        if not callable(raymarcher):
            raise ValueError('"raymarcher" has to be a "Callable" object.')

        self.raysampler = raysampler
        self.raymarcher = raymarcher
        self.device = device
        self.rayFreeze= rayFreeze
        self.rayState="Empty"
        self.frozenRays=False
    def getWeights(
        self, cameras: CamerasBase, volumetric_function: Callable, **kwargs
    ) -> Tuple[torch.Tensor, RayBundle]:
    # ) -> Tuple[torch.Tensor, Union[RayBundle, HeterogeneousRayBundle]]:
       # Ray BUndle heterogenous undalsindi ignored
        """
        Render a batch of images using a volumetric function
        represented as a callable (e.g. a Pytorch module).
        Args:
            cameras: A batch of cameras that render the scene. A `self.raysampler`
                takes the cameras as input and samples rays that pass through the
                domain of the volumetric function.
            volumetric_function: A `Callable` that accepts the parametrizations
                of the rendering rays and returns the densities and features
                at the respective 3D of the rendering rays. Please refer to
                the main class documentation for details.
        Returns:
            images: A tensor of shape `(minibatch, ..., feature_dim + opacity_dim)`
                containing the result of the rendering.
            ray_bundle: A `Union[RayBundle, HeterogeneousRayBundle]` containing
                the parametrizations of the sampled rendering rays.
        """

        if not callable(volumetric_function):
            raise ValueError('"volumetric_function" has to be a "Callable" object.')

        ray_bundle = self.raysampler(
                cameras=cameras, volumetric_function=volumetric_function, **kwargs
        )
        rays_densities, rays_features = volumetric_function(
            ray_bundle=ray_bundle, cameras=cameras, **kwargs
        )
        # ray_densities - minibatch x ... x n_pts_per_ray x density_dim
        # ray_features - minibatch x ... x n_pts_per_ray x feature_dim

        # finally, march along the sampled rays to obtain the renders

        rays_densities = rays_densities[..., 0]

        eps = 1e-10
        surface_thickness = 1
        absorption = _shifted_cumprod(
            (1.0 + eps) - rays_densities, shift=surface_thickness
        )
        weights = rays_densities * absorption
        del absorption
        
        return ray_bundle, weights

    def forward(
        self, cameras: CamerasBase, volumetric_function: Callable, stratified=True, depth=False,add_input_samples=False,maskRays=False,mask=False,**kwargs
    ) -> Tuple[torch.Tensor, RayBundle]:
    # ) -> Tuple[torch.Tensor, Union[RayBundle, HeterogeneousRayBundle]]:
       # Ray BUndle heterogenous undalsindi ignored
        """
        Render a batch of images using a volumetric function
        represented as a callable (e.g. a Pytorch module).
        Args:
            cameras: A batch of cameras that render the scene. A `self.raysampler`
                takes the cameras as input and samples rays that pass through the
                domain of the volumetric function.
            volumetric_function: A `Callable` that accepts the parametrizations
                of the rendering rays and returns the densities and features
                at the respective 3D of the rendering rays. Please refer to
                the main class documentation for details.
        Returns:
            images: A tensor of shape `(minibatch, ..., feature_dim + opacity_dim)`
                containing the result of the rendering.
            ray_bundle: A `Union[RayBundle, HeterogeneousRayBundle]` containing
                the parametrizations of the sampled rendering rays.
        """

        if not callable(volumetric_function):
            raise ValueError('"volumetric_function" has to be a "Callable" object.')


        # pyre-fixme[23]: Unable to unpack `object` into 2 values.
        
        if self.rayState=="Empty" : 
            if self.rayFreeze:
              self.rayState="Occupied"
            if stratified:
                    with torch.no_grad():
                        coarseRay_bundle,weights=self.getWeights(cameras=cameras, volumetric_function=volumetric_function)
    
                        if maskRays:
                            maskVals=torch.where(sample_images_at_mc_locs(mask.cuda(), coarseRay_bundle.xys)[...,0])
                            coarseRay_bundle=RayBundle(origins=coarseRay_bundle.origins[maskVals].unsqueeze(0),
                                                           directions=coarseRay_bundle.directions[maskVals].unsqueeze(0),
                                                           lengths=coarseRay_bundle.lengths[maskVals].unsqueeze(0),
                                                           xys=coarseRay_bundle.xys[maskVals].unsqueeze(0), )
                            weights=weights[maskVals].unsqueeze(0)
                        
                        psampler = ProbabilisticRaysampler(weights.shape[-1],stratified=True, stratified_test=False,add_input_samples=add_input_samples)
                        if len(weights.shape)==4:
                            bsz=weights.shape[0]
                            ray_bundle=RayBundle( origins=coarseRay_bundle.origins.view(bsz,-1,3),directions=coarseRay_bundle.directions.view(bsz,-1,3),
                                                          lengths=coarseRay_bundle.lengths.view(bsz,-1,weights.shape[-1]),xys=coarseRay_bundle.xys.view(bsz,-1,2),)
                            ray_bundle=psampler(ray_bundle,weights)
                            ray_bundle = RayBundle(origins=ray_bundle.origins.view(coarseRay_bundle.origins.shape),
                                                           directions=ray_bundle.directions.view(coarseRay_bundle.directions.shape),
                                                           lengths=ray_bundle.lengths.view(coarseRay_bundle.lengths.shape),
                                                           xys=ray_bundle.xys.view(coarseRay_bundle.xys.shape), )
                        else:
                            ray_bundle = psampler(coarseRay_bundle, weights)
            else:
              with torch.no_grad():
                coarseRay_bundle,weights=self.getWeights(cameras=cameras, volumetric_function=volumetric_function)
                if maskRays:
                            maskVals=torch.where(sample_images_at_mc_locs(mask.to(self.device), coarseRay_bundle.xys)[...,0])
                            ray_bundle=RayBundle(origins=coarseRay_bundle.origins[maskVals].unsqueeze(0),
                                                           directions=coarseRay_bundle.directions[maskVals].unsqueeze(0),
                                                           lengths=coarseRay_bundle.lengths[maskVals].unsqueeze(0),
                                                           xys=coarseRay_bundle.xys[maskVals].unsqueeze(0), )
                            weights=weights[maskVals].unsqueeze(0)
                else:
                  ray_bundle=coarseRay_bundle
            self.frozenRays=ray_bundle    
            #import pdb;pdb.set_trace()
        rays_densities, rays_features = volumetric_function(
                ray_bundle=self.frozenRays, cameras=cameras, **kwargs
            )
        # ray_densities - minibatch x ... x n_pts_per_ray x density_dim
        # ray_features - minibatch x ... x n_pts_per_ray x feature_dim

        # finally, march along the sampled rays to obtain the renders
        if depth:
            images, weights = self.raymarcher(
                rays_densities=rays_densities,
                rays_features=rays_features,
                ray_bundle=self.frozenRays,
                **kwargs,
            )
            # images - minibatch x ... x (feature_dim + opacity_dim)
            normLengths=(ray_bundle.lengths-ray_bundle.lengths[...,0].unsqueeze(-1)).cpu().view(-1,ray_bundle.lengths.shape[-1])
            normLengths=normLengths/normLengths.max()
            argPt=torch.argmax(weights.view(-1,ray_bundle.lengths.shape[-1]),dim=1)

            depthMap=normLengths[torch.arange(argPt.shape[0]), argPt].view(ray_bundle.lengths.shape[:-1]).unsqueeze(-1)
            import pdb;pdb.set_trace()
            return images, ray_bundle, depthMap
        else:
            images, weights = self.raymarcher(
                rays_densities=rays_densities,
                rays_features=rays_features,
                ray_bundle=self.frozenRays,
                **kwargs,
            )
            return images, self.frozenRays, weights


class EmissionAbsorptionRaymarcherStratified(torch.nn.Module):
    """
    Raymarch using the Emission-Absorption (EA) algorithm.
    The algorithm independently renders each ray by analyzing density and
    feature values sampled at (typically uniformly) spaced 3D locations along
    each ray. The density values `rays_densities` are of shape
    `(..., n_points_per_ray)`, their values should range between [0, 1], and
    represent the opaqueness of each point (the higher the less transparent).
    The feature values `rays_features` of shape
    `(..., n_points_per_ray, feature_dim)` represent the content of the
    point that is supposed to be rendered in case the given point is opaque
    (i.e. its density -> 1.0).
    EA first utilizes `rays_densities` to compute the absorption function
    along each ray as follows::
        absorption = cumprod(1 - rays_densities, dim=-1)
    The value of absorption at position `absorption[..., k]` specifies
    how much light has reached `k`-th point along a ray since starting
    its trajectory at `k=0`-th point.
    Each ray is then rendered into a tensor `features` of shape `(..., feature_dim)`
    by taking a weighed combination of per-ray features `rays_features` as follows::
        weights = absorption * rays_densities
        features = (rays_features * weights).sum(dim=-2)
    Where `weights` denote a function that has a strong peak around the location
    of the first surface point that a given ray passes through.
    Note that for a perfectly bounded volume (with a strictly binary density),
    the `weights = cumprod(1 - rays_densities, dim=-1) * rays_densities`
    function would yield 0 everywhere. In order to prevent this,
    the result of the cumulative product is shifted `self.surface_thickness`
    elements along the ray direction.
    """

    def __init__(self, surface_thickness: int = 1,thresholdMode=False) -> None:
        """
        Args:
            surface_thickness: Denotes the overlap between the absorption
                function and the density function.
        """
        super().__init__()
        self.surface_thickness = surface_thickness
        self.thresholdMode=thresholdMode

    def forward(
        self,
        rays_densities: torch.Tensor,
        rays_features: torch.Tensor,
        eps: float = 1e-10,

        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            rays_densities: Per-ray density values represented with a tensor
                of shape `(..., n_points_per_ray, 1)` whose values range in [0, 1].
            rays_features: Per-ray feature values represented with a tensor
                of shape `(..., n_points_per_ray, feature_dim)`.
            eps: A lower bound added to `rays_densities` before computing
                the absorption function (cumprod of `1-rays_densities` along
                each ray). This prevents the cumprod to yield exact 0
                which would inhibit any gradient-based learning.
        Returns:
            features_opacities: A tensor of shape `(..., feature_dim+1)`
                that concatenates two tensors along the last dimension:
                    1) features: A tensor of per-ray renders
                        of shape `(..., feature_dim)`.
                    2) opacities: A tensor of per-ray opacity values
                        of shape `(..., 1)`. Its values range between [0, 1] and
                        denote the total amount of light that has been absorbed
                        for each ray. E.g. a value of 0 corresponds to the ray
                        completely passing through a volume. Please refer to the
                        `AbsorptionOnlyRaymarcher` documentation for the
                        explanation of the algorithm that computes `opacities`.
        """
        _check_raymarcher_inputs(
            rays_densities,
            rays_features,
            None,
            z_can_be_none=True,
            features_can_be_none=False,
            density_1d=True,
        )
        _check_density_bounds(rays_densities)
        rays_densities = rays_densities[..., 0]
        
        
        if self.thresholdMode:
            c1 = rays_densities*0
            c1[torch.where(rays_densities > 0.05)] = 1
            rays_densities = c1
        absorption = _shifted_cumprod(
            (1.0 + eps) - rays_densities, shift=self.surface_thickness
        )
        weights = rays_densities * absorption
        features = (weights[..., None] * rays_features).sum(dim=-2)
        opacities = 1.0 - torch.prod(1.0 - rays_densities, dim=-1, keepdim=True)
        #if self.backMode:
        del absorption
        #import pdb;pdb.set_trace()
        absorption2 = _shifted_cumprod(
            (1.0 + eps) - rays_densities.flip(-1), shift=self.surface_thickness
        )
        weights2 = rays_densities * absorption2.flip(-1)
          #features = (weights[..., None] * rays_features).sum(dim=-2)
          #opacities = 1.0 - torch.prod(1.0 - rays_densities, dim=-1, keepdim=True)
        
        return torch.cat((features, opacities), dim=-1), torch.cat([weights, weights2],dim=-1)


class ProbabilisticRaysampler(torch.nn.Module):
    """
    Implements the importance sampling of points along rays.
    The input is a `RayBundle` object with a `ray_weights` tensor
    which specifies the probabilities of sampling a point along each ray.
    This raysampler is used for the fine rendering pass of NeRF.
    As such, the forward pass accepts the RayBundle output by the
    raysampling of the coarse rendering pass. Hence, it does not
    take cameras as input.
    """

    def __init__(
        self,
        n_pts_per_ray: int,
        stratified: bool,
        stratified_test: bool,
        add_input_samples: bool = True,
    ):
        """
        Args:
            n_pts_per_ray: The number of points to sample along each ray.
            stratified: If `True`, the input `ray_weights` are assumed to be
                sampled at equidistant intervals.
            stratified_test: Same as `stratified` with the difference that this
                setting is applied when the module is in the `eval` mode
                (`self.training==False`).
            add_input_samples: Concatenates and returns the sampled values
                together with the input samples.
        """
        super().__init__()
        self._n_pts_per_ray = n_pts_per_ray
        self._stratified = stratified
        self._stratified_test = stratified_test
        self._add_input_samples = add_input_samples

    def forward(
        self,
        input_ray_bundle: RayBundle,
        ray_weights: torch.Tensor,
        **kwargs,
    ) -> RayBundle:
        """
        Args:
            input_ray_bundle: An instance of `RayBundle` specifying the
                source rays for sampling of the probability distribution.
            ray_weights: A tensor of shape
                `(..., input_ray_bundle.legths.shape[-1])` with non-negative
                elements defining the probability distribution to sample
                ray points from.
        Returns:
            ray_bundle: A new `RayBundle` instance containing the input ray
                points together with `n_pts_per_ray` additional sampled
                points per ray.
        """

        # Calculate the mid-points between the ray depths.
        z_vals = input_ray_bundle.lengths
        batch_size = z_vals.shape[0]

        # Carry out the importance sampling.
        with torch.no_grad():
            z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(
                z_vals_mid.view(-1, z_vals_mid.shape[-1]),
                ray_weights.view(-1, ray_weights.shape[-1])[..., 1:-1],
                self._n_pts_per_ray,
                det=not (
                    (self._stratified and self.training)
                    or (self._stratified_test and not self.training)
                ),
            ).view(batch_size, z_vals.shape[1], self._n_pts_per_ray)

        if self._add_input_samples:
            # Add the new samples to the input ones.
            z_vals = torch.cat((z_vals, z_samples), dim=-1)
        else:
            z_vals = z_samples
        # Resort by depth.
        z_vals, _ = torch.sort(z_vals, dim=-1)

        return RayBundle(
            origins=input_ray_bundle.origins,
            directions=input_ray_bundle.directions,
            lengths=z_vals,
            xys=input_ray_bundle.xys,
        )
