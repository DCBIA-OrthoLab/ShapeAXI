import pandas as pd
import numpy as np

import os
import torch

from torchvision import transforms

from pl_bolts.transforms.dataset_normalizations import (
    imagenet_normalization
)

import sys

import platform
system = platform.system()
if system == 'Windows':
  code_path = '\\'.join(os.path.dirname(os.path.abspath(__file__)).split('\\')[:-1])
else:
  code_path = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1])
sys.path.append(code_path)

import utils

from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk


# File which is a composition of transformations to be applied during training

def Threshold(vtkdata, labels, threshold_min, threshold_max, invert=False):
    # Thresholds a vtk data object based on a label array
    threshold = vtk.vtkThreshold()
    threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, labels)
    threshold.SetInputData(vtkdata)
    threshold.SetLowerThreshold(threshold_min)
    threshold.SetUpperThreshold(threshold_max)
    threshold.SetInvert(invert)
    threshold.Update()
    geometry = vtk.vtkGeometryFilter()
    geometry.SetInputData(threshold.GetOutput())
    geometry.Update()

    return geometry.GetOutput()

def ensure_array(arr):
    # Ensures that the input is an array
    if arr is not None:
        if hasattr(arr, "__len__"):
            return arr
        else:
            return [arr]
    else:
        return None

class UnitSurfTransform:
    # This transform is used to make sure that the surface is in the unit cube
    def __init__(self, scale_factor=None):
        self.scale_factor = scale_factor

    def __call__(self, surf):
        if isinstance(surf, torch.Tensor):
            return utils.GetUnitSurfT(surf)
        else:
            return utils.GetUnitSurf(surf)

class UnitSurfTransform2:
    def __init__(self, random_rotation=False):
        self.random_rotation = random_rotation

    def __call__(self, surf):
        surf = utils.GetUnitSurf(surf)
        if self.random_rotation:
            surf, _a, _v = utils.RandomRotation(surf)
        return surf

class RandomRotation:
    # This transform is used to make sure that the surface is in the unit cube
    def __call__(self, surf):
        if isinstance(surf, torch.Tensor):
            surf, _a, _v = utils.RandomRotationT(surf)      
        else:
            surf, _a, _v = utils.RandomRotation(surf)
        return surf

class RandomRemoveLabeledRegionTransform:
    # This transform is used to remove a random labeled region from the surface
    def __init__(self, surf_property = None, random_rotation=False, max_remove=1, exclude=None, prob=0.5):

        self.surf_property = surf_property
        self.random_rotation = random_rotation
        self.max_remove = max_remove
        self.exclude = ensure_array(exclude)
        self.prob = prob

    def __call__(self, surf):

        surf = utils.GetUnitSurf(surf)
        if self.random_rotation:
            surf, _a, _v = utils.RandomRotation(surf)

        if self.surf_property:
            surf_point_data = surf.GetPointData().GetScalars(self.surf_property) 
            # ## Remove crown
            unique, counts  = np.unique(surf_point_data, return_counts = True)

            for i in range(self.max_remove):        
                id_to_remove = np.random.choice(unique[:-1])   

                if np.random.rand() > self.prob and (self.exclude is not None and id_to_remove not in self.exclude):
                    surf = Threshold(surf, self.surf_property ,id_to_remove-0.5,id_to_remove+0.5, invert=True)        
            return surf

class TrainTransform:
    # This transform is used to make sure that the surface is in the unit cube
    def __init__(self, scale_factor=None):
        self.train_transform = transforms.Compose(
            [
                UnitSurfTransform(scale_factor=scale_factor),
                RandomRotation()
            ]
        )

    def __call__(self, surf):
        return self.train_transform(surf)


class EvalTransform:
    # This transform is used to make sure that the surface is in the unit cube
    def __init__(self, scale_factor=None):
        self.eval_transform = transforms.Compose(
            [
                UnitSurfTransform(scale_factor=scale_factor)
            ]
        )

    def __call__(self, surf):
        return self.eval_transform(surf)


class RandomRemoveTeethTransform:
    # This transform is used to remove teeth from the surface (used for the SaxiSegmentation model)
    def __init__(self, surf_property = None, random_rotation=False, max_remove=4):

        self.surf_property = surf_property
        self.random_rotation = random_rotation
        self.max_remove = max_remove

    def __call__(self, surf):

        surf = utils.GetUnitSurf(surf)
        
        if self.random_rotation:
            surf, _a, _v = utils.RandomRotation(surf)

        if self.surf_property:
            surf_point_data = surf.GetPointData().GetScalars(self.surf_property)
            # Remove crown
            unique, counts  = np.unique(surf_point_data, return_counts = True)

            for i in range(self.max_remove):        
                id_to_remove = np.random.choice(unique[:-1])            
                if id_to_remove not in [1,16,17,32] and np.random.rand() > 0.5:
                    surf = post_process.Threshold(surf, self.surf_property ,id_to_remove-0.5,id_to_remove+0.5, invert=True)        
            return surf
