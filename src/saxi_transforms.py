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

import src.utils as utils

from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk


# Explanation : file which is a composition of transformations to be applied during training

def Threshold(vtkdata, labels, threshold_min, threshold_max, invert=False):
    
    threshold = vtk.vtkThreshold()
    threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, labels)
    threshold.SetInputData(vtkdata)
    # threshold.ThresholdBetween(threshold_min,threshold_max)
    threshold.SetLowerThreshold(threshold_min)
    threshold.SetUpperThreshold(threshold_max)
    threshold.SetInvert(invert)
    threshold.Update()

    geometry = vtk.vtkGeometryFilter()
    geometry.SetInputData(threshold.GetOutput())
    geometry.Update()
    return geometry.GetOutput()

def ensure_array(arr):
    if arr is not None:
        if hasattr(arr, "__len__"):
            return arr
        else:
            return [arr]
    else:
        return None

# class UnitSurfTransform:
#     def __init__(self, scale_factor=None): #scale_factor=0.13043011372797356
#         self.scale_factor = scale_factor
#     def __call__(self, surf):
#         return utils.GetUnitSurf(surf)

class UnitSurfTransform:
    def __init__(self, scale_factor=None):
        self.scale_factor = scale_factor

    def __call__(self, surf):
        if isinstance(surf, torch.Tensor):
            return utils.GetUnitSurfT(surf)
        else:
            return utils.GetUnitSurf(surf)
            

# class RandomRotation:
#     def __call__(self, surf):
#         surf, _a, _v = utils.RandomRotation(surf)
#         return surf

class RandomRotation:
    def __call__(self, surf):
        if isinstance(surf, torch.Tensor):
            surf, _a, _v = utils.RandomRotationT(surf)      
        else:
            surf, _a, _v = utils.RandomRotation(surf)
        return surf





class RandomRemoveLabeledRegionTransform:

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
    def __init__(self, scale_factor=None):
        self.eval_transform = transforms.Compose(
            [
                UnitSurfTransform(scale_factor=scale_factor)
            ]
        )

    def __call__(self, surf):
        return self.eval_transform(surf)
