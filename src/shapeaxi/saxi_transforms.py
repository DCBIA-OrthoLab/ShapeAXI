import pandas as pd
import numpy as np
import os
import torch
from torch import nn
import pytorch3d
from torchvision import transforms
import sys
import platform
system = platform.system()
if system == 'Windows':
  code_path = '\\'.join(os.path.dirname(os.path.abspath(__file__)).split('\\')[:-1])
else:
  code_path = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1])
sys.path.append(code_path)
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
import vtk
from shapeaxi import utils

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
                TriangleFilter(),
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
                UnitSurfTransform(scale_factor=scale_factor),
                TriangleFilter()
            ]
        )

    def __call__(self, surf):
        return self.eval_transform(surf)


class TriangleFilter:
    def __call__(self, surf):
        triangleFilter = vtk.vtkTriangleFilter()
        triangleFilter.SetInputData(surf)

        triangleFilter.Update()

        return triangleFilter.GetOutput()

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


class RotationTransform:
    def __call__(self, verts, rotation_matrix):
        verts = torch.transpose(torch.mm(rotation_matrix,torch.transpose(verts,0,1)),0,1)
        return verts

class RandomRotationTransform:
    def __call__(self, verts):
        rotation_matrix = pytorch3d.transforms.random_rotation()
        rotation_transform = RotationTransform()
        verts = rotation_transform(verts,rotation_matrix)
        return verts

class ApplyRotationTransform:
    def __init__(self):            
        self.rotation_matrix = pytorch3d.transforms.random_rotation()

    def __call__(self, verts):
        rotation_transform = RotationTransform()
        verts = rotation_transform(verts,self.rotation_matrix)
        return verts
    
    def change_rotation(self):
        self.rotation_matrix = pytorch3d.transforms.random_rotation()

class GaussianNoisePointTransform:
    def __init__(self, mean=0.0, std = 0.1):            
        self.mean = mean
        self.std = std

    def __call__(self, verts):
        noise = np.random.normal(loc=self.mean, scale=self.std, size=verts.shape)
        verts = verts + noise
        verts = verts.type(torch.float32)
        return verts

class NormalizePointTransform:
    def __call__(self, verts, scale_factor=1.0):
        bounds_max_v = [0.0] * 3
        v = torch.Tensor(verts)
        bounds = torch.tensor([torch.max(v[:,0]),torch.max(v[:,1]),torch.max(v[:,2])])
        bounds_max_v[0] = bounds[0]
        bounds_max_v[1] = bounds[1]
        bounds_max_v[2] = bounds[2]
        scale_factor = torch.tensor(bounds_max_v)

        verts = torch.multiply(v, 1/scale_factor)
        verts = verts.type(torch.float32)

        return verts


class CenterTransform:
    def __call__(self, verts,mean_arr = None):
        #calculate bounding box
        mean_v = [0.0] * 3

        v = torch.Tensor(verts)

        bounds = torch.tensor([torch.min(v[:,0]),torch.max(v[:,0]),torch.min(v[:,1]),torch.max(v[:,1]),torch.min(v[:,2]),torch.max(v[:,2])])

        mean_v[0] = (bounds[0] + bounds[1])/2.0
        mean_v[1] = (bounds[2] + bounds[3])/2.0
        mean_v[2] = (bounds[4] + bounds[5])/2.0
        
        #centering points of the shape
        mean_arr = torch.tensor(mean_v)

        verts = verts - mean_arr
        return verts 

class AvgPoolImages(nn.Module):
    def __init__(self, nbr_images = 12):
        super().__init__()
        self.nbr_images = nbr_images
        self.avg_pool = nn.AvgPool1d(self.nbr_images)

    def forward(self,x):
        x = x.permute(0,2,1)
        output = self.avg_pool(x)
        output = output.squeeze(dim=2)

        return output

############################################################################### ICOCONV PART ############################################################################################################

class GaussianNoise(nn.Module):
    def __init__(self, mean=0.0, std=0.01):
        super(GaussianNoise, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        if(self.training):
            return x + torch.normal(self.mean, self.std,size=x.shape, device=x.device)*(x!=0) # add noise on sphere (not on background)
        return x
    

