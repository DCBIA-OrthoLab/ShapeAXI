#CUDA_VISIBLE_DEVICES=0

import os 
import sys
sys.path.insert(0, './Classification')
sys.path.insert(0, './../Classification')

import numpy as np
import cv2

import torch
from torch import nn
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50

import monai
import pandas as pd


from net import IcoConvNet
from data import BrainIBISDataModule

from transformation import RandomRotationTransform, GaussianNoisePointTransform, NormalizePointTransform, CenterTransform


import numpy as np
import random
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pytorch_lightning as pl 

from vtk.util.numpy_support import vtk_to_numpy
import vtk

import nibabel as nib
from fsl.data import gifti
from tqdm import tqdm
from sklearn.utils import class_weight

import utils
from utils import ReadSurf, PolyDataToTensors

import pandas as pd

import plotly.express as pd

# datastructures
from pytorch3d.structures import Meshes

# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRendererWithFragments, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, SoftPhongShader, AmbientLights, PointLights, TexturesUV, TexturesVertex,
)
from pytorch3d.vis.plotly_vis import plot_scene

class Classification_for_left_path(nn.Module):
    def __init__(self,classification_layer,xR,demographic):
        super().__init__()
        self.classification_layer = classification_layer
        self.xR = xR
        self.demographic = demographic

    def forward(self,xL):
        l = [xL,self.xR,self.demographic]
        x = torch.cat(l,dim=1)
        x = self.classification_layer(x)
        return x

class Classification_for_right_path(nn.Module):
    def __init__(self,classification_layer,xL,demographic):
        super().__init__()
        self.classification_layer = classification_layer
        self.xL = xL
        self.demographic = demographic

    def forward(self,xR):
        l = [self.xL,xR,self.demographic]
        x = torch.cat(l,dim=1)
        x = self.classification_layer(x)     
        return x       

##############################################################################################Parameters

brain = '107524_V06.csv' #choose your subject
hemisphere = 'left'#'left','right'

experiment = 'Experiment0' #Name of your experiment (in the checkpoint directory)
epoch = 'epoch=0-val_loss=1.71.ckpt' #Name of the epoch (if you have multiple epochs)

pretrained = False #True,False
batch_size = 5
num_workers = 12 
image_size = 224
noise_lvl = 0.01
dropout_lvl = 0.2
num_epochs = 1000
ico_lvl = 2
if ico_lvl == 1:
    radius = 1.76 
elif ico_lvl == 2:
    radius = 1
lr = 1e-4

#parameters for GaussianNoiseTransform
mean = 0
std = 0.01

#parameters for EarlyStopping
min_delta_early_stopping = 0.00
patience_early_stopping = 30

#Paths
path_data = "/ASD/Autism/IBIS/Proc_Data/IBIS_sa_eacsf_thickness"

data_train = "../Data/V06-12.csv"
data_val = "../Data/V06-12.csv"
data_test = "../Data/V06-12.csv"

path_ico_left = '../3DObject/sphere_f327680_v163842.vtk'
path_ico_right = '../3DObject/sphere_f327680_v163842.vtk'
list_path_ico = [path_ico_left,path_ico_right]


###Demographics
list_demographic = ['Gender','MRI_Age','AmygdalaLeft','HippocampusLeft','LatVentsLeft','ICV','Crbm_totTissLeft','Cblm_totTissLeft','AmygdalaRight','HippocampusRight','LatVentsRight','Crbm_totTissRight','Cblm_totTissRight']#MLR
#List of used demographics 

###Transformation
list_train_transform = []    
list_train_transform.append(CenterTransform())
list_train_transform.append(NormalizePointTransform())
list_train_transform.append(RandomRotationTransform())
list_train_transform.append(GaussianNoisePointTransform(mean,std))
list_train_transform.append(NormalizePointTransform())

train_transform = monai.transforms.Compose(list_train_transform)

list_val_and_test_transform = []    
list_val_and_test_transform.append(CenterTransform())
list_val_and_test_transform.append(NormalizePointTransform())

val_and_test_transform = monai.transforms.Compose(list_val_and_test_transform)

###Layer
Layer = 'IcoConv2D' #'Att','IcoConv2D','IcoConv1D','IcoLinear'
#Choose between these 4 choices to choose what kind of model you want to use. 

##############################################################################################

###Get number of images
list_nb_verts_ico = [12,42]
nb_images = list_nb_verts_ico[ico_lvl-1]


#device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#Creation of Dataset
brain_data = BrainIBISDataModule(batch_size,list_demographic,path_data,data_train,data_val,data_test,list_path_ico,train_transform = train_transform,val_and_test_transform =val_and_test_transform,num_workers=num_workers)#MLR
nbr_features = brain_data.get_features()
weights = brain_data.get_weigths()
nbr_demographic = brain_data.get_nbr_demographic()
nbr_brain = brain_data.test_dataset.__len__()#
brain_data.test_dataloader()

#Load model
path_model = '/work/ugor/source/IcoConvNet-classification/Checkpoint/'+experiment+'/'+epoch
model = IcoConvNet(Layer,pretrained,nbr_features,nbr_demographic,dropout_lvl,image_size,noise_lvl,ico_lvl,batch_size,weights,radius=radius,lr=lr)
checkpoint = torch.load(path_model)
#checkpoint = torch.load(path_model,map_location=torch.device('cpu')) 
model.load_state_dict(checkpoint['state_dict'])
model = model.to(device)


classification_layer = model.Classification
n_targ = 1
targets = [ClassifierOutputTarget(n_targ)]






VL, FL, VFL, FFL, VR, FR, VFR, FFR, demographic, Y = brain_data.test_dataset.__getitem__(0)
VL = VL.unsqueeze(dim=0).to(device)
FL = FL.unsqueeze(dim=0).to(device)
VFL = VFL.unsqueeze(dim=0).to(device)
FFL = FFL.unsqueeze(dim=0).to(device)
VR = VR.unsqueeze(dim=0).to(device)
FR = FR.unsqueeze(dim=0).to(device)
VFR = VFR.unsqueeze(dim=0).to(device)
FFR = FFR.unsqueeze(dim=0).to(device)
demographic = demographic.unsqueeze(dim=0).to(device)


# x = model((VL, FL, VFL, FFL, VR, FR, VFR, FFR,demographic))
# print('Inside correct condition: ',x)

xL, PF = model.render(VL,FL,VFL,FFL)
xR, PF = model.render(VR,FR,VFR,FFR)




if hemisphere == 'left':
    input_tensor_cam = xL
    xR = model.poolingR(model.IcosahedronConv2dR(model.TimeDistributedR(xR))) 
    classifier = Classification_for_left_path(classification_layer,xR,demographic)
    model_cam = nn.Sequential(model.TimeDistributedL, model.IcosahedronConv2dL, model.poolingL,classifier)
else:
    input_tensor_cam = xR
    xL = model.poolingL(model.IcosahedronConv2dL(model.TimeDistributedL(xL))) 
    classifier = Classification_for_right_path(classification_layer,xL,demographic)
    model_cam = nn.Sequential(model.TimeDistributedR, model.IcosahedronConv2dR, model.poolingR,classifier)


target_layers = [model_cam[0].module.layer4[-1]]
cam = GradCAM(model=model_cam, target_layers=target_layers)


grayscale_cam = torch.Tensor(cam(input_tensor=input_tensor_cam, targets=targets))

name_save = 'gradcam.pt'
torch.save(grayscale_cam,'Saved_gradcam/'+name_save)
