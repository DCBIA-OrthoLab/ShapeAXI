import sys
import os
import numpy as np
import cv2
import torch
from torch import nn
from torchvision.models import resnet50
import monai
import pandas as pd
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


from .utils import ReadSurf, PolyDataToTensors
from .saxi_nets import IcoConv
from .saxi_dataset import BrainIBISDataModule

def get_epoch(name_model):
    epoch = ''
    for char in name_model[6:]:
        if char != '-':
            epoch += char
        else :
            break
    return epoch

##############################################################################################


experiment = 'Experiment0' #Name of your experiment (in the checkpoint directory)
epoch = 'epoch=0-val_loss=1.71.ckpt' #Name of the epoch (if you have multiple epochs)

batch_size = 5
alpha = 1
depth = 2
num_workers = 12 #6-12
image_size = 224
noise_lvl = 0.03
dropout_lvl = 0.2
num_epochs = 100
ico_lvl = 2
if ico_lvl == 1:
    radius = 1.76 
elif ico_lvl == 2:
    radius = 1
lr = 1e-4
pretrained = False #True,False

mean = 0
std = 0.01



#Paths
path_data = "/ASD/Autism/IBIS/Proc_Data/IBIS_sa_eacsf_thickness"

data_train = "../Data/V06-12_train.csv"
data_val = "../Data/V06-12_val.csv"
data_test = "../Data/V06-12_test.csv"

path_ico_left = '../3DObject/sphere_f327680_v163842.vtk'
path_ico_right = '../3DObject/sphere_f327680_v163842.vtk'  


list_path_ico = [path_ico_left,path_ico_right]

###Demographics
list_demographic = ['Gender','MRI_Age','AmygdalaLeft','HippocampusLeft','LatVentsLeft','ICV','Crbm_totTissLeft','Cblm_totTissLeft','AmygdalaRight','HippocampusRight','LatVentsRight','Crbm_totTissRight','Cblm_totTissRight']#MLR


#Transform
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
Layer = 'IcoConv2D' #'Att,'IcoConv2D','IcoConv1D','IcoLinear'
#Choose between these 3 choices to choose what kind of Layer we want to use
##############################################################################################



###Get number of images
list_nb_verts_ico = [12,42]
nb_images = list_nb_verts_ico[ico_lvl-1]

###Dataset
brain_data = BrainIBISDataModule(batch_size,list_demographic,path_data,data_train,data_val,data_test,list_path_ico,train_transform = train_transform,val_and_test_transform =val_and_test_transform,num_workers=num_workers)#MLR
nbr_features = brain_data.get_features()
weights = brain_data.get_weigths()
nbr_demographic = brain_data.get_nbr_demographic()




path_model = '../Checkpoint/'+experiment+'/'+epoch
model = IcoConv(Layer,pretrained,nbr_features,nbr_demographic,dropout_lvl,image_size,noise_lvl,ico_lvl,batch_size,weights,radius=radius,lr=lr)
checkpoint = torch.load(path_model)
dict_state_dict = dict(checkpoint['state_dict'].items())
if 'loss_train.weight' in dict_state_dict.keys():
    model.loss_train.weight = dict_state_dict['loss_train.weight']
    model.loss_val.weight = dict_state_dict['loss_val.weight']
model.load_state_dict(checkpoint['state_dict'])


trainer = Trainer(max_epochs=num_epochs,accelerator="gpu")

trainer.test(model, datamodule=brain_data)

