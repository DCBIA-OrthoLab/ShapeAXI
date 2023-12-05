from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence as pack_sequence, pad_packed_sequence as unpack_sequence
import pytorch_lightning as pl
from torchvision import transforms
import sys
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
import random
from torch.nn.functional import normalize
import nibabel as nib
from fsl.data import gifti
from tqdm import tqdm
from sklearn.utils import class_weight
from pl_bolts.transforms.dataset_normalizations import (
    imagenet_normalization
)
import platform
system = platform.system()
if system == 'Windows':
  code_path = '\\'.join(os.path.dirname(os.path.abspath(__file__)).split('\\')[:-1])
else:
  code_path = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1])
sys.path.append(code_path)


from . import utils, post_process

#File which manages the dataset for saxi

class SaxiDataset(Dataset):
    #This class is designed to make it easier to work with 3D surface data stored in files
    #It provides methods for loading and preprocessing the data and allows for flexible configurations depending on the specific use case
    def __init__(self, df, mount_point="./", transform=None, surf_column="surf", surf_property=None, class_column=None, scalar_column=None, **kwargs):
        self.df = df
        self.mount_point = mount_point
        self.transform = transform
        self.surf_column = surf_column
        self.surf_property = surf_property
        self.class_column = class_column

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        #Get item function for the dataset
        surf = self.getSurf(idx)

        if self.transform:
            surf = self.transform(surf)
    
        surf = utils.ComputeNormals(surf)
        color_normals = torch.tensor(vtk_to_numpy(utils.GetColorArray(surf, "Normals"))).to(torch.float32)/255.0
        verts = utils.PolyDataToTensors(surf)[0]
        faces = utils.PolyDataToTensors(surf)[1]

        if self.surf_property:            
            faces_pid0 = faces[:,0:1]
            surf_point_data = surf.GetPointData().GetScalars(self.surf_property)
            surf_point_data = torch.tensor(vtk_to_numpy(surf_point_data)).to(torch.float32)            
            surf_point_data_faces = torch.take(surf_point_data, faces_pid0)            
            surf_point_data_faces[surf_point_data_faces==-1] = 33            
            return verts, faces, surf_point_data_faces, color_normals

        if self.class_column:
            cl = torch.tensor(self.df.iloc[idx][self.class_column], dtype=torch.int64)
            return verts, faces, color_normals, cl

        return verts, faces, color_normals

    def getSurf(self, idx):
        surf_path = f'{self.mount_point}/{self.df.iloc[idx][self.surf_column]}'
        return utils.ReadSurf(surf_path)



class SaxiDataModule(pl.LightningDataModule):
    #It provides a structured and configurable way to load, preprocess, and organize 3D surface data for machine learning tasks, based on the specific requirements of the model type
    def __init__(self, df_train, df_val, df_test, mount_point="./", batch_size=256, num_workers=4, surf_column="surf", class_column='Classification', model='SaxiClassification', surf_property=None, scalar_column=None, train_transform=None, valid_transform=None, test_transform=None, drop_last=False):
        super().__init__()
        self.df_train = df_train
        self.df_val = df_val   
        self.df_test = df_test     
        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.surf_column = surf_column
        self.class_column = class_column
        self.scalar_column = scalar_column
        self.surf_property = surf_property        
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform
        self.drop_last = drop_last
        self.model = model

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        self.train_ds = SaxiDataset(self.df_train, self.mount_point, surf_column=self.surf_column, surf_property=self.surf_property, class_column=self.class_column, scalar_column=self.scalar_column, transform=self.train_transform)
        self.val_ds = SaxiDataset(self.df_val, self.mount_point, surf_column=self.surf_column, surf_property=self.surf_property, class_column=self.class_column, scalar_column=self.scalar_column, transform=self.valid_transform)
        self.test_ds = SaxiDataset(self.df_test, self.mount_point, surf_column=self.surf_column, surf_property=self.surf_property, class_column=self.class_column, scalar_column=self.scalar_column, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, collate_fn=self.pad_verts_faces)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, collate_fn=self.pad_verts_faces)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=1, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, collate_fn=self.pad_verts_faces)

    def pad_verts_faces(self, batch):
        # Collate function for the dataloader to know how to comine the data
        if self.model == 'SaxiClassification' or self.model == 'SaxiRegression':
            verts = [v for v, f, cn, l in batch]
            faces = [f for v, f, cn, l in batch]        
            color_normals = [cn for v, f, cn, l in batch]
            labels = [l for v, f, cn, l in batch]  
            
            verts = pad_sequence(verts, batch_first=True, padding_value=0.0)        
            faces = pad_sequence(faces, batch_first=True, padding_value=-1)        
            color_normals = pad_sequence(color_normals, batch_first=True, padding_value=0.0)
            labels = torch.tensor(labels)
            
            return verts, faces, color_normals, labels

        elif self.model == 'SaxiSegmentation':
            verts = [v for v, f, vdf, cn in batch]
            faces = [f for v, f, vdf, cn in batch]        
            verts_data_faces = [vdf for v, f, vdf, cn in batch]        
            color_normals = [cn for v, f, vdf, cn in batch]        
            
            verts = pad_sequence(verts, batch_first=True, padding_value=0.0)        
            faces = pad_sequence(faces, batch_first=True, padding_value=-1)
            verts_data_faces = torch.cat(verts_data_faces)
            color_normals = pad_sequence(color_normals, batch_first=True, padding_value=0.0)

            return verts, faces, verts_data_faces, color_normals


class SaxiTimepointsDataset(Dataset):
    # It is designed to handle time series data, where each time point represents 3D surface data.
    # It provides methods for loading and preprocessing the data for individual time points, allowing for flexible configurations depending on the specific use case, such as handling different surface properties or class labels.
    def __init__(self, df, mount_point = "./", transform=None, surf_column="surf", surf_property=None, class_column=None, scalar_column=None, **kwargs):
        self.df = df
        self.mount_point = mount_point
        self.transform = transform
        self.surf_column = surf_column
        self.surf_property = surf_property
        self.class_column = class_column

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        # Get specific item function for the dataset
        surf = self.getSurf(idx)

        if self.transform:
            surf = self.transform(surf)

        surf = utils.ComputeNormals(surf)
        color_normals = torch.tensor(vtk_to_numpy(utils.GetColorArray(surf, "Normals"))).to(torch.float32)/255.0
        verts = utils.PolyDataToTensors(surf)[0]
        faces = utils.PolyDataToTensors(surf)[1]

        if self.surf_property:            

            faces_pid0 = faces[:,0:1]
            surf_point_data = surf.GetPointData().GetScalars(self.surf_property)
            
            surf_point_data = torch.tensor(vtk_to_numpy(surf_point_data)).to(torch.float32)            
            surf_point_data_faces = torch.take(surf_point_data, faces_pid0)            

            surf_point_data_faces[surf_point_data_faces==-1] = 33            

            return verts, faces, surf_point_data_faces, color_normals

        if self.class_column:
            cl = torch.tensor(self.df.iloc[idx][self.class_column], dtype=torch.int64)
            return verts, faces, color_normals, cl

        return verts, faces, color_normals

    def getSurf(self, idx):
        # Load the 3D data surface
        surf_path = f'{self.mount_point}/{self.df.iloc[idx][self.surf_column]}'
        return utils.ReadSurf(surf_path)



class BrainIBISDataset(Dataset):
    def __init__(self,df,list_demographic,list_path_ico,transform = None,version=None,column_subject_ID='Subject_ID',column_age='Age',name_class ='ASD_administered'):
        self.df = df
        self.list_demographic = list_demographic
        self.list_path_ico = list_path_ico
        self.transform = transform
        self.version = version
        self.column_subject_ID = column_subject_ID
        self.column_age = column_age
        self.name_class = name_class

    def __len__(self):
        return(len(self.df)) 

    def __getitem__(self,idx):
        #Get item for each hemisphere (left and right)
        vertsL, facesL, vertex_featuresL, face_featuresL,demographic,Y = self.getitem_per_hemisphere('left', idx)
        vertsR, facesR, vertex_featuresR, face_featuresR,demographic,Y = self.getitem_per_hemisphere('right', idx)
        return  vertsL, facesL, vertex_featuresL, face_featuresL, vertsR, facesR, vertex_featuresR, face_featuresR,demographic, Y 
    
    def data_to_tensor(self,path):
        data = open(path,"r").read().splitlines()
        data = torch.tensor([float(ele) for ele in data])
        return data

    def getitem_per_hemisphere(self,hemisphere,idx):
        #Load Data
        row = self.df.loc[idx]
        path_left_eacsf = row['PathLeftEACSF']
        path_right_eacsf = row['PathRightEACSF']
        path_left_sa = row['PathLeftSa']
        path_right_sa = row['PathRightSa']
        path_left_thickness = row['PathLeftThickness']
        path_right_thickness = row['PathRightThickness']

        l_features = []

        if hemisphere == 'left':
            l_features.append(self.data_to_tensor(path_left_eacsf).unsqueeze(dim=1))
            l_features.append(self.data_to_tensor(path_left_sa).unsqueeze(dim=1))
            l_features.append(self.data_to_tensor(path_left_thickness).unsqueeze(dim=1))
        else:
            l_features.append(self.data_to_tensor(path_right_eacsf).unsqueeze(dim=1))
            l_features.append(self.data_to_tensor(path_right_sa).unsqueeze(dim=1))
            l_features.append(self.data_to_tensor(path_right_thickness).unsqueeze(dim=1))

        vertex_features = torch.cat(l_features,dim=1)

        #Demographics
        demographic_values = [float(row[name]) for name in self.list_demographic]
        demographic = torch.tensor(demographic_values)

        #Y
        Y = torch.tensor([int(row[self.name_class])])

        #Load  Icosahedron

        if hemisphere == 'left':
            reader = utils.ReadSurf(self.list_path_ico[0])
        else:
            reader = utils.ReadSurf(self.list_path_ico[1])
        verts, faces, edges = utils.PolyDataToTensors(reader)

        nb_faces = len(faces)

        #Transformations
        if self.transform:        
            verts = self.transform(verts)

        #Face Features
        faces_pid0 = faces[:,0:1]         
    
        offset = torch.zeros((nb_faces,vertex_features.shape[1]), dtype=int) + torch.Tensor([i for i in range(vertex_features.shape[1])]).to(torch.int64)
        faces_pid0_offset = offset + torch.multiply(faces_pid0, vertex_features.shape[1])      
        
        face_features = torch.take(vertex_features,faces_pid0_offset)

        return verts, faces,vertex_features,face_features,demographic, Y



class BrainIBISDataModule(pl.LightningDataModule):
    def __init__(self,batch_size,list_demographic,data_train,data_val,data_test,list_path_ico,train_transform=None,val_and_test_transform=None, num_workers=6, pin_memory=False, persistent_workers=False,name_class='ASD_administered'):
        super().__init__()
        self.batch_size = batch_size 
        self.list_demographic = list_demographic
        self.data_train = data_train
        self.data_val = data_val
        self.data_test = data_test
        self.list_path_ico = list_path_ico
        self.train_transform = train_transform
        self.val_and_test_transform = val_and_test_transform
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.name_class = name_class

        ### weights computing
        self.weights = []
        self.df_train = pd.read_csv(self.data_train)
        self.df_val = pd.read_csv(self.data_val)
        self.df_test = pd.read_csv(self.data_test)
        self.weights = self.class_weights()

        self.setup()

    
    def class_weights(self):
        class_weights_train = self.compute_class_weights(self.data_train)
        class_weights_val = self.compute_class_weights(self.data_val)
        class_weights_test = self.compute_class_weights(self.data_test)
        return [class_weights_train, class_weights_val, class_weights_test]

    def compute_class_weights(self, data_file):
        df = pd.read_csv(data_file)
        y = np.array(df.loc[:, self.name_class])
        labels = np.unique(y)
        class_weights = torch.tensor(class_weight.compute_class_weight('balanced', classes=labels, y=y)).to(torch.float32)
        return class_weights

    def setup(self,stage=None):
        # Assign train/val datasets for use in dataloaders
        self.train_dataset = BrainIBISDataset(self.df_train,self.list_demographic,self.list_path_ico,self.train_transform)
        self.val_dataset = BrainIBISDataset(self.df_val,self.list_demographic,self.list_path_ico,self.val_and_test_transform)
        self.test_dataset = BrainIBISDataset(self.df_test,self.list_demographic,self.list_path_ico,self.val_and_test_transform)

        VL, FL, VFL, FFL,VR, FR, VFR, FFR, demographic, Y = self.train_dataset.__getitem__(0)
        self.nbr_features = VFL.shape[1]
        self.nbr_demographic = demographic.shape[0]

    def train_dataloader(self):    
        return DataLoader(self.train_dataset,batch_size=self.batch_size,shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=self.persistent_workers, drop_last=True)

    def repeat_subject(self,df,final_size):
        n = len(df)
        q,r = final_size//n,final_size%n
        list_df = [df for i in range(q)]
        list_df.append(df[:r])
        new_df = pd.concat(list_df).reset_index().drop(['index'],axis=1)
        return new_df

    def val_dataloader(self):
        return DataLoader(self.val_dataset,batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=self.persistent_workers, drop_last=True)        

    def test_dataloader(self):
        return DataLoader(self.test_dataset,batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=self.persistent_workers, drop_last=True)

    def get_features(self):
        return self.nbr_features

    def get_weigths(self):
        return self.weights
    
    def get_nbr_demographic(self):
        return self.nbr_demographic

