from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import torch
import subprocess
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence as pack_sequence, pad_packed_sequence as unpack_sequence

import lightning as L
from lightning.pytorch.core import LightningDataModule

from torchvision import transforms
import sys
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
import vtk
import random
import torch.nn.functional as F
from torch.nn.functional import normalize
import nibabel as nib
from fsl.data import gifti
from tqdm import tqdm
from sklearn.utils import class_weight
import platform
import json
system = platform.system()
if system == 'Windows':
  code_path = '\\'.join(os.path.dirname(os.path.abspath(__file__)).split('\\')[:-1])
else:
  code_path = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1])
sys.path.append(code_path)

import monai
from monai.data import DataLoader as monai_DataLoader
from shapeaxi import utils
import shapeaxi.saxi_transforms as saxi_transforms

#####################################################################################################################################################################################
#                                                                                                                                                                                   #
#                                                                     Classification, Regression, Segmentation                                                                      #
#                                                                                                                                                                                   #
#####################################################################################################################################################################################


class SaxiDataset(Dataset):
    #This class is designed to make it easier to work with 3D surface data stored in files
    #It provides methods for loading and preprocessing the data and allows for flexible configurations depending on the specific use case
    def __init__(self, df, mount_point="./", transform=None, surf_column="surf", surf_property=None, class_column=None, scalar_column=None, CN=True, **kwargs):
        self.df = df
        self.mount_point = mount_point
        self.transform = transform
        self.surf_column = surf_column
        self.surf_property = surf_property
        self.class_column = class_column
        self.scalar_column = scalar_column
        self.CN = CN

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        #Get item function for the dataset
        surf = self.getSurf(idx)

        if self.transform:
            surf = self.transform(surf)

        verts, faces = utils.PolyDataToTensors_v_f(surf)

        if self.CN:
            surf = utils.ComputeNormals(surf)
            color_normals = torch.tensor(vtk_to_numpy(utils.GetColorArray(surf, "Normals"))).to(torch.float32)/255.0
        
        if self.surf_property:            
            faces_pid0 = faces[:,0:1]
            surf_point_data = surf.GetPointData().GetScalars(self.surf_property)
            surf_point_data = torch.tensor(vtk_to_numpy(surf_point_data)).to(torch.float32)            
            surf_point_data_faces = torch.take(surf_point_data, faces_pid0)            
            surf_point_data_faces[surf_point_data_faces==-1] = 33            

            if self.CN:                
                return verts, faces, surf_point_data_faces, color_normals
            
            return verts, faces, surf_point_data_faces

        if self.class_column:
            cl = torch.tensor(self.df.iloc[idx][self.class_column], dtype=torch.int64)
            if self.CN:
                return verts, faces, color_normals, cl
            else:
                return verts, faces, cl
            
        if self.scalar_column:
            scalar = torch.tensor(self.df.iloc[idx][self.scalar_column], dtype=torch.float32)
            if self.CN:
                return verts, faces, color_normals, scalar
            else:
                return verts, faces, scalar

        if self.CN:
            return verts, faces, color_normals
        return verts, faces

    def getSurf(self, idx):
        surf_path = os.path.join(self.mount_point, self.df.iloc[idx][self.surf_column])
        return utils.ReadSurf(surf_path)
    def getSurfPath(self, idx):
        return self.df.iloc[idx][self.surf_column]


class SaxiDataModule(LightningDataModule):
    #It provides a structured and configurable way to load, preprocess, and organize 3D surface data for machine learning tasks, based on the specific requirements of the model type
    def __init__(self, df_train, df_val, df_test, mount_point="./", batch_size=256, num_workers=4, surf_column="surf", class_column=None , surf_property=None, scalar_column=None, train_transform=None, valid_transform=None, test_transform=None, drop_last=False):
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

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        self.train_ds = SaxiDataset(self.df_train, self.mount_point, surf_column=self.surf_column, surf_property=self.surf_property, class_column=self.class_column, scalar_column=self.scalar_column, transform=self.train_transform)
        self.val_ds = SaxiDataset(self.df_val, self.mount_point, surf_column=self.surf_column, surf_property=self.surf_property, class_column=self.class_column, scalar_column=self.scalar_column, transform=self.valid_transform)
        self.test_ds = SaxiDataset(self.df_test, self.mount_point, surf_column=self.surf_column, surf_property=self.surf_property, class_column=self.class_column, scalar_column=self.scalar_column, transform=self.test_transform)

    def pad_verts_faces(self, batch):
        # Collate function for the dataloader to know how to comine the data
        if self.class_column or self.scalar_column:
            verts = [v for v, f, cn, l in batch]
            faces = [f for v, f, cn, l in batch]        
            color_normals = [cn for v, f, cn, l in batch]
            labels = [l for v, f, cn, l in batch]  
            
            verts = pad_sequence(verts, batch_first=True, padding_value=0.0)        
            faces = pad_sequence(faces, batch_first=True, padding_value=-1)        
            color_normals = pad_sequence(color_normals, batch_first=True, padding_value=0.0)
            labels = torch.tensor(labels)
            
            return verts, faces, color_normals, labels
        
        else:
            verts = [v for v, f, cn in batch]
            faces = [f for v, f, cn in batch]        
            color_normals = [cn for v, f, cn in batch]            
            
            verts = pad_sequence(verts, batch_first=True, padding_value=0.0)        
            faces = pad_sequence(faces, batch_first=True, padding_value=-1)        
            color_normals = pad_sequence(color_normals, batch_first=True, padding_value=0.0)            
            
            return verts, faces, color_normals
    
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, shuffle=True, drop_last=self.drop_last, collate_fn=self.pad_verts_faces)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, collate_fn=self.pad_verts_faces)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=1, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, collate_fn=self.pad_verts_faces)
        

class SaxiDataModuleVF(LightningDataModule):
    #It provides a structured and configurable way to load, preprocess, and organize 3D surface data for machine learning tasks, based on the specific requirements of the model type
    def __init__(self, df_train, df_val, df_test, mount_point="./", batch_size=256, num_workers=4, surf_column="surf", class_column=None , surf_property=None, scalar_column=None, train_transform=None, valid_transform=None, test_transform=None, drop_last=False):
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

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        self.train_ds = SaxiDataset(self.df_train, self.mount_point, surf_column=self.surf_column, surf_property=self.surf_property, class_column=self.class_column, scalar_column=self.scalar_column, transform=self.train_transform, CN=False)
        self.val_ds = SaxiDataset(self.df_val, self.mount_point, surf_column=self.surf_column, surf_property=self.surf_property, class_column=self.class_column, scalar_column=self.scalar_column, transform=self.valid_transform, CN=False)
        self.test_ds = SaxiDataset(self.df_test, self.mount_point, surf_column=self.surf_column, surf_property=self.surf_property, class_column=self.class_column, scalar_column=self.scalar_column, transform=self.test_transform, CN=False)

    def pad_verts_faces(self, batch):
        # Collate function for the dataloader to know how to comine the data

        if self.class_column:
            verts = [v for v, f, c  in batch]
            faces = [f for v, f, c in batch]
            classes = [c for v, f, c in batch]  
            
            verts = pad_sequence(verts, batch_first=True, padding_value=0.0)        
            faces = pad_sequence(faces, batch_first=True, padding_value=-1)
                
            return verts, faces, torch.tensor(classes)
        else:
        
            verts = [v for v, f  in batch]
            faces = [f for v, f in batch]
            
            verts = pad_sequence(verts, batch_first=True, padding_value=0.0)        
            faces = pad_sequence(faces, batch_first=True, padding_value=-1)
                
            return verts, faces


    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, shuffle=True, drop_last=self.drop_last, collate_fn=self.pad_verts_faces)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, collate_fn=self.pad_verts_faces)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=1, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, collate_fn=self.pad_verts_faces)


#####################################################################################################################################################################################
#                                                                                                                                                                                   #
#                                                                                   IcoConv                                                                                         #
#                                                                                                                                                                                   #
#####################################################################################################################################################################################


class SaxiIcoDataset(Dataset):
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
        # Load Data
        row = self.df.loc[idx]
        path_eacsf = row[f'Path{"Left" if hemisphere=="left" else "Right"}EACSF']
        path_sa = row[f'Path{"Left" if hemisphere=="left" else "Right"}Sa']
        path_thickness = row[f'Path{"Left" if hemisphere=="left" else "Right"}Thickness']

        l_features = [
            self.data_to_tensor(path_eacsf).unsqueeze(dim=1),
            self.data_to_tensor(path_sa).unsqueeze(dim=1),
            self.data_to_tensor(path_thickness).unsqueeze(dim=1)
        ]

        reader = utils.ReadSurf(self.list_path_ico[0 if hemisphere=="left" else 1])

        vertex_features = torch.cat(l_features,dim=1)

        #Demographics
        demographic_values = [float(row[name]) for name in self.list_demographic]
        demographic = torch.tensor(demographic_values)

        #Y
        Y = torch.tensor([int(row[self.name_class])])

        #Sphere per hemisphere
        verts, faces = utils.PolyDataToTensors_v_f(reader)
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


class SaxiIcoDataModule(LightningDataModule):
    def __init__(self,batch_size,list_demographic,data_train,data_val,data_test,list_path_ico,train_transform=None,val_and_test_transform=None, num_workers=6,name_class='ASD_administered'):
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
        self.train_dataset = SaxiIcoDataset(self.df_train,self.list_demographic,self.list_path_ico,self.train_transform,name_class = self.name_class)
        self.val_dataset = SaxiIcoDataset(self.df_val,self.list_demographic,self.list_path_ico,self.val_and_test_transform,name_class = self.name_class)
        self.test_dataset = SaxiIcoDataset(self.df_test,self.list_demographic,self.list_path_ico,self.val_and_test_transform,name_class = self.name_class)

        VL, FL, VFL, FFL,VR, FR, VFR, FFR, demographic, Y = self.train_dataset.__getitem__(0)
        self.nbr_demographic = demographic.shape[0]

    def repeat_subject(self,df,final_size):
        n = len(df)
        q,r = final_size//n,final_size%n
        list_df = [df for i in range(q)]
        list_df.append(df[:r])
        new_df = pd.concat(list_df).reset_index().drop(['index'],axis=1)
        return new_df
    
    def train_dataloader(self):    
        return DataLoader(self.train_dataset,batch_size=self.batch_size,shuffle=True, num_workers=self.num_workers, pin_memory=True, persistent_workers=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, persistent_workers=True, drop_last=True)        

    def test_dataloader(self):
        return DataLoader(self.test_dataset,batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, persistent_workers=True, drop_last=True)

    def get_weigths(self):
        return self.weights
    
    def get_nbr_demographic(self):
        return self.nbr_demographic


#####################################################################################################################################################################################
#                                                                                                                                                                                   #
#                                                                              Ring Freesurfer                                                                                      #
#                                                                                                                                                                                   #
#####################################################################################################################################################################################


class SaxiFreesurferDataset(Dataset):
    def __init__(self,df,transform = None, version=None, name_class ='fsqc_qc', freesurfer_path=None):
        self.df = df
        self.transform = transform
        self.version = version
        self.name_class = name_class
        self.freesurfer_path = freesurfer_path

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self,idx):
        #Get item for each hemisphere (left and right)
        vertsL, facesL, vertex_featuresL, face_featuresL, Y = self.getitem_per_hemisphere('L', idx)
        vertsR, facesR, vertex_featuresR, face_featuresR, Y = self.getitem_per_hemisphere('R', idx)
        return  vertsL, facesL, vertex_featuresL, face_featuresL, vertsR, facesR, vertex_featuresR, face_featuresR, Y 
    
    def data_to_tensor(self,path):
        data = nib.freesurfer.read_morph_data(path)
        data = data.byteswap().newbyteorder()
        data = torch.from_numpy(data).float()
        return data

    def set_wm_as_texture(self, sphere, wm_path):
        wm_surf = utils.ReadSurf(wm_path)
        wm_normals = utils.ComputeNormals(wm_surf)
        sphere.GetPointData().SetNormals(wm_normals.GetPointData().GetNormals())
        return sphere

    # Get the verts, faces, vertex_features, face_features and Y from an hemisphere
    def getitem_per_hemisphere(self, hemisphere, idx):
        row = self.df.loc[idx]
        sub_session = '_' + row['eventname']
        # sub_session = '_ses-' + row['eventname'].replace('_', '').replace('year', 'Year').replace('arm', 'Arm').replace('followup', 'FollowUp').replace('yArm','YArm')
        path_to_fs_data = os.path.join(self.freesurfer_path, row['Subject_ID'], row['Subject_ID'] + sub_session, 'surf')

        # Load Data
        hemisphere_prefix = 'lh' if hemisphere == 'L' else 'rh'
        path_sa = os.path.join(path_to_fs_data, f'{hemisphere_prefix}.area')
        path_thickness = os.path.join(path_to_fs_data, f'{hemisphere_prefix}.thickness')
        path_curvature = os.path.join(path_to_fs_data, f'{hemisphere_prefix}.curv')
        path_sulc = os.path.join(path_to_fs_data, f'{hemisphere_prefix}.sulc')

        paths = [path_sa, path_thickness, path_curvature, path_sulc]
        l_features = [self.data_to_tensor(path).unsqueeze(dim=1) for path in paths]

        sphere_path = os.path.join(path_to_fs_data, f'{hemisphere_prefix}.sphere.reg')
        sphere_vtk_path = os.path.join(path_to_fs_data, f'{hemisphere_prefix}.sphere.reg.vtk')
        wm_path = os.path.join(path_to_fs_data, f'{hemisphere_prefix}.white')
        wm_vtk_path = os.path.join(path_to_fs_data, f'{hemisphere_prefix}.white.vtk')

        paths = [path_sa, path_thickness, path_curvature, path_sulc, sphere_path, wm_path, sphere_vtk_path, wm_vtk_path]

        for path in paths:
            if not os.path.exists(path):
                print(f'File {path} does not exist')
                return

        if not os.path.exists(sphere_vtk_path):
            mris_command = f'mris_convert {sphere_path} {sphere_vtk_path}'
            subprocess.run(mris_command, shell=True)
        if not os.path.exists(wm_vtk_path):
            mris_command = f'mris_convert {wm_path} {wm_vtk_path}'
            subprocess.run(mris_command, shell=True)

        sphere = utils.ReadSurf(sphere_vtk_path)
        sphere = self.set_wm_as_texture(sphere, wm_vtk_path)

        vertex_features = torch.cat(l_features, dim=1)

        # Normalization of the features
        mean = torch.mean(vertex_features, dim=0)
        std = torch.std(vertex_features, dim=0)
        vertex_features = (vertex_features - mean) / std

        Y = torch.tensor([int(row[self.name_class])])

        # Sphere per hemisphere
        verts, faces = utils.PolyDataToTensors_v_f(sphere)

        #Transformations
        if self.transform:        
            verts = self.transform(verts)

        # Face Features
        faces_pid0 = faces[:,0:1]  

        face_features = torch.cat([torch.take(vf, faces_pid0) for vf in vertex_features.transpose(0, 1)], dim=-1)
        
        return verts, faces, vertex_features, face_features, Y
    
    def getSurf(self, idx):
        row = self.df.loc[idx]
        sub_session = '_' + row['eventname']
        lh_inflated_path = os.path.join(self.freesurfer_path, row['Subject_ID'], row['Subject_ID'] + sub_session, 'surf', 'lh_inflated.vtk')
        rh_inflated_path = os.path.join(self.freesurfer_path, row['Subject_ID'], row['Subject_ID'] + sub_session, 'surf', 'rh_inflated.vtk')

        if not lh_inflated_path or not rh_inflated_path:
            raise ValueError(f'File {lh_inflated_path} or/and {rh_inflated_path} does not exist')
        else:
            return utils.ReadSurf(lh_inflated_path), utils.ReadSurf(rh_inflated_path), os.path.join(row['Subject_ID'], row['Subject_ID'] + sub_session, 'surf', 'lh_inflated.vtk'), os.path.join(row['Subject_ID'], row['Subject_ID'] + sub_session, 'surf', 'rh_inflated.vtk')


class SaxiFreesurferDataModule(LightningDataModule):
    def __init__(self,batch_size,data_train,data_val,data_test,train_transform=None,val_and_test_transform=None, num_workers=6,name_class='fsqc_qc',freesurfer_path=None):
        super().__init__()
        self.batch_size = batch_size 
        self.data_train = data_train
        self.data_val = data_val
        self.data_test = data_test
        self.train_transform = train_transform
        self.val_and_test_transform = val_and_test_transform
        self.num_workers = num_workers
        self.name_class = name_class
        self.freesurfer_path = freesurfer_path

        self.weights = []
        self.df_train = pd.read_csv(self.data_train)
        self.df_val = pd.read_csv(self.data_val)
        self.df_test = pd.read_csv(self.data_test)
        self.weights = self.class_weights()

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
        self.train_dataset = SaxiFreesurferDataset(self.df_train,self.train_transform,name_class = self.name_class,freesurfer_path = self.freesurfer_path)
        self.val_dataset = SaxiFreesurferDataset(self.df_val,self.val_and_test_transform,name_class = self.name_class,freesurfer_path = self.freesurfer_path)
        self.test_dataset = SaxiFreesurferDataset(self.df_test,self.val_and_test_transform,name_class = self.name_class,freesurfer_path = self.freesurfer_path)

    def pad_verts_faces(self, batch):
        verts_l = [vl for vl, fl, vfl, ffl, vr, fr, vfr, ffr, y in batch]
        faces_l = [fl for vl, fl, vfl, ffl, vr, fr, vfr, ffr, y in batch]
        vertex_features_l = [vfl for vl, fl, vfl, ffl, vr, fr, vfr, ffr, y in batch]
        face_features_l = [ffl for vl, fl, vfl, ffl, vr, fr, vfr, ffr, y in batch]
        verts_r = [vr for vl, fl, vfl, ffl, vr, fr, vfr, ffr, y in batch]
        faces_r = [fr for vl, fl, vfl, ffl, vr, fr, vfr, ffr, y in batch]
        vertex_features_r = [vfr for vl, fl, vfl, ffl, vr, fr, vfr, ffr, y in batch]
        face_features_r = [ffr for vl, fl, vfl, ffl, vr, fr, vfr, ffr, y in batch]
        Y = [y for vl, fl, vfl, ffl, vr, fr, vfr, ffr, y in batch]     

        verts_l = pad_sequence(verts_l, batch_first=True, padding_value=0.0) 
        faces_l = pad_sequence(faces_l, batch_first=True, padding_value=-1)
        vertex_features_l = pad_sequence(vertex_features_l, batch_first=True, padding_value=0.0)
        face_features_l = torch.cat(face_features_l)
        verts_r = pad_sequence(verts_r, batch_first=True, padding_value=0.0)
        faces_r = pad_sequence(faces_r, batch_first=True, padding_value=-1)
        vertex_features_r = pad_sequence(vertex_features_r, batch_first=True, padding_value=0.0)
        face_features_r = torch.cat(face_features_r)
        Y = torch.tensor(Y)

        return verts_l, faces_l, vertex_features_l, face_features_l, verts_r, faces_r, vertex_features_r, face_features_r, Y
    
    def train_dataloader(self):  
        return monai_DataLoader(self.train_dataset,batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, drop_last=True, collate_fn=self.pad_verts_faces)

    def val_dataloader(self):
        return monai_DataLoader(self.val_dataset,batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True, collate_fn=self.pad_verts_faces)        

    def test_dataloader(self):
        return monai_DataLoader(self.test_dataset,batch_size=1, num_workers=self.num_workers, drop_last=True, collate_fn=self.pad_verts_faces)

    def get_weigths(self):
        return self.weights


#####################################################################################################################################################################################
#                                                                                                                                                                                   #
#                                                                    Ring Freesurfer with multiple timepoints                                                                       #
#                                                                                                                                                                                   #
#####################################################################################################################################################################################


class SaxiFreesurferMPDataModule(LightningDataModule):
    def __init__(self,batch_size,data_train,data_val,data_test,train_transform=None,val_and_test_transform=None, num_workers=6,name_class='fsqc_qc',freesurfer_path=None):
        super().__init__()
        self.batch_size = batch_size 
        self.data_train = data_train
        self.data_val = data_val
        self.data_test = data_test
        self.train_transform = train_transform
        self.val_and_test_transform = val_and_test_transform
        self.num_workers = num_workers
        self.name_class = name_class
        self.freesurfer_path = freesurfer_path

        self.weights = []
        self.df_train = pd.read_csv(self.data_train)
        self.df_val = pd.read_csv(self.data_val)
        self.df_test = pd.read_csv(self.data_test)
        self.weights = self.class_weights()

    
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
        self.train_dataset = SaxiFreesurferMPdataset(self.df_train,self.train_transform,name_class = self.name_class,freesurfer_path = self.freesurfer_path)
        self.val_dataset = SaxiFreesurferMPdataset(self.df_val,self.val_and_test_transform,name_class = self.name_class,freesurfer_path = self.freesurfer_path)
        self.test_dataset = SaxiFreesurferMPdataset(self.df_test,self.val_and_test_transform,name_class = self.name_class,freesurfer_path = self.freesurfer_path)
    
    def pad_verts_faces_through_timepoints(self, batch):
        padded_batch = {}
        timepoints = ['T1L', 'T2L', 'T3L', 'T1R', 'T2R', 'T3R']

        # Initialize the dictionary with empty lists for each timepoint
        for timepoint in timepoints:
            padded_batch[timepoint] = []
        padded_batch['Y'] = []

        for sub in batch:
            for timepoint in timepoints:
                value = sub.get(timepoint)
                padded_batch[timepoint].append(value)
            padded_batch['Y'].append(sub['Y'])
        
        # Pad the values for each timepoint
        for timepoint in timepoints:
            padded_batch[timepoint] = self.pad_verts_faces(padded_batch[timepoint])
        padded_batch['Y'] = torch.tensor(padded_batch['Y'])

        return padded_batch


    def pad_verts_faces(self, value):

        verts = [v for v, f, vf, ff in value]
        faces = [f for v, f, vf, ff in value]
        vertex_features = [vf for v, f, vf, ff in value]
        face_features = [ff for v, f, vf, ff in value]
       
        # Padding the sequences
        verts = pad_sequence(verts, batch_first=True, padding_value=0.0)
        faces = pad_sequence(faces, batch_first=True, padding_value=-1)
        vertex_features = pad_sequence(vertex_features, batch_first=True, padding_value=0.0)
        face_features = torch.cat(face_features) if face_features else torch.tensor([])  # Handle empty case

        return verts, faces, vertex_features, face_features 


    def train_dataloader(self):  
        return monai_DataLoader(self.train_dataset,batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, persistent_workers=True, drop_last=True, collate_fn=self.pad_verts_faces_through_timepoints)

    def val_dataloader(self):
        return monai_DataLoader(self.val_dataset,batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, persistent_workers=True, drop_last=True, collate_fn=self.pad_verts_faces_through_timepoints)        

    def test_dataloader(self):
        return monai_DataLoader(self.test_dataset,batch_size=1, num_workers=self.num_workers, pin_memory=True, persistent_workers=True, drop_last=True, collate_fn=self.pad_verts_faces_through_timepoints)

    def get_weigths(self):
        return self.weights


class SaxiFreesurferMPdataset(Dataset):
    def __init__(self, df, transform=None, version=None, name_class='fsqc_qc', freesurfer_path=None):
        self.df = df
        self.transform = transform
        self.version = version
        self.name_class = name_class
        self.freesurfer_path = freesurfer_path
        self.df_grouped = df.groupby('Subject_ID')
        self.keys = list(self.df_grouped.groups.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        subject_id = self.keys[idx]
        subject_data = self.df_grouped.get_group(subject_id)
        dicoL = self.getitem_per_hemisphere('L', subject_data)
        dicoR = self.getitem_per_hemisphere('R', subject_data)
        merged_dico = dicoL | dicoR
        return merged_dico

    def data_to_tensor(self, path):
        data = nib.freesurfer.read_morph_data(path)
        data = data.byteswap().newbyteorder()
        data = torch.from_numpy(data).float()
        return data

    def set_wm_as_texture(self, sphere, wm_path):
        normals_filter = vtk.vtkPolyDataNormals()
        normals_filter.SetInputData(sphere)
        normals_filter.ComputePointNormalsOn()
        normals_filter.ComputeCellNormalsOff()
        normals_filter.Update()

        output_with_normals = normals_filter.GetOutput()
        normals_array = output_with_normals.GetPointData().GetNormals()

        if normals_array is None:
            print(wm_path)
        elif normals_array.GetNumberOfTuples() == 0:
            wm_surf = utils.ReadSurf(wm_path)
            wm_normals = utils.ComputeNormals(wm_surf)
            sphere.GetPointData().SetScalars(wm_normals)

        return sphere


    def getitem_per_hemisphere(self, hemisphere, subject_data):
        dico = {}   
        for _, row in subject_data.iterrows():
            sub_session = '_' + row['eventname']
            # sub_session = '_ses-' + row['eventname'].replace('_', '').replace('year', 'Year').replace('arm', 'Arm').replace('followup', 'FollowUp').replace('yArm','YArm')
            path_to_fs_data = os.path.join(self.freesurfer_path, row['Subject_ID'], row['Subject_ID'] + sub_session, 'surf')

            hemisphere_prefix = 'lh' if hemisphere == 'L' else 'rh'
            path_sa = os.path.join(path_to_fs_data, f'{hemisphere_prefix}.area')
            path_thickness = os.path.join(path_to_fs_data, f'{hemisphere_prefix}.thickness')
            path_curvature = os.path.join(path_to_fs_data, f'{hemisphere_prefix}.curv')
            path_sulc = os.path.join(path_to_fs_data, f'{hemisphere_prefix}.sulc')

            paths = [path_sa, path_thickness, path_curvature, path_sulc]

            l_features = [data_to_tensor(path).unsqueeze(dim=1) for path in paths]

            for time_point in range(1, 4):
                key = f"T{time_point}{hemisphere}"
                if key not in dico:
                    dico[key] = (
                        torch.zeros((1, 3)),  # verts
                        torch.zeros((1, 3)),  # faces
                        torch.zeros((1, 4)),  # vertex_features
                        torch.zeros((1, 4)),  # face_features
                    )

            sphere_path = os.path.join(path_to_fs_data, f'{hemisphere_prefix}.sphere.reg')
            sphere_vtk_path = os.path.join(path_to_fs_data, f'{hemisphere_prefix}.sphere.reg.vtk')
            wm_path = os.path.join(path_to_fs_data, f'{hemisphere_prefix}.white')
            wm_vtk_path = os.path.join(path_to_fs_data, f'{hemisphere_prefix}.white.vtk')

            paths = [path_sa, path_thickness, path_curvature, path_sulc, sphere_path, wm_path, sphere_vtk_path, wm_vtk_path]

            for path in paths:
                if not os.path.exists(path):
                    print(f'File {path} does not exist')

            if not os.path.exists(sphere_vtk_path):
                mris_command = f'mris_convert {sphere_path} {sphere_vtk_path}'
                subprocess.run(mris_command, shell=True)
            if not os.path.exists(wm_vtk_path):
                mris_command = f'mris_convert {wm_path} {wm_vtk_path}'
                subprocess.run(mris_command, shell=True)

            sphere = utils.ReadSurf(sphere_vtk_path)
            sphere = self.set_wm_as_texture(sphere, wm_vtk_path)

            vertex_features = torch.cat(l_features, dim=1)

            Y = torch.tensor([int(row[self.name_class])])

            verts, faces = utils.PolyDataToTensors_v_f(sphere)

            if self.transform:        
                verts = self.transform(verts)

            faces_pid0 = faces[:,0:1]
            face_features = torch.cat([torch.take(vf, faces_pid0) for vf in vertex_features.transpose(0, 1)], dim=-1)

            if '2Year' in row['eventname']:
                time_point = 2
            elif '4Year' in row['eventname']:
                time_point = 3
            else:
                time_point = 1

            key = f"T{time_point}{hemisphere}"
            dico[key] = (verts, faces, vertex_features, face_features)

        if hemisphere == 'R':
            dico['Y'] = Y

        return dico


#####################################################################################################################################################################################
#                                                                                                                                                                                   #
#                                                                                     Octree                                                                                        #
#                                                                                                                                                                                   #
#####################################################################################################################################################################################

import ocnn
from ocnn.octree import Octree, Points

class SaxiOctreeDataset(Dataset):
    def __init__(self,df,transform=None, version=None, name_class ='fsqc_qc', freesurfer_path=None, normalize=False):
        self.df = df
        self.transform = transform
        self.version = version
        self.name_class = name_class
        self.freesurfer_path = freesurfer_path
        self.normalize = normalize

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        # Get item for each hemisphere (left and right)
        vertsL, vertex_featuresL, Y = self.getitem_per_hemisphere('L', idx)
        vertsR, vertex_featuresR, Y = self.getitem_per_hemisphere('R', idx)

        # vertsL, vertex_featuresL = self.match_sizes(vertsL, vertex_featuresL)
        # vertsR, vertex_featuresR = self.match_sizes(vertsR, vertex_featuresR)

        # print(f'VertsL: {vertsL.size()}', f'Vertex FeaturesL: {vertex_featuresL.size()}')
        # print(f'VertsR: {vertsR.size()}', f'Vertex FeaturesR: {vertex_featuresR.size()}')
        
        # Create Octree for each hemisphere
        octree_L = Octree(8)
        octree_L.build_octree(Points(vertsL, features=vertex_featuresL))
        # octree_L.build_octree(Points(vertsL))
        octree_R = Octree(8)
        octree_R.build_octree(Points(vertsR, features=vertex_featuresR))
        # octree_R.build_octree(Points(vertsR))

        # Return separate Octrees for left and right hemispheres
        return octree_L, octree_R, Y
    

    def match_sizes(self, verts, vertex_features):
        num_verts = verts.size(0)
        num_features = vertex_features.size(0)
        
        if num_verts > num_features:
            # Truncate verts
            verts = verts[:num_features, :]
        elif num_features > num_verts:
            # Truncate vertex_features
            vertex_features = vertex_features[:num_verts, :]
        
        return verts, vertex_features

    
    def data_to_tensor(self,path):
        data = nib.freesurfer.read_morph_data(path)
        data = data.byteswap().newbyteorder()
        data = torch.from_numpy(data).float()
        return data
    

    # Get the verts, faces, vertex_features, face_features and Y from an hemisphere
    def getitem_per_hemisphere(self, hemisphere, idx):
        row = self.df.loc[idx]
        # sub_session = '_' + row['eventname']
        sub_session = '_ses-' + row['eventname'].replace('_', '').replace('year', 'Year').replace('arm', 'Arm').replace('followup', 'FollowUp').replace('yArm','YArm')
        path_to_fs_data = os.path.join(self.freesurfer_path, row['Subject_ID'], row['Subject_ID'] + sub_session, 'surf')

        # Load Data
        hemisphere_prefix = 'lh' if hemisphere == 'L' else 'rh'
        path_sa = os.path.join(path_to_fs_data, f'{hemisphere_prefix}.area')
        path_thickness = os.path.join(path_to_fs_data, f'{hemisphere_prefix}.thickness')
        path_curvature = os.path.join(path_to_fs_data, f'{hemisphere_prefix}.curv')
        path_sulc = os.path.join(path_to_fs_data, f'{hemisphere_prefix}.sulc')

        paths = [path_sa, path_thickness, path_curvature, path_sulc]
        l_features = [self.data_to_tensor(path).unsqueeze(dim=1) for path in paths]

        vertex_features = torch.cat(l_features, dim=1)

        wm_vtk_path = os.path.join(path_to_fs_data, f'{hemisphere_prefix}.white.vtk')
        surf = utils.ReadSurf(wm_vtk_path)

        transform = saxi_transforms.UnitSurfTransform()
        surf_norm = transform(surf)

        verts, faces = utils.PolyDataToTensors_v_f(surf_norm)

        if self.normalize:
            mean, std = self.normalize_feature(self.df, hemisphere)
            vertex_features = (vertex_features - mean) / std

        Y = torch.tensor([int(row[self.name_class])])

        #Transformations
        if self.transform:        
            verts = self.transform(verts)

        if verts.size(0) != vertex_features.size(0):
            print(f'Verts: {verts.size(0)}', f'Vertex Features: {vertex_features.size(0)}')
            print(path_to_fs_data)

        return verts, vertex_features, Y
    

    def normalize_feature(self, df, hemisphere):
        # Initialize lists to store mean and std
        means = []
        stds = []

        # List of features to be standardized for each hemisphere
        mean_values = {
            'L': ['sulc_left_mean', 'curvature_left_mean', 'thickness_left_mean', 'area_left_mean'],
            'R': ['sulc_right_mean', 'curvature_right_mean', 'thickness_right_mean', 'area_right_mean']
        }

        std_values = {
            'L': ['sulc_left_std', 'curvature_left_std', 'thickness_left_std', 'area_left_std'],
            'R': ['sulc_right_std', 'curvature_right_std', 'thickness_right_std', 'area_right_std']
        }

        for mean in mean_values[hemisphere]:
            means.append(df[mean].mean())
        
        for std in std_values[hemisphere]:
            stds.append(df[std].mean())

        # Convert lists to tensors
        mean_tensor = torch.tensor(means)
        std_tensor = torch.tensor(stds)
        
        return mean_tensor, std_tensor
    

    def getSurf(self, idx):
        row = self.df.loc[idx]
        sub_session = '_' + row['eventname']
        lh_inflated_path = os.path.join(self.freesurfer_path, row['Subject_ID'], row['Subject_ID'] + sub_session, 'surf', 'lh_inflated.vtk')
        rh_inflated_path = os.path.join(self.freesurfer_path, row['Subject_ID'], row['Subject_ID'] + sub_session, 'surf', 'rh_inflated.vtk')

        if not lh_inflated_path or not rh_inflated_path:
            raise ValueError(f'File {lh_inflated_path} or/and {rh_inflated_path} does not exist')
        else:
            return utils.ReadSurf(lh_inflated_path), utils.ReadSurf(rh_inflated_path), os.path.join(row['Subject_ID'], row['Subject_ID'] + sub_session, 'surf', 'lh_inflated.vtk'), os.path.join(row['Subject_ID'], row['Subject_ID'] + sub_session, 'surf', 'rh_inflated.vtk')


class SaxiOctreeDataModule(LightningDataModule):
    def __init__(self,batch_size,data_train,data_val,data_test,train_transform=None,val_and_test_transform=None, num_workers=6,name_class='fsqc_qc',freesurfer_path=None):
        super().__init__()
        self.batch_size = batch_size 
        self.data_train = data_train
        self.data_val = data_val
        self.data_test = data_test
        self.train_transform = train_transform
        self.val_and_test_transform = val_and_test_transform
        self.num_workers = num_workers
        self.name_class = name_class
        self.freesurfer_path = freesurfer_path

        self.weights = []
        self.df_train = pd.read_csv(self.data_train)
        self.df_val = pd.read_csv(self.data_val)
        self.df_test = pd.read_csv(self.data_test)
        self.weights = self.class_weights()

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
        self.train_dataset = SaxiOctreeDataset(self.df_train,self.train_transform,name_class = self.name_class,freesurfer_path = self.freesurfer_path, normalize=True)
        self.val_dataset = SaxiOctreeDataset(self.df_val,self.val_and_test_transform,name_class = self.name_class,freesurfer_path = self.freesurfer_path)
        self.test_dataset = SaxiOctreeDataset(self.df_test,self.val_and_test_transform,name_class = self.name_class,freesurfer_path = self.freesurfer_path)

    def padding(self, batch):
        
        octree_L_batch = [item[0] for item in batch]
        octree_R_batch = [item[1] for item in batch]
        Y_batch = [item[2] for item in batch]

        # Merge octrees if needed
        merged_octree_L = ocnn.octree.merge_octrees(octree_L_batch)
        merged_octree_R = ocnn.octree.merge_octrees(octree_R_batch)

        # Construct neighbor indices
        merged_octree_L.construct_all_neigh()
        merged_octree_R.construct_all_neigh()

        # Convert labels to Tensor
        Y_tensor = torch.tensor(Y_batch)

        return merged_octree_L, merged_octree_R, Y_tensor

    def train_dataloader(self):  
        return monai_DataLoader(self.train_dataset,batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, drop_last=True, collate_fn=self.padding)

    def val_dataloader(self):
        return monai_DataLoader(self.val_dataset,batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True, collate_fn=self.padding)        

    def test_dataloader(self):
        return monai_DataLoader(self.test_dataset,batch_size=1, num_workers=self.num_workers, drop_last=True, collate_fn=self.padding)

    def get_weigths(self):
        return self.weights