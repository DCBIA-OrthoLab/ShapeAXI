import math
import numpy as np 
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchmetrics
import monai

import pandas as pd

import json
import os

import lightning as L
from typing import Tuple, Union

import pandas as pd

from pytorch3d.structures import (
    Meshes,
    Pointclouds,)

from pytorch3d.renderer import (
        FoVPerspectiveCameras, PerspectiveCameras, look_at_rotation, 
        RasterizationSettings, MeshRenderer, MeshRasterizer, MeshRendererWithFragments,
        HardPhongShader, AmbientLights, TexturesVertex
)
from pytorch3d.ops import (sample_points_from_meshes,
                           knn_points, 
                           knn_gather)

from pytorch3d.loss import (
    chamfer_distance,
    point_mesh_edge_distance, 
    point_mesh_face_distance
)

from pytorch3d.utils import ico_sphere
from torch.nn.utils.rnn import pad_sequence

import json
import os


from shapeaxi import utils
from shapeaxi.saxi_layers import *
from shapeaxi.saxi_nets import *
from shapeaxi.saxi_transforms import GaussianNoise, AvgPoolImages
from shapeaxi.colors import bcolors
from shapeaxi.saxi_losses import saxi_point_triangle_distance

import lightning as L
from lightning.pytorch.core import LightningModule

from diffusers.models.embeddings import Timesteps, GaussianFourierProjection, TimestepEmbedding
from diffusers import DDPMScheduler

class SaxiClassification(LightningModule):
    # Saxi classification network
    def __init__(self, **kwargs):
        super(SaxiClassification, self).__init__()
        self.save_hyperparameters()
        self.class_weights = None

        if hasattr(self.hparams, 'class_weights'):
            self.class_weights = torch.tensor(self.hparams.class_weights).to(torch.float32)            
        self.loss = nn.CrossEntropyLoss(weight=self.class_weights)
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.out_classes)

        if hasattr(monai.networks.nets, self.hparams.base_encoder):
            template_model = getattr(monai.networks.nets, self.hparams.base_encoder)
        elif hasattr(torchvision.models, self.hparams.base_encoder):
            template_model = getattr(torchvision.models, self.hparams.base_encoder)
        else:
            raise "{base_encoder} not in monai networks or torchvision".format(base_encoder=self.hparams.base_encoder)

        model_params = eval('dict(%s)' % self.hparams.base_encoder_params.replace('',''))
        self.convnet = template_model(**model_params)
        self.F = TimeDistributed(self.convnet)
        self.V = nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim)
        self.A = SelfAttention(input_dim=self.hparams.hidden_dim, hidden_dim=64)
        self.P = nn.Linear(self.hparams.hidden_dim, self.hparams.out_classes)        

        cameras = FoVPerspectiveCameras()

        raster_settings = RasterizationSettings(image_size=self.hparams.image_size, blur_radius=0, faces_per_pixel=1,max_faces_per_bin=200000)        
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        lights = AmbientLights()
        self.renderer = MeshRenderer(rasterizer=rasterizer,shader=HardPhongShader(cameras=cameras, lights=lights))
        self.ico_sphere(radius=self.hparams.radius, subdivision_level=self.hparams.subdivision_level)

    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("SaxiClassification")

        group.add_argument("--lr", type=float, default=1e-4)
        
        # Encoder parameters
        group.add_argument('--base_encoder', type=str, help='Base encoder for the feature extraction', default='resnet18')
        group.add_argument('--base_encoder_params', type=str, help='Base encoder parameters that are passed to build the feature extraction', default='pretrained=False,spatial_dims=2,n_input_channels=4,num_classes=512')
        group.add_argument('--hidden_dim', type=int, help='Hidden dimension for features output. Should match with output of base_encoder. Default value is 512', default=512)
        group.add_argument('--radius', type=float, help='Radius of icosphere', default=1.5)
        group.add_argument('--subdivision_level', type=int, help='Subdivision level for icosahedron', default=2)
        group.add_argument('--image_size', type=int, help='Image resolution size', default=256)
        group.add_argument('--out_classes', type=int, help='Output number of classes', default=4)

        return parent_parser


    def ico_sphere(self, radius=1.35, subdivision_level=1):
        # Create an icosphere
        ico_verts, ico_faces, ico_edges = utils.PolyDataToTensors(utils.CreateIcosahedronSubdivided(radius=radius, sl=subdivision_level))
        ico_verts = ico_verts.to(torch.float32)

        for idx, v in enumerate(ico_verts):
            if (torch.abs(torch.sum(v)) == radius):
                ico_verts[idx] = v + torch.normal(0.0, 1e-7, (3,))
        
        self.register_buffer("ico_verts", ico_verts)


    def configure_optimizers(self):
        # Configure the optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer


    def to(self, device=None):
        # Move the renderer to the specified device
        self.renderer = self.renderer.to(device)
        return super().to(device)


    def forward(self, X):
        # Forward pass
        x_f = self.F(X)
        x_v = self.V(x_f)
        x_a, x_s = self.A(x_f, x_v)
        x = self.P(x_a)
        
        return x, x_s


    def render(self, V, F, CN):
        # Render the input surface mesh to an image
        textures = TexturesVertex(verts_features=CN.to(torch.float32))
        meshes = Meshes(verts=V, faces=F, textures=textures)
        X = []
        PF = []

        for camera_position in self.ico_verts:
            camera_position = camera_position.unsqueeze(0)
            R = look_at_rotation(camera_position, device=self.device)  # (1, 3, 3)
            T = -torch.bmm(R.transpose(1, 2), camera_position[:,:,None])[:, :, 0]   # (1, 3)
            images = self.renderer(meshes_world=meshes.clone(), R=R, T=T)        
            fragments = self.renderer.rasterizer(meshes.clone())
            pix_to_face = fragments.pix_to_face
            zbuf = fragments.zbuf
            images = torch.cat([images[:,:,:,0:3], zbuf], dim=-1)
            images = images.permute(0,3,1,2)
            pix_to_face = pix_to_face.permute(0,3,1,2)
            X.append(images.unsqueeze(1))
            PF.append(pix_to_face.unsqueeze(1))
        
        X = torch.cat(X, dim=1)
        PF = torch.cat(PF, dim=1)        

        return X, PF

    def training_step(self, train_batch, batch_idx):
        # Training step
        V, F, CN, Y = train_batch
        V = V.to(self.device, non_blocking=True)
        F = F.to(self.device, non_blocking=True)        
        CN = CN.to(self.device, non_blocking=True)
        X, PF = self.render(V, F, CN)
        x, _ = self(X)
        loss = self.loss(x, Y)
        batch_size = V.shape[0]
        self.log('train_loss', loss, batch_size=batch_size)
        self.accuracy(x, Y)
        self.log("train_acc", self.accuracy, batch_size=batch_size)

        return loss


    def validation_step(self, val_batch, batch_idx):
        # Validation step
        V, F, CN, Y = val_batch
        V = V.to(self.device, non_blocking=True)
        F = F.to(self.device, non_blocking=True)        
        CN = CN.to(self.device, non_blocking=True)
        X, PF = self.render(V, F, CN)
        x, _ = self(X)
        loss = self.loss(x, Y)
        batch_size = V.shape[0]
        self.log('val_loss', loss, batch_size=batch_size, sync_dist=True)
        self.accuracy(x, Y)
        self.log("val_acc", self.accuracy, batch_size=batch_size, sync_dist=True)


#####################################################################################################################################################################################
#                                                                                                                                                                                   #
#                                                                                        Regression                                                                                 #
#                                                                                                                                                                                   #
#####################################################################################################################################################################################


class SaxiRegression(LightningModule):
    # Saxi regression network
    def __init__(self, **kwargs):

        super(SaxiRegression, self).__init__()
        self.save_hyperparameters()
        
        self.loss = nn.L1Loss(reduction='sum')

        if hasattr(monai.networks.nets, self.hparams.base_encoder):
            template_model = getattr(monai.networks.nets, self.hparams.base_encoder)
        elif hasattr(torchvision.models, self.hparams.base_encoder):
            template_model = getattr(torchvision.models, self.hparams.base_encoder)
        else:
            raise "{base_encoder} not in monai networks or torchvision".format(base_encoder=self.hparams.base_encoder)

        model_params = eval('dict(%s)' % self.hparams.base_encoder_params.replace(' ',''))
        
        self.convnet = template_model(**model_params)

        self.F = TimeDistributed(self.convnet)
        self.V = nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim)
        self.A = SelfAttention(in_units=self.hparams.hidden_dim, out_units=64)
        self.P = nn.Linear(self.hparams.hidden_dim, self.hparams.out_features)        

        cameras = FoVPerspectiveCameras()

        raster_settings = RasterizationSettings(
            image_size=self.hparams.image_size, 
            blur_radius=0, 
            faces_per_pixel=1,
            max_faces_per_bin=200000
        )        
        rasterizer = MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        )
        lights = AmbientLights()
        self.renderer = MeshRenderer(
                rasterizer=rasterizer,
                shader=HardPhongShader(cameras=cameras, lights=lights)
        )

        self.ico_sphere(radius=self.hparams.radius, subdivision_level=self.hparams.subdivision_level)
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("SaxiRegression")

        group.add_argument("--lr", type=float, default=1e-4)
        
        # Encoder parameters
        group.add_argument('--base_encoder', type=str, help='Base encoder for the feature extraction', default='resnet18')
        group.add_argument('--base_encoder_params', type=str, help='Base encoder parameters that are passed to build the feature extraction', default='pretrained=False,spatial_dims=2,n_input_channels=4,num_classes=512')
        group.add_argument('--hidden_dim', type=int, help='Hidden dimension for features output. Should match with output of base_encoder. Default value is 512', default=512)
        group.add_argument('--radius', type=float, help='Radius of icosphere', default=1.5)
        group.add_argument('--subdivision_level', type=int, help='Subdivision level for icosahedron', default=2)
        group.add_argument('--image_size', type=int, help='Image resolution size', default=256)
        group.add_argument('--out_classes', type=int, help='Output number of classes', default=4)

        return parent_parser


    def ico_sphere(self, radius=1.35, subdivision_level=1):
        # Create an icosphere
        ico_verts, ico_faces, ico_edges = utils.PolyDataToTensors(utils.CreateIcosahedronSubdivided(radius=radius, sl=subdivision_level))
        ico_verts = ico_verts.to(torch.float32)

        for idx, v in enumerate(ico_verts):
            if (torch.abs(torch.sum(v)) == radius):
                ico_verts[idx] = v + torch.normal(0.0, 1e-7, (3,))
        
        self.register_buffer("ico_verts", ico_verts)


    def configure_optimizers(self):
        # Configure the optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer


    def to(self, device=None):
        # Move the renderer to the specified device
        self.renderer = self.renderer.to(device)
        return super().to(device)


    def forward(self, X):
        # Forward pass
        x_f = self.F(X)
        x_v = self.V(x_f)
        x_a, x_s = self.A(x_f, x_v)
        x = self.P(x_a)
        
        return x


    def render(self, V, F, CN):
        # Render the input surface mesh to an image
        textures = TexturesVertex(verts_features=CN.to(torch.float32))
        meshes = Meshes(verts=V, faces=F, textures=textures)
        X = []
        PF = []

        for camera_position in self.ico_verts:
            camera_position = camera_position.unsqueeze(0)
            R = look_at_rotation(camera_position, device=self.device)  # (1, 3, 3)
            T = -torch.bmm(R.transpose(1, 2), camera_position[:,:,None])[:, :, 0]   # (1, 3)
            images = self.renderer(meshes_world=meshes.clone(), R=R, T=T)        
            fragments = self.renderer.rasterizer(meshes.clone())
            pix_to_face = fragments.pix_to_face
            zbuf = fragments.zbuf
            images = torch.cat([images[:,:,:,0:3], zbuf], dim=-1)
            images = images.permute(0,3,1,2)
            pix_to_face = pix_to_face.permute(0,3,1,2)
            X.append(images.unsqueeze(1))
            PF.append(pix_to_face.unsqueeze(1))
        X = torch.cat(X, dim=1)
        PF = torch.cat(PF, dim=1)        

        return X, PF

    def training_step(self, train_batch, batch_idx):
        # Training step
        V, F, CN, Y = train_batch
        V = V.to(self.device, non_blocking=True)
        F = F.to(self.device, non_blocking=True)        
        CN = CN.to(self.device, non_blocking=True)
        X, PF = self.render(V, F, CN)
        x = self(X)
        loss = self.loss(x, Y)
        batch_size = V.shape[0]
        self.log('train_loss', loss, batch_size=batch_size)

        return loss

    def validation_step(self, val_batch, batch_idx):
        # Validation step
        V, F, CN, Y = val_batch
        V = V.to(self.device, non_blocking=True)
        F = F.to(self.device, non_blocking=True)        
        CN = CN.to(self.device, non_blocking=True)
        X, PF = self.render(V, F, CN)
        x = self(X)
        loss = self.loss(x, Y)
        batch_size = V.shape[0]
        self.log('val_loss', loss, batch_size=batch_size, sync_dist=True)


#####################################################################################################################################################################################
#                                                                                                                                                                                   #
#                                                                                          Monai                                                                                    #
#                                                                                                                                                                                   #
#####################################################################################################################################################################################


class MonaiUNet(LightningModule):
    # Monai UNet network
    def __init__(self, args = None, out_channels=3, class_weights=None, image_size=320, radius=1.35, subdivision_level=1, train_sphere_samples=4):

        super(MonaiUNet, self).__init__()        
        
        self.save_hyperparameters()        
        self.args = args
        
        self.out_channels = out_channels
        self.class_weights = None
        if(class_weights is not None):
            self.class_weights = torch.tensor(class_weights).to(torch.float32)
            
        self.loss = monai.losses.DiceCELoss(include_background=False, to_onehot_y=True, softmax=True, ce_weight=self.class_weights)
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=34)

        unet = monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=4,   # images: torch.cuda.FloatTensor[batch_size,224,224,4]
            out_channels=out_channels, 
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
        self.model = TimeDistributed(unet)

        ico_verts, ico_faces, ico_edges = utils.PolyDataToTensors(utils.CreateIcosahedronSubdivided(radius=radius, sl=subdivision_level))
        ico_verts = ico_verts.to(torch.float32)

        for idx, v in enumerate(ico_verts):
            if (torch.abs(torch.sum(v)) == radius):
                ico_verts[idx] = v + torch.normal(0.0, 1e-7, (3,))

        
        self.register_buffer("ico_verts", ico_verts)

        cameras = FoVPerspectiveCameras()
        raster_settings = RasterizationSettings(
            image_size=image_size, 
            blur_radius=0, 
            faces_per_pixel=1,
            max_faces_per_bin=200000
        )        
        rasterizer = MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        )
        lights = AmbientLights()
        self.renderer = MeshRenderer(
                rasterizer=rasterizer,
                shader=HardPhongShader(cameras=cameras, lights=lights)
        )

    def configure_optimizers(self):
        # Configure the optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr)
        return optimizer
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("MonaiUNet")

        group.add_argument("--lr", type=float, default=1e-4)
        
        # Encoder parameters
        group.add_argument('--base_encoder', type=str, help='Base encoder for the feature extraction', default='resnet18')
        group.add_argument('--base_encoder_params', type=str, help='Base encoder parameters that are passed to build the feature extraction', default='pretrained=False,spatial_dims=2,n_input_channels=4,num_classes=512')
        group.add_argument('--hidden_dim', type=int, help='Hidden dimension for features output. Should match with output of base_encoder. Default value is 512', default=512)
        group.add_argument('--radius', type=float, help='Radius of icosphere', default=1.5)
        group.add_argument('--subdivision_level', type=int, help='Subdivision level for icosahedron', default=2)
        group.add_argument('--image_size', type=int, help='Image resolution size', default=256)
        group.add_argument('--out_classes', type=int, help='Output number of classes', default=4)

        return parent_parser

    def to(self, device=None):
        # Move the renderer to the specified device
        self.renderer = self.renderer.to(device)
        return super().to(device)

    def forward(self, x):
        # Forward pass
        V, F, CN = x
        X, PF = self.render(V, F, CN)
        x = self.model(X)
        return x, X, PF


    def render(self, V, F, CN):
        # Render the input surface mesh to an image
        textures = TexturesVertex(verts_features=CN)
        meshes = Meshes(verts=V, faces=F, textures=textures)        
        X = []
        PF = []

        for camera_position in self.ico_verts:
            camera_position = camera_position.unsqueeze(0)
            R = look_at_rotation(camera_position, device=self.device)  # (1, 3, 3)
            T = -torch.bmm(R.transpose(1, 2), camera_position[:,:,None])[:, :, 0]   # (1, 3)
            images = self.renderer(meshes_world=meshes.clone(), R=R, T=T)
            fragments = self.renderer.rasterizer(meshes.clone())
            pix_to_face = fragments.pix_to_face
            zbuf = fragments.zbuf
            images = torch.cat([images[:,:,:,0:3], zbuf], dim=-1)
            images = images.permute(0,3,1,2)
            pix_to_face = pix_to_face.permute(0,3,1,2)
            X.append(images.unsqueeze(1))
            PF.append(pix_to_face.unsqueeze(1))
        
        X = torch.cat(X, dim=1)
        PF = torch.cat(PF, dim=1)        

        return X, PF

    def training_step(self, train_batch, batch_idx):
        # Training step
        V, F, YF, CN = train_batch
        V = V.to(self.device, non_blocking=True)
        F = F.to(self.device, non_blocking=True)
        YF = YF.to(self.device, non_blocking=True)
        CN = CN.to(self.device, non_blocking=True).to(torch.float32)
        x, X, PF = self((V, F, CN))
        y = torch.take(YF, PF).to(torch.int64)*(PF >= 0)
        x = x.permute(0, 2, 1, 3, 4) #batch, time, channels, H, W -> batch, channels, time, H, W
        y = y.permute(0, 2, 1, 3, 4)
        loss = self.loss(x, y)
        batch_size = V.shape[0]
        self.log('train_loss', loss, batch_size=batch_size)
        self.accuracy(torch.argmax(x, dim=1, keepdim=True).reshape(-1, 1), y.reshape(-1, 1).to(torch.int32))
        self.log("train_acc", self.accuracy, batch_size=batch_size)

        return loss


    def validation_step(self, val_batch, batch_idx):
        # Validation step   
        V, F, YF, CN = val_batch
        V = V.to(self.device, non_blocking=True)
        F = F.to(self.device, non_blocking=True)
        YF = YF.to(self.device, non_blocking=True)
        CN = CN.to(self.device, non_blocking=True).to(torch.float32)
        x, X, PF = self((V, F, CN))
        y = torch.take(YF, PF).to(torch.int64)*(PF >= 0)
        x = x.permute(0, 2, 1, 3, 4) #batch, time, channels, H, W -> batch, channels, time, H, W
        y = y.permute(0, 2, 1, 3, 4)
        loss = self.loss(x, y)
        batch_size = V.shape[0]
        self.accuracy(torch.argmax(x, dim=1, keepdim=True).reshape(-1, 1), y.reshape(-1, 1).to(torch.int32))
        self.log("val_acc", self.accuracy, batch_size=batch_size, sync_dist=True)
        self.log('val_loss', loss, batch_size=batch_size, sync_dist=True)


    def test_step(self, val_batch, batch_idx):
        V, F, YF, CN = val_batch
        x, X, PF = self((V, F, CN))
        y = torch.take(YF, PF).to(torch.int64)*(PF >= 0)
        x = x.permute(0, 2, 1, 3, 4) #batch, time, channels, h, w -> batch, channels, time, h, w
        y = y.permute(0, 2, 1, 3, 4) 
        loss = self.loss(x, y)
        self.accuracy(torch.argmax(x, dim=1, keepdim=True).reshape(-1, 1), y.reshape(-1, 1).to(torch.int32))        
        return {'test_loss': loss, 'test_correct': self.accuracy}


#####################################################################################################################################################################################
#                                                                                                                                                                                   #
#                                                                                         IcoConv                                                                                   #
#                                                                                                                                                                                   #
#####################################################################################################################################################################################


class SaxiIcoClassification(LightningModule):
    def __init__(self, **kwargs):
        super(SaxiIcoClassification, self).__init__()
        self.save_hyperparameters()
        self.y_pred = []
        self.y_true = []

        ico_sphere = utils.CreateIcosahedronSubdivided(self.hparams.radius, self.hparams.subdivision_level)
        ico_sphere_verts, ico_sphere_faces, self.ico_sphere_edges = utils.PolyDataToTensors(ico_sphere)
        self.ico_sphere_verts = ico_sphere_verts
        self.ico_sphere_edges = np.array(self.ico_sphere_edges)
        R=[]
        T=[]
        for coords_cam in self.ico_sphere_verts.tolist():
            camera_position = torch.FloatTensor([coords_cam])
            R_current = look_at_rotation(camera_position)
            # check if camera coords vector and up vector for R are collinear
            if torch.equal(torch.cross(camera_position,torch.tensor([[0.,1.,0.]])),torch.tensor([[0., 0., 0.]])):
               R_current = look_at_rotation(camera_position, up = torch.tensor([[0.0, 0.0, 1.0]]),)
            T_current = -torch.bmm(R_current.transpose(1, 2), camera_position[:,:,None])[:, :, 0]   # (1, 3)

            R.append(R_current)
            T.append(T_current)
        self.R=torch.cat(R)
        self.T=torch.cat(T)
        self.nbr_cam = len(self.R)

        self.drop = nn.Dropout(p=self.hparams.dropout_lvl)

        # Left path
        self.create_network('L', self.hparams.out_size)
        # Right path
        self.create_network('R', self.hparams.out_size)

        #Demographics
        self.normalize = nn.BatchNorm1d(self.hparams.nbr_demographic)

        #Final layer
        self.Classification = nn.Linear(2*self.hparams.out_size+self.hparams.nbr_demographic, 2)

        #Loss
        self.loss_train = nn.CrossEntropyLoss(weight=self.hparams.weights[0])
        self.loss_val = nn.CrossEntropyLoss(weight=self.hparams.weights[1])
        self.loss_test = nn.CrossEntropyLoss(weight=self.hparams.weights[2])

        #Accuracy
        self.train_accuracy = torchmetrics.Accuracy('multiclass',num_classes=self.hparams.out_classes,average='macro')
        self.val_accuracy = torchmetrics.Accuracy('multiclass',num_classes=self.hparams.out_classes,average='macro')
        
        # Initialize a perspective camera.
        self.hparams.cameras = FoVPerspectiveCameras()

        # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
        raster_settings = RasterizationSettings(
            image_size=self.hparams.image_size,
            blur_radius=0,
            faces_per_pixel=1,
            max_faces_per_bin=100000
        )

        lights = AmbientLights()
        rasterizer = MeshRasterizer(
                cameras=self.hparams.cameras,
                raster_settings=raster_settings
        )

        self.hparams.phong_renderer = MeshRendererWithFragments(
            rasterizer=rasterizer,
            shader=HardPhongShader(cameras=self.hparams.cameras, lights=lights)
        )
        
    
    def to(self, device=None):
        # Move the renderer to the specified device
        self.hparams.phong_renderer = self.hparams.phong_renderer.to(device)
        return super().to(device)

    
    # Create an icosphere
    def create_network(self, side, out_size):

        if hasattr(monai.networks.nets, self.hparams.base_encoder):
            template_model = getattr(monai.networks.nets, self.hparams.base_encoder)
        elif hasattr(torchvision.models, self.hparams.base_encoder):
            template_model = getattr(torchvision.models, self.hparams.base_encoder)
        else:
            raise "{base_encoder} not in monai networks or torchvision".format(base_encoder=self.hparams.base_encoder)

        model_params = eval('dict(%s)' % self.hparams.base_encoder_params.replace(' ',''))
        
        self.convnet = template_model(**model_params)
        setattr(self, f'TimeDistributed{side}', TimeDistributed(self.convnet))

        if self.hparams.layer == 'Att':
            setattr(self, f'WV{side}', nn.Linear(self.hparams.hidden_dim, out_size))
            setattr(self, f'Attention{side}', SelfAttention(self.hparams.hidden_dim, out_size))

        elif self.hparams.layer in {'IcoConv2D', 'IcoConv1D', 'IcoLinear'}:
            if self.hparams.layer == 'IcoConv2D':
                conv_layer = nn.Conv2d(self.hparams.hidden_dim, out_size, kernel_size=(3, 3), stride=2, padding=0)
            elif self.hparams.layer == 'IcoConv1D':
                conv_layer = nn.Conv1d(self.hparams.hidden_dim, out_size, 7)
            else:
                conv_layer = nn.Linear(self.hparams.hidden_dim * 7, out_size)
            icosahedron = IcosahedronConv2d(conv_layer, self.ico_sphere_verts, self.ico_sphere_edges)
            avgpool = AvgPoolImages(nbr_images=self.nbr_cam)
            setattr(self, f'IcosahedronConv2d{side}', icosahedron)
            setattr(self, f'pooling{side}', avgpool)

        else:
            raise f"{self.hparams.layer} not in IcoConv2D, IcoConv1D, or IcoLinear"


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer


    def forward(self, x):
        VL, FL, VFL, FFL, VR, FR, VFR, FFR, demographic = x
        ###To Device
        VL = VL.to(self.device,non_blocking=True)
        FL = FL.to(self.device,non_blocking=True)
        VFL = VFL.to(self.device,non_blocking=True)
        FFL = FFL.to(self.device,non_blocking=True)
        VR = VR.to(self.device,non_blocking=True)
        FR = FR.to(self.device,non_blocking=True)
        VFR = VFR.to(self.device,non_blocking=True)
        FFR = FFR.to(self.device,non_blocking=True)
        demographic = demographic.to(self.device,non_blocking=True)
        ###Resnet18+Ico+Concatenation
        x = self.get_features(VL,FL,VFL,FFL,VR,FR,VFR,FFR,demographic)
        ###Last classification layer
        x = self.drop(x)
        x = self.Classification(x)

        return x
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("SaxiIcoClassification")
        
        group.add_argument("--lr", type=float, default=1e-4)
        group.add_argument("--out_classes", type=int, default=2)
        group.add_argument("--out_size", type=int, default=256)
        group.add_argument('--dropout_lvl',  type=float, help='Dropout level (default: 0.2)', default=0.2)

        # Encoder parameters
        group.add_argument('--base_encoder', type=str, help='Base encoder for the feature extraction', default='resnet18')
        group.add_argument('--base_encoder_params', type=str, help='Base encoder parameters that are passed to build the feature extraction', default='pretrained=False,spatial_dims=2,n_input_channels=1,num_classes=512')
        group.add_argument('--hidden_dim', type=int, help='Hidden dimension for features output. Should match with output of base_encoder. Default value is 512', default=512)
        group.add_argument('--radius', type=float, help='Radius of icosphere', default=1.5)
        group.add_argument('--subdivision_level', type=int, help='Subdivision level for icosahedron', default=2)
        group.add_argument('--image_size', type=int, help='Image resolution size', default=256)

        return parent_parser


    def get_features(self,VL,FL,VFL,FFL,VR,FR,VFR,FFR,demographic):
        #########Left path
        xL, PF = self.render(VL,FL,VFL,FFL)   
        xL = self.TimeDistributedL(xL)
        if self.Is_it_Icolayer(self.hparams.layer):
            xL = self.IcosahedronConv2dL(xL)
            xL = self.poolingL(xL)
        else:
            valuesL = self.WVL(xL)
            xL, score = self.AttentionL(xL,valuesL)            
        ###########Right path
        xR, PF = self.render(VR,FR,VFR,FFR)
        xR = self.TimeDistributedR(xR)
        if self.Is_it_Icolayer(self.hparams.layer):
            xR = self.IcosahedronConv2dR(xR)
            xR = self.poolingR(xR)
        else:
            valuesR = self.WVR(xR)
            xR,score = self.AttentionR(xR,valuesR)   
        #Concatenation   
        demographic = self.normalize(demographic)
        l_left_right = [xL,xR,demographic]
        x = torch.cat(l_left_right,dim=1)

        return x


    def render(self,V,F,VF,FF):
        textures = TexturesVertex(verts_features=VF)
        meshes = Meshes(
            verts=V,
            faces=F,
            textures=textures
        )
        PF = []
        for i in range(self.nbr_cam):
            pix_to_face = self.GetView(meshes,i)
            PF.append(pix_to_face.unsqueeze(dim=1))

        PF = torch.cat(PF, dim=1)
        l_features = []
        for index in range(FF.shape[-1]):
            l_features.append(torch.take(FF[:,:,index],PF)*(PF >= 0)) # take each feature for each pictures
        x = torch.cat(l_features,dim=2)

        return x, PF


    def training_step(self, train_batch, batch_idx):
        VL, FL, VFL, FFL, VR, FR, VFR, FFR, demographic, Y = train_batch
        x = self((VL, FL, VFL, FFL, VR, FR, VFR, FFR, demographic))
        Y = Y.squeeze(dim=1)  
        loss = self.loss_train(x,Y)
        self.log('train_loss', loss) 
        predictions = torch.argmax(x, dim=1, keepdim=True)
        self.train_accuracy(predictions.reshape(-1, 1), Y.reshape(-1, 1))
        self.log("train_acc", self.train_accuracy)           

        return loss


    def validation_step(self,val_batch,batch_idx):
        VL, FL, VFL, FFL, VR, FR, VFR, FFR, demographic, Y = val_batch
        x = self((VL, FL, VFL, FFL, VR, FR, VFR, FFR, demographic))
        Y = Y.squeeze(dim=1) 
        loss = self.loss_val(x,Y)
        self.log('val_loss', loss)
        predictions = torch.argmax(x, dim=1, keepdim=True)
        val_acc = self.val_accuracy(predictions.reshape(-1, 1), Y.reshape(-1, 1))
        self.log("val_acc", val_acc)   

        return val_acc


    def test_step(self,test_batch,batch_idx):
        VL, FL, VFL, FFL, VR, FR, VFR, FFR, demographic, Y = test_batch
        x = self((VL, FL, VFL, FFL, VR, FR, VFR, FFR, demographic))
        Y = Y.squeeze(dim=1)     
        loss = self.loss_test(x,Y)
        self.log('test_loss', loss, batch_size=self.hparams.batch_size)
        predictions = torch.argmax(x, dim=1, keepdim=True)
        output = [predictions,Y]

        return output


    def test_epoch_end(self,input_test):
        y_pred = []
        y_true = []
        for ele in input_test:
            y_pred += ele[0].tolist()
            y_true += ele[1].tolist()
        target_names = ['No ASD','ASD']
        self.y_pred =y_pred
        self.y_true =y_true
        #Classification report
        print(self.y_pred)
        print(self.y_true)
        print(classification_report(self.y_true, self.y_pred, target_names=target_names))


    def GetView(self,meshes,index):
        phong_renderer = self.hparams.phong_renderer.to(self.device)
        R = self.R[index][None].to(self.device)
        T = self.T[index][None].to(self.device)
        _, fragments = phong_renderer(meshes.clone(),R=R,T=T)
        pix_to_face = fragments.pix_to_face
        pix_to_face = pix_to_face.permute(0,3,1,2)

        return pix_to_face


    def get_y_for_report_classification(self):
        #This function could be called only after test step was done
        return (self.y_pred,self.hparams.y_true)
    

    def Is_it_Icolayer(self,layer):
        return (layer[:3] == 'Ico')


#####################################################################################################################################################################################
#                                                                                                                                                                                   #
#                                                                                    IcoConv Freesurfer                                                                             #
#                                                                                                                                                                                   #
#####################################################################################################################################################################################


class SaxiIcoClassification_fs(LightningModule):
    def __init__(self, **kwargs):
        super(SaxiIcoClassification_fs, self).__init__()
        self.save_hyperparameters()
        self.y_pred = []
        self.y_true = []

        ico_sphere = utils.CreateIcosahedronSubdivided(self.hparams.radius, self.hparams.subdivision_level)
        ico_sphere_verts, ico_sphere_faces, ico_sphere_edges = utils.PolyDataToTensors(ico_sphere)
        self.ico_sphere_verts = ico_sphere_verts
        self.ico_sphere_edges = ico_sphere_edges
        R=[]
        T=[]
        for coords_cam in self.ico_sphere_verts.tolist():
            camera_position = torch.FloatTensor([coords_cam])
            R_current = look_at_rotation(camera_position)
            # check if camera coords vector and up vector for R are collinear
            if torch.equal(torch.cross(camera_position,torch.tensor([[0.,1.,0.]])),torch.tensor([[0., 0., 0.]])):
               R_current = look_at_rotation(camera_position, up = torch.tensor([[0.0, 0.0, 1.0]]),)
            T_current = -torch.bmm(R_current.transpose(1, 2), camera_position[:,:,None])[:, :, 0]   # (1, 3)

            R.append(R_current)
            T.append(T_current)
        self.R=torch.cat(R)
        self.T=torch.cat(T)
        self.nbr_cam = len(self.R)

        self.drop = nn.Dropout(p=self.hparams.dropout_lvl)

        # Left path
        self.create_network('L', self.hparams.out_size)
        # Right path
        self.create_network('R', self.hparams.out_size)

        #Loss
        self.loss_train = nn.CrossEntropyLoss()
        self.loss_val = nn.CrossEntropyLoss()
        self.loss_test = nn.CrossEntropyLoss()

        #Final layer 
        self.Classification = nn.Linear(2*self.hparams.out_size, 2)

        #Accuracy
        self.train_accuracy = torchmetrics.Accuracy('multiclass',num_classes=self.hparams.out_classes,average='macro')
        self.val_accuracy = torchmetrics.Accuracy('multiclass',num_classes=self.hparams.out_classes,average='macro')
        
        # Initialize a perspective camera.
        self.hparams.cameras = FoVPerspectiveCameras()

        # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
        raster_settings = RasterizationSettings(image_size=self.hparams.image_size,blur_radius=0,faces_per_pixel=1,max_faces_per_bin=100000)

        lights = AmbientLights()
        rasterizer = MeshRasterizer(cameras=self.hparams.cameras,raster_settings=raster_settings)

        self.hparams.phong_renderer = MeshRendererWithFragments(rasterizer=rasterizer,shader=HardPhongShader(cameras=self.hparams.cameras, lights=lights))

    def to(self, device=None):
        # Move the renderer to the specified device
        self.hparams.phong_renderer = self.hparams.phong_renderer.to(device)
        return super().to(device)

    
    # Create an icosphere
    def create_network(self, side, out_size):

        if hasattr(monai.networks.nets, self.hparams.base_encoder):
            template_model = getattr(monai.networks.nets, self.hparams.base_encoder)
        elif hasattr(torchvision.models, self.hparams.base_encoder):
            template_model = getattr(torchvision.models, self.hparams.base_encoder)
        else:
            raise "{base_encoder} not in monai networks or torchvision".format(base_encoder=self.hparams.base_encoder)

        model_params = 'pretrained=False,spatial_dims=2,n_input_channels=1,num_classes=512'
        model_params = eval('dict(%s)' % self.hparams.base_encoder_params.replace(' ',''))
        
        self.convnet = template_model(**model_params)
        setattr(self, f'TimeDistributed{side}', TimeDistributed(self.convnet))

        if self.hparams.layer == 'IcoConv2D':
            conv_layer = nn.Conv2d(self.hparams.hidden_dim, out_size, kernel_size=(3, 3), stride=2, padding=0)
        elif self.hparams.layer == 'IcoConv1D':
            conv_layer = nn.Conv1d(self.hparams.hidden_dim, out_size, 7)
        else:
            conv_layer = nn.Linear(self.hparams.hidden_dim * 7, out_size)

        icosahedron = IcosahedronConv2d(conv_layer, self.ico_sphere_verts, self.ico_sphere_edges)
        avgpool = AvgPoolImages(nbr_images=self.nbr_cam)
        setattr(self, f'IcosahedronConv2d{side}', icosahedron)
        setattr(self, f'pooling{side}', avgpool)


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer


    def forward(self, x):
        VL, FL, VFL, FFL, VR, FR, VFR, FFR = x
        ###To Device
        VL = VL.to(self.device,non_blocking=True)
        FL = FL.to(self.device,non_blocking=True)
        VFL = VFL.to(self.device,non_blocking=True)
        FFL = FFL.to(self.device,non_blocking=True)
        VR = VR.to(self.device,non_blocking=True)
        FR = FR.to(self.device,non_blocking=True)
        VFR = VFR.to(self.device,non_blocking=True)
        FFR = FFR.to(self.device,non_blocking=True)
        ###Resnet18+Ico+Concatenation
        xL = self.get_features(VL,FL,VFL,FFL,'L')
        xR = self.get_features(VR,FR,VFR,FFR,'R')
        l_left_right = [xL,xR]
        x = torch.cat(l_left_right,dim=1)
        # ###Last classification layer
        x = self.drop(x)
        x = self.Classification(x)

        return x
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("SaxiIcoClassification_fs")
        
        group.add_argument("--lr", type=float, default=1e-4)
        group.add_argument("--fs_path", type=str, default=None)
        group.add_argument("--out_classes", type=int, default=2)
        group.add_argument("--out_size", type=int, default=256)
        group.add_argument('--dropout_lvl',  type=float, help='Dropout level (default: 0.2)', default=0.2)

        # Encoder parameters
        group.add_argument('--base_encoder', type=str, help='Base encoder for the feature extraction', default='resnet18')
        group.add_argument('--base_encoder_params', type=str, help='Base encoder parameters that are passed to build the feature extraction', default='pretrained=False,spatial_dims=2,n_input_channels=1,num_classes=512')
        group.add_argument('--hidden_dim', type=int, help='Hidden dimension for features output. Should match with output of base_encoder. Default value is 512', default=512)
        group.add_argument('--radius', type=float, help='Radius of icosphere', default=1.5)
        group.add_argument('--subdivision_level', type=int, help='Subdivision level for icosahedron', default=2)
        group.add_argument('--image_size', type=int, help='Image resolution size', default=256)
        group.add_argument('--layer', type=str, help='Layer type for the IcosahedronConv2d', default='IcoConv2D')

        return parent_parser


    def get_features(self,V,F,VF,FF,side):
        x, PF = self.render(V,F,VF,FF)  
        x = getattr(self, f'TimeDistributed{side}')(x)
        x = getattr(self, f'IcosahedronConv2d{side}')(x)
        x = getattr(self, f'pooling{side}')(x) 
        return x


    def render(self,V,F,VF,FF):
        textures = TexturesVertex(verts_features=VF[:, :, :3])
        meshes = Meshes(
            verts=V,
            faces=F,
            textures=textures
        )
        PF = []
        for i in range(self.nbr_cam):
            pix_to_face = self.GetView(meshes,i)
            PF.append(pix_to_face.unsqueeze(dim=1))
        PF = torch.cat(PF, dim=1)

        l_features = []
        for index in range(FF.shape[-1]):
            l_features.append(torch.take(FF[:,index],PF)*(PF >= 0)) # take each feature for each pictures
        x = torch.cat(l_features,dim=2)

        return x, PF


    def training_step(self, train_batch, batch_idx):
        VL, FL, VFL, FFL, VR, FR, VFR, FFR, Y = train_batch
        x = self((VL, FL, VFL, FFL, VR, FR, VFR, FFR))
        loss = self.loss_train(x,Y)
        self.log('train_loss', loss) 
        predictions = torch.argmax(x, dim=1, keepdim=True)
        self.train_accuracy(predictions, Y.reshape(-1, 1))
        self.log("train_acc", self.train_accuracy, batch_size=self.hparams.batch_size)           

        return loss


    def validation_step(self,val_batch,batch_idx):
        VL, FL, VFL, FFL, VR, FR, VFR, FFR, Y = val_batch
        x = self((VL, FL, VFL, FFL, VR, FR, VFR, FFR))
        loss = self.loss_val(x,Y)
        self.log('val_loss', loss)
        predictions = torch.argmax(x, dim=1, keepdim=True)
        val_acc = self.val_accuracy(predictions, Y.reshape(-1, 1))
        self.log("val_acc", val_acc, batch_size=self.hparams.batch_size)

        return val_acc


    def test_step(self,test_batch,batch_idx):
        VL, FL, VFL, FFL, VR, FR, VFR, FFR, Y = test_batch
        x = self((VL, FL, VFL, FFL, VR, FR, VFR, FFR))
        loss = self.loss_test(x,Y)
        self.log('test_loss', loss, batch_size=self.hparams.batch_size)
        predictions = torch.argmax(x, dim=1, keepdim=True)
        output = [predictions,Y]

        return output


    def test_epoch_end(self,input_test):
        y_pred = []
        y_true = []
        for ele in input_test:
            y_pred += ele[0].tolist()
            y_true += ele[1].tolist()
        # target_names = ['No ASD','ASD']
        target_names = ['No QC','QC']
        self.y_pred = y_pred
        self.y_true = y_true
        #Classification report
        print(self.y_pred)
        print(self.y_true)
        print(classification_report(self.y_true, self.y_pred, target_names=target_names))


    def GetView(self,meshes,index):
        phong_renderer = self.hparams.phong_renderer.to(self.device)
        R = self.R[index][None].to(self.device)
        T = self.T[index][None].to(self.device)
        _, fragments = phong_renderer(meshes.clone(),R=R,T=T)
        pix_to_face = fragments.pix_to_face
        pix_to_face = pix_to_face.permute(0,3,1,2)

        return pix_to_face


    def get_y_for_report_classification(self):
        #This function could be called only after test step was done
        return (self.y_pred,self.hparams.y_true)
    

    def Is_it_Icolayer(self,layer):
        return (layer[:3] == 'Ico')


#####################################################################################################################################################################################
#                                                                                                                                                                                   #
#                                                                                       Segmentation                                                                                #
#                                                                                                                                                                                   #
#####################################################################################################################################################################################


class SaxiSegmentation(LightningModule):
    # Saxi segmentation network
    def __init__(self, **kwargs):
        super(SaxiSegmentation, self).__init__()
        self.save_hyperparameters()      

        self.class_weights = None
        if hasattr(self.hparams, 'class_weights'):
            self.class_weights = torch.tensor(self.hparams.class_weights).to(torch.float32)       
            
        self.loss = monai.losses.DiceCELoss(include_background=False, to_onehot_y=True, softmax=True, ce_weight=self.class_weights)
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.out_channels)

        unet = monai.networks.nets.UNet(spatial_dims=2,in_channels=4,out_channels=self.hparams.out_channels, channels=(16, 32, 64, 128, 256),strides=(2, 2, 2, 2),num_res_units=2,)
        self.model = TimeDistributed(unet)

        ico_verts, ico_faces, ico_edges = utils.PolyDataToTensors(utils.CreateIcosahedronSubdivided(radius=self.hparams.radius, sl=self.hparams.subdivision_level))
        ico_verts = ico_verts.to(torch.float32)

        for idx, v in enumerate(ico_verts):
            if (torch.abs(torch.sum(v)) == self.hparams.radius):
                ico_verts[idx] = v + torch.normal(0.0, 1e-7, (3,))

        
        self.register_buffer("ico_verts", ico_verts)

        cameras = FoVPerspectiveCameras()
        raster_settings = RasterizationSettings(image_size=self.hparams.image_size, blur_radius=0, faces_per_pixel=1, max_faces_per_bin=200000)        
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

        lights = AmbientLights()
        self.renderer = MeshRenderer(rasterizer=rasterizer, shader=HardPhongShader(cameras=cameras, lights=lights))

    def configure_optimizers(self):
        # Configure the optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def to(self, device=None):
        # Move the renderer to the specified device
        self.renderer = self.renderer.to(device)
        return super().to(device)

    def forward(self, x):
        # Forward pass
        V, F, CN = x
        X, PF = self.render(V, F, CN)
        x = self.model(X)
        return x, X, PF

    def render(self, V, F, CN):
        # Render the input surface mesh to an image
        textures = TexturesVertex(verts_features=CN)
        meshes = Meshes(verts=V, faces=F, textures=textures)        
        X = []
        PF = []

        for camera_position in self.ico_verts:
            camera_position = camera_position.unsqueeze(0)
            R = look_at_rotation(camera_position, device=self.device)  # (1, 3, 3)
            T = -torch.bmm(R.transpose(1, 2), camera_position[:,:,None])[:, :, 0]   # (1, 3)
            images = self.renderer(meshes_world=meshes.clone(), R=R, T=T)
            fragments = self.renderer.rasterizer(meshes.clone())
            pix_to_face = fragments.pix_to_face
            zbuf = fragments.zbuf
            images = torch.cat([images[:,:,:,0:3], zbuf], dim=-1)
            images = images.permute(0,3,1,2)
            pix_to_face = pix_to_face.permute(0,3,1,2)
            X.append(images.unsqueeze(1))
            PF.append(pix_to_face.unsqueeze(1))
        
        X = torch.cat(X, dim=1)
        PF = torch.cat(PF, dim=1)        

        return X, PF
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("SaxiSegmentation")

        group.add_argument("--lr", type=float, default=1e-4)
        group.add_argument("--crown_segmentation", type=bool, default=False)
        
        # Encoder parameters
        group.add_argument('--base_encoder', type=str, help='Base encoder for the feature extraction', default='resnet18')
        group.add_argument('--base_encoder_params', type=str, help='Base encoder parameters that are passed to build the feature extraction', default='pretrained=False,spatial_dims=2,n_input_channels=4,num_classes=512')
        group.add_argument('--hidden_dim', type=int, help='Hidden dimension for features output. Should match with output of base_encoder. Default value is 512', default=512)
        group.add_argument('--radius', type=float, help='Radius of icosphere', default=1.5)
        group.add_argument('--subdivision_level', type=int, help='Subdivision level for icosahedron', default=2)
        group.add_argument('--image_size', type=int, help='Image resolution size', default=256)
        group.add_argument('--out_channels', type=int, help='Output number of classes', default=4)

        return parent_parser

    def training_step(self, train_batch, batch_idx):
        # Training step
        V, F, YF, CN = train_batch
        V = V.to(self.device, non_blocking=True)
        F = F.to(self.device, non_blocking=True)
        YF = YF.to(self.device, non_blocking=True)
        CN = CN.to(self.device, non_blocking=True).to(torch.float32)
        x, X, PF = self((V, F, CN))
        y = torch.take(YF, PF).to(torch.int64)*(PF >= 0)
        x = x.permute(0, 2, 1, 3, 4) #batch, time, channels, H, W -> batch, channels, time, H, W
        y = y.permute(0, 2, 1, 3, 4)
        loss = self.loss(x, y)
        batch_size = V.shape[0]
        self.log('train_loss', loss, batch_size=batch_size)
        self.accuracy(torch.argmax(x, dim=1, keepdim=True).reshape(-1, 1), y.reshape(-1, 1).to(torch.int32))
        self.log("train_acc", self.accuracy, batch_size=batch_size)

        return loss


    def validation_step(self, val_batch, batch_idx):
        # Validation step
        V, F, YF, CN = val_batch
        V = V.to(self.device, non_blocking=True)
        F = F.to(self.device, non_blocking=True)
        YF = YF.to(self.device, non_blocking=True)
        CN = CN.to(self.device, non_blocking=True).to(torch.float32)
        x, X, PF = self((V, F, CN))
        y = torch.take(YF, PF).to(torch.int64)*(PF >= 0)
        x = x.permute(0, 2, 1, 3, 4) #batch, time, channels, H, W -> batch, channels, time, H, W
        y = y.permute(0, 2, 1, 3, 4)
        loss = self.loss(x, y)
        batch_size = V.shape[0]
        self.accuracy(torch.argmax(x, dim=1, keepdim=True).reshape(-1, 1), y.reshape(-1, 1).to(torch.int32))
        self.log("val_acc", self.accuracy, batch_size=batch_size, sync_dist=True)
        self.log('val_loss', loss, batch_size=batch_size, sync_dist=True)


    def test_step(self, batch, batch_idx):
        # Test step
        V, F, YF, CN = val_batch
        x, X, PF = self(V, F, CN)
        y = torch.take(YF, PF).to(torch.int64)*(PF >= 0)
        x = x.permute(0, 2, 1, 3, 4) #batch, time, channels, h, w -> batch, channels, time, h, w
        y = y.permute(0, 2, 1, 3, 4) 
        loss = self.loss(x, y)
        self.accuracy(torch.argmax(x, dim=1, keepdim=True).reshape(-1, 1), y.reshape(-1, 1).to(torch.int32))        
        return {'test_loss': loss, 'test_correct': self.accuracy}


#####################################################################################################################################################################################
#                                                                                                                                                                                   #
#                                                                                 Dental Model Segmentation                                                                         #
#                                                                                                                                                                                   #
#####################################################################################################################################################################################


class DentalModelSeg(LightningModule):
    def __init__(self, image_size=320, radius=1.35, subdivision_level=2, train_sphere_samples=4, custom_model=None, device='cuda:0'):
        super(DentalModelSeg, self).__init__()        
        self.save_hyperparameters()        
        self.config_path = os.path.join(os.path.dirname(__file__), "dental_model_path.json")
        self.out_channels = 34
        self.class_weights = None
        self.custom_model = custom_model

        model = self.create_unet_model()

        if custom_model is None:
             # Use the default UNet model
            model_loaded = self.dental_model_seg(model)
            self.model = TimeDistributed(model_loaded)
        else:
            # Use the provided custom model
            self.custom_model = self.dental_model_seg_args(model)
            self.model = TimeDistributed(self.custom_model)
        

        ico = utils.CreateIcosahedronSubdivided(self.hparams.radius, self.hparams.subdivision_level)

        ico_verts, ico_faces, ico_edges = utils.PolyDataToTensors(ico)
        ico_verts = ico_verts.to(torch.float32)


        for idx, v in enumerate(ico_verts):
            if (torch.abs(torch.sum(v)) == radius):
                ico_verts[idx] = v + torch.normal(0.0, 1e-7, (3,))
        
        self.register_buffer("ico_verts", ico_verts)

        cameras = FoVPerspectiveCameras()
        raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0, faces_per_pixel=1, max_faces_per_bin=200000)        
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

        lights = AmbientLights()
        self.renderer = MeshRenderer(rasterizer=rasterizer, shader=HardPhongShader(cameras=cameras, lights=lights))

    def create_unet_model(self):
        model = monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=4,   # images: torch.cuda.FloatTensor[batch_size,224,224,4]
            out_channels=self.out_channels, 
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
        return model

    def dental_model_seg(self, model):
        with open(self.config_path, 'r') as f:
            config = json.load(f)
            model_url = config.get('dental', {}).get('url')
            if not model_url:
                print("Error: Model URL not found in the config file.")
                return None
            state_dict = torch.hub.load_state_dict_from_url(model_url)
            model.load_state_dict(state_dict)
            print(bcolors.INFO, "Model loaded successfully", bcolors.ENDC)
            return model
    
    def dental_model_seg_args(self, model):
        state_dict = torch.load(self.custom_model)
        model.load_state_dict(state_dict)
        print(bcolors.INFO, "Model loaded successfully", bcolors.ENDC)
        return model

    def to(self, device=None):
        # Move the renderer to the specified device
        self.renderer = self.renderer.to(device)
        return super().to(device)

    def forward(self, x):
        # Forward pass
        V, F, CN = x
        X, PF = self.render(V, F, CN)
        x = self.model(X)
        return x, X, PF

    def render(self, V, F, CN):
        # Render the input surface mesh to an image
        textures = TexturesVertex(verts_features=CN)
        meshes = Meshes(verts=V, faces=F, textures=textures)        
        X = []
        PF = []

        for camera_position in self.ico_verts:
            camera_position = camera_position.unsqueeze(0)
            R = look_at_rotation(camera_position, device=self.device)  # (1, 3, 3)
            T = -torch.bmm(R.transpose(1, 2), camera_position[:,:,None])[:, :, 0]   # (1, 3)
            images = self.renderer(meshes_world=meshes.clone(), R=R, T=T)
            fragments = self.renderer.rasterizer(meshes.clone())
            pix_to_face = fragments.pix_to_face
            zbuf = fragments.zbuf
            images = torch.cat([images[:,:,:,0:3], zbuf], dim=-1)
            images = images.permute(0,3,1,2)
            pix_to_face = pix_to_face.permute(0,3,1,2)
            X.append(images.unsqueeze(1))
            PF.append(pix_to_face.unsqueeze(1))
        
        X = torch.cat(X, dim=1)
        PF = torch.cat(PF, dim=1)        

        return X, PF
    
    def configure_optimizers(self):
        pass

    def training_step(self):
        pass

    def validation_step(self):
        pass

    def test_step(self):
        pass



class SaxiMHA(LightningModule):
    def __init__(self, **kwargs):
        super(SaxiMHA, self).__init__()
        self.save_hyperparameters()
        self.y_pred = []
        self.y_true = []

        # Create the icosahedrons form each level
        ico_12 = utils.CreateIcosahedron(self.hparams.radius) # 12 vertices

        if self.hparams.subdivision_level == 2:
            ico = utils.SubdividedIcosahedron(ico_12,2,self.hparams.radius) # 42 vertices

        elif self.hparams.subdivision_level == 3:
            ico = utils.SubdividedIcosahedron(ico_12,2,self.hparams.radius) # 42 vertices
            ico = utils.SubdividedIcosahedron(ico,2,self.hparams.radius) # 162 vertices
        
        ico_sphere_verts, ico_sphere_faces, ico_sphere_edges = utils.PolyDataToTensors(ico)

        self.ico_sphere_verts = ico_sphere_verts
        self.ico_sphere_edges = ico_sphere_edges

        R=[]
        T=[]
        for coords_cam in self.ico_sphere_verts.tolist():
            camera_position = torch.FloatTensor([coords_cam])
            R_current = look_at_rotation(camera_position)
            # check if camera coords vector and up vector for R are collinear
            if torch.equal(torch.cross(camera_position,torch.tensor([[0.,1.,0.]])),torch.tensor([[0., 0., 0.]])):
               R_current = look_at_rotation(camera_position, up = torch.tensor([[0.0, 0.0, 1.0]]),)
            T_current = -torch.bmm(R_current.transpose(1, 2), camera_position[:,:,None])[:, :, 0]   # (1, 3)

            R.append(R_current)
            T.append(T_current)

        self.R=torch.cat(R)
        self.T=torch.cat(T)
        self.nbr_cam = len(self.R)

        self.drop = nn.Dropout(p=self.hparams.dropout_lvl)

        # Left network
        self.create_network('L')
        # Right network
        self.create_network('R')

        # Loss
        self.loss_train = nn.CrossEntropyLoss()
        self.loss_val = nn.CrossEntropyLoss()
        self.loss_test = nn.CrossEntropyLoss()

        # Final layer
        self.Classification = nn.Linear(2*ico.GetNumberOfPoints(), self.hparams.out_classes)

        #vAccuracy
        self.train_accuracy = torchmetrics.Accuracy('multiclass',num_classes=self.hparams.out_classes,average='macro')
        self.val_accuracy = torchmetrics.Accuracy('multiclass',num_classes=self.hparams.out_classes,average='macro')
        
        # Initialize a perspective camera.
        self.hparams.cameras = FoVPerspectiveCameras()

        # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
        raster_settings = RasterizationSettings(image_size=self.hparams.image_size,blur_radius=0,faces_per_pixel=1,max_faces_per_bin=100000)

        lights = AmbientLights()
        rasterizer = MeshRasterizer(cameras=self.hparams.cameras,raster_settings=raster_settings)

        self.hparams.phong_renderer = MeshRendererWithFragments(rasterizer=rasterizer,shader=HardPhongShader(cameras=self.hparams.cameras, lights=lights))


    def to(self, device=None):
        # Move the renderer to the specified device
        self.hparams.phong_renderer = self.hparams.phong_renderer.to(device)
        return super().to(device)

    
    def create_network(self, side):
        # Create an icosphere
        if hasattr(monai.networks.nets, self.hparams.base_encoder):
            template_model = getattr(monai.networks.nets, self.hparams.base_encoder)
        elif hasattr(torchvision.models, self.hparams.base_encoder):
            template_model = getattr(torchvision.models, self.hparams.base_encoder)
        else:
            raise "{base_encoder} not in monai networks or torchvision".format(base_encoder=self.hparams.base_encoder)

        model_params = eval('dict(%s)' % self.hparams.base_encoder_params.replace(' ',''))

        self.embed_dim = 512
        self.num_heads = 8
        
        self.convnet = template_model(**model_params)
        setattr(self, f'TimeDistributed{side}', TimeDistributed(self.convnet))
        setattr(self, f'MHA{side}', MultiHeadAttentionModule(self.embed_dim, self.num_heads, batch_first=True))
        setattr(self, f'W{side}', nn.Linear(self.hparams.hidden_dim, self.hparams.out_size))
        setattr(self, f'Attention{side}', SelfAttention(self.hparams.hidden_dim, self.hparams.out_size, dim=2))


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def forward(self, x):
        VL, FL, VFL, FFL, VR, FR, VFR, FFR = x
        # TimeDistributed
        xL = self.get_features(VL,FL,VFL,FFL,'L')
        xR = self.get_features(VR,FR,VFR,FFR,'R')
        l_left_right = [xL,xR]
        x = torch.cat(l_left_right,dim=1)
        # Last classification layer
        x = self.drop(x)
        x = self.Classification(x)

        return x

    def get_features(self,V,F,VF,FF,side):
        x, PF = self.render(V,F,VF,FF)  
        x = getattr(self, f'TimeDistributed{side}')(x)
        x, _ = getattr(self, f'MHA{side}')(x,x,x)
        values = getattr(self, f'W{side}')(x)
        x, _ = getattr(self, f'Attention{side}')(x,values)
        return x


    def render(self,V,F,VF,FF):
        # textures = TexturesVertex(verts_features=VF[:, :, :3])
        if VF.shape[-1] < 3:
            padding = torch.zeros(VF.shape[0], VF.shape[1], 3 - VF.shape[-1], device=VF.device)
            VF = torch.cat([VF, padding], dim=-1)
        elif VF.shape[-1] > 3:
            VF = VF[:, :, :3]
        textures = TexturesVertex(verts_features=VF)

        meshes = Meshes(
            verts=V,
            faces=F,
            textures=textures
        )
        PF = []
        for i in range(self.nbr_cam):
            pix_to_face = self.GetView(meshes,i)
            PF.append(pix_to_face.unsqueeze(dim=1))
        PF = torch.cat(PF, dim=1)

        l_features = []
        for index in range(FF.shape[-1]):
            l_features.append(torch.take(FF[:,index],PF)*(PF >= 0)) # take each feature for each pictures
        x = torch.cat(l_features,dim=2)

        return x, PF
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("SaxiMHA")
        
        group.add_argument("--lr", type=float, default=1e-4)
        group.add_argument("--fs_path", type=str, default=None)
        group.add_argument("--out_classes", type=int, default=2)
        group.add_argument("--out_size", type=int, default=256)
        group.add_argument('--dropout_lvl',  type=float, help='Dropout level (default: 0.2)', default=0.2)

        # Encoder parameters
        group.add_argument('--base_encoder', type=str, help='Base encoder for the feature extraction', default='resnet18')
        group.add_argument('--base_encoder_params', type=str, help='Base encoder parameters that are passed to build the feature extraction', default='pretrained=False,spatial_dims=2,n_input_channels=1,num_classes=512')
        group.add_argument('--hidden_dim', type=int, help='Hidden dimension for features output. Should match with output of base_encoder. Default value is 512', default=512)
        group.add_argument('--radius', type=float, help='Radius of icosphere', default=1.5)
        group.add_argument('--subdivision_level', type=int, help='Subdivision level for icosahedron', default=2)
        group.add_argument('--image_size', type=int, help='Image resolution size', default=256)

        return parent_parser


    def training_step(self, train_batch, batch_idx):
        VL, FL, VFL, FFL, VR, FR, VFR, FFR, Y = train_batch
        x = self((VL, FL, VFL, FFL, VR, FR, VFR, FFR))
        loss = self.loss_train(x,Y)
        self.log('train_loss', loss) 
        predictions = torch.argmax(x, dim=1, keepdim=True)
        self.train_accuracy(predictions, Y.reshape(-1, 1))
        self.log("train_acc", self.train_accuracy, batch_size=self.hparams.batch_size)           

        return loss


    def validation_step(self,val_batch,batch_idx):
        VL, FL, VFL, FFL, VR, FR, VFR, FFR, Y = val_batch
        x = self((VL, FL, VFL, FFL, VR, FR, VFR, FFR))
        loss = self.loss_val(x,Y)
        self.log('val_loss', loss, sync_dist=True)
        predictions = torch.argmax(x, dim=1, keepdim=True)
        val_acc = self.val_accuracy(predictions, Y.reshape(-1, 1))
        self.log("val_acc", val_acc, batch_size=self.hparams.batch_size)

        return val_acc


    def test_step(self,test_batch,batch_idx):
        VL, FL, VFL, FFL, VR, FR, VFR, FFR, Y = test_batch
        x = self((VL, FL, VFL, FFL, VR, FR, VFR, FFR))
        loss = self.loss_test(x,Y)
        self.log('test_loss', loss, batch_size=self.hparams.batch_size)
        predictions = torch.argmax(x, dim=1, keepdim=True)
        output = [predictions,Y]

        return output


    def test_epoch_end(self,input_test):
        y_pred = []
        y_true = []
        for ele in input_test:
            y_pred += ele[0].tolist()
            y_true += ele[1].tolist()
        target_names = ['No QC','QC']
        self.y_pred = y_pred
        self.y_true = y_true
        #Classification report
        print(self.y_pred)
        print(self.y_true)
        print(classification_report(self.y_true, self.y_pred, target_names=target_names))


    def GetView(self,meshes,index):
        phong_renderer = self.hparams.phong_renderer.to(self.device)
        R = self.R[index][None].to(self.device)
        T = self.T[index][None].to(self.device)
        _, fragments = phong_renderer(meshes.clone(),R=R,T=T)
        pix_to_face = fragments.pix_to_face
        pix_to_face = pix_to_face.permute(0,3,1,2)

        return pix_to_face


    def get_y_for_report_classification(self):
        #This function could be called only after test step was done
        return (self.y_pred,self.hparams.y_true)
    

    def Is_it_Icolayer(self,layer):
        return (layer[:3] == 'Ico')


class SaxiRing_QC(LightningModule):
    def __init__(self, **kwargs):
        super(SaxiRing_QC, self).__init__()
        self.save_hyperparameters()
        self.y_pred = []
        self.y_true = []

        # Create the icosahedrons form each level
        ico_12 = utils.CreateIcosahedron(self.hparams.radius) # 12 vertices
        ico_42 = utils.SubdividedIcosahedron(ico_12,2,self.hparams.radius) # 42 vertices
        ico_162 = utils.SubdividedIcosahedron(ico_42,2,self.hparams.radius) # 162 vertices

        # Get the neighbors to go form level N to level N-1
        ring_neighs_42 = utils.GetPreservedPointIds(ico_12,ico_42)
        ring_neighs_162 = utils.GetPreservedPointIds(ico_42,ico_162)

        # Create the down blocks to go from 162 -> 42 -> 12
        self.down1 = AttentionRings(self.hparams.hidden_dim, self.hparams.hidden_dim, self.hparams.hidden_dim, ring_neighs_162)
        self.down2 = AttentionRings(self.hparams.hidden_dim, self.hparams.hidden_dim, self.hparams.hidden_dim, ring_neighs_42)

        ico_sphere_verts, ico_sphere_faces, ico_sphere_edges = utils.PolyDataToTensors(ico_162)

        self.ico_sphere_verts = ico_sphere_verts
        self.ico_sphere_edges = ico_sphere_edges

        R=[]
        T=[]
        for coords_cam in self.ico_sphere_verts.tolist():
            camera_position = torch.FloatTensor([coords_cam])
            R_current = look_at_rotation(camera_position)
            # check if camera coords vector and up vector for R are collinear
            if torch.equal(torch.cross(camera_position,torch.tensor([[0.,1.,0.]])),torch.tensor([[0., 0., 0.]])):
               R_current = look_at_rotation(camera_position, up = torch.tensor([[0.0, 0.0, 1.0]]),)
            T_current = -torch.bmm(R_current.transpose(1, 2), camera_position[:,:,None])[:, :, 0]   # (1, 3)

            R.append(R_current)
            T.append(T_current)

        self.R=torch.cat(R)
        self.T=torch.cat(T)
        self.nbr_cam = len(self.R)

        self.drop = nn.Dropout(p=self.hparams.dropout_lvl)

        # Left network
        self.create_network('L')
        # Right network
        self.create_network('R')

        # Loss
        self.loss_train = nn.CrossEntropyLoss()
        self.loss_val = nn.CrossEntropyLoss()
        self.loss_test = nn.CrossEntropyLoss()

        # Pooling layer
        self.W = nn.Linear(self.hparams.hidden_dim, self.hparams.out_size)
        self.Att = SelfAttention(self.hparams.hidden_dim, self.hparams.out_size, dim=2)
        
        # Final layer
        self.Classification = nn.Linear(2*ico_12.GetNumberOfPoints(), self.hparams.out_classes)

        #vAccuracy
        self.train_accuracy = torchmetrics.Accuracy('multiclass',num_classes=self.hparams.out_classes,average='macro')
        self.val_accuracy = torchmetrics.Accuracy('multiclass',num_classes=self.hparams.out_classes,average='macro')
        
        # Initialize a perspective camera.
        self.hparams.cameras = FoVPerspectiveCameras()

        # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
        raster_settings = RasterizationSettings(image_size=self.hparams.image_size,blur_radius=0,faces_per_pixel=1,max_faces_per_bin=100000)

        lights = AmbientLights()
        rasterizer = MeshRasterizer(cameras=self.hparams.cameras,raster_settings=raster_settings)

        self.hparams.phong_renderer = MeshRendererWithFragments(rasterizer=rasterizer,shader=HardPhongShader(cameras=self.hparams.cameras, lights=lights))


    def to(self, device=None):
        # Move the renderer to the specified device
        self.hparams.phong_renderer = self.hparams.phong_renderer.to(device)
        return super().to(device)

    
    def create_network(self, side):
        # Create an icosphere
        if hasattr(monai.networks.nets, self.hparams.base_encoder):
            template_model = getattr(monai.networks.nets, self.hparams.base_encoder)
        elif hasattr(torchvision.models, self.hparams.base_encoder):
            template_model = getattr(torchvision.models, self.hparams.base_encoder)
        else:
            raise "{base_encoder} not in monai networks or torchvision".format(base_encoder=self.hparams.base_encoder)

        model_params = eval('dict(%s)' % self.hparams.base_encoder_params.replace(' ',''))
        
        self.convnet = template_model(**model_params)
        setattr(self, f'TimeDistributed{side}', TimeDistributed(self.convnet))


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def forward(self, x):
        VL, FL, VFL, FFL, VR, FR, VFR, FFR = x
        VL = VL.to(self.device,non_blocking=True)
        FL = FL.to(self.device,non_blocking=True)
        VFL = VFL.to(self.device,non_blocking=True)
        FFL = FFL.to(self.device,non_blocking=True)
        VR = VR.to(self.device,non_blocking=True)
        FR = FR.to(self.device,non_blocking=True)
        VFR = VFR.to(self.device,non_blocking=True)
        FFR = FFR.to(self.device,non_blocking=True)
        # TimeDistributed
        xL = self.get_features(VL,FL,VFL,FFL,'L')
        xR = self.get_features(VR,FR,VFR,FFR,'R')
        xL, scoreL = self.down1(xL)
        xL, scoreL = self.down2(xL)
        xR, scoreR = self.down1(xR)
        xR, scoreR = self.down2(xR)
        # Add attention layer
        valuesL = self.W(xL)
        valuesR = self.W(xR)
        xL, score = self.Att(xL,valuesL)
        xR, score = self.Att(xR,valuesR)
        l_left_right = [xL,xR]
        x = torch.cat(l_left_right,dim=1)
        # Last classification layer
        x = self.drop(x)
        x = self.Classification(x)

        return x


    def get_features(self,V,F,VF,FF,side):
        x, PF = self.render(V,F,VF,FF)  
        x = getattr(self, f'TimeDistributed{side}')(x)
        return x


    def render(self,V,F,VF,FF):
        if VF.shape[-1] < 3:
            padding = torch.zeros(VF.shape[0], VF.shape[1], 3 - VF.shape[-1], device=VF.device)
            VF = torch.cat([VF, padding], dim=-1)
        elif VF.shape[-1] > 3:
            VF = VF[:, :, :3]
        textures = TexturesVertex(verts_features=VF[:, :, :3])
        meshes = Meshes(
            verts=V,
            faces=F,
            textures=textures
        )
        PF = []
        for i in range(self.nbr_cam):
            pix_to_face = self.GetView(meshes,i)
            PF.append(pix_to_face.unsqueeze(dim=1))
        PF = torch.cat(PF, dim=1)

        l_features = []
        for index in range(FF.shape[-1]):
            l_features.append(torch.take(FF[:,index],PF)*(PF >= 0)) # take each feature for each pictures
        x = torch.cat(l_features,dim=2)

        return x, PF
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("SaxiMHA")
        
        group.add_argument("--lr", type=float, default=1e-4)
        group.add_argument("--fs_path", type=str, default=None)
        group.add_argument("--out_classes", type=int, default=2)
        group.add_argument("--out_size", type=int, default=256)
        group.add_argument('--dropout_lvl',  type=float, help='Dropout level (default: 0.2)', default=0.2)

        # Encoder parameters
        group.add_argument('--base_encoder', type=str, help='Base encoder for the feature extraction', default='resnet18')
        group.add_argument('--base_encoder_params', type=str, help='Base encoder parameters that are passed to build the feature extraction', default='pretrained=False,spatial_dims=2,n_input_channels=1,num_classes=512')
        group.add_argument('--hidden_dim', type=int, help='Hidden dimension for features output. Should match with output of base_encoder. Default value is 512', default=512)
        group.add_argument('--radius', type=float, help='Radius of icosphere', default=1.5)
        group.add_argument('--subdivision_level', type=int, help='Subdivision level for icosahedron', default=2)
        group.add_argument('--image_size', type=int, help='Image resolution size', default=256)

        return parent_parser


    def training_step(self, train_batch, batch_idx):
        VL, FL, VFL, FFL, VR, FR, VFR, FFR, Y = train_batch
        x = self((VL, FL, VFL, FFL, VR, FR, VFR, FFR))
        loss = self.loss_train(x,Y)
        self.log('train_loss', loss) 
        predictions = torch.argmax(x, dim=1, keepdim=True)
        self.train_accuracy(predictions, Y.reshape(-1, 1))
        self.log("train_acc", self.train_accuracy, batch_size=self.hparams.batch_size)           

        return loss


    def validation_step(self,val_batch,batch_idx):
        VL, FL, VFL, FFL, VR, FR, VFR, FFR, Y = val_batch
        x = self((VL, FL, VFL, FFL, VR, FR, VFR, FFR))
        loss = self.loss_val(x,Y)
        self.log('val_loss', loss)
        predictions = torch.argmax(x, dim=1, keepdim=True)
        val_acc = self.val_accuracy(predictions, Y.reshape(-1, 1))
        self.log("val_acc", val_acc, batch_size=self.hparams.batch_size)

        return val_acc


    def test_step(self,test_batch,batch_idx):
        VL, FL, VFL, FFL, VR, FR, VFR, FFR, Y = test_batch
        x = self((VL, FL, VFL, FFL, VR, FR, VFR, FFR))
        loss = self.loss_test(x,Y)
        self.log('test_loss', loss, batch_size=self.hparams.batch_size)
        predictions = torch.argmax(x, dim=1, keepdim=True)
        output = [predictions,Y]

        return output


    def test_epoch_end(self,input_test):
        y_pred = []
        y_true = []
        for ele in input_test:
            y_pred += ele[0].tolist()
            y_true += ele[1].tolist()
        target_names = ['No QC','QC']
        self.y_pred = y_pred
        self.y_true = y_true
        #Classification report
        print(self.y_pred)
        print(self.y_true)
        print(classification_report(self.y_true, self.y_pred, target_names=target_names))


    def GetView(self,meshes,index):
        phong_renderer = self.hparams.phong_renderer.to(self.device)
        R = self.R[index][None].to(self.device)
        T = self.T[index][None].to(self.device)
        _, fragments = phong_renderer(meshes.clone(),R=R,T=T)
        pix_to_face = fragments.pix_to_face
        pix_to_face = pix_to_face.permute(0,3,1,2)

        return pix_to_face


    def get_y_for_report_classification(self):
        #This function could be called only after test step was done
        return (self.y_pred,self.hparams.y_true)
    

    def Is_it_Icolayer(self,layer):
        return (layer[:3] == 'Ico')


class SaxiAE(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = MHAEncoder(input_dim=self.hparams.input_dim, 
                                      embed_dim=self.hparams.embed_dim, 
                                      num_heads=self.hparams.num_heads, 
                                      output_dim=self.hparams.output_dim, 
                                      sample_levels=self.hparams.sample_levels, 
                                      dropout=self.hparams.dropout, 
                                      K=self.hparams.K)
        
        self.decoder = MHADecoder(input_dim=self.hparams.output_dim,                                       
                                      embed_dim=self.hparams.embed_dim, 
                                      output_dim=self.hparams.input_dim, 
                                      num_heads=self.hparams.num_heads, 
                                      sample_levels=len(self.hparams.sample_levels),
                                      K=self.hparams.K, 
                                      dropout=self.hparams.dropout)
        
        self.ff_mu = FeedForward(self.hparams.output_dim, hidden_dim=self.hparams.output_dim, dropout=self.hparams.dropout)
        self.ff_sigma = FeedForward(self.hparams.output_dim, hidden_dim=self.hparams.output_dim, dropout=self.hparams.dropout)
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("SaxiAE")

        group.add_argument("--lr", type=float, default=1e-4)
        group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=0.01)
        
        group.add_argument("--input_dim", type=int, default=3, help='Input dimension for the encoder')
        group.add_argument("--embed_dim", type=int, default=128, help='Embedding dimension')
        group.add_argument("--num_heads", type=int, default=32, help='Number of attention heads')
        group.add_argument("--output_dim", type=int, default=256, help='Output dimension from the encoder')
        group.add_argument("--sample_levels", type=int, default=[16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64], nargs="+", help='Number of sampling levels in the encoder')
        group.add_argument("--sample_level_loss", type=int, default=1000, help='Number of samples for loss')

        
        group.add_argument("--hidden_dim", type=int, default=64, help='Hidden dimension size')
        group.add_argument("--dropout", type=float, default=0.1, help='Dropout rate')
        
        # Decoder parameters
        group.add_argument("--K", type=int, default=64, help='Top K nearest neighbors to consider in the encoder')        
        group.add_argument("--loss_chamfer_weight", type=float, default=1.0, help='Loss weight for the chamfer distance')
        # group.add_argument("--loss_point_triangle_weight", type=float, default=0.1, help='Loss weight for the point to nearest face plane distance')
        group.add_argument("--loss_mesh_face_weight", type=float, default=1.0, help='Loss weight for the mesh face distance')
        group.add_argument("--loss_mesh_edge_weight", type=float, default=1.0, help='Loss weight for the mesh edge distance')        
        group.add_argument("--loss_dist_weight", type=float, default=0.01, help='Loss weight for the distance between points, should be maximized to try an avoid clumps')        

        return parent_parser
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.ff_mu.parameters()) + list(self.ff_sigma.parameters()),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)        
        return optimizer
    
    def create_mesh(self, V, F):
        return Meshes(verts=V, faces=F)
    
    def create_mesh_from_points(self, X):
        dists = knn_points(X, X, K=3)
        F = dists.idx
        return Meshes(verts=X, faces=F)
    
    def sample_points_from_meshes(self, x_mesh, Ns, return_normals=False):
        if return_normals:
            x, x_N = sample_points_from_meshes(x_mesh, Ns, return_normals=True)
            return x, x_N
        return sample_points_from_meshes(x_mesh, Ns)
    
    def sample_points(self, x, Ns):
        return self.decoder.sample_points(x, Ns)

    def compute_loss(self, X_mesh, X_hat, step="train", sync_dist=False):
        
        X = self.sample_points_from_meshes(X_mesh, self.hparams.sample_level_loss)
        
        loss_chamfer, _ = chamfer_distance(X, X_hat, batch_reduction="mean", point_reduction="sum")        

        X_hat_PC = Pointclouds(X_hat)
        loss_point_mesh_face = point_mesh_face_distance(X_mesh, X_hat_PC)
        loss_point_mesh_edge = point_mesh_edge_distance(X_mesh, X_hat_PC)


        dists = knn_points(X_hat, X_hat, K=self.hparams.K)
        loss_dist = 1.0/(torch.sum(dists.dists) + 1e-6)

        loss = loss_chamfer*self.hparams.loss_chamfer_weight + loss_point_mesh_face*self.hparams.loss_mesh_face_weight + loss_point_mesh_edge*self.hparams.loss_mesh_edge_weight + loss_dist*self.hparams.loss_dist_weight

        self.log(f"{step}_loss", loss, sync_dist=sync_dist)        
        self.log(f"{step}_loss_chamfer", loss_chamfer, sync_dist=sync_dist)
        self.log(f"{step}_loss_point_mesh_face", loss_point_mesh_face, sync_dist=sync_dist)
        self.log(f"{step}_loss_point_mesh_edge", loss_point_mesh_edge, sync_dist=sync_dist)
        self.log(f"{step}_loss_dist", loss_dist, sync_dist=sync_dist)

        return loss

    def training_step(self, train_batch, batch_idx):
        V, F = train_batch
        
        X_mesh = self.create_mesh(V, F)

        X = self.sample_points_from_meshes(X_mesh, self.hparams.sample_levels[0])

        X_hat = self(X)

        loss = self.compute_loss(X_mesh, X_hat)

        return loss

    def validation_step(self, val_batch, batch_idx):
        
        V, F = val_batch
        
        X_mesh = self.create_mesh(V, F)

        X = self.sample_points_from_meshes(X_mesh, self.hparams.sample_levels[0])

        X_hat = self(X)

        self.compute_loss(X_mesh, X_hat, step="val", sync_dist=True)

    def sampling(self, z_mu: torch.Tensor, z_sigma: torch.Tensor) -> torch.Tensor:        
        eps = torch.randn_like(z_sigma)
        z_vae = z_mu + eps * z_sigma
        return z_vae

    def forward(self, X):        
        h, w = self.encoder(X)
        z_mu = self.ff_mu(h)
        z_sigma = self.ff_sigma(h)
        z = self.sampling(z_mu, z_sigma)
        X_hat = self.decoder(z)
        return X_hat
    
class SaxiIdxAE(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = MHAIdxEncoder(input_dim=self.hparams.input_dim, 
                                     output_dim=self.hparams.stages[-1], 
                                     K=self.hparams.K, 
                                     num_heads=self.hparams.num_heads, 
                                     stages=self.hparams.stages, 
                                     dropout=self.hparams.dropout, 
                                     pooling_factor=self.hparams.pooling_factor, 
                                     pooling_hidden_dim=self.hparams.pooling_hidden_dim,
                                     score_pooling=self.hparams.score_pooling,
                                     feed_forward_hidden_dim=self.hparams.feed_forward_hidden_dim, 
                                     use_skip_connection=self.hparams.use_skip_connection)
        
        self.decoder = MHAIdxDecoder(input_dim=self.hparams.stages[-1], 
                                     output_dim=self.hparams.output_dim, 
                                     K=self.hparams.K[::-1], 
                                     num_heads=self.hparams.num_heads[::-1], 
                                     stages=self.hparams.stages[::-1], 
                                     dropout=self.hparams.dropout, 
                                     pooling_hidden_dim=self.hparams.pooling_hidden_dim[::-1] if self.hparams.pooling_hidden_dim is not None else None,
                                     feed_forward_hidden_dim=self.hparams.feed_forward_hidden_dim[::-1] if self.hparams.feed_forward_hidden_dim is not None else None,
                                     use_skip_connection=self.hparams.use_skip_connection)
        
        self.ff_mu = FeedForward(self.hparams.stages[-1], hidden_dim=self.hparams.stages[-1], dropout=self.hparams.dropout)
        self.ff_sigma = FeedForward(self.hparams.stages[-1], hidden_dim=self.hparams.stages[-1], dropout=self.hparams.dropout)

        self.loss_fn = nn.MSELoss(reduction='sum')
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("SaxiIdxAE")

        
        group.add_argument("--lr", type=float, default=1e-4)
        group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=0.01)
        
        # Encoder/Decoder params
        group.add_argument("--num_samples", type=int, default=8192, help='Number of samples to take from the mesh to start the encoding')
        group.add_argument("--input_dim", type=int, default=3, help='Input dimension for the encoder')
        group.add_argument("--output_dim", type=int, default=3, help='Output dimension of the model')
        group.add_argument("--K", type=int, nargs="*", default=[27, 27], help='Number of K neighbors for each stage')
        group.add_argument("--num_heads", type=int, nargs="*", default=[64, 128], help='Number of attention heads per stage the encoder')
        group.add_argument("--stages", type=int, nargs="*", default=[64, 128], help='Dimension per stage')
        group.add_argument("--dropout", type=float, default=0.1, help='Dropout rate')
        group.add_argument("--pooling_factor", type=float, nargs="*", default=[0.5, 0.5], help='Pooling factor')
        group.add_argument("--score_pooling", type=int, default=0, help='Use score base pooling')
        group.add_argument("--pooling_hidden_dim", type=int, nargs="*", default=[32, 64], help='Hidden dim for the pooling layer')
        group.add_argument("--feed_forward_hidden_dim", type=int, nargs="*", default=[32, 64], help='Hidden dim for the Residual FeedForward layer')
        group.add_argument("--use_skip_connection", type=int, default=0, help='Use skip connections, i.e., unet style network')

        # group.add_argument("--loss_mse_weight", type=float, default=1.0, help='Loss weight for the chamfer distance')
        group.add_argument("--loss_chamfer_weight", type=float, default=1.0, help='Loss weight for the chamfer distance')
        group.add_argument("--loss_mesh_face_weight", type=float, default=1.0, help='Loss weight for the mesh face distance')
        group.add_argument("--loss_mesh_edge_weight", type=float, default=1.0, help='Loss weight for the mesh edge distance')
        

        return parent_parser
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.ff_mu.parameters()) + list(self.ff_sigma.parameters()),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)        
        return optimizer
    
    def create_mesh(self, V, F):
        return Meshes(verts=V, faces=F)
    
    def create_mesh_from_points(self, X):
        dists = knn_points(X, X, K=3)
        F = dists.idx
        return Meshes(verts=X, faces=F)
    
    def sample_points_from_meshes(self, x_mesh, Ns, return_normals=False):
        return sample_points_from_meshes(x_mesh, Ns, return_normals=return_normals)
    
    def sample_points(self, x, Ns):
        """
        Samples Ns points from each batch in a tensor of shape (Bs, N, F).

        Args:
            x (torch.Tensor): Input tensor of shape (Bs, N, F).
            Ns (int): Number of points to sample from each batch.

        Returns:
            torch.Tensor: Output tensor of shape (Bs, Ns, F).
        """
        Bs, N, F = x.shape

        # Generate random indices for sampling
        indices = torch.randint(low=0, high=N, size=(Bs, Ns), device=x.device).unsqueeze(-1)

        # Gather the sampled points
        x = knn_gather(x, indices).squeeze(-2).contiguous()

        return x, indices

    def compute_loss(self, X_mesh, X, X_hat, step="train", sync_dist=False):
        
        # X = self.sample_points_from_meshes(X_mesh, self.hparams.num_samples)
        
        loss_chamfer, _ = chamfer_distance(X, X_hat, batch_reduction="mean", point_reduction="sum")        

        # loss_mse = self.loss_fn(X, X_hat)

        X_hat_PC = Pointclouds(X_hat)
        loss_point_mesh_face = point_mesh_face_distance(X_mesh, X_hat_PC)
        loss_point_mesh_edge = point_mesh_edge_distance(X_mesh, X_hat_PC)

        # loss = loss_mse*self.hparams.loss_mse_weight + loss_point_mesh_face*self.hparams.loss_mesh_face_weight + loss_point_mesh_edge*self.hparams.loss_mesh_edge_weight
        loss = loss_chamfer*self.hparams.loss_chamfer_weight + loss_point_mesh_face*self.hparams.loss_mesh_face_weight + loss_point_mesh_edge*self.hparams.loss_mesh_edge_weight

        self.log(f"{step}_loss", loss, sync_dist=sync_dist)        
        # self.log(f"{step}_loss_mse", loss_mse, sync_dist=sync_dist)
        self.log(f"{step}_loss_chamfer", loss_chamfer, sync_dist=sync_dist)
        self.log(f"{step}_loss_point_mesh_face", loss_point_mesh_face, sync_dist=sync_dist)
        self.log(f"{step}_loss_point_mesh_edge", loss_point_mesh_edge, sync_dist=sync_dist)

        return loss

    def training_step(self, train_batch, batch_idx):
        V, F = train_batch
        
        X_mesh = self.create_mesh(V, F)

        X = self.sample_points_from_meshes(X_mesh, self.hparams.num_samples)

        X_hat = self(X)

        loss = self.compute_loss(X_mesh, X, X_hat)

        return loss

    def validation_step(self, val_batch, batch_idx):
        
        V, F = val_batch
        
        X_mesh = self.create_mesh(V, F)

        X = self.sample_points_from_meshes(X_mesh, self.hparams.num_samples)

        X_hat = self(X)

        self.compute_loss(X_mesh, X, X_hat, step="val", sync_dist=True)

    def sampling(self, z_mu: torch.Tensor, z_sigma: torch.Tensor) -> torch.Tensor:        
        eps = torch.randn_like(z_sigma)
        z_vae = z_mu + eps * z_sigma
        return z_vae

    def forward(self, x):
        skip_connections = None

        if self.hparams.use_skip_connection:
            h, unpooling_idxs, skip_connections = self.encoder(x, x)
        else:
            h, unpooling_idxs = self.encoder(x, x)
        
        z_mu = self.ff_mu(h)
        z_sigma = self.ff_sigma(h)
        z = self.sampling(z_mu, z_sigma)

        return self.decoder(z, unpooling_idxs, skip_connections)


class SaxiMHAClassification(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = MHAEncoder(input_dim=self.hparams.input_dim, 
                                      embed_dim=self.hparams.embed_dim, 
                                      num_heads=self.hparams.num_heads, 
                                      output_dim=self.hparams.output_dim, 
                                      sample_levels=self.hparams.sample_levels, 
                                      dropout=self.hparams.dropout, 
                                      K=self.hparams.K)
        
        self.mha = MHA_KNN(embed_dim=self.hparams.output_dim, num_heads=self.hparams.output_dim, K=self.hparams.output_dim, dropout=self.hparams.dropout, return_v=True)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(self.hparams.output_dim*self.hparams.sample_levels[-1]*2, self.hparams.num_classes)
        self.loss = nn.CrossEntropyLoss()
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("SaxiMHAClassification")

        group.add_argument("--lr", type=float, default=1e-4)
        group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=0.01)
        
        # Encoder parameters
        
        group.add_argument("--input_dim", type=int, default=3, help='Input dimension for the encoder')
        group.add_argument("--embed_dim", type=int, default=256, help='Embedding dimension')
        group.add_argument("--K", type=int, default=32, help='Top K nearest neighbors to consider in the encoder')
        group.add_argument("--num_heads", type=int, default=256, help='Number of attention heads for the encoder')
        group.add_argument("--output_dim", type=int, default=32, help='Output dimension from the encoder')        
        group.add_argument("--sample_levels", type=int, default=[40962, 10242, 2562, 642, 162], nargs="+", help='Number of sampling levels in the encoder')                
        group.add_argument("--dropout", type=float, default=0.1, help='Dropout rate')
        
        # classification parameters
        group.add_argument("--num_classes", type=int, default=4, help='Number of output classes')

        return parent_parser
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)        
        return optimizer
    
    def create_mesh(self, V, F):
        return Meshes(verts=V, faces=F)
    
    def sample_points_from_meshes(self, x_mesh, Ns, return_normals=False):
        if return_normals:
            x, x_N = sample_points_from_meshes(x_mesh, Ns, return_normals=True)
            return x, x_N
        return sample_points_from_meshes(x_mesh, Ns)

    def compute_loss(self, X_hat, Y):
        return self.loss(X_hat, Y)

    def training_step(self, train_batch, batch_idx):
        V, F = train_batch
        
        X_mesh = self.create_mesh(V, F)
        X_hat, _ = self(X_mesh)
        loss = self.compute_loss(X_mesh, X_hat)
        
        self.log("train_loss", loss)        

        return loss

    def validation_step(self, val_batch, batch_idx):
        
        V, F = val_batch
        
        X_mesh = self.create_mesh(V, F)
        X_hat, _ = self(X_mesh)

        loss = self.compute_loss(X_mesh, X_hat)
        
        self.log("val_loss", loss, sync_dist=True)

    def forward(self, X_mesh):
        X = self.sample_points_from_meshes(X_mesh, self.hparams.sample_levels[-1])
        x, x_w = self.encoder(X)
        x, x_v = self.mha(x)
        x = torch.cat([x, x_v], dim=1)
        x = self.flatten(x)
        x = self.fc(x)
        return x, x_w
    

class SaxiMHAFBClassification(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = MHAEncoder(input_dim=self.hparams.input_dim, 
                                      hidden_dim=self.hparams.hidden_dim,
                                      embed_dim=self.hparams.embed_dim, 
                                      num_heads=self.hparams.num_heads, 
                                      output_dim=self.hparams.output_dim, 
                                      sample_levels=self.hparams.sample_levels, 
                                      dropout=self.hparams.dropout, 
                                      K=self.hparams.K)
        
        
        self.attn = SelfAttention(self.hparams.output_dim, self.hparams.hidden_dim)
        self.ff = FeedForward(self.hparams.output_dim, hidden_dim=self.hparams.hidden_dim, dropout=self.hparams.dropout)

        effnet = monai.networks.nets.EfficientNetBN('efficientnet-b0', spatial_dims=2, in_channels=4, num_classes=self.hparams.output_dim)
        self.convnet = TimeDistributed(effnet)
        self.mha_fb = nn.MultiheadAttention(self.hparams.output_dim, self.hparams.num_heads, dropout=self.hparams.dropout, batch_first=True)         
        self.ff_fb = FeedForward(self.hparams.output_dim, hidden_dim=self.hparams.hidden_dim, dropout=self.hparams.dropout)
        self.attn_fb = SelfAttention(self.hparams.output_dim, self.hparams.hidden_dim)

        self.fc = nn.Linear(self.hparams.output_dim*2, self.hparams.num_classes)
        
        cameras = FoVPerspectiveCameras()

        raster_settings = RasterizationSettings(image_size=self.hparams.image_size, blur_radius=0, faces_per_pixel=1,max_faces_per_bin=200000)        
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        lights = AmbientLights()
        self.renderer = MeshRenderer(rasterizer=rasterizer,shader=HardPhongShader(cameras=cameras, lights=lights))
        self.ico_sphere(radius=self.hparams.radius, subdivision_level=self.hparams.subdivision_level)
        
        self.loss = nn.CrossEntropyLoss()
        
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.num_classes)

        centers = torch.tensor([12.5000, 37.5000, 62.5000, 87.5000], dtype=torch.float32)
        self.register_buffer("centers", centers)
        widths = torch.tensor([12.5000, 12.5000, 12.5000, 12.5000], dtype=torch.float32)
        self.register_buffer("widths", widths)
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("SaxiMHAFBClassification")

        group.add_argument("--lr", type=float, default=1e-4)
        group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=0.01)
        
        # Encoder parameters
        
        group.add_argument("--input_dim", type=int, default=3, help='Input dimension for the encoder')
        group.add_argument("--embed_dim", type=int, default=256, help='Embedding dimension')
        group.add_argument("--hidden_dim", type=int, default=64, help='Embedding dimension')
        group.add_argument("--image_size", type=int, default=224, help='Image size for rendering')
        group.add_argument("--radius", type=float, default=1.35, help='Radius of the icosphere/camera positions')
        group.add_argument("--subdivision_level", type=int, default=2, help='Subdivision level of the ico sphere')
        group.add_argument("--K", type=int, default=128, help='Top K nearest neighbors to consider in the encoder')
        group.add_argument("--num_heads", type=int, default=256, help='Number of attention heads for the encoder')
        group.add_argument("--output_dim", type=int, default=256, help='Output dimension from the encoder')        
        group.add_argument("--sample_levels", type=int, default=[4096, 2048, 512, 128], nargs="+", help='Number of sampling levels in the encoder')                
        group.add_argument("--dropout", type=float, default=0.1, help='Dropout rate')
        
        # classification parameters
        group.add_argument("--num_classes", type=int, default=4, help='Number of output classes')

        return parent_parser
    
    def ico_sphere(self, radius=1.1, subdivision_level=1):
        # Create an icosphere
        ico_verts, ico_faces, ico_edges = utils.PolyDataToTensors(utils.CreateIcosahedronSubdivided(radius=radius, sl=subdivision_level))
        ico_verts = ico_verts.to(torch.float32)

        for idx, v in enumerate(ico_verts):
            if (torch.abs(torch.sum(v)) == radius):
                ico_verts[idx] = v + torch.tensor([-1.2447e-05, -3.7212e-06, -1.5617e-06])
        
        self.register_buffer("ico_verts", ico_verts)

    def to(self, device=None):
        # Move the renderer to the specified device
        self.renderer = self.renderer.to(device)
        return super().to(device)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)        
        return optimizer
    
    def create_mesh(self, V, F, CN=None):
        
        if CN is not None:
            textures = TexturesVertex(verts_features=CN.to(torch.float32))
            return Meshes(verts=V, faces=F, textures=textures)
        return Meshes(verts=V, faces=F)
    
    def sample_points_from_meshes(self, x_mesh, Ns, return_normals=False):
        if return_normals:
            x, x_N = sample_points_from_meshes(x_mesh, Ns, return_normals=True)
            return x, x_N
        return sample_points_from_meshes(x_mesh, Ns)
    
    def render(self, meshes):
        # Render the input surface mesh to an image
        
        X = []
        PF = []

        for camera_position in self.ico_verts:
            camera_position = camera_position.unsqueeze(0)
            R = look_at_rotation(camera_position, device=self.device)  # (1, 3, 3)
            T = -torch.bmm(R.transpose(1, 2), camera_position[:,:,None])[:, :, 0]   # (1, 3)
            images = self.renderer(meshes_world=meshes.clone(), R=R, T=T)        
            fragments = self.renderer.rasterizer(meshes.clone())
            pix_to_face = fragments.pix_to_face
            zbuf = fragments.zbuf
            images = torch.cat([images[:,:,:,0:3], zbuf], dim=-1)
            images = images.permute(0,3,1,2)
            pix_to_face = pix_to_face.permute(0,3,1,2)
            X.append(images.unsqueeze(1))
            PF.append(pix_to_face.unsqueeze(1))
        
        X = torch.cat(X, dim=1)
        PF = torch.cat(PF, dim=1)        

        return X, PF

    def compute_loss(self, X_hat, Y):
        return self.loss(X_hat, Y)
    
    def soft_class_probabilities(self, values):
        # Calculate unscaled probabilities using a Gaussian-like function
        # Here, we use the negative squared distance scaled by width as logits
        values = values.unsqueeze(-1)
        logits = -(values - self.centers) ** 2 / (2 * self.widths ** 2)
        
        # Apply softmax to convert logits into probabilities
        probabilities = F.softmax(logits, dim=1)
        
        return probabilities

    def training_step(self, train_batch, batch_idx):
        V, F, CN, Y = train_batch

        Y = self.soft_class_probabilities(Y)
        
        X_mesh = self.create_mesh(V, F, CN)
        X_hat, _, _ = self(X_mesh)
        loss = self.compute_loss(X_hat, Y)
        
        self.log("train_loss", loss)
        self.accuracy(X_hat, torch.argmax(Y, dim=1))
        self.log("train_acc", self.accuracy, batch_size=V.shape[0], sync_dist=True) 

        return loss

    def validation_step(self, val_batch, batch_idx):
        
        V, F, CN, Y = val_batch

        Y = self.soft_class_probabilities(Y)
        
        X_mesh = self.create_mesh(V, F, CN)
        X_hat, _, _ = self(X_mesh)

        loss = self.compute_loss(X_hat, Y)
        
        self.log("val_loss", loss, sync_dist=True)
        self.accuracy(X_hat, torch.argmax(Y, dim=1))
        self.log("val_acc", self.accuracy, batch_size=V.shape[0], sync_dist=True)

    def forward(self, X_mesh):
        X = self.sample_points_from_meshes(X_mesh, self.hparams.sample_levels[0])
        
        x, x_w = self.encoder(X)        
        x = self.ff(x)
        x, x_s = self.attn(x, x)        


        X_views, X_PF = self.render(X_mesh)
        x_fb = self.convnet(X_views)
        x_fb = self.ff_fb(x_fb)
        x_fb, x_fb_mha_s = self.mha_fb(x_fb, x_fb, x_fb)
        x_fb, x_fb_s = self.attn_fb(x_fb, x_fb)

        x = torch.cat([x, x_fb], dim=1)

        x = self.fc(x)
        return x, x_w, X

class SaxiMHAClassificationSingle(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()        
        
        self.mlp_in = ProjectionHead(self.hparams.input_dim, hidden_dim=self.hparams.hidden_dim, output_dim=self.hparams.embed_dim, dropout=self.hparams.dropout)
        self.mha = MHA_KNN(embed_dim=self.hparams.embed_dim, num_heads=self.hparams.num_heads, K=self.hparams.K, dropout=self.hparams.dropout, use_direction=False, return_weights=True)
        self.ff = Residual(FeedForward(self.hparams.embed_dim, hidden_dim=self.hparams.hidden_dim, dropout=self.hparams.dropout))
        self.mlp_out = ProjectionHead(self.hparams.embed_dim, hidden_dim=self.hparams.hidden_dim, output_dim=self.hparams.output_dim, dropout=self.hparams.dropout)
        
        self.fc = nn.Linear(self.hparams.output_dim, self.hparams.num_classes)
        self.loss = nn.CrossEntropyLoss()

        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.num_classes)
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("SaxiMHAClassificationSingle")

        group.add_argument("--lr", type=float, default=1e-4)
        group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=0.01)
        
        # Encoder parameters
        
        group.add_argument("--input_dim", type=int, default=3, help='Input dimension for the encoder')
        group.add_argument("--hidden_dim", type=int, default=64, help='Input dimension for the encoder')
        group.add_argument("--embed_dim", type=int, default=128, help='Embedding dimension')
        group.add_argument("--K", type=int, default=1024, help='Top K nearest neighbors to consider in the encoder')
        group.add_argument("--num_heads", type=int, default=128, help='Number of attention heads for the encoder')
        group.add_argument("--output_dim", type=int, default=128, help='Output dimension from the encoder')        
        group.add_argument("--sample_level", type=int, default=1024, help='Sampling level')                
        group.add_argument("--dropout", type=float, default=0.1, help='Dropout rate')
        
        # classification parameters
        group.add_argument("--num_classes", type=int, default=None, help='Number of output classes', required=True)

        return parent_parser
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)        
        return optimizer
    
    def create_mesh(self, V, F, CN=None):
        
        if CN is not None:
            textures = TexturesVertex(verts_features=CN.to(torch.float32))
            return Meshes(verts=V, faces=F, textures=textures)
        return Meshes(verts=V, faces=F)
    
    def sample_points_from_meshes(self, x_mesh, Ns, return_normals=False):
        if return_normals:
            x, x_N = sample_points_from_meshes(x_mesh, Ns, return_normals=True)
            return x, x_N
        return sample_points_from_meshes(x_mesh, Ns)

    def compute_loss(self, X_hat, Y):
        return self.loss(X_hat, Y)

    def training_step(self, train_batch, batch_idx):
        V, F, CN, Y = train_batch
        
        X_mesh = self.create_mesh(V, F)
        X_hat, _ = self(X_mesh)
        loss = self.compute_loss(X_hat, Y)
        
        batch_size = V.shape[0]
        self.log("train_loss", loss, batch_size=batch_size)
        self.accuracy(X_hat, Y)
        self.log("train_acc", self.accuracy, batch_size=batch_size)

        return loss

    def validation_step(self, val_batch, batch_idx):
        
        V, F, CN, Y = val_batch
        
        X_mesh = self.create_mesh(V, F)
        X_hat, _ = self(X_mesh)

        loss = self.compute_loss(X_hat, Y)
        
        batch_size = V.shape[0]
        self.log("val_loss", loss, sync_dist=True, batch_size=batch_size)
        self.accuracy(X_hat, Y)
        self.log("val_acc", self.accuracy, batch_size=batch_size)

    def forward(self, X_mesh):
        x = self.sample_points_from_meshes(X_mesh, self.hparams.sample_level)
        
        x = self.mlp_in(x)
        x, x_w = self.mha(x)
        x = self.ff(x)
        x = self.mlp_out(x)
        x = torch.mean(x, dim=1)
        x = self.fc(x)

        return x, x_w
    

class SaxiMHAFBRegression(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = MHAEncoder(input_dim=self.hparams.input_dim, 
                                      hidden_dim=self.hparams.hidden_dim,
                                      embed_dim=self.hparams.embed_dim, 
                                      num_heads=self.hparams.num_heads, 
                                      output_dim=self.hparams.output_dim, 
                                      sample_levels=self.hparams.sample_levels, 
                                      dropout=self.hparams.dropout, 
                                      K=self.hparams.K)
        self.ff = FeedForward(self.hparams.output_dim, hidden_dim=self.hparams.hidden_dim, dropout=self.hparams.dropout)
        self.attn = SelfAttention(self.hparams.output_dim, self.hparams.hidden_dim)

        effnet = monai.networks.nets.EfficientNetBN('efficientnet-b0', spatial_dims=2, in_channels=4, num_classes=self.hparams.output_dim)
        self.convnet = TimeDistributed(effnet)
        self.mha_fb = nn.MultiheadAttention(self.hparams.output_dim, self.hparams.num_heads, dropout=self.hparams.dropout, batch_first=True)         
        self.ff_fb = FeedForward(self.hparams.output_dim, hidden_dim=self.hparams.hidden_dim, dropout=self.hparams.dropout)
        self.attn_fb = SelfAttention(self.hparams.output_dim, self.hparams.hidden_dim)

        self.fc = nn.Linear(self.hparams.output_dim*2, 1)
        
        cameras = FoVPerspectiveCameras()

        raster_settings = RasterizationSettings(image_size=self.hparams.image_size, blur_radius=0, faces_per_pixel=1,max_faces_per_bin=200000)        
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        lights = AmbientLights()
        self.renderer = MeshRenderer(rasterizer=rasterizer,shader=HardPhongShader(cameras=cameras, lights=lights))
        self.ico_sphere(radius=self.hparams.radius, subdivision_level=self.hparams.subdivision_level)
        
        # self.loss = nn.CrossEntropyLoss()
        self.loss = nn.MSELoss()
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("SaxiMHAFBRegression")

        group.add_argument("--lr", type=float, default=1e-4)
        group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=0.01)
        
        # Encoder parameters
        
        group.add_argument("--input_dim", type=int, default=3, help='Input dimension for the encoder')
        group.add_argument("--embed_dim", type=int, default=256, help='Embedding dimension')
        group.add_argument("--hidden_dim", type=int, default=64, help='Embedding dimension')
        group.add_argument("--image_size", type=int, default=224, help='Image size for rendering')
        group.add_argument("--radius", type=float, default=1.35, help='Radius of the icosphere/camera positions')
        group.add_argument("--subdivision_level", type=int, default=2, help='Subdivision level of the ico sphere')
        group.add_argument("--K", type=int, default=128, help='Top K nearest neighbors to consider in the encoder')
        group.add_argument("--num_heads", type=int, default=256, help='Number of attention heads for the encoder')
        group.add_argument("--output_dim", type=int, default=256, help='Output dimension from the encoder')        
        group.add_argument("--sample_levels", type=int, default=[4096, 2048, 512, 128], nargs="+", help='Number of sampling levels in the encoder')                
        group.add_argument("--dropout", type=float, default=0.1, help='Dropout rate')
        
        # classification parameters
        # group.add_argument("--num_classes", type=int, default=4, help='Number of output classes')

        return parent_parser
    
    def ico_sphere(self, radius=1.1, subdivision_level=1):
        # Create an icosphere
        ico_verts, ico_faces, ico_edges = utils.PolyDataToTensors(utils.CreateIcosahedronSubdivided(radius=radius, sl=subdivision_level))
        ico_verts = ico_verts.to(torch.float32)

        for idx, v in enumerate(ico_verts):
            if (torch.abs(torch.sum(v)) == radius):
                ico_verts[idx] = v + torch.tensor([-1.2447e-05, -3.7212e-06, -1.5617e-06])
        
        self.register_buffer("ico_verts", ico_verts)

    def to(self, device=None):
        # Move the renderer to the specified device
        self.renderer = self.renderer.to(device)
        return super().to(device)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)        
        return optimizer
    
    def create_mesh(self, V, F, CN=None):
        
        if CN is not None:
            textures = TexturesVertex(verts_features=CN.to(torch.float32))
            return Meshes(verts=V, faces=F, textures=textures)
        
        return Meshes(verts=V, faces=F)
    
    def sample_points_from_meshes(self, x_mesh, Ns, return_normals=False):
        if return_normals:
            x, x_N = sample_points_from_meshes(x_mesh, Ns, return_normals=True)
            return x, x_N
        return sample_points_from_meshes(x_mesh, Ns)
    
    def render(self, meshes):
        # Render the input surface mesh to an image
        
        X = []
        PF = []

        for camera_position in self.ico_verts:
            camera_position = camera_position.unsqueeze(0)
            R = look_at_rotation(camera_position, device=self.device)  # (1, 3, 3)
            T = -torch.bmm(R.transpose(1, 2), camera_position[:,:,None])[:, :, 0]   # (1, 3)
            images = self.renderer(meshes_world=meshes.clone(), R=R, T=T)        
            fragments = self.renderer.rasterizer(meshes.clone())
            pix_to_face = fragments.pix_to_face
            zbuf = fragments.zbuf
            images = torch.cat([images[:,:,:,0:3], zbuf], dim=-1)
            images = images.permute(0,3,1,2)
            pix_to_face = pix_to_face.permute(0,3,1,2)
            X.append(images.unsqueeze(1))
            PF.append(pix_to_face.unsqueeze(1))
        
        X = torch.cat(X, dim=1)
        PF = torch.cat(PF, dim=1)        

        return X, PF

    def compute_loss(self, X_hat, Y):
        Y = Y.unsqueeze(-1).to(torch.float32)
        return self.loss(X_hat, Y)

    def training_step(self, train_batch, batch_idx):
        V, F, CN, Y = train_batch
        
        X_mesh = self.create_mesh(V, F, CN)
        X_hat, _, _ = self(X_mesh)
        loss = self.compute_loss(X_hat, Y)
        
        self.log("train_loss", loss)
        # self.log("train_acc", self.accuracy, batch_size=V.shape[0], sync_dist=True) 

        return loss

    def validation_step(self, val_batch, batch_idx):
        
        V, F, CN, Y = val_batch
        
        X_mesh = self.create_mesh(V, F, CN)
        X_hat, _, _ = self(X_mesh)

        loss = self.compute_loss(X_hat, Y)
        
        self.log("val_loss", loss, sync_dist=True)

    def forward(self, X_mesh):
        X = self.sample_points_from_meshes(X_mesh, self.hparams.sample_levels[0])
        
        x, x_w = self.encoder(X)        
        x = self.ff(x)
        x, x_s = self.attn(x, x)

        X_views, X_PF = self.render(X_mesh)
        x_fb = self.convnet(X_views)
        x_fb = self.ff_fb(x_fb)
        x_fb, x_fb_mha_s = self.mha_fb(x_fb, x_fb, x_fb)
        x_fb, x_fb_s = self.attn_fb(x_fb, x_fb)

        x = torch.cat([x, x_fb], dim=1)

        x = self.fc(x)
        return x, x_w, X
    


class SaxiMHAFBRegression_V(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()


        self.encoder = MHAEncoder_V(input_dim=self.hparams.input_dim, 
                                    output_dim=self.hparams.output_dim,
                                    K=self.hparams.K,
                                    num_heads=self.hparams.num_heads,
                                    stages=self.hparams.stages,
                                    dropout=self.hparams.dropout,
                                    pooling_factor=self.hparams.pooling_factor)
        
        self.attn = SelfAttention(self.hparams.output_dim, self.hparams.hidden_dim)

        effnet = monai.networks.nets.EfficientNetBN('efficientnet-b0', spatial_dims=2, in_channels=4, num_classes=self.hparams.output_dim)
        self.convnet = TimeDistributed(effnet)
        self.mha_fb = nn.MultiheadAttention(self.hparams.output_dim, self.hparams.num_heads[-1], dropout=self.hparams.dropout, batch_first=True)
        self.attn_fb = SelfAttention(self.hparams.output_dim, self.hparams.hidden_dim)

        self.fc = nn.Linear(self.hparams.output_dim*2, 1)
        
        cameras = FoVPerspectiveCameras()

        raster_settings = RasterizationSettings(image_size=self.hparams.image_size, blur_radius=0, faces_per_pixel=1,max_faces_per_bin=200000)        
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        lights = AmbientLights()
        self.renderer = MeshRenderer(rasterizer=rasterizer,shader=HardPhongShader(cameras=cameras, lights=lights))
        self.ico_sphere(radius=self.hparams.radius, subdivision_level=self.hparams.subdivision_level)
        
        # self.loss = nn.CrossEntropyLoss()
        self.loss = nn.MSELoss()
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("SaxiMHAFBRegression")

        group.add_argument("--lr", type=float, default=1e-4)
        group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=0.01)
        
        # Encoder parameters

        group.add_argument("--input_dim", type=int, default=3, help='Input dimension for the encoder')
        group.add_argument("--embed_dim", type=int, default=256, help='Embedding dimension')
        group.add_argument("--hidden_dim", type=int, default=64, help='Embedding dimension')
        group.add_argument("--image_size", type=int, default=224, help='Image size for rendering')
        group.add_argument("--radius", type=float, default=1.35, help='Radius of the icosphere/camera positions')
        group.add_argument("--subdivision_level", type=int, default=2, help='Subdivision level of the ico sphere')
        group.add_argument("--K", type=int, nargs="+",  default=[27, 125, 125], help='Top K nearest neighbors to consider in the encoder')
        group.add_argument("--num_heads", type=int, default=[32, 64, 128], help='Number of attention heads for the encoder')
        group.add_argument("--stages", type=int, nargs="+", default=[32, 64, 128], help='Number of attention heads for the encoder')
        group.add_argument("--pooling_factor", type=int, nargs="+", default=[0.25, 0.25, 0.25], help='Number of attention heads for the encoder')
        
        group.add_argument("--output_dim", type=int, default=256, help='Output dimension from the encoder')        
        group.add_argument("--sample_level", type=int, default=4096, help='Number of sampling levels in the encoder')                
        group.add_argument("--dropout", type=float, default=0.1, help='Dropout rate')
        
        # classification parameters
        # group.add_argument("--num_classes", type=int, default=4, help='Number of output classes')

        return parent_parser
    
    def ico_sphere(self, radius=1.1, subdivision_level=1):
        # Create an icosphere
        ico_verts, ico_faces, ico_edges = utils.PolyDataToTensors(utils.CreateIcosahedronSubdivided(radius=radius, sl=subdivision_level))
        ico_verts = ico_verts.to(torch.float32)

        for idx, v in enumerate(ico_verts):
            if (torch.abs(torch.sum(v)) == radius):
                ico_verts[idx] = v + torch.tensor([-1.2447e-05, -3.7212e-06, -1.5617e-06])
        
        self.register_buffer("ico_verts", ico_verts)

    def to(self, device=None):
        # Move the renderer to the specified device
        self.renderer = self.renderer.to(device)
        return super().to(device)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)        
        return optimizer
    
    def create_mesh(self, V, F, CN=None):
        
        if CN is not None:
            textures = TexturesVertex(verts_features=CN.to(torch.float32))
            return Meshes(verts=V, faces=F, textures=textures)
        
        return Meshes(verts=V, faces=F)
    
    def sample_points_from_meshes(self, x_mesh, Ns, return_normals=False):
        if return_normals:
            x, x_N = sample_points_from_meshes(x_mesh, Ns, return_normals=True)
            return x, x_N
        return sample_points_from_meshes(x_mesh, Ns)
    
    def render(self, meshes):
        # Render the input surface mesh to an image
        
        X = []
        PF = []

        for camera_position in self.ico_verts:
            camera_position = camera_position.unsqueeze(0)
            R = look_at_rotation(camera_position, device=self.device)  # (1, 3, 3)
            T = -torch.bmm(R.transpose(1, 2), camera_position[:,:,None])[:, :, 0]   # (1, 3)
            images = self.renderer(meshes_world=meshes.clone(), R=R, T=T)        
            fragments = self.renderer.rasterizer(meshes.clone())
            pix_to_face = fragments.pix_to_face
            zbuf = fragments.zbuf
            images = torch.cat([images[:,:,:,0:3], zbuf], dim=-1)
            images = images.permute(0,3,1,2)
            pix_to_face = pix_to_face.permute(0,3,1,2)
            X.append(images.unsqueeze(1))
            PF.append(pix_to_face.unsqueeze(1))
        
        X = torch.cat(X, dim=1)
        PF = torch.cat(PF, dim=1)        

        return X, PF

    def compute_loss(self, X_hat, Y):
        Y = Y.unsqueeze(-1).to(torch.float32)
        return self.loss(X_hat, Y)

    def training_step(self, train_batch, batch_idx):
        V, F, CN, Y = train_batch
        
        X_mesh = self.create_mesh(V, F, CN)
        X_hat, _, _ = self(X_mesh)
        loss = self.compute_loss(X_hat, Y)
        
        self.log("train_loss", loss)
        # self.log("train_acc", self.accuracy, batch_size=V.shape[0], sync_dist=True) 

        return loss

    def validation_step(self, val_batch, batch_idx):
        
        V, F, CN, Y = val_batch
        
        X_mesh = self.create_mesh(V, F, CN)
        X_hat, _, _ = self(X_mesh)

        loss = self.compute_loss(X_hat, Y)
        
        self.log("val_loss", loss, sync_dist=True)

    def forward(self, X_mesh):
        X = self.sample_points_from_meshes(X_mesh, self.hparams.sample_level)
        
        x, x_v, x_s_idx = self.encoder(X, X)
        x, x_s = self.attn(x, x)

        X_views, X_PF = self.render(X_mesh)
        x_fb = self.convnet(X_views)
        x_fb, x_fb_mha_s = self.mha_fb(x_fb, x_fb, x_fb)
        x_fb, x_fb_s = self.attn_fb(x_fb, x_fb)

        x = torch.cat([x, x_fb], dim=1)

        x = self.fc(x)
        return x, x_s, X




#####################################################################################################################################################################################
#                                                                                                                                                                                   #
#                                                                                SaxiRing Multiple TimePoints                                                                       #
#                                                                                                                                                                                   #
#####################################################################################################################################################################################


class SaxiRingMT(LightningModule):
    def __init__(self, **kwargs):
        super(SaxiRingMT, self).__init__()
        self.save_hyperparameters()
        self.y_pred = []
        self.y_true = []


        # Create the icosahedrons form each level
        ico_12 = utils.CreateIcosahedron(self.hparams.radius) # 12 vertices
        ico_42 = utils.SubdividedIcosahedron(ico_12,2,self.hparams.radius) # 42 vertices
        ico_162 = utils.SubdividedIcosahedron(ico_42,2,self.hparams.radius) # 162 vertices

        # Get the neighbors to go form level N to level N-1
        ring_neighs_42 = utils.GetPreservedPointIds(ico_12,ico_42)
        ring_neighs_162 = utils.GetPreservedPointIds(ico_42,ico_162)

        # Create the down blocks to go from 162 -> 42 -> 12
        self.down1 = AttentionRings(self.hparams.hidden_dim, self.hparams.hidden_dim, self.hparams.hidden_dim, ring_neighs_162)
        self.down2 = AttentionRings(self.hparams.hidden_dim, self.hparams.hidden_dim, self.hparams.hidden_dim, ring_neighs_42)

        ico_sphere_verts, ico_sphere_faces, ico_sphere_edges = utils.PolyDataToTensors(ico_162)

        self.ico_sphere_verts = ico_sphere_verts
        self.ico_sphere_edges = ico_sphere_edges

        R=[]
        T=[]
        for coords_cam in self.ico_sphere_verts.tolist():
            camera_position = torch.FloatTensor([coords_cam])
            R_current = look_at_rotation(camera_position)
            # check if camera coords vector and up vector for R are collinear
            if torch.equal(torch.cross(camera_position,torch.tensor([[0.,1.,0.]])),torch.tensor([[0., 0., 0.]])):
               R_current = look_at_rotation(camera_position, up = torch.tensor([[0.0, 0.0, 1.0]]),)
            T_current = -torch.bmm(R_current.transpose(1, 2), camera_position[:,:,None])[:, :, 0]   # (1, 3)

            R.append(R_current)
            T.append(T_current)

        self.R=torch.cat(R)
        self.T=torch.cat(T)
        self.nbr_cam = len(self.R)

        self.drop = nn.Dropout(p=self.hparams.dropout_lvl)

        # Left network
        self.create_network('L')
        # Right network
        self.create_network('R')

        # Loss
        self.loss_train = nn.CrossEntropyLoss()
        self.loss_val = nn.CrossEntropyLoss()
        self.loss_test = nn.CrossEntropyLoss()

        # Pooling layer
        self.W = nn.Linear(self.hparams.hidden_dim, self.hparams.out_size)
        self.Att = SelfAttention(self.hparams.hidden_dim, self.hparams.out_size, dim=2)
        
        # Final layer
        self.Classification = nn.Linear(2*ico_12.GetNumberOfPoints()*3, self.hparams.out_classes) # 3 timepoints so *3

        #vAccuracy
        self.train_accuracy = torchmetrics.Accuracy('multiclass',num_classes=self.hparams.out_classes,average='macro')
        self.val_accuracy = torchmetrics.Accuracy('multiclass',num_classes=self.hparams.out_classes,average='macro')
        
        # Initialize a perspective camera.
        self.hparams.cameras = FoVPerspectiveCameras()

        # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
        raster_settings = RasterizationSettings(image_size=self.hparams.image_size,blur_radius=0,faces_per_pixel=1,max_faces_per_bin=100000)

        lights = AmbientLights()
        rasterizer = MeshRasterizer(cameras=self.hparams.cameras,raster_settings=raster_settings)

        self.hparams.phong_renderer = MeshRendererWithFragments(rasterizer=rasterizer,shader=HardPhongShader(cameras=self.hparams.cameras, lights=lights))


    def to(self, device=None):
        # Move the renderer to the specified device
        self.hparams.phong_renderer = self.hparams.phong_renderer.to(device)
        return super().to(device)

    
    def create_network(self, side):
        # Create an icosphere
        if hasattr(monai.networks.nets, self.hparams.base_encoder):
            template_model = getattr(monai.networks.nets, self.hparams.base_encoder)
        elif hasattr(torchvision.models, self.hparams.base_encoder):
            template_model = getattr(torchvision.models, self.hparams.base_encoder)
        else:
            raise "{base_encoder} not in monai networks or torchvision".format(base_encoder=self.hparams.base_encoder)

        model_params = eval('dict(%s)' % self.hparams.base_encoder_params.replace(' ',''))
        
        self.convnet = template_model(**model_params)
        setattr(self, f'TimeDistributed{side}', TimeDistributed(self.convnet))


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer


    def forward(self, x):  
        T1L, T2L, T3L, T1R, T2R, T3R = x
        
        all_xL = []
        all_xR = []

        # Process right timepoints
        for V, F, VF, FF in [T1L, T2L, T3L]:
            xL, scoreL = self.get_features(V, F, VF, FF, 'L')
            all_xL.append(xL)

        # Process right timepoints
        for V, F, VF, FF in [T1R, T2R, T3R]:
            xR, scoreR = self.get_features(V, F, VF, FF, 'R')
            all_xR.append(xR)

        xL = torch.cat(all_xL, dim=1)  # Output shape is (batch, 12*3, features)
        xR = torch.cat(all_xR, dim=1)

        # Add attention layer
        valuesL = self.W(xL)
        valuesR = self.W(xR)
        xL, score = self.Att(xL, valuesL)  # Output shape is (batch, features)
        xR, score = self.Att(xR, valuesR) 

        l_left_right = [xL, xR]
        x = torch.cat(l_left_right, dim=1)  # Output shape is (batch, 2*features)

        # Last classification layer
        x = self.drop(x)
        x = self.Classification(x)

        return x
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("SaxiRing")
        
        group.add_argument("--lr", type=float, default=1e-4)
        group.add_argument("--fs_path", type=str, default=None)
        group.add_argument("--out_classes", type=int, default=2)
        group.add_argument("--out_size", type=int, default=256)
        group.add_argument('--dropout_lvl',  type=float, help='Dropout level (default: 0.2)', default=0.2)

        # Encoder parameters
        group.add_argument('--base_encoder', type=str, help='Base encoder for the feature extraction', default='resnet18')
        group.add_argument('--base_encoder_params', type=str, help='Base encoder parameters that are passed to build the feature extraction', default='pretrained=False,spatial_dims=2,n_input_channels=1,num_classes=512')
        group.add_argument('--hidden_dim', type=int, help='Hidden dimension for features output. Should match with output of base_encoder. Default value is 512', default=512)
        group.add_argument('--radius', type=float, help='Radius of icosphere', default=1.5)
        group.add_argument('--subdivision_level', type=int, help='Subdivision level for icosahedron', default=2)
        group.add_argument('--image_size', type=int, help='Image resolution size', default=256)

        return parent_parser


    def get_features(self,V,F,VF,FF,side):
        x, PF = self.render(V,F,VF,FF)  
        x = getattr(self, f'TimeDistributed{side}')(x)
        x, score = self.down1(x) # Output shape is (batch, 42, features)
        x, score = self.down2(x) # Output shape is (batch, 12, features)
        return x, score


    def render(self,V,F,VF,FF):
        textures = TexturesVertex(verts_features=VF[:, :, :3])
        meshes = Meshes(
            verts=V,
            faces=F,
            textures=textures
        )
        PF = []
        for i in range(self.nbr_cam):
            pix_to_face = self.GetView(meshes,i)
            PF.append(pix_to_face.unsqueeze(dim=1))
        PF = torch.cat(PF, dim=1)

        l_features = []
        for index in range(FF.shape[-1]):
            l_features.append(torch.take(FF[:,index],PF)*(PF >= 0)) # take each feature for each pictures
        x = torch.cat(l_features,dim=2)

        return x, PF


    def training_step(self, train_batch, batch_idx):
        # Unpack the batch
        T1L = train_batch['T1L']
        T2L = train_batch['T2L']
        T3L = train_batch['T3L']
        T1R = train_batch['T1R']
        T2R = train_batch['T2R']
        T3R = train_batch['T3R']
        Y = train_batch['Y']

        x = self((T1L, T2L, T3L, T1R, T2R, T3R))

        loss = self.loss_train(x, Y)
        self.log('train_loss', loss)
        predictions = torch.argmax(x, dim=1, keepdim=True)
        self.train_accuracy(predictions, Y.reshape(-1, 1))
        self.log("train_acc", self.train_accuracy, batch_size=self.hparams.batch_size)
        return loss

    def validation_step(self, val_batch, batch_idx):
        # Unpack the batch
        T1L = val_batch['T1L']
        T2L = val_batch['T2L']
        T3L = val_batch['T3L']
        T1R = val_batch['T1R']
        T2R = val_batch['T2R']
        T3R = val_batch['T3R']
        Y = val_batch['Y']

        # Forward pass
        x = self((T1L, T2L, T3L, T1R, T2R, T3R))
        # Compute loss
        loss = self.loss_val(x, Y)
        self.log('val_loss', loss)
        # Calculate predictions and accuracy
        predictions = torch.argmax(x, dim=1, keepdim=True)
        val_acc = self.val_accuracy(predictions, Y.reshape(-1, 1))
        self.log("val_acc", val_acc, batch_size=self.hparams.batch_size)
        
        return val_acc


    def test_step(self, test_batch, batch_idx):
        # Unpack the batch
        T1L = test_batch['T1L']
        T2L = test_batch['T2L']
        T3L = test_batch['T3L']
        T1R = test_batch['T1R']
        T2R = test_batch['T2R']
        T3R = test_batch['T3R']
        Y = test_batch['Y']
        
        x = self((T1L, T2L, T3L, T1R, T2R, T3R))
        loss = self.loss_test(x, Y)
        self.log('test_loss', loss, batch_size=self.hparams.batch_size)
        predictions = torch.argmax(x, dim=1, keepdim=True)
        output = [predictions, Y]
        return output


    def test_epoch_end(self,input_test):
        y_pred = []
        y_true = []
        for ele in input_test:
            y_pred += ele[0].tolist()
            y_true += ele[1].tolist()
        target_names = ['No QC','QC']
        self.y_pred = y_pred
        self.y_true = y_true


    def GetView(self,meshes,index):
        phong_renderer = self.hparams.phong_renderer.to(self.device)
        R = self.R[index][None].to(self.device)
        T = self.T[index][None].to(self.device)
        _, fragments = phong_renderer(meshes.clone(),R=R,T=T)
        pix_to_face = fragments.pix_to_face
        pix_to_face = pix_to_face.permute(0,3,1,2)

        return pix_to_face


    def get_y_for_report_classification(self):
        #This function could be called only after test step was done
        return (self.y_pred,self.hparams.y_true)
    

    def Is_it_Icolayer(self,layer):
        return (layer[:3] == 'Ico')




class SaxiOctree(LightningModule):
    def __init__(self, **kwargs):
        super(SaxiOctree, self).__init__()
        self.save_hyperparameters()
        self.y_pred = []
        self.y_true = []

        
        self.features = self.create_network()
        self.drop = nn.Dropout(p=self.hparams.dropout)
        self.Classification = nn.Linear(self.hparams.out_channels, self.hparams.out_classes)


        # Loss
        self.loss_train = nn.CrossEntropyLoss()
        self.loss_val = nn.CrossEntropyLoss()
        self.loss_test = nn.CrossEntropyLoss()

        #vAccuracy
        self.train_accuracy = torchmetrics.Accuracy('multiclass',num_classes=self.hparams.out_classes,average='macro')
        self.val_accuracy = torchmetrics.Accuracy('multiclass',num_classes=self.hparams.out_classes,average='macro')

    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("SaxiOctree")
        
        group.add_argument("--lr", type=float, default=1e-4)
        group.add_argument("--out_classes", type=int, default=2)

        # Octree params
        group.add_argument("--in_channels", type=int, default=3)
        group.add_argument("--dropout", type=float, default=0.1)
        group.add_argument("--out_channels", type=int, default=1280)
        group.add_argument('--input_feature', type=str, help='Type of features to get from the octree', default='P')
        group.add_argument('--resblock_num', type=int, help='Number of residual blocks', default=1)
        group.add_argument('--stages', type=int, help='Number of stages', default=3)
        group.add_argument('--depth', type=int, help='Start depth', default=16)
        group.add_argument('--radius', type=float, help='Radius of icosphere', default=1.5)
        group.add_argument('--subdivision_level', type=int, help='Subdivision level for icosahedron', default=2)
        group.add_argument('--image_size', type=int, help='Image resolution size', default=256)

        return parent_parser
    
    def create_network(self):
        import ocnn
        return ocnn.models.ResNet(in_channels=self.hparams.in_channels, out_channels=self.hparams.out_channels, resblock_num=self.hparams.resblock_num, stages=self.hparams.stages, nempty=False)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def forward(self, O):

        # TimeDistributed
        X = O.get_input_feature(self.hparams.input_feature).to(torch.float)

        z = self.features(X, octree=O, depth=self.hparams.depth)
        x = self.Classification(z)

        return x

    def training_step(self, train_batch, batch_idx):
        O, Y = train_batch
        x = self(O)
        loss = self.loss_train(x, Y)
        self.log('train_loss', loss) 
        predictions = torch.argmax(x, dim=1, keepdim=True)
        self.train_accuracy(predictions, Y.reshape(-1, 1))
        self.log("train_acc", self.train_accuracy, batch_size=self.hparams.batch_size)           

        return loss

    def validation_step(self,val_batch,batch_idx):
        O, Y = val_batch
        x = self(O)
        loss = self.loss_val(x, Y)
        self.log('val_loss', loss, sync_dist=True)
        predictions = torch.argmax(x, dim=1, keepdim=True)
        val_acc = self.val_accuracy(predictions, Y.reshape(-1, 1))
        self.log("val_acc", val_acc, batch_size=self.hparams.batch_size, sync_dist=True)


    def test_step(self,test_batch,batch_idx):
        OL, OR, Y = test_batch
        x = self((OL, OR, Y))
        loss = self.loss_test(x,Y)
        self.log('test_loss', loss, batch_size=self.hparams.batch_size)
        predictions = torch.argmax(x, dim=1, keepdim=True)
        output = [predictions,Y]

        return output


    def test_epoch_end(self,input_test):
        y_pred = []
        y_true = []
        for ele in input_test:
            y_pred += ele[0].tolist()
            y_true += ele[1].tolist()
        target_names = ['No QC','QC']
        self.y_pred = y_pred
        self.y_true = y_true
        #Classification report
        print(self.y_pred)
        print(self.y_true)
        print(classification_report(self.y_true, self.y_pred, target_names=target_names))




## DEPRECATED

class SaxiRing(LightningModule):
    def __init__(self, **kwargs):
        super(SaxiRing, self).__init__()
        self.save_hyperparameters()
        self.y_pred = []
        self.y_true = []

        # Left network
        self.create_network('L')
        # Right network
        self.create_network('R')

        if self.hparams.subdivision_level >= 2:
            # Get the neighbors to go form level N to level N-1
            ring_neighs_42 = utils.GetPreservedPointIds(self.ico_12,self.ico_42)
            # Create the down blocks to go from 42 -> 12
            self.down2 = AttentionRings(self.hparams.hidden_dim, self.hparams.hidden_dim, self.hparams.hidden_dim, self.ring_neighs_42)

        if self.hparams.subdivision_level == 3:
            ring_neighs_162 = utils.GetPreservedPointIds(self.ico_42,self.ico_162)
            # Create the down blocks to go from 162 -> 42
            self.down1 = AttentionRings(self.hparams.hidden_dim, self.hparams.hidden_dim, self.hparams.hidden_dim, self.ring_neighs_162) 
        
        R=[]
        T=[]
        for coords_cam in self.ico_sphere_verts.tolist():
            camera_position = torch.FloatTensor([coords_cam])
            R_current = look_at_rotation(camera_position)
            # check if camera coords vector and up vector for R are collinear
            if torch.equal(torch.cross(camera_position,torch.tensor([[0.,1.,0.]])),torch.tensor([[0., 0., 0.]])):
               R_current = look_at_rotation(camera_position, up = torch.tensor([[0.0, 0.0, 1.0]]),)
            T_current = -torch.bmm(R_current.transpose(1, 2), camera_position[:,:,None])[:, :, 0]   # (1, 3)

            R.append(R_current)
            T.append(T_current)

        self.R=torch.cat(R)
        self.T=torch.cat(T)
        self.nbr_cam = len(self.R)

        # Loss
        self.loss_train = nn.CrossEntropyLoss()
        self.loss_val = nn.CrossEntropyLoss()
        self.loss_test = nn.CrossEntropyLoss()

        # Dropout
        self.drop = nn.Dropout(p=self.hparams.dropout_lvl)
        
        # Final layer
        self.Classification = nn.Linear(2*self.ico_12.GetNumberOfPoints(), self.hparams.out_classes)

        #vAccuracy
        self.train_accuracy = torchmetrics.Accuracy('multiclass',num_classes=self.hparams.out_classes,average='macro')
        self.val_accuracy = torchmetrics.Accuracy('multiclass',num_classes=self.hparams.out_classes,average='macro')
        
        # Initialize a perspective camera.
        self.hparams.cameras = FoVPerspectiveCameras()

        # We will also create a Phong renderer. This is simpler and only needs to render one face per pixel.
        raster_settings = RasterizationSettings(image_size=self.hparams.image_size,blur_radius=0,faces_per_pixel=1,max_faces_per_bin=100000)

        lights = AmbientLights()
        rasterizer = MeshRasterizer(cameras=self.hparams.cameras,raster_settings=raster_settings)

        self.hparams.phong_renderer = MeshRendererWithFragments(rasterizer=rasterizer,shader=HardPhongShader(cameras=self.hparams.cameras, lights=lights))


    def to(self, device=None):
        # Move the renderer to the specified device
        self.hparams.phong_renderer = self.hparams.phong_renderer.to(device)
        return super().to(device)

    
    def create_network(self, side):
        # Create an icosphere
        if hasattr(monai.networks.nets, self.hparams.base_encoder):
            template_model = getattr(monai.networks.nets, self.hparams.base_encoder)
        elif hasattr(torchvision.models, self.hparams.base_encoder):
            template_model = getattr(torchvision.models, self.hparams.base_encoder)
        else:
            raise "{base_encoder} not in monai networks or torchvision".format(base_encoder=self.hparams.base_encoder)

        model_params = eval('dict(%s)' % self.hparams.base_encoder_params.replace(' ',''))
        
        self.convnet = template_model(**model_params)

        self.ico_12 = utils.CreateIcosahedron(self.hparams.radius)
        
        if self.hparams.subdivision_level == 1:
            ico = self.ico_12

        if self.hparams.subdivision_level >= 2:
            self.ico_42 = utils.SubdividedIcosahedron(self.ico_12,2,self.hparams.radius)
            self.ring_neighs_42 = utils.GetPreservedPointIds(self.ico_12,self.ico_42)
            ico = self.ico_42

        if self.hparams.subdivision_level == 3:
            self.ico_162 = utils.SubdividedIcosahedron(self.ico_42,2,self.hparams.radius)
            self.ring_neighs_162 = utils.GetPreservedPointIds(self.ico_42,self.ico_162)
            ico = self.ico_162
        
        else:
            raise ValueError(f"{self.hparams.subdivision_level} subdivision level not supported, you have to choose between 1, 2 or 3")
        
        self.ico_sphere_verts, self.ico_sphere_faces, self.ico_sphere_edges = utils.PolyDataToTensors(ico)

        setattr(self, f'TimeDistributed{side}', TimeDistributed(self.convnet))
        setattr(self, f'W{side}', nn.Linear(self.hparams.hidden_dim, self.hparams.out_size))
        setattr(self, f'Attention{side}', SelfAttention(self.hparams.hidden_dim, self.hparams.out_size, dim=2))


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def forward(self, x):
        VL, FL, VFL, FFL, VR, FR, VFR, FFR = x
        # TimeDistributed
        xL = self.get_features(VL,FL,VFL,FFL,'L')
        xR = self.get_features(VR,FR,VFR,FFR,'R')
        l_left_right = [xL,xR]
        x = torch.cat(l_left_right,dim=1)
        # Last classification layer
        x = self.drop(x)
        x = self.Classification(x)

        return x


    def get_features(self,V,F,VF,FF,side):
        x, PF = self.render(V,F,VF,FF)  
        x = getattr(self, f'TimeDistributed{side}')(x)
        if self.hparams.subdivision_level == 3:
            x, score = self.down1(x)
            x, score = self.down2(x)
        elif self.hparams.subdivision_level == 2:
            x, score = self.down2(x)
        values = getattr(self, f'W{side}')(x)
        x, _ = getattr(self, f'Attention{side}')(x,values)
        
        return x
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("SaxiRing")
        
        group.add_argument("--lr", type=float, default=1e-4)
        group.add_argument("--fs_path", type=str, default=None)
        group.add_argument("--out_classes", type=int, default=2)
        group.add_argument("--out_size", type=int, default=256)
        group.add_argument('--dropout_lvl',  type=float, help='Dropout level (default: 0.2)', default=0.2)

        # Encoder parameters
        group.add_argument('--base_encoder', type=str, help='Base encoder for the feature extraction', default='resnet18')
        group.add_argument('--base_encoder_params', type=str, help='Base encoder parameters that are passed to build the feature extraction', default='pretrained=False,spatial_dims=2,n_input_channels=1,num_classes=512')
        group.add_argument('--hidden_dim', type=int, help='Hidden dimension for features output. Should match with output of base_encoder. Default value is 512', default=512)
        group.add_argument('--radius', type=float, help='Radius of icosphere', default=1.5)
        group.add_argument('--subdivision_level', type=int, help='Subdivision level for icosahedron', default=2)
        group.add_argument('--image_size', type=int, help='Image resolution size', default=256)

        return parent_parser


    def render(self,V,F,VF,FF):
        # textures = TexturesVertex(verts_features=VF[:, :, :3])

        dummy_textures = [torch.ones((v.shape[0], 3), device=v.device) for v in V]  # (V, C) for each mesh
        dummy_textures = torch.stack(dummy_textures)  # (N, V, C)

        textures = TexturesVertex(verts_features=dummy_textures)
        meshes = Meshes(verts=V, faces=F, textures=textures)

        PF = []
        for i in range(self.nbr_cam):
            pix_to_face = self.GetView(meshes,i)
            PF.append(pix_to_face.unsqueeze(dim=1))
        PF = torch.cat(PF, dim=1)

        l_features = []
        for index in range(FF.shape[-1]):
            l_features.append(torch.take(FF[:,index],PF)*(PF >= 0)) # take each feature for each pictures
        x = torch.cat(l_features,dim=2)

        return x, PF


    def training_step(self, train_batch, batch_idx):
        VL, FL, VFL, FFL, VR, FR, VFR, FFR, Y = train_batch
        x = self((VL, FL, VFL, FFL, VR, FR, VFR, FFR))
        loss = self.loss_train(x,Y)
        self.log('train_loss', loss) 
        predictions = torch.argmax(x, dim=1, keepdim=True)
        self.train_accuracy(predictions, Y.reshape(-1, 1))
        self.log("train_acc", self.train_accuracy, batch_size=self.hparams.batch_size)           

        return loss


    def validation_step(self,val_batch,batch_idx):
        VL, FL, VFL, FFL, VR, FR, VFR, FFR, Y = val_batch
        x = self((VL, FL, VFL, FFL, VR, FR, VFR, FFR))
        loss = self.loss_val(x,Y)
        self.log('val_loss', loss)
        predictions = torch.argmax(x, dim=1, keepdim=True)
        val_acc = self.val_accuracy(predictions, Y.reshape(-1, 1))
        self.log("val_acc", val_acc, batch_size=self.hparams.batch_size)

        return val_acc


    def test_step(self,test_batch,batch_idx):
        VL, FL, VFL, FFL, VR, FR, VFR, FFR, Y = test_batch
        x = self((VL, FL, VFL, FFL, VR, FR, VFR, FFR))
        loss = self.loss_test(x,Y)
        self.log('test_loss', loss, batch_size=self.hparams.batch_size)
        predictions = torch.argmax(x, dim=1, keepdim=True)
        output = [predictions,Y]

        return output


    def test_epoch_end(self,input_test):
        y_pred = []
        y_true = []
        for ele in input_test:
            y_pred += ele[0].tolist()
            y_true += ele[1].tolist()
        target_names = ['No QC','QC']
        self.y_pred = y_pred
        self.y_true = y_true
        #Classification report
        print(self.y_pred)
        print(self.y_true)
        print(classification_report(self.y_true, self.y_pred, target_names=target_names))


    def GetView(self,meshes,index):
        phong_renderer = self.hparams.phong_renderer.to(self.device)
        R = self.R[index][None].to(self.device)
        T = self.T[index][None].to(self.device)
        _, fragments = phong_renderer(meshes.clone(),R=R,T=T)
        pix_to_face = fragments.pix_to_face
        pix_to_face = pix_to_face.permute(0,3,1,2)

        return pix_to_face


    def get_y_for_report_classification(self):
        #This function could be called only after test step was done
        return (self.y_pred,self.hparams.y_true)
    

    def Is_it_Icolayer(self,layer):
        return (layer[:3] == 'Ico')
    
class SaxiRingClassification(LightningModule):
    # Saxi classification network
    def __init__(self, **kwargs):
        super(SaxiRingClassification, self).__init__()
        self.save_hyperparameters()
        self.class_weights = None

        self.create_network()

        if hasattr(self.hparams, 'class_weights'):
            self.class_weights = torch.tensor(self.hparams.class_weights).to(torch.float32)       

        self.loss = nn.CrossEntropyLoss(weight=self.class_weights)
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.out_classes)

        if self.hparams.subdivision_level == 2:
            # Get the neighbors to go form level N to level N-1
            ring_neighs_42 = utils.GetPreservedPointIds(self.ico_12,self.ico_42)
            # Create the down blocks to go from 42 -> 12
            self.down2 = AttentionRings(self.hparams.hidden_dim, self.hparams.hidden_dim, self.hparams.hidden_dim, self.ring_neighs_42)

        elif self.hparams.subdivision_level == 3:
            ring_neighs_162 = utils.GetPreservedPointIds(self.ico_42,self.ico_162)
            # Create the down blocks to go from 162 -> 42
            self.down1 = AttentionRings(self.hparams.hidden_dim, self.hparams.hidden_dim, self.hparams.hidden_dim, self.ring_neighs_162) 
            # Get the neighbors to go form level N to level N-1
            ring_neighs_42 = utils.GetPreservedPointIds(self.ico_12,self.ico_42)
            # Create the down blocks to go from 42 -> 12
            self.down2 = AttentionRings(self.hparams.hidden_dim, self.hparams.hidden_dim, self.hparams.hidden_dim, self.ring_neighs_42)

        # Layers of the network
        self.TimeD = TimeDistributed(self.convnet)
        self.W = nn.Linear(self.hparams.hidden_dim, self.hparams.out_size)
        self.Att = SelfAttention(self.hparams.hidden_dim, self.hparams.out_size, dim=2)  
        self.Drop = nn.Dropout(p=self.hparams.dropout_lvl)
        self.Classification = nn.Linear(self.ico_12.GetNumberOfPoints(), self.hparams.out_classes)  

        cameras = PerspectiveCameras()

        raster_settings = RasterizationSettings(image_size=self.hparams.image_size, blur_radius=0, faces_per_pixel=1,max_faces_per_bin=200000)        
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        lights = AmbientLights()
        self.renderer = MeshRenderer(rasterizer=rasterizer,shader=HardPhongShader(cameras=cameras, lights=lights))
    
    
    def create_network(self):

        if hasattr(monai.networks.nets, self.hparams.base_encoder):
            template_model = getattr(monai.networks.nets, self.hparams.base_encoder)
        elif hasattr(torchvision.models, self.hparams.base_encoder):
            template_model = getattr(torchvision.models, self.hparams.base_encoder)
        else:
            raise "{base_encoder} not in monai networks or torchvision".format(base_encoder=self.hparams.base_encoder)

        model_params = eval('dict(%s)' % self.hparams.base_encoder_params.replace(' ',''))
        self.convnet = template_model(**model_params)

        self.ico_12 = utils.CreateIcosahedron(self.hparams.radius)
        
        if self.hparams.subdivision_level == 1:
            ico = self.ico_12

        elif self.hparams.subdivision_level == 2:
            self.ico_42 = utils.SubdividedIcosahedron(self.ico_12,2,self.hparams.radius)
            self.ring_neighs_42 = utils.GetPreservedPointIds(self.ico_12,self.ico_42)
            ico = self.ico_42

        else:
            self.ico_42 = utils.SubdividedIcosahedron(self.ico_12,2,self.hparams.radius)
            self.ring_neighs_42 = utils.GetPreservedPointIds(self.ico_12,self.ico_42)
            self.ico_162 = utils.SubdividedIcosahedron(self.ico_42,2,self.hparams.radius)
            self.ring_neighs_162 = utils.GetPreservedPointIds(self.ico_42,self.ico_162)
            ico = self.ico_162
        
        ico_sphere_verts, ico_sphere_faces, ico_sphere_edges = utils.PolyDataToTensors(ico)
        ico_verts = ico_sphere_verts.to(torch.float32)

        for idx, v in enumerate(ico_verts):
            if (torch.abs(torch.sum(v)) == self.hparams.radius):
                ico_verts[idx] = v + torch.tensor([-1.2447e-05, -3.7212e-06, -1.5617e-06])
        
        self.register_buffer("ico_verts", ico_verts)
    
    def create_mesh(self, V, F, CN=None):
        
        if CN is not None:
            textures = TexturesVertex(verts_features=CN.to(torch.float32))
            return Meshes(verts=V, faces=F, textures=textures)
        return Meshes(verts=V, faces=F)
    
    def sample_points_from_meshes(self, x_mesh, Ns, return_normals=False):
        if return_normals:
            x, x_N = sample_points_from_meshes(x_mesh, Ns, return_normals=True)
            return x, x_N
        return sample_points_from_meshes(x_mesh, Ns)
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("SaxiRingClassification")

        group.add_argument("--lr", type=float, default=1e-4)
        
        # Encoder parameters
        group.add_argument('--base_encoder', type=str, help='Base encoder for the feature extraction', default='resnet18')
        group.add_argument('--base_encoder_params', type=str, help='Base encoder parameters that are passed to build the feature extraction', default='pretrained=False,spatial_dims=2,n_input_channels=1,num_classes=512')
        group.add_argument('--hidden_dim', type=int, help='Hidden dimension for features output. Should match with output of base_encoder. Default value is 512', default=512)
        group.add_argument('--out_size', type=int, help='Output size for the attention', default=256)
        group.add_argument('--radius', type=float, help='Radius of icosphere', default=1.2)
        group.add_argument('--subdivision_level', type=int, help='Subdivision level for icosahedron', default=3)
        group.add_argument('--image_size', type=int, help='Image resolution size', default=128)
        group.add_argument('--dropout_lvl', type=float, help='Dropout', default=0.1)
        group.add_argument('--out_classes', type=int, help='Output number of classes', default=4)

        return parent_parser
    
    def configure_optimizers(self):
        # Configure the optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer


    def to(self, device=None):
        # Move the renderer to the specified device
        self.renderer = self.renderer.to(device)
        return super().to(device)


    def forward(self, x):
        # Forward pass
        x = self.TimeD(x)
        if self.hparams.subdivision_level == 3:
            x, score = self.down1(x)
            x, score = self.down2(x)
        elif self.hparams.subdivision_level == 2:
            x, score = self.down2(x)
        else:
            x = self.W(x)
        value = self.W(x)
        x, x_s = self.Att(x, value)
        x = self.Drop(x)
        x = self.Classification(x)
        
        return x, x_s


    def render(self, V, F, CN):
        # Render the input surface mesh to an image
        textures = TexturesVertex(verts_features=CN)
        meshes = Meshes(verts=V, faces=F, textures=textures)
        X = []
        PF = []

        for camera_position in self.ico_verts:
            camera_position = camera_position.unsqueeze(0)
            camera_position = camera_position.to(self.device)
            R = look_at_rotation(camera_position, device=self.device)  # (1, 3, 3)
            T = -torch.bmm(R.transpose(1, 2), camera_position[:,:,None])[:, :, 0]   # (1, 3)
            images = self.renderer(meshes_world=meshes.clone(), R=R, T=T)        
            fragments = self.renderer.rasterizer(meshes.clone())
            pix_to_face = fragments.pix_to_face
            zbuf = fragments.zbuf
            images = torch.cat([images[:,:,:,0:3], zbuf], dim=-1)
            images = images.permute(0,3,1,2)
            pix_to_face = pix_to_face.permute(0,3,1,2)
            X.append(images.unsqueeze(1))
            PF.append(pix_to_face.unsqueeze(1))
        
        X = torch.cat(X, dim=1)
        PF = torch.cat(PF, dim=1)        

        return X, PF

    def training_step(self, train_batch, batch_idx):
        # Training step
        V, F, CN, Y = train_batch
        V = V.to(self.device, non_blocking=True)
        F = F.to(self.device, non_blocking=True)        
        CN = CN.to(self.device, non_blocking=True)
        X, PF = self.render(V, F, CN)
        x, _ = self(X)
        loss = self.loss(x, Y)
        batch_size = V.shape[0]
        self.log('train_loss', loss, batch_size=batch_size)
        self.accuracy(x, Y)
        self.log("train_acc", self.accuracy, batch_size=batch_size)

        return loss


    def validation_step(self, val_batch, batch_idx):
        # Validation step
        V, F, CN, Y = val_batch
        V = V.to(self.device, non_blocking=True)
        F = F.to(self.device, non_blocking=True)        
        CN = CN.to(self.device, non_blocking=True)
        X, PF = self.render(V, F, CN)
        x, _ = self(X)
        loss = self.loss(x, Y)
        batch_size = V.shape[0]
        self.log('val_loss', loss, batch_size=batch_size, sync_dist=True)
        self.accuracy(x, Y)
        self.log("val_acc", self.accuracy, batch_size=batch_size, sync_dist=True)


class SaxiDenoiseUnet(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = MHAIdxEncoder(input_dim=self.hparams.input_dim, 
                                     output_dim=self.hparams.stages[-1], 
                                     K=self.hparams.K, 
                                     num_heads=self.hparams.num_heads, 
                                     stages=self.hparams.stages, 
                                     dropout=self.hparams.dropout, 
                                     pooling_factor=self.hparams.pooling_factor, 
                                     pooling_hidden_dim=self.hparams.pooling_hidden_dim,
                                     score_pooling=self.hparams.score_pooling,
                                     feed_forward_hidden_dim=self.hparams.feed_forward_hidden_dim, 
                                     use_skip_connection=self.hparams.use_skip_connection, 
                                     use_direction=self.hparams.use_direction, 
                                     use_layer_norm=self.hparams.use_layer_norm)
        
        self.decoder = MHAIdxDecoder(input_dim=self.hparams.stages[-1], 
                                     output_dim=self.hparams.output_dim, 
                                     K=self.hparams.K[::-1], 
                                     num_heads=self.hparams.num_heads[::-1], 
                                     stages=self.hparams.stages[::-1], 
                                     dropout=self.hparams.dropout, 
                                     pooling_hidden_dim=self.hparams.pooling_hidden_dim[::-1] if self.hparams.pooling_hidden_dim is not None else None,
                                     feed_forward_hidden_dim=self.hparams.feed_forward_hidden_dim[::-1] if self.hparams.feed_forward_hidden_dim is not None else None,
                                     use_skip_connection=self.hparams.use_skip_connection,
                                     use_direction=self.hparams.use_direction, 
                                     use_layer_norm=self.hparams.use_layer_norm)

        self.loss_fn = nn.MSELoss(reduction='sum')
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("SaxiDenoiseUnet")

        
        group.add_argument("--lr", type=float, default=1e-4)
        group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=0.01)
        
        # Encoder/Decoder params
        group.add_argument("--num_samples", type=int, default=1000, help='Number of samples to take from the mesh to start the encoding')
        group.add_argument("--input_dim", type=int, default=3, help='Input dimension for the encoder')
        group.add_argument("--output_dim", type=int, default=3, help='Output dimension of the model')
        group.add_argument("--K", type=int, nargs="*", default=[(96, 32), (96, 32)], help='Number of K neighbors for each stage. If tuple (K_neighbors, Farthest_K_neighbors)')
        group.add_argument("--num_heads", type=int, nargs="*", default=[64, 128], help='Number of attention heads per stage the encoder')
        group.add_argument("--stages", type=int, nargs="*", default=[64, 128], help='Dimension per stage')
        group.add_argument("--dropout", type=float, default=0.1, help='Dropout rate')
        group.add_argument("--pooling_factor", type=float, nargs="*", default=[0.75, 0.75], help='Pooling factor')
        group.add_argument("--score_pooling", type=int, default=0, help='Use score base pooling')
        group.add_argument("--pooling_hidden_dim", type=int, nargs="*", default=[32, 64], help='Hidden dim for the pooling layer')
        group.add_argument("--feed_forward_hidden_dim", type=int, nargs="*", default=[32, 64], help='Hidden dim for the Residual FeedForward layer')
        group.add_argument("--use_skip_connection", type=int, default=1, help='Use skip connections, i.e., unet style network')
        group.add_argument("--use_layer_norm", type=int, default=1, help='Use layer norm')
        group.add_argument("--use_direction", type=int, default=1, help='Use direction instead of position')

        return parent_parser
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(list(self.encoder.parameters()) + list(self.decoder.parameters()),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)        
        return optimizer
    
    def create_mesh(self, V, F):
        return Meshes(verts=V, faces=F)
    
    def sample_points_from_meshes(self, x_mesh, Ns, return_normals=False):
        return sample_points_from_meshes(x_mesh, Ns, return_normals=return_normals)

    def compute_loss(self, X, X_hat, step="train", sync_dist=False):

        loss = self.loss_fn(X, X_hat)

        self.log(f"{step}_loss", loss, sync_dist=sync_dist)

        return loss
    
    def corrupt(self, x, amount):
        """Corrupt the input `x` by mixing it with noise according to `amount`"""
        noise = torch.rand_like(x)
        amount = amount.view(-1, 1, 1)  # Sort shape so broadcasting works
        return x * (1 - amount) + noise * amount

    def training_step(self, train_batch, batch_idx):
        V, F = train_batch
        
        X_mesh = self.create_mesh(V, F)

        X = self.sample_points_from_meshes(X_mesh, self.hparams.num_samples)

        noise_amount = torch.rand(X.shape[0]).to(self.device)  # Pick random noise amounts

        noisy_X = self.corrupt(X, noise_amount)  # Create our noisy x

        X_hat = self(noisy_X)

        loss = self.compute_loss(X, X_hat)

        return loss

    def validation_step(self, val_batch, batch_idx):
        
        V, F = val_batch
        
        X_mesh = self.create_mesh(V, F)

        X = self.sample_points_from_meshes(X_mesh, self.hparams.num_samples)

        noise_amount = amount = torch.linspace(0, 1, X.shape[0]).to(self.device)

        noisy_X = self.corrupt(X, noise_amount)  

        X_hat = self(noisy_X)

        self.compute_loss(X, X_hat, step="val", sync_dist=True)

    def forward(self, x: torch.tensor):

        skip_connections = None

        if self.hparams.use_skip_connection:
            z, unpooling_idxs, skip_connections = self.encoder(x, x)
        else:
            z, unpooling_idxs = self.encoder(x, x)

        return self.decoder(z, unpooling_idxs, skip_connections)

class SaxiDDPMUnet(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = MHAIdxEncoder(input_dim=self.hparams.input_dim, 
                                     output_dim=self.hparams.stages[-1], 
                                     K=self.hparams.K, 
                                     num_heads=self.hparams.num_heads, 
                                     stages=self.hparams.stages, 
                                     dropout=self.hparams.dropout, 
                                     pooling_factor=self.hparams.pooling_factor, 
                                     pooling_hidden_dim=self.hparams.pooling_hidden_dim,
                                     score_pooling=self.hparams.score_pooling,
                                     feed_forward_hidden_dim=self.hparams.feed_forward_hidden_dim, 
                                     use_skip_connection=self.hparams.use_skip_connection, 
                                     use_direction=self.hparams.use_direction, 
                                     use_layer_norm=self.hparams.use_layer_norm, 
                                     time_embed_dim=self.hparams.time_embed_dim)
        
        self.decoder = MHAIdxDecoder(input_dim=self.hparams.stages[-1], 
                                     output_dim=self.hparams.output_dim, 
                                     K=self.hparams.K[::-1], 
                                     num_heads=self.hparams.num_heads[::-1], 
                                     stages=self.hparams.stages[::-1], 
                                     dropout=self.hparams.dropout, 
                                     pooling_hidden_dim=self.hparams.pooling_hidden_dim[::-1] if self.hparams.pooling_hidden_dim is not None else None,
                                     feed_forward_hidden_dim=self.hparams.feed_forward_hidden_dim[::-1] if self.hparams.feed_forward_hidden_dim is not None else None,
                                     use_skip_connection=self.hparams.use_skip_connection,
                                     use_direction=self.hparams.use_direction, 
                                     use_layer_norm=self.hparams.use_layer_norm,
                                     time_embed_dim=self.hparams.time_embed_dim)

        self.loss_fn = nn.MSELoss(reduction='sum')

        # time
        if self.hparams.time_embedding_type == "fourier":
            self.time_proj = GaussianFourierProjection(embedding_size=self.hparams.stages[0], scale=16)
            timestep_input_dim = 2 * self.hparams.stages[0]
        elif self.hparams.time_embedding_type == "positional":
            self.time_proj = Timesteps(self.hparams.stages[0], self.hparams.flip_sin_to_cos, self.hparams.freq_shift)
            timestep_input_dim = self.hparams.stages[0]
        elif self.hparams.time_embedding_type == "learned":
            self.time_proj = nn.Embedding(self.hparams.num_train_timesteps, self.stages[0])
            timestep_input_dim = self.hparams.stages[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, self.hparams.time_embed_dim)

        self.noise_scheduler = DDPMScheduler(num_train_timesteps=self.hparams.num_train_steps, beta_schedule="squaredcos_cap_v2")
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("SaxiIdxAE")

        
        group.add_argument("--lr", type=float, default=1e-4)
        group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=0.01)
        
        # Encoder/Decoder params
        group.add_argument("--num_samples", type=int, default=1000, help='Number of samples to take from the mesh to start the encoding')
        group.add_argument("--input_dim", type=int, default=3, help='Input dimension for the encoder')
        group.add_argument("--output_dim", type=int, default=3, help='Output dimension of the model')
        group.add_argument("--K", type=int, nargs="*", default=[(64, 64), (64, 64)], help='Number of K neighbors for each stage')
        group.add_argument("--num_heads", type=int, nargs="*", default=[64, 128], help='Number of attention heads per stage the encoder')
        group.add_argument("--stages", type=int, nargs="*", default=[64, 128], help='Dimension per stage')
        group.add_argument("--dropout", type=float, default=0.1, help='Dropout rate')
        group.add_argument("--pooling_factor", type=float, nargs="*", default=[0.75, 0.75], help='Pooling factor')
        group.add_argument("--score_pooling", type=int, default=0, help='Use score base pooling')
        group.add_argument("--pooling_hidden_dim", type=int, nargs="*", default=[32, 64], help='Hidden dim for the pooling layer')
        group.add_argument("--feed_forward_hidden_dim", type=int, nargs="*", default=[32, 64], help='Hidden dim for the Residual FeedForward layer')
        group.add_argument("--use_skip_connection", type=int, default=1, help='Use skip connections, i.e., unet style network')
        group.add_argument("--use_layer_norm", type=int, default=1, help='Use layer norm')
        group.add_argument("--use_direction", type=int, default=0, help='Use direction instead of position')
        group.add_argument("--num_train_steps", type=int, default=1000, help='Number of training steps')
        

        group.add_argument("--time_embedding_type", type=str, default='positional', help='Time embedding type', choices=['fourier', 'positional', 'learned'])
        group.add_argument("--time_embed_dim", type=int, default=128, help='Time embedding dimension')
        group.add_argument("--flip_sin_to_cos", type=int, default=1, help='Whether to flip sin to cos for Fourier time embedding.')
        group.add_argument("--freq_shift", type=int, default=0, help='Frequency shift for Fourier time embedding.')
        

        return parent_parser
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(list(self.encoder.parameters()) + list(self.decoder.parameters()),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)        
        return optimizer
    
    def create_mesh(self, V, F):
        return Meshes(verts=V, faces=F)
    
    def sample_points_from_meshes(self, x_mesh, Ns, return_normals=False):
        return sample_points_from_meshes(x_mesh, Ns, return_normals=return_normals)

    def compute_loss(self, X, X_hat, step="train", sync_dist=False):

        loss = self.loss_fn(X, X_hat)

        self.log(f"{step}_loss", loss, sync_dist=sync_dist)

        return loss
    
    # def corrupt(self, x, amount):
    #     """Corrupt the input `x` by mixing it with noise according to `amount`"""
    #     noise = torch.rand_like(x)
    #     amount = amount.view(-1, 1, 1)  # Sort shape so broadcasting works
    #     return x * (1 - amount) + noise * amount

    def training_step(self, train_batch, batch_idx):
        V, F = train_batch
        
        X_mesh = self.create_mesh(V, F)

        X = self.sample_points_from_meshes(X_mesh, self.hparams.num_samples)

        noise = torch.randn_like(X).to(self.device)

        timesteps = torch.randint(0, self.hparams.num_train_steps - 1, (X.shape[0],)).long().to(self.device)

        noisy_X = self.noise_scheduler.add_noise(X, noise, timesteps)

        X_hat = self(noisy_X, timesteps)

        loss = self.compute_loss(noise, X_hat)

        return loss

    def validation_step(self, val_batch, batch_idx):
        
        V, F = val_batch
        
        X_mesh = self.create_mesh(V, F)

        X = self.sample_points_from_meshes(X_mesh, self.hparams.num_samples)

        noise = torch.randn_like(X).to(self.device)
        
        timesteps = torch.randint(0, self.hparams.num_train_steps - 1, (X.shape[0],)).long().to(self.device)

        noisy_X = self.noise_scheduler.add_noise(X, noise, timesteps)

        X_hat = self(noisy_X, timesteps)

        self.compute_loss(noise, X_hat, step="val", sync_dist=True)

    def sampling(self, z_mu: torch.Tensor, z_sigma: torch.Tensor) -> torch.Tensor:        
        eps = torch.randn_like(z_sigma)
        z_vae = z_mu + eps * z_sigma
        return z_vae

    def forward(self, x: torch.tensor, timestep: Union[torch.Tensor, float, int]):

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=self.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(self.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(x.shape[0], dtype=timesteps.dtype, device=timesteps.device)

        t_emb = self.time_proj(timesteps)
        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        skip_connections = None

        if self.hparams.use_skip_connection:
            z, unpooling_idxs, skip_connections = self.encoder(x, x, time=emb)
        else:
            z, unpooling_idxs = self.encoder(x, x, time=emb)

        return self.decoder(z, unpooling_idxs, skip_connections, time=emb)