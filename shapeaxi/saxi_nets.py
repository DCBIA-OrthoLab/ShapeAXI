import math
import numpy as np 
import torch
from torch import Tensor, nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import torchmetrics
import monai

import pandas as pd

import pytorch_lightning as pl

from pytorch3d.structures import (
    Meshes,
    Pointclouds)
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
    point_mesh_face_distance,
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)

import json
import os

from shapeaxi import utils
from shapeaxi.saxi_layers import IcosahedronConv2d, TimeDistributed, SelfAttention, Residual, FeedForward, MHA, UnpoolMHA, SmoothAttention, SmoothMHA, ProjectionHead, UnpoolMHA_KNN, MHA_KNN
from shapeaxi.saxi_transforms import GaussianNoise, AvgPoolImages
from shapeaxi.colors import bcolors
from shapeaxi.saxi_losses import saxi_point_triangle_distance

import lightning as L
from lightning.pytorch.core import LightningModule

#####################################################################################################################################################################################
#                                                                                                                                                                                   #
#                                                                                       Classification                                                                              #
#                                                                                                                                                                                   #
#####################################################################################################################################################################################


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
        self.A = SelfAttention(in_units=self.hparams.hidden_dim, out_units=64)
        self.P = nn.Linear(self.hparams.hidden_dim, self.hparams.out_classes)        

        cameras = FoVPerspectiveCameras()

        raster_settings = RasterizationSettings(image_size=self.hparams.image_size, blur_radius=0, faces_per_pixel=1,max_faces_per_bin=200000)        
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        lights = AmbientLights()
        self.renderer = MeshRenderer(rasterizer=rasterizer,shader=HardPhongShader(cameras=cameras, lights=lights))
        self.ico_sphere(radius=self.hparams.radius, subdivision_level=self.hparams.subdivision_level)


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
        self.noise = GaussianNoise(mean=0.0, std=self.hparams.noise_lvl)


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
        self.noise = GaussianNoise(mean=0.0, std=self.hparams.noise_lvl)

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
    def __init__(self, args = None, out_channels=3, class_weights=None, image_size=320, radius=1.35, subdivision_level=1, train_sphere_samples=4):
        super(SaxiSegmentation, self).__init__()        
        self.save_hyperparameters()        
        self.args = args
        
        self.out_channels = out_channels
        self.class_weights = None
        if(class_weights is not None):
            self.class_weights = torch.tensor(class_weights).to(torch.float32)
            
        self.loss = monai.losses.DiceCELoss(include_background=False, to_onehot_y=True, softmax=True, ce_weight=self.class_weights)
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.out_channels)

        unet = monai.networks.nets.UNet(spatial_dims=2,in_channels=4,out_channels=out_channels, channels=(16, 32, 64, 128, 256),strides=(2, 2, 2, 2),num_res_units=2,)
        self.model = TimeDistributed(unet)

        ico_verts, ico_faces, ico_edges = utils.PolyDataToTensors(utils.CreateIcosahedronSubdivided(radius=radius, sl=subdivision_level))
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

    def configure_optimizers(self):
        # Configure the optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr)
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
        self.config_path = os.path.join(os.path.dirname(__file__), "config.json")
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


#####################################################################################################################################################################################
#                                                                                                                                                                                   #
#                                                                                   SaxiRing Brain Data                                                                             #
#                                                                                                                                                                                   #
#####################################################################################################################################################################################


class AttentionRing(nn.Module):
    def __init__(self, in_units, out_units, neigh_orders):
        super().__init__()
        # neigh_order: (Nviews previous level, Neighbors next level)
        self.neigh_orders = neigh_orders
        self.Att = SelfAttention(in_units, out_units, dim=2)

    def forward(self, query, values):
        # Apply attention to the input sequence
        # query: (batch, views, features)
        # values: (batch, views, features)

        #Some neighbours are 7 and some are 6 so we repeat the first one to reach 7
        for data in self.neigh_orders:
            if len(data) == 6:
                repeat_neighbor = data[0]
                data.append(repeat_neighbor)
        
        query = query[:, self.neigh_orders] # (batch, Nv_{n-1}, Idx_{n}, features)
        values = values[:, self.neigh_orders] # (batch, Nv_{n-1}, Idx_{n}, features)

        context_vector, score = self.Att(query, values)

        return context_vector, score


class AttentionRings(nn.Module):
    def __init__(self, in_units, out_v_units, out_q_units, neigh_orders):
        super(AttentionRings, self).__init__()
        self.Q = nn.Linear(in_units, out_q_units)
        self.Att = AttentionRing(in_units, out_v_units, neigh_orders)
        
    def forward(self, x):
        query = self.Q(x)
        context_vector, score = self.Att(query, x)
        return context_vector, score


class SaxiRing(LightningModule):
    def __init__(self, **kwargs):
        super(SaxiRing, self).__init__()
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
#                                                                                   SaxiRing Teeth Data                                                                             #
#                                                                                                                                                                                   #
#####################################################################################################################################################################################


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

        raster_settings = RasterizationSettings(image_size=self.hparams.image_size, blur_radius=0, faces_per_pixel=1,max_faces_per_bin=None)        
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
        self.timepoints = ['T1', 'T2', 'T3']

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
        for timepoint in self.timepoints:
            left_side = f'{timepoint}L'
            right_side = f'{timepoint}R'
            VL, FL, VFL, FFL, Y = train_batch[left_side]
            VR, FR, VFR, FFR, Y = train_batch[right_side]
            x = self((VL, FL, VFL, FFL, VR, FR, VFR, FFR))
            loss = self.loss_train(x,Y)
            self.log('train_loss', loss) 
            predictions = torch.argmax(x, dim=1, keepdim=True)
            self.train_accuracy(predictions, Y.reshape(-1, 1))
            self.log("train_acc", self.train_accuracy, batch_size=self.hparams.batch_size)        
            return loss


    def validation_step(self,val_batch,batch_idx):
        for timepoint in self.timepoints:
            left_side = f'{timepoint}L'
            right_side = f'{timepoint}R'
            VL, FL, VFL, FFL, Y = val_batch[left_side]
            VR, FR, VFR, FFR, Y = val_batch[right_side]
            x = self((VL, FL, VFL, FFL, VR, FR, VFR, FFR))
            loss = self.loss_val(x,Y)
            self.log('val_loss', loss)
            predictions = torch.argmax(x, dim=1, keepdim=True)
            val_acc = self.val_accuracy(predictions, Y.reshape(-1, 1))
            self.log("val_acc", val_acc, batch_size=self.hparams.batch_size)
            return val_acc


    def test_step(self,test_batch,batch_idx):
        for timepoint in self.timepoints:
            left_side = f'{timepoint}L'
            right_side = f'{timepoint}R'
            VL, FL, VFL, FFL, Y = val_batch[left_side]
            VR, FR, VFR, FFR, Y = val_batch[right_side]
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

class SaxiMHAEncoder(nn.Module):
    def __init__(self, input_dim=3, embed_dim=256, num_heads=256, K=32, output_dim=256, sample_levels=[40962, 10242, 2562, 642, 162], dropout=0.1, return_sorted=True):
        super(SaxiMHAEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.K = K
        self.sample_levels = sample_levels
        self.dropout = dropout
        self.return_sorted = return_sorted

        self.embedding = nn.Linear(input_dim, embed_dim)

        for i, sl in enumerate(sample_levels):
            setattr(self, f"mha_{i}", MHA_KNN(embed_dim=embed_dim, num_heads=num_heads, K=K, dropout=dropout))
        
        self.output = nn.Linear(embed_dim, output_dim)
    
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
        
    def forward(self, x):
        
        x = self.embedding(x)
        
        for i, sl in enumerate(self.sample_levels):
            
            if i > 0:
                # sample points for the next level, a.k.a. downsample/pooling
                x, _ = self.sample_points(x, sl)
            # the mha will compute an optimal representation for the current level
            mha = getattr(self, f"mha_{i}")
            x = mha(x)

        #output layer
        x = self.output(x)
        return x
    

class SaxiMHADecoder(nn.Module):
    def __init__(self, input_dim=3, embed_dim=128, output_dim=3, num_heads=4, K=4, dropout=0.1, return_sorted=True):
        super(SaxiMHADecoder, self).__init__()

        self.input_dim = input_dim        
        self.K = K
        self.embed_dim = embed_dim
        self.num_heads = num_heads        
        # self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.return_sorted = return_sorted

        self.embedding = nn.Linear(input_dim, embed_dim, bias=False)

        self.unpool_mha_knn = UnpoolMHA_KNN(MHA_KNN(embed_dim=embed_dim, num_heads=num_heads, K=self.K, dropout=dropout))

        self.output = nn.Linear(embed_dim, output_dim, bias=False)
        
    
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
        
    def forward(self, x, sample_levels=1):

        x = self.embedding(x)
        
        for i in range(sample_levels):
            x = self.unpool_mha_knn(x)
        
        x = self.output(x)

        return x
        
    

class SaxiAE(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = SaxiMHAEncoder(input_dim=self.hparams.input_dim, 
                                      embed_dim=self.hparams.embed_dim, 
                                      num_heads=self.hparams.num_heads, 
                                      output_dim=self.hparams.output_dim, 
                                      sample_levels=self.hparams.sample_levels, 
                                      hidden_dim=self.hparams.hidden_dim, 
                                      dropout=self.hparams.dropout, 
                                      K=self.hparams.K_encoder)
        
        self.decoder = SaxiMHADecoder(input_dim=self.hparams.output_dim, 
                                      sample_levels=self.hparams.sample_levels[::-1], 
                                      embed_dim=self.hparams.embed_dim, 
                                      output_dim=self.hparams.input_dim, 
                                      num_heads=self.hparams.num_heads, 
                                      K=self.hparams.K_decoder, 
                                      hidden_dim=self.hparams.hidden_dim, 
                                      dropout=self.hparams.dropout)
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("SaxiAE")

        group.add_argument("--lr", type=float, default=1e-4)
        group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=0.01)
        
        # Encoder parameters
        
        group.add_argument("--input_dim", type=int, default=3, help='Input dimension for the encoder')
        group.add_argument("--embed_dim", type=int, default=256, help='Embedding dimension')
        group.add_argument("--num_heads", type=int, default=32, help='Number of attention heads')
        group.add_argument("--output_dim", type=int, default=256, help='Output dimension from the encoder')
        group.add_argument("--sample_levels", type=int, default=[10242, 2562, 642], nargs="+", help='Number of sampling levels in the encoder')
        group.add_argument("--sample_level_loss", type=int, default=1000, help='Number of samples to compute loss')
        
        group.add_argument("--hidden_dim", type=int, default=64, help='Hidden dimension size')
        group.add_argument("--dropout", type=float, default=0.5, help='Dropout rate')
        
        # Decoder parameters
        group.add_argument("--K_encoder", type=int, default=8, help='Top K nearest neighbors to consider in the encoder')
        group.add_argument("--K_decoder", type=int, default=8, help='Top K nearest neighbors to consider in the decoder steps')
        group.add_argument("--loss_chamfer_weight", type=float, default=1.0, help='Loss weight for the chamfer distance')
        # group.add_argument("--loss_point_triangle_weight", type=float, default=0.1, help='Loss weight for the point to nearest face plane distance')
        group.add_argument("--loss_mesh_face_weight", type=float, default=1.0, help='Loss weight for the mesh face distance')
        group.add_argument("--loss_mesh_edge_weight", type=float, default=1.0, help='Loss weight for the mesh edge distance')
        group.add_argument("--loss_decoder_dist_weight", type=float, default=0.01, help='Loss weight for the edge distance during decoder stage')
        
        # group.add_argument("--loss_laplacian_weight", type=float, default=1.0, help='Loss weight for the laplacian smoothing')
        # group.add_argument("--loss_edge_weight", type=float, default=10.0, help='Loss weight for the mesh edge distance')
        # group.add_argument("--loss_normal_weight", type=float, default=0.01, help='Loss weight for the mesh normal distance')

        return parent_parser
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(list(self.encoder.parameters()) + list(self.decoder.parameters()),
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

    def compute_loss(self, X_mesh, X_hat, total_dist, step="train", sync_dist=False):
        # We compare the two sets of pointclouds by computing (a) the chamfer loss

        # X, X_N = self.sample_points_from_meshes(X_mesh, self.hparams.sample_level_loss, return_normals=True)

        # X_hat, X_hat_N = X_hat[..., :3], X_hat[..., 3:]
        # X_hat_N = X_hat_N / torch.norm(X_hat_N, dim=-1, keepdim=True)

        # loss_chamfer, loss_chamfer_N = chamfer_distance(X, X_hat, x_normals=X_N, y_normals=X_hat_N, batch_reduction="sum", point_reduction="sum")


        X = self.sample_points_from_meshes(X_mesh, self.hparams.sample_level_loss)
        loss_chamfer, _ = chamfer_distance(X, X_hat)
        
        
        # loss_point_mesh_face = saxi_point_triangle_distance(X, X_hat) + saxi_point_triangle_distance(X_hat, X_hat, ignore_first=True, K_triangle=9, randomize=True)  
        # loss_point_triangle = saxi_point_triangle_distance(X, X_hat)

        # Generate pseudo faces for the input point cloud, i.e, for each point find the 3 closest neighbors
        # dists = knn_points(X, X, K=3)
        # X_F = dists.idx
        
        # Find the nearest neighbors in the predicted mesh
        # dists = knn_points(X, X_hat, K=1)
        # Sort the predicted points based on the indices of the nearest neighbors
        # X_hat_sorted = knn_gather(X_hat, dists.idx).squeeze(-2).contiguous()

        # X_hat_mesh = Meshes(verts=X_hat_sorted, faces=X_F)

         # and (b) the edge length of the predicted mesh
        # loss_edge = mesh_edge_loss(X_hat_mesh)
    
        # mesh laplacian smoothing
        # loss_laplacian = mesh_laplacian_smoothing(X_hat_mesh, method="uniform")

        X_hat = Pointclouds(X_hat)
        loss_point_mesh_face = point_mesh_face_distance(X_mesh, X_hat)
        loss_point_mesh_edge = point_mesh_edge_distance(X_mesh, X_hat)

        # loss = (loss_chamfer + loss_chamfer_N)*self.hparams.loss_chamfer_weight + loss_point_triangle*self.hparams.loss_point_triangle_weight + loss_laplacian*self.hparams.loss_laplacian_weight + loss_edge*self.hparams.loss_edge_weight 
        # loss = (loss_chamfer + loss_chamfer_N)*self.hparams.loss_chamfer_weight + loss_point_mesh_face*self.hparams.loss_mesh_face_weight + loss_point_mesh_edge*self.hparams.loss_mesh_edge_weight + 1.0/(total_dist + 1e-6)*self.hparams.loss_decoder_dist_weight
        # loss = (loss_chamfer + loss_chamfer_N)*self.hparams.loss_chamfer_weight + 1.0/(total_dist + 1e-6)*self.hparams.loss_decoder_dist_weight + loss_point_triangle*self.hparams.loss_point_triangle_weight

        loss = loss_chamfer*self.hparams.loss_chamfer_weight + loss_point_mesh_face*self.hparams.loss_mesh_face_weight + loss_point_mesh_edge*self.hparams.loss_mesh_edge_weight + 1.0/(total_dist + 1e-6)*self.hparams.loss_decoder_dist_weight

        self.log(f"{step}_loss", loss, sync_dist=sync_dist)
        # self.log(f"{step}_loss_point_triangle", loss_point_triangle, sync_dist=sync_dist)
        self.log(f"{step}_loss_chamfer", loss_chamfer, sync_dist=sync_dist)
        # self.log(f"{step}_loss_chamfer_N", loss_chamfer_N, sync_dist=sync_dist)
        # self.log(f"{step}_loss_laplacian", loss_laplacian, sync_dist=sync_dist)        
        # self.log(f"{step}_loss_edge", loss_edge, sync_dist=sync_dist)
        self.log(f"{step}_loss_point_mesh_face", loss_point_mesh_face, sync_dist=sync_dist)
        self.log(f"{step}_loss_point_mesh_edge", loss_point_mesh_edge, sync_dist=sync_dist)
        self.log(f"{step}_total_dist", total_dist, sync_dist=sync_dist)

        return loss

    def training_step(self, train_batch, batch_idx):
        V, F = train_batch
        
        X_mesh = self.create_mesh(V, F)

        # X, X_N = self.sample_points_from_meshes(X_mesh, self.hparams.sample_levels[0], return_normals=True)        
        # X = torch.cat([X, X_N], dim=-1)

        X = self.sample_points_from_meshes(X_mesh, self.hparams.sample_levels[0])

        X_hat, z, total_dist = self(X)

        loss = self.compute_loss(X_mesh, X_hat, total_dist)

        return loss

    def validation_step(self, val_batch, batch_idx):
        
        V, F = val_batch
        
        X_mesh = self.create_mesh(V, F)

        # X, X_N = self.sample_points_from_meshes(X_mesh, self.hparams.sample_levels[0], return_normals=True)
        # X = torch.cat([X, X_N], dim=-1)

        X = self.sample_points_from_meshes(X_mesh, self.hparams.sample_levels[0])

        X_hat, z, total_dist = self(X)

        self.compute_loss(X_mesh, X_hat, total_dist, step="val", sync_dist=True)

    def forward(self, X):        
        z = self.encoder(X)
        X_hat, total_dist = self.decoder(z)
        return X_hat, z, total_dist
    

class SaxiMHAClassification(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = SaxiMHAEncoder(input_dim=self.hparams.input_dim, 
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
        group = parent_parser.add_argument_group("SaxiAE")

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
        x = self.encoder(X)
        x, x_v = self.mha(x)
        x = torch.cat([x, x_v], dim=1)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    

class SaxiD(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        self.decoder = SaxiMHADecoder(input_dim=self.hparams.output_dim,                                       
                                      embed_dim=self.hparams.embed_dim, 
                                      output_dim=self.hparams.input_dim, 
                                      num_heads=self.hparams.num_heads, 
                                      K=self.hparams.K, 
                                      dropout=self.hparams.dropout)
        
        # self.loss = nn.MSELoss(reduction='sum')
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("SaxiAE")

        group.add_argument("--lr", type=float, default=1e-4)
        group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=0.01)
        
        # Encoder parameters
        
        group.add_argument("--input_dim", type=int, default=3, help='Input dimension for the encoder')
        group.add_argument("--embed_dim", type=int, default=256, help='Embedding dimension')
        group.add_argument("--num_heads", type=int, default=256, help='Number of attention heads')
        group.add_argument("--output_dim", type=int, default=3, help='Output dimension from the encoder')
        group.add_argument("--start_samples", type=int, default=64, help='Starting number of samples for the reconstruction')
        group.add_argument("--end_samples", type=int, default=128, help='Number of samples for the reconstruction. start and end form the range of n samples used during training')
        group.add_argument("--sample_levels", type=int, default=4, help='Number of sampling levels, i.e., max_samples=2^sample_levels*start_samples')        
        
        # group.add_argument("--hidden_dim", type=int, default=128, help='Hidden dimension size')
        group.add_argument("--dropout", type=float, default=0.1, help='Dropout rate')
        # Decoder parameters        
        group.add_argument("--K", type=int, default=32, help='Top K nearest neighbors to consider in the decoder')

        # group.add_argument("--loss_dist_weight", type=float, default=0.01, help='Loss weight for the edge distance during decoder stage')
        group.add_argument("--loss_chamfer_weight", type=float, default=1.0, help='Loss weight for the chamfer distance')
        group.add_argument("--loss_mesh_face_weight", type=float, default=1.0, help='Loss weight for the mesh face distance')
        group.add_argument("--loss_mesh_edge_weight", type=float, default=1.0, help='Loss weight for the mesh edge distance')

        return parent_parser
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.decoder.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        return optimizer
    
    def create_mesh(self, V, F):
        return Meshes(verts=V, faces=F)
    
    def create_mesh_from_points(self, X):
        dists = knn_points(X, X, K=3)
        F = dists.idx
        return Meshes(verts=X, faces=F)
    
    def sample_points(self, x, Ns):
        return self.decoder.sample_points(x, Ns)
    
    def sample_points_from_meshes(self, x_mesh, Ns, return_normals=False):
        if return_normals:
            x, x_N = sample_points_from_meshes(x_mesh, Ns, return_normals=True)
            return x, x_N
        return sample_points_from_meshes(x_mesh, Ns)

    def compute_loss(self, X_mesh, X_hat, step="train", sync_dist=False):
        
        
        ns = int(math.pow(2.0, self.hparams.sample_levels)*self.hparams.start_samples)
        X = self.sample_points_from_meshes(X_mesh, ns)
        
        loss_chamfer, _ = chamfer_distance(X, X_hat, batch_reduction="mean", point_reduction="sum")
        # loss = loss_chamfer        
        
        # X_hat_ordered = knn_gather(X_hat, knn_points(X, X_hat, K=1).idx).squeeze(-2).contiguous()

        X_hat = Pointclouds(X_hat)
        loss_point_mesh_face = point_mesh_face_distance(X_mesh, X_hat)
        loss_point_mesh_edge = point_mesh_edge_distance(X_mesh, X_hat)

        loss = loss_chamfer*self.hparams.loss_chamfer_weight + loss_point_mesh_face*self.hparams.loss_mesh_face_weight + loss_point_mesh_edge*self.hparams.loss_mesh_edge_weight

        self.log(f"{step}_loss", loss, sync_dist=sync_dist)        
        self.log(f"{step}_loss_chamfer", loss_chamfer, sync_dist=sync_dist)
        self.log(f"{step}_loss_point_mesh_face", loss_point_mesh_face, sync_dist=sync_dist)
        self.log(f"{step}_loss_point_mesh_edge", loss_point_mesh_edge, sync_dist=sync_dist)

        return loss

    def training_step(self, train_batch, batch_idx):
        V, F = train_batch
        
        X_mesh = self.create_mesh(V, F)

        sl = torch.randint(self.hparams.start_samples, self.hparams.end_samples, (1,)).item()

        X = self.sample_points_from_meshes(X_mesh, sl)
        
        X_hat = self.decoder(X)

        loss = self.compute_loss(X_mesh, X_hat)

        return loss

    def validation_step(self, val_batch, batch_idx):
        
        V, F = val_batch
        
        X_mesh = self.create_mesh(V, F)

        X = self.sample_points_from_meshes(X_mesh, self.hparams.start_samples)
        
        X_hat = self.decoder(X, self.hparams.sample_levels)

        self.compute_loss(X_mesh, X_hat, step="val", sync_dist=True)

    def forward(self, X):                
        return self.decoder(X)