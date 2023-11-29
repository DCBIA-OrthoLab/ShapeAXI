import math
import numpy as np 
import torch
from torch import Tensor, nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import torchmetrics
import monai
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import plot_scene
import pytorch_lightning as pl
from pytorch3d.vis.plotly_vis import plot_scene
import plotly.express as px
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
import seaborn
import cv2
from pytorch3d.renderer import (
        FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
        RasterizationSettings, MeshRenderer, MeshRasterizer, MeshRendererWithFragments, BlendParams,
        SoftSilhouetteShader, HardPhongShader, SoftPhongShader, AmbientLights, PointLights, TexturesUV, TexturesVertex, TexturesAtlas
)
import requests
from io import BytesIO
import json
import os

import utils
from IcoConcOperator import IcosahedronConv1d, IcosahedronConv2d, IcosahedronLinear
from saxi_transforms import GaussianNoise, MaxPoolImages, AvgPoolImages, SelfAttention, Identity, TimeDistributed


class ProjectionHead(nn.Module):
    # Projection MLP
    def __init__(self, input_dim=1280, hidden_dim=1280, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False)
        )

    def forward(self, x):
        x_v = self.model(x)
        return x, x_v


class TimeDistributed(nn.Module):
    # Wrapper to apply a module to each time step of a sequence
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module
 
    def forward(self, input_seq):
        assert len(input_seq.size()) > 2
 
        # reshape input data --> (samples * timesteps, input_size)
        # squash timesteps

        size = input_seq.size()
        batch_size = size[0]
        time_steps = size[1]
        size_reshape = [batch_size*time_steps] + list(size[2:])
        reshaped_input = input_seq.contiguous().view(size_reshape)
        output = self.module(reshaped_input)
        output_size = output.size()
        output_size = [batch_size, time_steps] + list(output_size[1:])
        output = output.contiguous().view(output_size)

        return output


class SelfAttention(nn.Module):
    # Self attention layer
    def __init__(self, in_units, out_units):
        super().__init__()
        self.W1 = nn.Linear(in_units, out_units)
        self.V = nn.Linear(out_units, 1)

    def forward(self, query, values):        
        score = nn.Sigmoid()(self.V(nn.Tanh()(self.W1(query))))
        attention_weights = score/torch.sum(score, dim=1,keepdim=True)
        context_vector = attention_weights * values
        context_vector = torch.sum(context_vector, dim=1)
        return context_vector, score



#################################################################################### SEGMENTATION PART ########################################################################

class SaxiSegmentation(pl.LightningModule):
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

        ico_verts, ico_faces, ico_edges = utils.PolyDataToTensors(utils.CreateIcosahedron(radius=radius, sl=subdivision_level))
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



########################################################################## CLASSIFICATION PART ######################################################################################

class SaxiClassification(pl.LightningModule):
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
        ico_verts, ico_faces, ico_edges = utils.PolyDataToTensors(utils.CreateIcosahedron(radius=radius, sl=subdivision_level))
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




################################################################################ SAXIREGRESSION PART ################################################################################


class SaxiRegression(pl.LightningModule):
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
        ico_verts, ico_faces, ico_edges = utils.PolyDataToTensors(utils.CreateIcosahedron(radius=radius, sl=subdivision_level))
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



#################################################################################### MONAIUNET PART #########################################################################################

class MonaiUNet(pl.LightningModule):
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

        ico_verts, ico_faces, ico_edges = utils.PolyDataToTensors(utils.CreateIcosahedron(radius=radius, sl=subdivision_level))
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


########################################################################################### ICOCONV PART ########################################################################################

class SaxiIcoClassification(pl.LightningModule):
    def __init__(self, **kwargs):
        super(SaxiIcoClassification, self).__init__()
        self.save_hyperparameters()
        self.y_pred = []
        self.y_true = []

        ico_sphere = utils.CreateIcosahedron(self.hparams.radius, self.hparams.ico_lvl)
        ico_sphere_verts, ico_sphere_faces, self.hparams.ico_sphere_edges = utils.PolyDataToTensors(ico_sphere)
        self.ico_sphere_verts = ico_sphere_verts
        self.ico_sphere_edges = np.array(self.hparams.ico_sphere_edges)
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

        out_size = 256
        # Left path
        self.create_network('L', out_size)
        # Right path
        self.create_network('R', out_size)

        #Demographics
        self.normalize = nn.BatchNorm1d(self.hparams.nbr_demographic)

        #Final layer
        self.Classification = nn.Linear(2*out_size+self.hparams.nbr_demographic, 2)

        #Loss
        self.loss_train = nn.CrossEntropyLoss(weight=self.hparams.weights[0])
        self.loss_val = nn.CrossEntropyLoss(weight=self.hparams.weights[1])
        self.loss_test = nn.CrossEntropyLoss(weight=self.hparams.weights[2])

        #Accuracy
        self.train_accuracy = torchmetrics.Accuracy('multiclass',num_classes=2,average='macro')
        self.val_accuracy = torchmetrics.Accuracy('multiclass',num_classes=2,average='macro')
        
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

    
    # Create an icosphere
    def create_network(self, side, out_size):
        if hasattr(torchvision.models, self.hparams.base_encoder):
            template_model = getattr(torchvision.models, self.hparams.base_encoder)
            print('Torvision network')
        else:
            raise f"{self.hparams.base_encoder} not in monai networks or torchvision"

        model_params = eval('dict(%s)' % self.hparams.base_encoder_params.replace('', ''))
        # Remove of the parameters that are not used by the torchvision model
        model_params.pop('n_input_channels', None)
        model_params.pop('spatial_dims', None)
        model_params.pop('pretrained', None)
        self.convnet = template_model(**model_params)
        setattr(self, f'TimeDistributed{side}', TimeDistributed(self.convnet))
        output_size = model_params.get('num_classes', None)

        if self.hparams.layer == 'Att':
            setattr(self, f'WV{side}', nn.Linear(output_size, out_size))
            setattr(self, f'Attention{side}', SelfAttention(output_size, 128))

        elif self.hparams.layer in {'IcoConv2D', 'IcoConv1D', 'IcoLinear'}:
            if self.hparams.layer == 'IcoConv2D':
                conv_layer = nn.Conv2d(output_size, out_size, kernel_size=(3, 3), stride=2, padding=0)
            elif self.hparams.layer == 'IcoConv1D':
                conv_layer = nn.Conv1d(output_size, out_size, 7)
            else:
                conv_layer = nn.Linear(output_size * 7, out_size)
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
        B,NV,C,L,W = xL.size()
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



#################################################################### DENTAL MODEL SEGMENTATION PART #########################################################################################

class DentalModelSeg(pl.LightningModule):
    def __init__(self, args=None, out_channels=3 ,config_path = "./", class_weights=None, image_size=320, radius=1.35, subdivision_level=1, train_sphere_samples=4):
        super(DentalModelSeg, self).__init__()        
        self.save_hyperparameters()        
        self.args = args
        self.config_path = config_path
        self.out_channels = out_channels
        self.class_weights = None
        if(class_weights is not None):
            self.class_weights = torch.tensor(class_weights).to(torch.float32)
            
        self.loss = monai.losses.DiceCELoss(include_background=False, to_onehot_y=True, softmax=True, ce_weight=self.class_weights)
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=34)

        #Initialize and load model
        unet = self.create_unet_model()
        self.model = self.dental_model_seg(unet)
        self.model = TimeDistributed(unet)

        ico_verts, ico_faces, ico_edges = utils.PolyDataToTensors(utils.CreateIcosahedron(radius=radius, sl=subdivision_level))
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
            print("Model loaded successfully")
            return model

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
        pass

    def validation_step(self, val_batch, batch_idx):
        pass

    def test_step(self, val_batch, batch_idx):
        pass
