import math
import numpy as np 

import torch
from torch import Tensor, nn

import torchvision
from torchvision import models
from torchvision import transforms
import torchmetrics

import src.utils as utils

import monai
from pytorch3d.renderer import (
        FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
        RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
        SoftSilhouetteShader, HardPhongShader, SoftPhongShader, AmbientLights, PointLights, TexturesUV, TexturesVertex, TexturesAtlas
)
from pytorch3d.structures import Meshes

import pytorch_lightning as pl

class ProjectionHead(nn.Module):
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
    def __init__(self, in_units, out_units):
        #super(SelfAttention, self).__init__()
        super().__init__()


        self.W1 = nn.Linear(in_units, out_units)
        self.V = nn.Linear(out_units, 1)

    def forward(self, query, values):        

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)

        score = nn.Sigmoid()(self.V(nn.Tanh()(self.W1(query))))
        
        attention_weights = score/torch.sum(score, dim=1,keepdim=True)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector, score

class SaxiSegmentation(pl.LightningModule):
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr)
        return optimizer

    def to(self, device=None):
        self.renderer = self.renderer.to(device)
        return super().to(device)

    def forward(self, x):

        V, F, CN = x
        
        X, PF = self.render(V, F, CN)
        x = self.model(X)
        
        return x, X, PF

    def render(self, V, F, CN):

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

        V, F, YF, CN = val_batch

        x, X, PF = self(V, F, CN)
        y = torch.take(YF, PF).to(torch.int64)*(PF >= 0)
        
        x = x.permute(0, 2, 1, 3, 4) #batch, time, channels, h, w -> batch, channels, time, h, w
        y = y.permute(0, 2, 1, 3, 4) 

        loss = self.loss(x, y)

        self.accuracy(torch.argmax(x, dim=1, keepdim=True).reshape(-1, 1), y.reshape(-1, 1).to(torch.int32))        

        return {'test_loss': loss, 'test_correct': self.accuracy}

class SaxiClassification(pl.LightningModule):
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

        model_params = eval('dict(%s)' % self.hparams.base_encoder_params.replace(' ',''))
        
        self.convnet = template_model(**model_params)

        self.F = TimeDistributed(self.convnet)
        self.V = nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim)
        self.A = SelfAttention(in_units=self.hparams.hidden_dim, out_units=64)
        self.P = nn.Linear(self.hparams.hidden_dim, self.hparams.out_classes)        

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
        ico_verts, ico_faces, ico_edges = utils.PolyDataToTensors(utils.CreateIcosahedron(radius=radius, sl=subdivision_level))
        ico_verts = ico_verts.to(torch.float32)

        for idx, v in enumerate(ico_verts):
            if (torch.abs(torch.sum(v)) == radius):
                ico_verts[idx] = v + torch.normal(0.0, 1e-7, (3,))
        
        self.register_buffer("ico_verts", ico_verts)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def to(self, device=None):
        self.renderer = self.renderer.to(device)
        return super().to(device)

    def forward(self, X):
        x_f = self.F(X)
        x_v = self.V(x_f)
        x_a, x_s = self.A(x_f, x_v)
        x = self.P(x_a)
        
        return x, x_s

    def render(self, V, F, CN):
        
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

    # def test_step(self, test_batch, batch_idx):

    #     V, F, CN, Y = test_batch

    #     V = V.to(self.device, non_blocking=True)
    #     F = F.to(self.device, non_blocking=True)        
    #     CN = CN.to(self.device, non_blocking=True)

    #     X, PF = self.render(V, F, CN)
    #     x, _ = self(X)

    #     batch_size = V.shape[0]
    #     self.accuracy(x, Y)
    #     self.log("test_acc", self.accuracy, batch_size=batch_size, sync_dist=True)

    #     return 1.0

class SaxiRegression(pl.LightningModule):
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
        ico_verts, ico_faces, ico_edges = utils.PolyDataToTensors(utils.CreateIcosahedron(radius=radius, sl=subdivision_level))
        ico_verts = ico_verts.to(torch.float32)

        for idx, v in enumerate(ico_verts):
            if (torch.abs(torch.sum(v)) == radius):
                ico_verts[idx] = v + torch.normal(0.0, 1e-7, (3,))
        
        self.register_buffer("ico_verts", ico_verts)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def to(self, device=None):
        self.renderer = self.renderer.to(device)
        return super().to(device)

    def forward(self, X):
        x_f = self.F(X)
        x_v = self.V(x_f)
        x_a, x_s = self.A(x_f, x_v)
        x = self.P(x_a)
        
        return x

    def render(self, V, F, CN):
        
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

        V, F, CN, Y = val_batch

        V = V.to(self.device, non_blocking=True)
        F = F.to(self.device, non_blocking=True)        
        CN = CN.to(self.device, non_blocking=True)

        X, PF = self.render(V, F, CN)
        x = self(X)
            
        loss = self.loss(x, Y)

        batch_size = V.shape[0]
        self.log('val_loss', loss, batch_size=batch_size, sync_dist=True)




class MonaiUNet(pl.LightningModule):
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr)
        return optimizer

    def to(self, device=None):
        self.renderer = self.renderer.to(device)
        return super().to(device)

    def forward(self, x):
        V, F, CN = x
        X, PF = self.render(V, F, CN)
        x = self.model(X)
        return x, X, PF


    def render(self, V, F, CN):

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