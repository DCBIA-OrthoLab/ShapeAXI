import math
import numpy as np 
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
import torchmetrics
import monai

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


#####################################################################################################################################################################################
#                                                                                                                                                                                   #
#                                                                                   SaxiRing Brain Data                                                                             #
#                                                                                                                                                                                   #
#####################################################################################################################################################################################


class AttentionRing(nn.Module):
    def __init__(self, in_units, out_units, neigh_orders):
        super().__init__()
        self.num_heads = 8
        # neigh_order: (Nviews previous level, Neighbors next level)
        self.neigh_orders = neigh_orders
        #MHA
        # self.MHA = MultiHeadAttentionModule(in_units, self.num_heads, batch_first=True)
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

        # x, _ = self.MHA(query, values, values)
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


#####################################################################################################################################################################################
#                                                                                                                                                                                   #
#                                                                                Multi Head Attention                                                                               #
#                                                                                                                                                                                   #
#####################################################################################################################################################################################


class MultiHeadAttentionModule(nn.Module):
    def __init__(self, embed_dim, num_heads, batch_first=True):
        super(MultiHeadAttentionModule, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=batch_first)

    def forward(self, query, key, value, attn_mask=None):
        attn_output, attn_output_weights = self.multihead_attn(query, key, value, attn_mask=attn_mask)
        return attn_output, attn_output_weights


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



#####################################################################################################################################################################################
#                                                                                                                                                                                   #
#                                       Architecture (SaxiRing) ued to train the model of Quality Control which is available on Github                                              #
#                                                                                                                                                                                   #
#####################################################################################################################################################################################



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


class SaxiMHAEncoder(nn.Module):
    def __init__(self, input_dim=3, embed_dim=256, hidden_dim=64, num_heads=256, K=32, output_dim=256, sample_levels=[40962, 10242, 2562, 642, 162], dropout=0.1, return_sorted=True):
        super(SaxiMHAEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.K = K
        self.sample_levels = sample_levels
        self.dropout = dropout
        self.return_sorted = return_sorted

        self.embedding = nn.Linear(input_dim, embed_dim)

        for i, sl in enumerate(sample_levels):
            setattr(self, f"mha_{i}", MHA_KNN(embed_dim=embed_dim, num_heads=num_heads, K=K, return_weights=True, dropout=dropout))
            setattr(self, f"ff_{i}", Residual(FeedForward(embed_dim, hidden_dim=hidden_dim, dropout=dropout)))
        
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

        weights = torch.zeros(x.shape[0], x.shape[1], device=x.device)
        idx = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1)
        
        indices = []
        
        for i, sl in enumerate(self.sample_levels):
            
            if i > 0:
                # select the first sl points a.k.a. downsample/pooling                
                x, x_i = self.sample_points(x, sl)

                # initialize idx with the index of the current level
                idx = x_i
                
                for idx_prev in reversed(indices): # go through the list of the previous ones in reverse
                    idx = knn_gather(idx_prev, idx).squeeze(-2).contiguous() # using the indices of the previous level update idx, at the end idx should have the indices of the first level
                
                idx = idx.squeeze(-1)
                indices.append(x_i)
            
            # the mha will select optimal points from the input
            x, x_w = getattr(self, f"mha_{i}")(x)
            x = getattr(self, f"ff_{i}")(x)
            
            weights.scatter_add_(1, idx, x_w)

        #output layer
        x = self.output(x)
        return x, weights
    

class SaxiMHADecoder(nn.Module):
    def __init__(self, input_dim=3, embed_dim=128, output_dim=3, num_heads=4, sample_levels=1, K=4, dropout=0.1, return_sorted=True):
        super(SaxiMHADecoder, self).__init__()

        self.input_dim = input_dim        
        self.K = K
        self.embed_dim = embed_dim
        self.num_heads = num_heads        
        self.sample_levels = sample_levels
        self.dropout = dropout
        self.return_sorted = return_sorted

        self.embedding = nn.Linear(input_dim, embed_dim, bias=False)

        for i in range(sample_levels):
            setattr(self, f"unpool_mha_knn_{i}", UnpoolMHA_KNN(MHA_KNN(embed_dim=embed_dim, num_heads=num_heads, K=K, dropout=dropout)))

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
        
    def forward(self, x):

        x = self.embedding(x)
        
        for i in range(self.sample_levels):
            x = getattr(self, f"unpool_mha_knn_{i}")(x)            
        
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
                                      dropout=self.hparams.dropout, 
                                      K=self.hparams.K)
        
        self.decoder = SaxiMHADecoder(input_dim=self.hparams.output_dim,                                       
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

        self.encoder = SaxiMHAEncoder(input_dim=self.hparams.input_dim, 
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
    

class SaxiD(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        self.decoder = SaxiMHADecoder(input_dim=self.hparams.output_dim,                                       
                                      embed_dim=self.hparams.embed_dim, 
                                      output_dim=self.hparams.input_dim, 
                                      num_heads=self.hparams.num_heads, 
                                      sample_levels=self.hparams.sample_levels,
                                      K=self.hparams.K, 
                                      dropout=self.hparams.dropout)
        
        # self.loss = nn.MSELoss(reduction='sum')
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("SaxiD")

        group.add_argument("--lr", type=float, default=1e-4)
        group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=0.0001)
        group.add_argument('--momentum', help='Momentum for optimizer', type=float, default=0.9)
        
        # Encoder parameters
        
        group.add_argument("--input_dim", type=int, default=3, help='Input dimension for the encoder')
        group.add_argument("--embed_dim", type=int, default=256, help='Embedding dimension')
        group.add_argument("--num_heads", type=int, default=256, help='Number of attention heads')
        group.add_argument("--output_dim", type=int, default=3, help='Output dimension from the encoder')
        group.add_argument("--start_samples", type=int, default=128, help='Starting number of samples for the reconstruction')
        group.add_argument("--end_samples", type=int, default=156, help='Number of samples for the reconstruction. start and end form the range of n samples used during training')
        group.add_argument("--sample_levels", type=int, default=4, help='Number of sampling levels, i.e., max_samples=2^sample_levels*start_samples')        
        
        # group.add_argument("--hidden_dim", type=int, default=128, help='Hidden dimension size')
        group.add_argument("--dropout", type=float, default=0.1, help='Dropout rate')
        # Decoder parameters        
        group.add_argument("--K", type=int, default=32, help='Top K nearest neighbors to consider in the decoder')

        # group.add_argument("--loss_dist_weight", type=float, default=0.01, help='Loss weight for the edge distance during decoder stage')
        group.add_argument("--loss_chamfer_weight", type=float, default=1.0, help='Loss weight for the chamfer distance')
        group.add_argument("--loss_mesh_face_weight", type=float, default=1.0, help='Loss weight for the mesh face distance')
        group.add_argument("--loss_mesh_edge_weight", type=float, default=1.0, help='Loss weight for the mesh edge distance')
        group.add_argument("--loss_repulsion_weight", type=float, default=1.0, help='Loss weight for the mesh edge distance')

        return parent_parser
    
    def configure_optimizers(self):
        # optimizer = optim.AdamW(self.decoder.parameters(),
        #                         lr=self.hparams.lr,
        #                         weight_decay=self.hparams.weight_decay)
        optimizer = optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)
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
        
        loss_chamfer, loss_repulsion = self.chamfer_with_repulsion_loss(X, X_hat, batch_reduction="mean", point_reduction="sum")
        # loss = loss_chamfer        
        
        # X_hat_ordered = knn_gather(X_hat, knn_points(X, X_hat, K=1).idx).squeeze(-2).contiguous()

        X_hat = Pointclouds(X_hat)
        loss_point_mesh_face = point_mesh_face_distance(X_mesh, X_hat)
        loss_point_mesh_edge = point_mesh_edge_distance(X_mesh, X_hat)

        loss = loss_chamfer*self.hparams.loss_chamfer_weight + loss_point_mesh_face*self.hparams.loss_mesh_face_weight + loss_point_mesh_edge*self.hparams.loss_mesh_edge_weight + loss_repulsion*self.hparams.loss_repulsion_weight

        self.log(f"{step}_loss", loss, sync_dist=sync_dist)        
        self.log(f"{step}_loss_chamfer", loss_chamfer, sync_dist=sync_dist)
        self.log(f"{step}_loss_repulsion", loss_repulsion, sync_dist=sync_dist)
        self.log(f"{step}_loss_point_mesh_face", loss_point_mesh_face, sync_dist=sync_dist)
        self.log(f"{step}_loss_point_mesh_edge", loss_point_mesh_edge, sync_dist=sync_dist)

        return loss
    
    def chamfer_with_repulsion_loss(self, pred, target, batch_reduction="mean", point_reduction="sum"):
        """
        Compute Chamfer loss with an additional repulsion term to penalize closely packed points.

        Args:
        - pred (torch.Tensor): Predicted point cloud of shape (Bs, Ns, 3).
        - target (torch.Tensor): Target point cloud of shape (Bs, Nt, 3).
        - repulsion_weight (float): Weight for the repulsion loss term.

        Returns:
        - total_loss (torch.Tensor): Combined Chamfer and repulsion loss.
        """
        # Compute Chamfer Loss using pytorch3d's chamfer_distance function
        chamfer_loss, _ = chamfer_distance(pred, target, batch_reduction="mean", point_reduction="sum")

        # Find k-nearest neighbors in the predicted point cloud
        dists = knn_points(pred, pred, K=5)

        # Compute Repulsion Loss
        # knn_dist has shape (Bs, Ns, k) where each entry is the distance to one of the k-nearest neighbors
        # We ignore the distance to itself (distance of zero) by starting from the second closest neighbor
        repulsion_loss = 1.0 / (dists.dists[:, :, 1:] + 1e-8)  # Avoid division by zero
        repulsion_loss = repulsion_loss.mean()

        # Combine Chamfer Loss and Repulsion Loss
        return chamfer_loss, repulsion_loss

    def training_step(self, train_batch, batch_idx):
        V, F = train_batch
        
        X_mesh = self.create_mesh(V, F)

        sl = torch.randint(self.hparams.start_samples, self.hparams.end_samples, (1,)).item()

        X = self.sample_points_from_meshes(X_mesh, sl)        
        
        X_hat = self(X)

        loss = self.compute_loss(X_mesh, X_hat)

        return loss

    def validation_step(self, val_batch, batch_idx):
        
        V, F = val_batch
        
        X_mesh = self.create_mesh(V, F)

        X = self.sample_points_from_meshes(X_mesh, self.hparams.start_samples)
        
        X_hat = self(X)

        self.compute_loss(X_mesh, X_hat, step="val", sync_dist=True)

    def forward(self, X):                
        return self.decoder(X)
    

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

        self.encoder = SaxiMHAEncoder(input_dim=self.hparams.input_dim, 
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

        # Left network
        self.create_network('L')
        # Right network
        self.create_network('R')

        # Loss
        self.loss_train = nn.CrossEntropyLoss()
        self.loss_val = nn.CrossEntropyLoss()
        self.loss_test = nn.CrossEntropyLoss()

        # Dropout
        self.drop = nn.Dropout(p=self.hparams.dropout_lvl)

        # Final layer
        self.Classification = nn.Linear(2560, self.hparams.out_classes)

        #vAccuracy
        self.train_accuracy = torchmetrics.Accuracy('multiclass',num_classes=self.hparams.out_classes,average='macro')
        self.val_accuracy = torchmetrics.Accuracy('multiclass',num_classes=self.hparams.out_classes,average='macro')


    def create_network(self, side):
        self.resnet = ocnn.models.ResNet(in_channels=7, out_channels=1280, resblock_num=1, stages=3, nempty=False)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def forward(self, x):
        OL, OR, Y = x
        # TimeDistributed
        xL = self.get_features(OL,'L')
        xR = self.get_features(OR,'R')
        l_left_right = [xL,xR]
        x = torch.cat(l_left_right,dim=1)
        # Last classification layer
        x = self.drop(x)
        x = self.Classification(x)

        return x

    def get_features(self,octree,side):
        x = self.resnet(octree.get_input_feature('FP').to(torch.float), octree=octree, depth=8)
        return x

    def training_step(self, train_batch, batch_idx):
        OL, OR, Y = train_batch
        x = self((OL, OR, Y))
        loss = self.loss_train(x,Y)
        self.log('train_loss', loss) 
        predictions = torch.argmax(x, dim=1, keepdim=True)
        self.train_accuracy(predictions, Y.reshape(-1, 1))
        self.log("train_acc", self.train_accuracy, batch_size=self.hparams.batch_size)           

        return loss

    def validation_step(self,val_batch,batch_idx):
        OL, OR, Y = val_batch
        x = self((OL, OR, Y))
        loss = self.loss_val(x,Y)
        self.log('val_loss', loss)
        predictions = torch.argmax(x, dim=1, keepdim=True)
        val_acc = self.val_accuracy(predictions, Y.reshape(-1, 1))
        self.log("val_acc", val_acc, batch_size=self.hparams.batch_size)

        return val_acc


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




############################################################################################################################################################################


# from typing import List, Optional
# import ocnn
# import dwconv


# class OctreeT(Octree):

#   def __init__(self, octree: Octree, patch_size: int = 24, dilation: int = 4,
#                nempty: bool = True, max_depth: Optional[int] = None,
#                start_depth: Optional[int] = None, **kwargs):
#     super().__init__(octree.depth, octree.full_depth)
#     self.__dict__.update(octree.__dict__)

#     self.patch_size = patch_size
#     self.dilation = dilation  # TODO dilation as a list
#     self.nempty = nempty
#     self.max_depth = max_depth or self.depth
#     self.start_depth = start_depth or self.full_depth
#     self.invalid_mask_value = -1e3
#     assert self.start_depth > 1

#     self.block_num = patch_size * dilation
#     self.nnum_t = self.nnum_nempty if nempty else self.nnum
#     self.nnum_a = ((self.nnum_t / self.block_num).ceil() * self.block_num).int()

#     num = self.max_depth + 1
#     self.batch_idx = [None] * num
#     self.patch_mask = [None] * num
#     self.dilate_mask = [None] * num
#     self.rel_pos = [None] * num
#     self.dilate_pos = [None] * num
#     self.build_t()

#   def build_t(self):
#     for d in range(self.start_depth, self.max_depth + 1):
#       self.build_batch_idx(d)
#       self.build_attn_mask(d)
#       self.build_rel_pos(d)

#   def build_batch_idx(self, depth: int):
#     batch = self.batch_id(depth, self.nempty)
#     self.batch_idx[depth] = self.patch_partition(batch, depth, self.batch_size)

#   def build_attn_mask(self, depth: int):
#     batch = self.batch_idx[depth]
#     mask = batch.view(-1, self.patch_size)
#     self.patch_mask[depth] = self._calc_attn_mask(mask)

#     mask = batch.view(-1, self.patch_size, self.dilation)
#     mask = mask.transpose(1, 2).reshape(-1, self.patch_size)
#     self.dilate_mask[depth] = self._calc_attn_mask(mask)

#   def _calc_attn_mask(self, mask: torch.Tensor):
#     attn_mask = mask.unsqueeze(2) - mask.unsqueeze(1)
#     attn_mask = attn_mask.masked_fill(attn_mask != 0, self.invalid_mask_value)
#     return attn_mask

#   def build_rel_pos(self, depth: int):
#     key = self.key(depth, self.nempty)
#     key = self.patch_partition(key, depth)
#     x, y, z, _ = ocnn.octree.key2xyz(key, depth)
#     xyz = torch.stack([x, y, z], dim=1)

#     xyz = xyz.view(-1, self.patch_size, 3)
#     self.rel_pos[depth] = xyz.unsqueeze(2) - xyz.unsqueeze(1)

#     xyz = xyz.view(-1, self.patch_size, self.dilation, 3)
#     xyz = xyz.transpose(1, 2).reshape(-1, self.patch_size, 3)
#     self.dilate_pos[depth] = xyz.unsqueeze(2) - xyz.unsqueeze(1)

#   def patch_partition(self, data: torch.Tensor, depth: int, fill_value=0):
#     num = self.nnum_a[depth] - self.nnum_t[depth]
#     tail = data.new_full((num,) + data.shape[1:], fill_value)
#     return torch.cat([data, tail], dim=0)

#   def patch_reverse(self, data: torch.Tensor, depth: int):
#     return data[:self.nnum_t[depth]]


# class MLP(torch.nn.Module):

#   def __init__(self, in_features: int, hidden_features: Optional[int] = None,
#                out_features: Optional[int] = None, activation=torch.nn.GELU,
#                drop: float = 0.0, **kwargs):
#     super().__init__()
#     self.in_features = in_features
#     self.out_features = out_features or in_features
#     self.hidden_features = hidden_features or in_features

#     self.fc1 = torch.nn.Linear(self.in_features, self.hidden_features)
#     self.act = activation()
#     self.fc2 = torch.nn.Linear(self.hidden_features, self.out_features)
#     self.drop = torch.nn.Dropout(drop, inplace=True)

#   def forward(self, data: torch.Tensor):
#     data = self.fc1(data)
#     data = self.act(data)
#     data = self.drop(data)
#     data = self.fc2(data)
#     data = self.drop(data)
#     return data


# class OctreeDWConvBn(torch.nn.Module):

#   def __init__(self, in_channels: int, kernel_size: List[int] = [3],
#                stride: int = 1, nempty: bool = False):
#     super().__init__()
#     self.conv = dwconv.OctreeDWConv(
#         in_channels, kernel_size, nempty, use_bias=False)
#     self.bn = torch.nn.BatchNorm1d(in_channels)

#   def forward(self, data: torch.Tensor, octree: Octree, depth: int):
#     out = self.conv(data, octree, depth)
#     out = self.bn(out)
#     return out


# class RPE(torch.nn.Module):

#   def __init__(self, patch_size: int, num_heads: int, dilation: int = 1):
#     super().__init__()
#     self.patch_size = patch_size
#     self.num_heads = num_heads
#     self.dilation = dilation
#     self.pos_bnd = self.get_pos_bnd(patch_size)
#     self.rpe_num = 2 * self.pos_bnd + 1
#     self.rpe_table = torch.nn.Parameter(torch.zeros(3*self.rpe_num, num_heads))
#     torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

#   def get_pos_bnd(self, patch_size: int):
#     return int(0.8 * patch_size * self.dilation**0.5)

#   def xyz2idx(self, xyz: torch.Tensor):
#     mul = torch.arange(3, device=xyz.device) * self.rpe_num
#     xyz = xyz.clamp(-self.pos_bnd, self.pos_bnd)
#     idx = xyz + (self.pos_bnd + mul)
#     return idx

#   def forward(self, xyz):
#     idx = self.xyz2idx(xyz)
#     out = self.rpe_table.index_select(0, idx.reshape(-1))
#     out = out.view(idx.shape + (-1,)).sum(3)
#     out = out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
#     return out

#   def extra_repr(self) -> str:
#     return 'num_heads={}, pos_bnd={}, dilation={}'.format(
#             self.num_heads, self.pos_bnd, self.dilation)  # noqa


# class OctreeAttention(torch.nn.Module):

#   def __init__(self, dim: int, patch_size: int, num_heads: int,
#                qkv_bias: bool = True, qk_scale: Optional[float] = None,
#                attn_drop: float = 0.0, proj_drop: float = 0.0,
#                dilation: int = 1, use_rpe: bool = True):
#     super().__init__()
#     self.dim = dim
#     self.patch_size = patch_size
#     self.num_heads = num_heads
#     self.dilation = dilation
#     self.use_rpe = use_rpe
#     self.scale = qk_scale or (dim // num_heads) ** -0.5

#     self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
#     self.attn_drop = torch.nn.Dropout(attn_drop)
#     self.proj = torch.nn.Linear(dim, dim)
#     self.proj_drop = torch.nn.Dropout(proj_drop)
#     self.softmax = torch.nn.Softmax(dim=-1)

#     # NOTE: self.rpe is not used in the original experiments of my paper. When
#     # releasing the code, I added self.rpe because I observed that it could
#     # stablize the training process and improve the performance on ScanNet by
#     # 0.3 to 0.5; on the other datasets, the improvements are more marginal. So
#     # it is not indispensible, and can be removed by setting `use_rpe` as False.
#     self.rpe = RPE(patch_size, num_heads, dilation) if use_rpe else None

#   def forward(self, data: torch.Tensor, octree: OctreeT, depth: int):
#     H = self.num_heads
#     K = self.patch_size
#     C = self.dim
#     D = self.dilation

#     # patch partition
#     data = octree.patch_partition(data, depth)
#     if D > 1:  # dilation
#       rel_pos = octree.dilate_pos[depth]
#       mask = octree.dilate_mask[depth]
#       data = data.view(-1, K, D, C).transpose(1, 2).reshape(-1, C)
#     else:
#       rel_pos = octree.rel_pos[depth]
#       mask = octree.patch_mask[depth]
#     data = data.view(-1, K, C)

#     # qkv
#     qkv = self.qkv(data).reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4)
#     q, k, v = qkv[0], qkv[1], qkv[2]      # (N, H, K, C')
#     q = q * self.scale

#     # attn
#     attn = q @ k.transpose(-2, -1)        # (N, H, K, K)
#     attn = self.apply_rpe(attn, rel_pos)  # (N, H, K, K)
#     attn = attn + mask.unsqueeze(1)
#     attn = self.softmax(attn)
#     attn = self.attn_drop(attn)
#     data = (attn @ v).transpose(1, 2).reshape(-1, C)

#     # patch reverse
#     if D > 1:  # dilation
#       data = data.view(-1, D, K, C).transpose(1, 2).reshape(-1, C)
#     data = octree.patch_reverse(data, depth)

#     # ffn
#     data = self.proj(data)
#     data = self.proj_drop(data)
#     return data

#   def apply_rpe(self, attn, rel_pos):
#     if self.use_rpe:
#       attn = attn + self.rpe(rel_pos)
#     return attn

#   def extra_repr(self) -> str:
#     return 'dim={}, patch_size={}, num_heads={}, dilation={}'.format(
#             self.dim, self.patch_size, self.num_heads, self.dilation)  # noqa


# class OctFormerBlock(torch.nn.Module):

#   def __init__(self, dim: int, num_heads: int, patch_size: int = 32,
#                dilation: int = 0, mlp_ratio: float = 4.0, qkv_bias: bool = True,
#                qk_scale: Optional[float] = None, attn_drop: float = 0.0,
#                proj_drop: float = 0.0, drop_path: float = 0.0, nempty: bool = True,
#                activation: torch.nn.Module = torch.nn.GELU, **kwargs):
#     super().__init__()
#     self.norm1 = torch.nn.LayerNorm(dim)
#     self.attention = OctreeAttention(dim, patch_size, num_heads, qkv_bias,
#                                      qk_scale, attn_drop, proj_drop, dilation)
#     self.norm2 = torch.nn.LayerNorm(dim)
#     self.mlp = MLP(dim, int(dim * mlp_ratio), dim, activation, proj_drop)
#     self.drop_path = ocnn.nn.OctreeDropPath(drop_path, nempty)
#     self.cpe = OctreeDWConvBn(dim, nempty=nempty)

#   def forward(self, data: torch.Tensor, octree: OctreeT, depth: int):
#     data = self.cpe(data, octree, depth) + data
#     attn = self.attention(self.norm1(data), octree, depth)
#     data = data + self.drop_path(attn, octree, depth)
#     ffn = self.mlp(self.norm2(data))
#     data = data + self.drop_path(ffn, octree, depth)
#     return data


# class OctFormerStage(torch.nn.Module):

#   def __init__(self, dim: int, num_heads: int, patch_size: int = 32,
#                dilation: int = 0, mlp_ratio: float = 4.0, qkv_bias: bool = True,
#                qk_scale: Optional[float] = None, attn_drop: float = 0.0,
#                proj_drop: float = 0.0, drop_path: float = 0.0, nempty: bool = True,
#                activation: torch.nn.Module = torch.nn.GELU, interval: int = 6,
#                use_checkpoint: bool = True, num_blocks: int = 2,
#                octformer_block=OctFormerBlock, **kwargs):
#     super().__init__()
#     self.num_blocks = num_blocks
#     self.use_checkpoint = use_checkpoint
#     self.interval = interval  # normalization interval
#     self.num_norms = (num_blocks - 1) // self.interval

#     self.blocks = torch.nn.ModuleList([octformer_block(
#         dim=dim, num_heads=num_heads, patch_size=patch_size,
#         dilation=1 if (i % 2 == 0) else dilation,
#         mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
#         attn_drop=attn_drop, proj_drop=proj_drop,
#         drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
#         nempty=nempty, activation=activation) for i in range(num_blocks)])
#     # self.norms = torch.nn.ModuleList([
#     #     torch.nn.BatchNorm1d(dim) for _ in range(self.num_norms)])

#   def forward(self, data: torch.Tensor, octree: OctreeT, depth: int):
#     for i in range(self.num_blocks):
#       if self.use_checkpoint and self.training:
#         data = checkpoint(self.blocks[i], data, octree, depth)
#       else:
#         data = self.blocks[i](data, octree, depth)
#       # if i % self.interval == 0 and i != 0:
#       #   data = self.norms[(i - 1) // self.interval](data)
#     return data


# class PatchEmbed(torch.nn.Module):

#   def __init__(self, in_channels: int = 3, dim: int = 96, num_down: int = 2,
#                nempty: bool = True, **kwargs):
#     super().__init__()
#     self.num_stages = num_down
#     self.delta_depth = -num_down
#     channels = [int(dim * 2**i) for i in range(-self.num_stages, 1)]

#     self.convs = torch.nn.ModuleList([ocnn.modules.OctreeConvBnRelu(
#         in_channels if i == 0 else channels[i], channels[i], kernel_size=[3],
#         stride=1, nempty=nempty) for i in range(self.num_stages)])
#     self.downsamples = torch.nn.ModuleList([ocnn.modules.OctreeConvBnRelu(
#         channels[i], channels[i+1], kernel_size=[2], stride=2, nempty=nempty)
#         for i in range(self.num_stages)])
#     self.proj = ocnn.modules.OctreeConvBnRelu(
#         channels[-1], dim, kernel_size=[3], stride=1, nempty=nempty)

#   def forward(self, data: torch.Tensor, octree: Octree, depth: int):
#     for i in range(self.num_stages):
#       depth_i = depth - i
#       data = self.convs[i](data, octree, depth_i)
#       data = self.downsamples[i](data, octree, depth_i)
#     data = self.proj(data, octree, depth_i - 1)
#     return data


# class Downsample(torch.nn.Module):

#   def __init__(self, in_channels: int, out_channels: int,
#                kernel_size: List[int] = [2], nempty: bool = True):
#     super().__init__()
#     self.norm = torch.nn.BatchNorm1d(out_channels)
#     self.conv = ocnn.nn.OctreeConv(in_channels, out_channels, kernel_size,
#                                    stride=2, nempty=nempty, use_bias=True)

#   def forward(self, data: torch.Tensor, octree: Octree, depth: int):
#     data = self.conv(data, octree, depth)
#     data = self.norm(data)
#     return data


# class OctFormer(torch.nn.Module):

#   def __init__(self, in_channels: int,
#                channels: List[int] = [96, 192, 384, 384],
#                num_blocks: List[int] = [2, 2, 18, 2],
#                num_heads: List[int] = [6, 12, 24, 24],
#                patch_size: int = 26, dilation: int = 4, drop_path: float = 0.5,
#                nempty: bool = True, stem_down: int = 2, **kwargs):
#     super().__init__()
#     self.patch_size = patch_size
#     self.dilation = dilation
#     self.nempty = nempty
#     self.num_stages = len(num_blocks)
#     self.stem_down = stem_down
#     drop_ratio = torch.linspace(0, drop_path, sum(num_blocks)).tolist()

#     self.patch_embed = PatchEmbed(in_channels, channels[0], stem_down, nempty)
#     self.layers = torch.nn.ModuleList([OctFormerStage(
#         dim=channels[i], num_heads=num_heads[i], patch_size=patch_size,
#         drop_path=drop_ratio[sum(num_blocks[:i]):sum(num_blocks[:i+1])],
#         dilation=dilation, nempty=nempty, num_blocks=num_blocks[i],)
#         for i in range(self.num_stages)])
#     self.downsamples = torch.nn.ModuleList([Downsample(
#         channels[i], channels[i + 1], kernel_size=[2],
#         nempty=nempty) for i in range(self.num_stages - 1)])

#   def forward(self, data: torch.Tensor, octree: Octree, depth: int):
#     data = self.patch_embed(data, octree, depth)
#     depth = depth - self.stem_down   # current octree depth
#     octree = OctreeT(octree, self.patch_size, self.dilation, self.nempty,
#                      max_depth=depth, start_depth=depth-self.num_stages+1)
#     features = {}
#     for i in range(self.num_stages):
#       depth_i = depth - i
#       data = self.layers[i](data, octree, depth_i)
#       features[depth_i] = data
#       if i < self.num_stages - 1:
#         data = self.downsamples[i](data, octree, depth_i)
#     return features




# class ClsHeader(torch.nn.Module):
#   def __init__(self, out_channels: int, in_channels: int,
#                nempty: bool = False, dropout: float = 0.5):
#     super().__init__()
#     self.global_pool = ocnn.nn.OctreeGlobalPool(nempty)
#     self.cls_header = torch.nn.Sequential(
#         ocnn.modules.FcBnRelu(in_channels, 256),
#         torch.nn.Dropout(p=dropout),
#         torch.nn.Linear(256, out_channels))

#   def forward(self, data: torch.Tensor, octree: Octree, depth: int):
#     data = self.global_pool(data, octree, depth)
#     logit = self.cls_header(data)
#     return logit


# class OctFormerCls(torch.nn.Module):

#   def __init__(self, in_channels: int, out_channels: int,
#                channels: List[int] = [96, 192, 384, 384],
#                num_blocks: List[int] = [2, 2, 18, 2],
#                num_heads: List[int] = [6, 12, 24, 24],
#                patch_size: int = 32, dilation: int = 4,
#                drop_path: float = 0.5, nempty: bool = True,
#                stem_down: int = 2, head_drop: float = 0.5, **kwargs):
#     super().__init__()
#     self.backbone = OctFormer(
#         in_channels, channels, num_blocks, num_heads, patch_size, dilation,
#         drop_path, nempty, stem_down)
#     self.head = ClsHeader(
#         out_channels, channels[-1], nempty, head_drop)
#     self.apply(self.init_weights)

#   def init_weights(self, m):
#     if isinstance(m, torch.nn.Linear):
#       torch.nn.init.trunc_normal_(m.weight, std=0.02)
#       if isinstance(m, torch.nn.Linear) and m.bias is not None:
#         torch.nn.init.constant_(m.bias, 0)

#   def forward(self, data: torch.Tensor, octree: Octree, depth: int):
#     features = self.backbone(data, octree, depth)
#     curr_depth = min(features.keys())
#     output = self.head(features[curr_depth], octree, curr_depth)
#     return output




# class SaxiOctreeFormer(LightningModule):
#     def __init__(self, **kwargs):
#         super(SaxiOctree, self).__init__()
#         self.save_hyperparameters()
#         self.y_pred = []
#         self.y_true = []

#         # Left network
#         self.create_network('L')
#         # Right network
#         self.create_network('R')

#         # Loss
#         self.loss_train = nn.CrossEntropyLoss()
#         self.loss_val = nn.CrossEntropyLoss()
#         self.loss_test = nn.CrossEntropyLoss()

#         # Dropout
#         self.drop = nn.Dropout(p=self.hparams.dropout_lvl)

#         # Final layer
#         self.Classification = nn.Linear(2560, self.hparams.out_classes)

#         #vAccuracy
#         self.train_accuracy = torchmetrics.Accuracy('multiclass',num_classes=self.hparams.out_classes,average='macro')
#         self.val_accuracy = torchmetrics.Accuracy('multiclass',num_classes=self.hparams.out_classes,average='macro')


#     def create_network(self, side):
#         # self.resnet = ocnn.models.ResNet(in_channels=7, out_channels=1280, resblock_num=1, stages=3, nempty=False)
#         self.octformercls = OctFormerCls(in_channels=7, out_channels=1280, channels=[96, 192, 384, 384], num_blocks=[2, 2, 18, 2], num_heads=[6, 12, 24, 24], patch_size=32, dilation=4, drop_path=0.5, nempty=True, stem_down=2, head_drop=0.5)

#     def configure_optimizers(self):
#         optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
#         return optimizer

#     def forward(self, x):
#         OL, OR, Y = x
#         # TimeDistributed
#         xL = self.get_features(OL,'L')
#         xR = self.get_features(OR,'R')
#         l_left_right = [xL,xR]
#         x = torch.cat(l_left_right,dim=1)

#         # Last classification layer
#         x = self.drop(x)
#         x = self.Classification(x)

#         return x

#     def get_features(self,octree,side):
#         x = self.octformercls(octree.get_input_feature('FP').to(torch.float), octree=octree, depth=8)
#         return x

#     def training_step(self, train_batch, batch_idx):
#         OL, OR, Y = train_batch
#         x = self((OL, OR, Y))
#         loss = self.loss_train(x,Y)
#         self.log('train_loss', loss) 
#         predictions = torch.argmax(x, dim=1, keepdim=True)
#         self.train_accuracy(predictions, Y.reshape(-1, 1))
#         self.log("train_acc", self.train_accuracy, batch_size=self.hparams.batch_size)           

#         return loss

#     def validation_step(self,val_batch,batch_idx):
#         OL, OR, Y = val_batch
#         x = self((OL, OR, Y))
#         loss = self.loss_val(x,Y)
#         self.log('val_loss', loss)
#         predictions = torch.argmax(x, dim=1, keepdim=True)
#         val_acc = self.val_accuracy(predictions, Y.reshape(-1, 1))
#         self.log("val_acc", val_acc, batch_size=self.hparams.batch_size)

#         return val_acc


#     def test_step(self,test_batch,batch_idx):
#         OL, OR, Y = test_batch
#         x = self((OL, OR, Y))
#         loss = self.loss_test(x,Y)
#         self.log('test_loss', loss, batch_size=self.hparams.batch_size)
#         predictions = torch.argmax(x, dim=1, keepdim=True)
#         output = [predictions,Y]

#         return output


#     def test_epoch_end(self,input_test):
#         y_pred = []
#         y_true = []
#         for ele in input_test:
#             y_pred += ele[0].tolist()
#             y_true += ele[1].tolist()
#         target_names = ['No QC','QC']
#         self.y_pred = y_pred
#         self.y_true = y_true
#         #Classification report
#         print(self.y_pred)
#         print(self.y_true)
#         print(classification_report(self.y_true, self.y_pred, target_names=target_names))