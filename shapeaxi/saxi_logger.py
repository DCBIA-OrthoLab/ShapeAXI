from lightning.pytorch.callbacks import Callback
import torchvision
import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# This file contains custom callbacks for logging and visualizing images during training within a PyTorch Lightning-based deep learning workflow

#####################################################################################################################################################################################
#                                                                                                                                                                                   #
#                                                                                       Tensorboard                                                                                 #
#                                                                                                                                                                                   #
#####################################################################################################################################################################################

class SaxiImageLoggerTensorboard(Callback):
    # Periodically log visualizations of input images during training based on the value of log_steps
    def __init__(self, num_images=12, log_steps=10):
        self.log_steps = log_steps
        self.num_images = num_images
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):        
        # This function is called at the end of each training batch
        if batch_idx % self.log_steps == 0:
                
                V, F, CN, Y = batch

                batch_size = V.shape[0]
                num_images = min(batch_size, self.num_images)

                V = V.to(pl_module.device, non_blocking=True)
                F = F.to(pl_module.device, non_blocking=True)                
                CN = CN.to(pl_module.device, non_blocking=True).to(torch.float32)

                with torch.no_grad():
                    # Render the input surface mesh to an image
                    X, PF = pl_module.render(V[0:1], F[0:1], CN[0:1])
                    
                    grid_X = torchvision.utils.make_grid(X[0, 0:num_images, 0:3, :, :])#Grab the first image, RGB channels only, X, Y. The time dimension is on dim=1
                    trainer.logger.experiment.add_image('X_normals', grid_X, pl_module.global_step)

                    grid_X = torchvision.utils.make_grid(X[0, 0:num_images, 3:, :, :])#Grab the depth map. The time dimension is on dim=1
                    trainer.logger.experiment.add_image('X_depth', grid_X, pl_module.global_step)


class SaxiImageLoggerTensorboardIco_fs(Callback):
    def __init__(self,num_features = 3 , num_images=12, log_steps=10,mean=0,std=0.015):
        self.num_features = num_features
        self.log_steps = log_steps
        self.num_images = num_images
        self.mean = mean
        self.std = std

    def grid_images(self,images):
        grid_images = torchvision.utils.make_grid(images[0, 0:self.num_images, 0:self.num_features+1, :, :])
        numpy_grid_images = grid_images.cpu().numpy()
        return numpy_grid_images

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):        

        if batch_idx % self.log_steps == 0:

            VL, FL, VFL, FFL, VR, FR, VFR, FFR, Y = batch

            VL = VL.to(pl_module.device,non_blocking=True)
            FL = FL.to(pl_module.device,non_blocking=True)
            VFL = VFL.to(pl_module.device,non_blocking=True)
            FFL = FFL.to(pl_module.device,non_blocking=True)
            VR = VR.to(pl_module.device,non_blocking=True)
            FR = FR.to(pl_module.device,non_blocking=True)
            VFR = VFR.to(pl_module.device,non_blocking=True)
            FFR = FFR.to(pl_module.device,non_blocking=True)

            with torch.no_grad():

                images, PF = pl_module.render(VL, FL, VFL, FFL) 
                ###Add because we only have 2 features      
                t_zeros = torch.ones(images[:,:,:1].shape).to(pl_module.device,non_blocking=True)*(images[:,:,:1] > 0.0)

                images = torch.cat([images,t_zeros],dim=2)     
                numpy_grid_images = self.grid_images(images)
                trainer.logger.experiment.add_image('Image features', numpy_grid_images, pl_module.global_step)

                images_noiseM = pl_module.noise(images)
                numpy_grid_images_noiseM = self.grid_images(images_noiseM)
                trainer.logger.experiment.add_image('Image + noise M ', numpy_grid_images_noiseM, pl_module.global_step)    


class SaxiImageLoggerTensorboardIco(Callback):
    def __init__(self,num_features = 3 , num_images=12, log_steps=10,mean=0,std=0.015):
        self.num_features = num_features
        self.log_steps = log_steps
        self.num_images = num_images
        self.mean = mean
        self.std = std

    def grid_images(self,images):
        grid_images = torchvision.utils.make_grid(images[0, 0:self.num_images, 0:self.num_features+1, :, :])
        numpy_grid_images = grid_images.cpu().numpy()
        return numpy_grid_images

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):        

        if batch_idx % self.log_steps == 0:

            VL, FL, VFL, FFL, VR, FR, VFR, FFR,demographics, Y = batch

            VL = VL.to(pl_module.device,non_blocking=True)
            FL = FL.to(pl_module.device,non_blocking=True)
            VFL = VFL.to(pl_module.device,non_blocking=True)
            FFL = FFL.to(pl_module.device,non_blocking=True)
            VR = VR.to(pl_module.device,non_blocking=True)
            FR = FR.to(pl_module.device,non_blocking=True)
            VFR = VFR.to(pl_module.device,non_blocking=True)
            FFR = FFR.to(pl_module.device,non_blocking=True)

            with torch.no_grad():

                images, PF = pl_module.render(VL, FL, VFL, FFL) 
                ###Add because we only have 2 features      
                t_zeros = torch.ones(images[:,:,:1].shape).to(pl_module.device,non_blocking=True)*(images[:,:,:1] > 0.0)

                images = torch.cat([images,t_zeros],dim=2)     
                numpy_grid_images = grid_images(images)
                trainer.logger.experiment.add_image('Image features', numpy_grid_images, pl_module.global_step)

                images_noiseM = pl_module.noise(images)
                numpy_grid_images_noiseM = grid_images(images_noiseM)
                trainer.logger.experiment.add_image('Image + noise M ', numpy_grid_images_noiseM, pl_module.global_step)    
    

class SaxiImageLoggerTensorboardSegmentation(Callback):
    # Another custom callback used especially for the Segmentation part for logging and visualizing images during training within a PyTorch Lightning-based deep learning workflow
    def __init__(self, num_images=12, log_steps=10):
        self.log_steps = log_steps
        self.num_images = num_images
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):        

        if batch_idx % self.log_steps == 0:

                V, F, YF, CN = batch
                batch_size = V.shape[0]
                num_images = min(batch_size, self.num_images)

                V = V.to(pl_module.device, non_blocking=True)
                F = F.to(pl_module.device, non_blocking=True)
                YF = YF.to(pl_module.device, non_blocking=True)
                CN = CN.to(pl_module.device, non_blocking=True).to(torch.float32)

                with torch.no_grad():
                    # Render the input surface mesh to an image
                    x, X, PF = pl_module((V[0:1], F[0:1], CN[0:1]))

                    y = torch.take(YF, PF).to(torch.int64)*(PF >= 0) # YF=input, pix_to_face=index. shape of y = shape of pix_to_face
                    x = torch.argmax(x, dim=2, keepdim=True)
                    
                    grid_X = torchvision.utils.make_grid(X[0, 0:num_images, 0:3, :, :])#Grab the first image, RGB channels only, X, Y. The time dimension is on dim=1
                    trainer.logger.experiment.add_image('X_normals', grid_X, pl_module.global_step)

                    grid_X = torchvision.utils.make_grid(X[0, 0:num_images, 3:, :, :])#Grab the depth map. The time dimension is on dim=1
                    trainer.logger.experiment.add_image('X_depth', grid_X, pl_module.global_step)
                    
                    grid_x = torchvision.utils.make_grid(x[0, 0:num_images, 0:1, :, :]/pl_module.out_channels)# The time dimension is on dim 1 grab only the first one
                    trainer.logger.experiment.add_image('x', grid_x, pl_module.global_step)

                    grid_y = torchvision.utils.make_grid(y[0, 0:num_images, :, :, :]/pl_module.out_channels)# The time dimension here is swapped after the permute and is on dim=2. It will grab the first image
                    grid_y = grid_y.cpu().numpy()
                    trainer.logger.experiment.add_image('Y', grid_y, pl_module.global_step)


#####################################################################################################################################################################################
#                                                                                                                                                                                   #
#                                                                                         Neptune                                                                                   #
#                                                                                                                                                                                   #
#####################################################################################################################################################################################


class SaxiImageLoggerNeptune(Callback):
    # This callback logs images for visualization during training, with the ability to log images to the Neptune logging system for easy monitoring and analysis
    def __init__(self, num_images=12, log_steps=10):
        self.log_steps = log_steps
        self.num_images = num_images

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx): 
        # This function is called at the end of each training batch
        if batch_idx % self.log_steps == 0:

            V, F, CN, Y = batch

            batch_size = V.shape[0]
            num_images = min(batch_size, self.num_images)

            V = V.to(pl_module.device, non_blocking=True)
            F = F.to(pl_module.device, non_blocking=True)                
            CN = CN.to(pl_module.device, non_blocking=True).to(torch.float32)

            with torch.no_grad():
                # Render the input surface mesh to an image
                X, PF = pl_module.render(V[0:1], F[0:1], CN[0:1])
                
                grid_X = torchvision.utils.make_grid(X[0, 0:num_images, 0:3, :, :])#Grab the first image, RGB channels only, X, Y. The time dimension is on dim=1
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_X.permute(1, 2, 0).cpu().numpy())
                trainer.logger.experiment["images/x"].upload(fig)
                plt.close()
                
                grid_X = torchvision.utils.make_grid(X[0, 0:num_images, 3:, :, :])#Grab the depth map. The time dimension is
                fig = plt.figure(figsize=(7, 9))
                ax = plt.imshow(grid_X.permute(1, 2, 0).cpu().numpy())
                trainer.logger.experiment["images/x_depth"].upload(fig)
                plt.close()


class SaxiImageLoggerNeptune_Ico_fs(Callback):
    # This callback logs images for visualization during training, with the ability to log images to the Neptune logging system for easy monitoring and analysis
    def __init__(self, num_images=12, log_steps=10):
        self.log_steps = log_steps
        self.num_images = num_images

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx): 
        # This function is called at the end of each training batch
        if batch_idx % self.log_steps == 0:

            VL, FL, VFL, FFL, VR, FR, VFR, FFR, Y = batch
            num_images = min(VL.shape[1], self.num_images)

            VL = VL.to(pl_module.device,non_blocking=True)
            FL = FL.to(pl_module.device,non_blocking=True)
            VFL = VFL.to(pl_module.device,non_blocking=True).to(torch.float32)
            FFL = FFL.to(pl_module.device,non_blocking=True)
            VR = VR.to(pl_module.device,non_blocking=True)
            FR = FR.to(pl_module.device,non_blocking=True)
            VFR = VFR.to(pl_module.device,non_blocking=True).to(torch.float32)
            FFR = FFR.to(pl_module.device,non_blocking=True)

            with torch.no_grad():
                # Render the input surface mesh to an image
                XL, PFL = pl_module.render(VL, FL, VFL, FFL)
                grid_XL = torchvision.utils.make_grid(XL[0, 0:num_images, 0:3, :, :], nrow=3, padding=0)#Grab the first image, RGB channels only, X, Y. The time dimension is on dim=1
                fig = plt.figure(figsize=(7, 9))
                grid_XL = grid_XL.permute(1, 2, 0)
                ax = plt.imshow(grid_XL.detach().cpu().numpy())
                trainer.logger.experiment["images/x"].upload(fig)
                plt.close()


class SaxiAELoggerNeptune(Callback):
    # This callback logs images for visualization during training, with the ability to log images to the Neptune logging system for easy monitoring and analysis
    def __init__(self, num_surf=1, log_steps=10):
        self.log_steps = log_steps
        self.num_surf = num_surf
        self.num_samples = 1000

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx): 
        # This function is called at the end of each training batch
        if batch_idx % self.log_steps == 0:

            with torch.no_grad():
                V, F = batch

                X_mesh = pl_module.create_mesh(V, F)
                X, X_N = pl_module.sample_points_from_meshes(X_mesh, pl_module.hparams.sample_levels[0], return_normals=True)
                X = torch.cat([X, X_N], dim=-1)

                X_hat, z = pl_module(X)

                X_samples = pl_module.sample_points_from_meshes(X_mesh, self.num_samples)
                X_samples_hat, _ = pl_module.encoder.sample_points(X_hat[0:1], self.num_samples)
                X_samples_orig, _ = pl_module.encoder.sample_points(X_mesh.verts_list()[0].unsqueeze(0), self.num_samples)
                
                fig = self.plot_pointclouds(X_samples_orig[0].cpu().numpy(), X_samples[0].cpu().numpy(), X_samples_hat[0].detach().cpu().numpy())
                trainer.logger.experiment["images/surf"].upload(fig)

    
    def plot_pointclouds(self, X, X_samples, X_hat):
    

        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}], [{'type': 'scatter3d'}, {}]]
        )

        # First scatter plot
        fig.add_trace(
            go.Scatter3d(x=X_samples[:,0], y=X_samples[:,1], z=X_samples[:,2], mode='markers', marker=dict(
                size=2,
                color=X_samples[:,2],                # set color to an array/list of desired values
                colorscale='Viridis',   # choose a colorscale
                opacity=0.8
            )),
            row=1, col=1
        )

        # Second scatter plot
        fig.add_trace(
            go.Scatter3d(x=X_hat[:,0], y=X_hat[:,1], z=X_hat[:,2], mode='markers', marker=dict(
                size=2,
                color=X_hat[:,2],                # set color to an array/list of desired values
                colorscale='Viridis',   # choose a colorscale
                opacity=0.8
            )),
            row=1, col=2
        )

        # Second scatter plot
        fig.add_trace(
            go.Scatter3d(x=X[:,0], y=X[:,1], z=X[:,2], mode='markers', marker=dict(
                size=2,
                color=X[:,2],                # set color to an array/list of desired values
                colorscale='Viridis',   # choose a colorscale
                opacity=0.8
            )),
            row=2, col=1
        )

        # Update the layout if necessary
        fig.update_layout(height=900, width=1200, title_text="Side-by-Side 3D Scatter Plots")

        return fig