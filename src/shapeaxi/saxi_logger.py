from lightning.pytorch.callbacks import Callback
import torchvision
import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np

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
                XL = XL.cpu().numpy()
                fig = self.create_figure(XL)
                # fig = self.create_figure(XL[0, 0:num_images, 0, :, :], XL[0, 0:num_images, 1, :, :], XL[0, 0:num_images, 2, :, :], XL[0, 0:num_images, 3, :, :])
                trainer.logger.experiment["images/features"].upload(fig)
          

    def create_figure(self, XL):
        nb_features = XL.shape[2]
        image_data_list = []

        # Iterate through features and gather image data
        for feature in range(nb_features):
            image_data = []
            for i in range(self.num_images):
                image_slice = XL[0, i, feature, :, :]
                image_data.append(image_slice)
            image_data_list.append(image_data)

        rows = (nb_features + 1) // 2
        cols = 2 if nb_features > 1 else 1

        subplot_titles = []
        subplot_titles.extend([f'Feature {i}' for i in range(nb_features + 1)])
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)

        # Add initial frames for each feature with shared coloraxis
        for idx, image_data in enumerate(image_data_list):
            row = (idx // cols) + 1
            col = (idx % cols) + 1
            fig.add_trace(go.Heatmap(z=image_data[0], coloraxis="coloraxis"), row=row, col=col)

        # Create frames for the animation
        frames = []
        for k in range(self.num_images):
            frame_data = []
            for idx, image_data in enumerate(image_data_list):
                frame_data.append(go.Heatmap(z=image_data[k], coloraxis="coloraxis"))
            frame = go.Frame(data=frame_data, name=str(k))
            frames.append(frame)

        # Add frames to the figure
        fig.frames = frames

        # Calculate the aspect ratio
        height, width = image_data_list[0][0].shape[:2]
        aspect_ratio = height / width

        # Determine global min and max values for consistent color scale
        vmin = min([min(data.min() for data in feature_data) for feature_data in image_data_list])
        vmax = max([max(data.max() for data in feature_data) for feature_data in image_data_list])

        # Update layout with animation settings and fixed aspect ratio
        fig.update_layout(
            autosize=False,
            width=1200,  # Adjust width as needed
            height=1200,  # Adjust height according to aspect ratio
            coloraxis={"colorscale": "jet",
                        "cmin": vmin,  # Set global min value for color scale
                        "cmax": vmax}, # Set global max value for color scale
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 500, "redraw": True},
                                        "fromcurrent": True, "mode": "immediate"}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate"}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }],
            sliders=[{
                "steps": [
                    {
                        "args": [[str(k)], {"frame": {"duration": 300, "redraw": True},
                                            "mode": "immediate"}],
                        "label": str(k),
                        "method": "animate"
                    } for k in range(image_data_list[0][0].shape[0])
                ],
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 20},
                    "prefix": "Frame:",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 300, "easing": "cubic-in-out"}
            }]
        )
        return fig





class SaxiImageLoggerNeptune_Ico_one_feature(Callback):
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
                XL = XL.cpu().numpy()
                fig = self.create_figure(XL[0, 0:num_images, 0, :, :])
                trainer.logger.experiment["images/features"].upload(fig)
          

    def create_figure(self, image_data1):
        fig = make_subplots(rows=1, cols=1, subplot_titles=('Thickness'))

        # Add initial frames for both images with shared coloraxis
        fig.add_trace(go.Heatmap(z=image_data1[0], coloraxis="coloraxis"), row=1, col=1)

        # Create frames for the animation
        frames = []
        for k in range(image_data1.shape[0]):
            frame = go.Frame(data=[
                go.Heatmap(z=image_data1[k], coloraxis="coloraxis"),
            ], name=str(k))
            frames.append(frame)

        # Add frames to the figure
        fig.frames = frames

        # Calculate the aspect ratio
        height, width = image_data1[0].shape[:2]
        aspect_ratio = height / width

        # Determine global min and max values for consistent color scale
        vmin = image_data1.min()
        vmax = image_data1.max()

        # Update layout with animation settings and fixed aspect ratio
        fig.update_layout(
            autosize=False,
            width=1200,  # Adjust width as needed
            height=1200,  # Adjust height according to aspect ratio
            coloraxis={"colorscale": "jet",
                        "cmin": vmin,  # Set global min value for color scale
                        "cmax": vmax}, # Set global max value for color scale
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 500, "redraw": True},
                                        "fromcurrent": True, "mode": "immediate"}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate"}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }],
            sliders=[{
                "steps": [
                    {
                        "args": [[str(k)], {"frame": {"duration": 300, "redraw": True},
                                            "mode": "immediate"}],
                        "label": str(k),
                        "method": "animate"
                    } for k in range(image_data1.shape[0])
                ],
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 20},
                    "prefix": "Frame:",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 300, "easing": "cubic-in-out"}
            }]
        )
        return fig


class SaxiAELoggerNeptune(Callback):
    # This callback logs images for visualization during training, with the ability to log images to the Neptune logging system for easy monitoring and analysis
    def __init__(self, num_surf=1, log_steps=10):
        self.log_steps = log_steps
        self.num_surf = num_surf
        self.num_samples = 4000

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx): 
        # This function is called at the end of each training batch
        if batch_idx % self.log_steps == 0:

            with torch.no_grad():
                V, F = batch

                X_mesh = pl_module.create_mesh(V, F)
                # X, X_N = pl_module.sample_points_from_meshes(X_mesh, pl_module.hparams.sample_levels[0], return_normals=True)
                # X = torch.cat([X, X_N], dim=-1)

                if hasattr(pl_module.hparams, 'start_samples'):
                    X = pl_module.sample_points_from_meshes(X_mesh, pl_module.hparams.start_samples)
                elif hasattr(pl_module.hparams, 'sample_levels'):
                    X = pl_module.sample_points_from_meshes(X_mesh, pl_module.hparams.sample_levels[0])
                elif hasattr(pl_module.hparams, 'num_samples'):
                    X = pl_module.sample_points_from_meshes(X_mesh, pl_module.hparams.num_samples)
                
                X_hat = pl_module(X)                

                X_samples = pl_module.sample_points_from_meshes(X_mesh, self.num_samples)
                X_samples_hat, _ = pl_module.sample_points(X_hat[0:1], self.num_samples)

                if hasattr(pl_module.hparams, 'start_samples'):
                    X_start_samples = pl_module.sample_points_from_meshes(X_mesh, pl_module.hparams.start_samples)
                elif hasattr(pl_module.hparams, 'sample_levels'):
                    X_start_samples = pl_module.sample_points_from_meshes(X_mesh, pl_module.hparams.sample_levels[-1])
                elif hasattr(pl_module.hparams, 'num_samples'):
                    X_start_samples = pl_module.sample_points_from_meshes(X_mesh, pl_module.hparams.num_samples)
                
                
                fig = self.plot_pointclouds(X_start_samples[0].cpu().numpy(), X_samples[0].cpu().numpy(), X_samples_hat[0].detach().cpu().numpy())
                trainer.logger.experiment["images/surf"].upload(fig)

    
    def plot_pointclouds(self, X, X_samples, X_hat):
    

        fig = make_subplots(
            rows=1, cols=3,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]]
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
            go.Scatter3d(x=X[:,0], y=X[:,1], z=X[:,2], mode='markers', marker=dict(
                size=2,
                color=X[:,2],                # set color to an array/list of desired values
                colorscale='Viridis',   # choose a colorscale
                opacity=0.8
            )),
            row=1, col=2
        )

        # Second scatter plot
        fig.add_trace(
            go.Scatter3d(x=X_hat[:,0], y=X_hat[:,1], z=X_hat[:,2], mode='markers', marker=dict(
                size=2,
                color=X_hat[:,2],                # set color to an array/list of desired values
                colorscale='Viridis',   # choose a colorscale
                opacity=0.8
            )),
            row=1, col=3
        )


        # Update the layout if necessary
        fig.update_layout(height=600, width=1200, title_text="Side-by-Side 3D Scatter Plots")

        return fig
    
class SaxiDenoiseUnetLoggerNeptune(Callback):
    # This callback logs images for visualization during training, with the ability to log images to the Neptune logging system for easy monitoring and analysis
    def __init__(self, num_surf=12, log_steps=10, num_steps=10):
        self.log_steps = log_steps
        self.num_surf = num_surf
        self.num_steps = num_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx): 
        # This function is called at the end of each training batch
        if pl_module.global_step % self.log_steps == 0:

            with torch.no_grad():
                V, F = batch

                n = min(V.shape[0], self.num_surf)

                V = V[0:n]
                F = F[0:n]

                X_mesh = pl_module.create_mesh(V, F)

                X = pl_module.sample_points_from_meshes(X_mesh, pl_module.hparams.num_samples)
                
                noise_amount = torch.linspace(0, 1, X.shape[0]).to(pl_module.device)

                X_noised = pl_module.corrupt(X, noise_amount)  

                X_hat = pl_module(X_noised)
                
                fig = self.plot_pointclouds(X.cpu().numpy(), X_noised.cpu().numpy(), X_hat.cpu().numpy())
                trainer.logger.experiment["images/surf"].upload(fig)

    def plot_diffusion(self, X):
        num_surf = len(X)
        specs_r = [{'type': 'scatter3d'} for _ in range(num_surf)]

        fig = make_subplots(
            rows=1, cols=num_surf,
            specs=[specs_r]
        )

        for idx, x in zip(range(num_surf), X):
            # First scatter plot
            fig.add_trace(
                go.Scatter3d(x=x[:,0], y=x[:,1], z=x[:,2], mode='markers', marker=dict(
                    size=2,
                    color=x[:,2],                # set color to an array/list of desired values
                    colorscale='Viridis',   # choose a colorscale
                    opacity=0.8
                )),
                row=1, col=idx+1
            )

        return fig


    def plot_pointclouds(self, X, X_noised, X_hat):

        num_surf = len(X)
        specs_r = [{'type': 'scatter3d'} for _ in range(num_surf)]

        fig = make_subplots(
            rows=3, cols=num_surf,
            specs=[specs_r, specs_r, specs_r]
        )

        for idx, x, x_noised, x_hat in zip(range(num_surf), X, X_noised, X_hat):
            # First scatter plot
            fig.add_trace(
                go.Scatter3d(x=x[:,0], y=x[:,1], z=x[:,2], mode='markers', marker=dict(
                    size=2,
                    color=x[:,2],                # set color to an array/list of desired values
                    colorscale='Viridis',   # choose a colorscale
                    opacity=0.8
                )),
                row=1, col=idx+1
            )

            # Second scatter plot
            fig.add_trace(
                go.Scatter3d(x=x_noised[:,0], y=x_noised[:,1], z=x_noised[:,2], mode='markers', marker=dict(
                    size=2,
                    color=x_noised[:,2],                # set color to an array/list of desired values
                    colorscale='Viridis',   # choose a colorscale
                    opacity=0.8
                )),
                row=2, col=idx+1
            )

            # Third scatter plot
            fig.add_trace(
                go.Scatter3d(x=x_hat[:,0], y=x_hat[:,1], z=x_hat[:,2], mode='markers', marker=dict(
                    size=2,
                    color=x_hat[:,2],                # set color to an array/list of desired values
                    colorscale='Viridis',   # choose a colorscale
                    opacity=0.8
                )),
                row=3, col=idx+1
            )

        # Update the layout if necessary
        fig.update_layout(height=900, width=1600, title_text="Side-by-Side 3D Scatter Plots")

        return fig

class SaxiDDPMLoggerNeptune(Callback):
    # This callback logs images for visualization during training, with the ability to log images to the Neptune logging system for easy monitoring and analysis
    def __init__(self, num_surf=5, log_steps=10, num_steps=5):
        self.log_steps = log_steps
        self.num_surf = num_surf
        self.num_steps = num_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx): 
        # This function is called at the end of each training batch
        if pl_module.global_step % self.log_steps == 0:
            
            pl_module.eval()
            with torch.no_grad():

                if isinstance(batch, tuple) or isinstance(batch, list):
                    V, F = batch        
                    X_mesh = pl_module.create_mesh(V, F)
                    X = pl_module.sample_points_from_meshes(X_mesh, pl_module.hparams.num_samples)
                    if hasattr(pl_module, 'sorter'):
                        X = pl_module.sorter(X)
                elif isinstance(batch, dict):
                    X = batch['pointcloud']
                else:
                    X = batch
                
                fig = self.plot_diffusion(X[0:self.num_surf].cpu().numpy())
                trainer.logger.experiment["images/batch"].upload(fig)
                

                # V, F = batch

                # n = min(V.shape[0], self.num_surf)

                # V = V[0:n]
                # F = F[0:n]

                # X_mesh = pl_module.create_mesh(V, F)

                # X = pl_module.sample_points_from_meshes(X_mesh, pl_module.hparams.num_samples)
                X = torch.randn((1, pl_module.hparams.num_samples, 3)).to(pl_module.device)
                
                if hasattr(pl_module, 'sample'):
                    pc, intermediates = pl_module.sample(intermediate_steps=self.num_steps)
                    
                    fig = self.plot_diffusion(torch.cat(intermediates, dim=0).cpu().numpy())
                    trainer.logger.experiment["images/intermediates"].upload(fig)

                elif hasattr(pl_module, 'inferer'):
                    
                    pl_module.noise_scheduler.set_timesteps(num_inference_steps=pl_module.hparams.num_train_steps)

                    context = None

                    if hasattr(pl_module, 'flow'):
                        context = pl_module.flow.sample(X.shape[0])

                    pc, intermediates = pl_module.inferer.sample(
                        input_noise=X, 
                        diffusion_model=pl_module, 
                        scheduler=pl_module.noise_scheduler, 
                        save_intermediates=True, 
                        intermediate_steps=pl_module.hparams.num_train_steps//self.num_steps, 
                        verbose=False,
                        conditioning=context
                    )

                    fig = self.plot_diffusion(torch.cat(intermediates, dim=0).cpu().numpy())
                    trainer.logger.experiment["images/intermediates"].upload(fig)

                else:

                    X_gen = []
                    X_orig_sample = []

                    num_diff_steps = int(pl_module.hparams.num_train_steps/self.num_steps)

                    for i, t in enumerate(range(pl_module.hparams.num_train_steps)):
                        residual = pl_module(X, t)  

                        scheduler_output = pl_module.noise_scheduler.step(residual, t, X)

                        X = scheduler_output.prev_sample
                        
                        if i % num_diff_steps == 0:
                            X_gen.append(X)

                        if i % num_diff_steps == 0:
                            X_orig_sample.append(scheduler_output.pred_original_sample)
                    
                    fig = self.plot_diffusion(torch.cat(X_gen, dim=0).cpu().numpy())
                    # fig = self.plot_pointclouds(X.cpu().numpy(), X_noised.cpu().numpy(), X_hat.cpu().numpy())
                    trainer.logger.experiment["images/prev_sample"].upload(fig)

                    fig = self.plot_diffusion(torch.cat(X_orig_sample, dim=0).cpu().numpy())
                    trainer.logger.experiment["images/pred_original_sample"].upload(fig)

    def plot_diffusion(self, X):
        num_surf = len(X)
        specs_r = [{'type': 'scatter3d'} for _ in range(num_surf)]

        fig = make_subplots(
            rows=1, cols=num_surf,
            specs=[specs_r]
        )

        for idx, x in zip(range(num_surf), X):
            # First scatter plot
            fig.add_trace(
                go.Scatter3d(x=x[:,0], y=x[:,1], z=x[:,2], mode='markers', marker=dict(
                    size=2,
                    color=x[:,2],                # set color to an array/list of desired values
                    colorscale='Viridis',   # choose a colorscale
                    opacity=0.8
                )),
                row=1, col=idx+1
            )

        return fig


    def plot_pointclouds(self, X, X_noised, X_hat):

        num_surf = len(X)
        specs_r = [{'type': 'scatter3d'} for _ in range(num_surf)]

        fig = make_subplots(
            rows=3, cols=num_surf,
            specs=[specs_r, specs_r, specs_r]
        )

        for idx, x, x_noised, x_hat in zip(range(num_surf), X, X_noised, X_hat):
            # First scatter plot
            fig.add_trace(
                go.Scatter3d(x=x[:,0], y=x[:,1], z=x[:,2], mode='markers', marker=dict(
                    size=2,
                    color=x[:,2],                # set color to an array/list of desired values
                    colorscale='Viridis',   # choose a colorscale
                    opacity=0.8
                )),
                row=1, col=idx+1
            )

            # Second scatter plot
            fig.add_trace(
                go.Scatter3d(x=x_noised[:,0], y=x_noised[:,1], z=x_noised[:,2], mode='markers', marker=dict(
                    size=2,
                    color=x_noised[:,2],                # set color to an array/list of desired values
                    colorscale='Viridis',   # choose a colorscale
                    opacity=0.8
                )),
                row=2, col=idx+1
            )

            # Third scatter plot
            fig.add_trace(
                go.Scatter3d(x=x_hat[:,0], y=x_hat[:,1], z=x_hat[:,2], mode='markers', marker=dict(
                    size=2,
                    color=x_hat[:,2],                # set color to an array/list of desired values
                    colorscale='Viridis',   # choose a colorscale
                    opacity=0.8
                )),
                row=3, col=idx+1
            )

        # Update the layout if necessary
        fig.update_layout(height=900, width=1600, title_text="Side-by-Side 3D Scatter Plots")

        return fig
    

class SaxiClassLoggerNeptune(Callback):
    # This callback logs images for visualization during training, with the ability to log images to the Neptune logging system for easy monitoring and analysis
    def __init__(self, num_surf=1, log_steps=10):
        self.log_steps = log_steps
        self.num_surf = num_surf
        self.num_samples = 1000

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx): 
        # This function is called at the end of each training batch
        if batch_idx % self.log_steps == 0:

            with torch.no_grad():
                if len(batch) == 3:
                    V, F, Y = batch
                else:
                    V, F, CN, Y = batch

                X_mesh = pl_module.create_mesh(V, F)                

                X_samples = pl_module.sample_points_from_meshes(X_mesh, self.num_samples)
                
                fig = self.plot_pointclouds(X_samples[0].cpu().numpy())
                trainer.logger.experiment["images/surf"].upload(fig)

    
    def plot_pointclouds(self, X):
    

        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{'type': 'scatter3d'}]]
        )

        # First scatter plot
        fig.add_trace(
            go.Scatter3d(x=X[:,0], y=X[:,1], z=X[:,2], mode='markers', marker=dict(
                size=2,
                color=X[:,2],                # set color to an array/list of desired values
                colorscale='Viridis',   # choose a colorscale
                opacity=0.8
            )),
            row=1, col=1
        )

        # Update the layout if necessary
        fig.update_layout(height=600, width=600, title_text="Side-by-Side 3D Scatter Plots")

        return fig
    


class SaxiClassMHAFBLoggerNeptune(Callback):
    # This callback logs images for visualization during training, with the ability to log images to the Neptune logging system for easy monitoring and analysis
    def __init__(self, num_surf=1, log_steps=10):
        self.log_steps = log_steps
        self.num_surf = num_surf
        self.num_samples = 1000
        self.num_images = 12

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx): 
        # This function is called at the end of each training batch
        if batch_idx % self.log_steps == 0:

            with torch.no_grad():
                V, F, CN, Y = batch

                X_mesh = pl_module.create_mesh(V, F, CN)

                X_samples = pl_module.sample_points_from_meshes(X_mesh, self.num_samples)
                
                fig = self.plot_pointclouds(X_samples[0].cpu().numpy())
                trainer.logger.experiment["images/surf"].upload(fig)

                X_fb, X_PF = pl_module.render(X_mesh)
                
                X_img = X_fb[0,:,0:3].permute(0,2,3,1).squeeze().cpu().numpy()
                X_img = (X_img - X_img.min()) / (X_img.max() - X_img.min())*255
                X_img_zbuf = X_fb[0,:,3:4].permute(0,2,3,1).squeeze().cpu().numpy()

                fig = self.create_figure(X_img[0:self.num_images], X_img_zbuf[0:self.num_images])
                trainer.logger.experiment["images/fb"].upload(fig)

    
    def plot_pointclouds(self, X):
    

        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{'type': 'scatter3d'}]]
        )

        # First scatter plot
        fig.add_trace(
            go.Scatter3d(x=X[:,0], y=X[:,1], z=X[:,2], mode='markers', marker=dict(
                size=2,
                color=X[:,2],                # set color to an array/list of desired values
                colorscale='Viridis',   # choose a colorscale
                opacity=0.8
            )),
            row=1, col=1
        )

        # Update the layout if necessary
        fig.update_layout(height=600, width=600, title_text="Side-by-Side 3D Scatter Plots")

        return fig
    
    def create_figure(self, image_data1, image_data2):
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Image 1', 'Image 2'))

        # Add initial frames for both images with shared coloraxis
        fig.add_trace(go.Image(z=image_data1[0]), row=1, col=1)
        fig.add_trace(go.Heatmap(z=image_data2[0], coloraxis="coloraxis"), row=1, col=2)

        # Create frames for the animation
        frames = []
        for k in range(image_data1.shape[0]):
            frame = go.Frame(data=[
                go.Image(z=image_data1[k]),
                go.Heatmap(z=image_data2[k], coloraxis="coloraxis")
            ], name=str(k))
            frames.append(frame)

        # Add frames to the figure
        fig.frames = frames

        # Calculate the aspect ratio
        height, width = image_data1[0].shape[:2]
        aspect_ratio = height / width

        # Determine global min and max values for consistent color scale
        # vmin = min(image_data1.min(), image_data2.min())
        # vmax = max(image_data1.max(), image_data2.max())
        vmin = image_data2.min()
        vmax = image_data2.max()

        # Update layout with animation settings and fixed aspect ratio
        fig.update_layout(
            autosize=False,
            width=1200,  # Adjust width as needed
            height=600,  # Adjust height according to aspect ratio
            coloraxis={"colorscale": "jet",
                    "cmin": vmin,  # Set global min value for color scale
                        "cmax": vmax},   # Set global max value for color scale},  # Set colorscale for the shared coloraxis
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 500, "redraw": True},
                                        "fromcurrent": True, "mode": "immediate"}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate"}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }],
            sliders=[{
                "steps": [
                    {
                        "args": [[str(k)], {"frame": {"duration": 300, "redraw": True},
                                            "mode": "immediate"}],
                        "label": str(k),
                        "method": "animate"
                    } for k in range(image_data1.shape[0])
                ],
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 20},
                    "prefix": "Frame:",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 300, "easing": "cubic-in-out"}
            }]
        )
        return fig 
    


class SaxiClassRingLoggerNeptune(Callback):
    # This callback logs images for visualization during training, with the ability to log images to the Neptune logging system for easy monitoring and analysis
    def __init__(self, num_surf=1, log_steps=10):
        self.log_steps = log_steps
        self.num_surf = num_surf
        self.num_samples = 1000
        self.num_images = 12

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx): 
        # This function is called at the end of each training batch
        if batch_idx % self.log_steps == 0:

            with torch.no_grad():
                V, F, CN, Y = batch

                X_mesh = pl_module.create_mesh(V, F, CN)

                X_samples = pl_module.sample_points_from_meshes(X_mesh, self.num_samples)
                
                fig = self.plot_pointclouds(X_samples[0].cpu().numpy())
                trainer.logger.experiment["images/surf"].upload(fig)

                X_fb, X_PF = pl_module.render(V, F, CN)
                
                X_img = X_fb[0,:,0:3].permute(0,2,3,1).squeeze().cpu().numpy()
                X_img = (X_img - X_img.min()) / (X_img.max() - X_img.min())*255
                X_img_zbuf = X_fb[0,:,3:4].permute(0,2,3,1).squeeze().cpu().numpy()

                fig = self.create_figure(X_img[0:self.num_images], X_img_zbuf[0:self.num_images])
                trainer.logger.experiment["images/fb"].upload(fig)

    
    def plot_pointclouds(self, X):
    

        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{'type': 'scatter3d'}]]
        )

        # First scatter plot
        fig.add_trace(
            go.Scatter3d(x=X[:,0], y=X[:,1], z=X[:,2], mode='markers', marker=dict(
                size=2,
                color=X[:,2],                # set color to an array/list of desired values
                colorscale='Viridis',   # choose a colorscale
                opacity=0.8
            )),
            row=1, col=1
        )

        # Update the layout if necessary
        fig.update_layout(height=600, width=600, title_text="Side-by-Side 3D Scatter Plots")

        return fig
    
    def create_figure(self, image_data1, image_data2):
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Image 1', 'Image 2'))

        # Add initial frames for both images with shared coloraxis
        fig.add_trace(go.Image(z=image_data1[0]), row=1, col=1)
        fig.add_trace(go.Heatmap(z=image_data2[0], coloraxis="coloraxis"), row=1, col=2)

        # Create frames for the animation
        frames = []
        for k in range(image_data1.shape[0]):
            frame = go.Frame(data=[
                go.Image(z=np.flip(image_data1[k], axis=0)),
                go.Heatmap(z=image_data2[k], coloraxis="coloraxis")
            ], name=str(k))
            frames.append(frame)

        # Add frames to the figure
        fig.frames = frames

        # Calculate the aspect ratio
        height, width = image_data1[0].shape[:2]
        aspect_ratio = height / width

        # Determine global min and max values for consistent color scale
        # vmin = min(image_data1.min(), image_data2.min())
        # vmax = max(image_data1.max(), image_data2.max())
        vmin = image_data2.min()
        vmax = image_data2.max()

        # Update layout with animation settings and fixed aspect ratio
        fig.update_layout(
            autosize=False,
            width=1200,  # Adjust width as needed
            height=600,  # Adjust height according to aspect ratio
            coloraxis={"colorscale": "jet",
                    "cmin": vmin,  # Set global min value for color scale
                        "cmax": vmax},   # Set global max value for color scale},  # Set colorscale for the shared coloraxis
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 500, "redraw": True},
                                        "fromcurrent": True, "mode": "immediate"}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate"}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }],
            sliders=[{
                "steps": [
                    {
                        "args": [[str(k)], {"frame": {"duration": 300, "redraw": True},
                                            "mode": "immediate"}],
                        "label": str(k),
                        "method": "animate"
                    } for k in range(image_data1.shape[0])
                ],
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 20},
                    "prefix": "Frame:",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 300, "easing": "cubic-in-out"}
            }]
        )
        return fig


class SaxiNeRFLoggerNeptune(Callback):
    # This callback logs images for visualization during training, with the ability to log images to the Neptune logging system for easy monitoring and analysis
    def __init__(self, log_steps = 10, *args, **kwargs):
        self.log_steps = log_steps

    @staticmethod
    def add_logger_specific_args(parent_parser):
        logger_group = parent_parser.add_argument_group(title='NeRF Logger')
        logger_group.add_argument('--log_steps', type=int, help='Log steps for the callback (neptune)', default=50)
        return parent_parser

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx): 
        # This function is called at the end of each training batch
        if pl_module.global_step % self.log_steps == 0:

            pl_module.eval()
            with torch.no_grad():

                images, poses = batch

                
                ray_origins, ray_directions = pl_module.generate_rays(poses)
                # ray_origins, ray_directions = pl_module.get_rays(poses[0])


                # rgb = []
                # depth = []
                # acc = []

                # for r_o, r_d in zip(torch.chunk(ray_origins, chunks=32, dim=1), torch.chunk(ray_directions, chunks=32, dim=1)):
                #     r, d, a = pl_module.render_rays(r_o, r_d)            
                #     rgb.append(r)
                #     depth.append(d)
                #     acc.append(a)

                # rgb = torch.cat(rgb, dim=1)
                # depth = torch.cat(depth, dim=1)
                # acc = torch.cat(acc, dim=1)

                rgb, depth, acc = pl_module.render_rays(ray_origins, ray_directions)

                rgb = rgb.reshape(1, pl_module.hparams.height, pl_module.hparams.width, 3).cpu().numpy()*255
                depth = depth.reshape(-1, pl_module.hparams.height, pl_module.hparams.width).cpu().numpy()
                acc = acc.reshape(-1, pl_module.hparams.height, pl_module.hparams.width).cpu().numpy()

                # rgb = rgb.reshape(-1, 3, pl_module.hparams.height, pl_module.hparams.width).permute(0, 2, 3, 1).cpu().numpy()*255
                # depth = depth.reshape(-1, pl_module.hparams.height, pl_module.hparams.width).cpu().numpy()
                # acc = acc.reshape(-1, pl_module.hparams.height, pl_module.hparams.width).cpu().numpy()
                

                fig = self.create_figure(images.cpu().numpy()*255, rgb, depth, acc)
                trainer.logger.experiment["images/NeRF"].upload(fig)

    def create_figure(self, images, rgb, depth, acc):
        fig = make_subplots(rows=2, cols=3, subplot_titles=('Images', '', '', 'RGB', 'Depth', 'Acc'))

        # Add initial frames for both images with shared coloraxis
        fig.add_trace(go.Image(z=images[0]), row=1, col=1)
        fig.add_trace(go.Image(z=rgb[0]), row=2, col=1)
        fig.add_trace(go.Heatmap(z=depth[0], coloraxis="coloraxis"), row=2, col=2)
        fig.add_trace(go.Heatmap(z=acc[0], coloraxis="coloraxis"), row=2, col=3)

        fig.update_layout(
            autosize=False,
            width=1200,  # Adjust width as needed
            height=1200,  # Adjust height according to aspect ratio
            coloraxis={"colorscale": "jet"},   # Set colorscale for the shared coloraxis
        )

        return fig
    
    def create_animation(self, rgb, depth, acc):
        fig = make_subplots(rows=1, cols=3, subplot_titles=('RGB', 'Depth', 'Acc'))

        # Add initial frames for both images with shared coloraxis
        fig.add_trace(go.Image(z=rgb[0]), row=1, col=1)
        fig.add_trace(go.Heatmap(z=depth[0], coloraxis="coloraxis"), row=1, col=2)
        fig.add_trace(go.Heatmap(z=acc[0], coloraxis="coloraxis"), row=1, col=3)

        # Create frames for the animation
        frames = []
        for k in range(rgb.shape[0]):
            frame = go.Frame(data=[
                go.Image(z=rgb[k]),
                go.Heatmap(z=depth[k], coloraxis="coloraxis"),
                go.Heatmap(z=acc[k], coloraxis="coloraxis")
            ], name=str(k))
            frames.append(frame)
        
        # Add frames to the figure
        fig.frames = frames

        # Update layout with animation settings and fixed aspect ratio
        fig.update_layout(
            autosize=False,
            width=1200,  # Adjust width as needed
            height=600,  # Adjust height according to aspect ratio
            coloraxis={"colorscale": "jet"},   # Set colorscale for the shared coloraxis
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 500, "redraw": True},
                                        "fromcurrent": True, "mode": "immediate"}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate"}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }],
            sliders=[{
                "steps": [
                    {
                        "args": [[str(k)], {"frame": {"duration": 300, "redraw": True},
                                            "mode": "immediate"}],
                        "label": str(k),
                        "method": "animate"
                    } for k in range(rgb.shape[0])
                ],
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 20},
                    "prefix": "Frame:",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 300, "easing": "cubic-in-out"}
            }]
        )
        return fig