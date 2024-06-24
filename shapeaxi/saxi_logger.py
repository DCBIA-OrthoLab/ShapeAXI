from pytorch_lightning.callbacks import Callback
import torchvision
import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


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

