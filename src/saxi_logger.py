from pytorch_lightning.callbacks import Callback
import torchvision
import torch

# This file saxi_train uses SaxiImageLogger which is a callback intended for logging images to the PyTorch Lightning logger during training

class SaxiImageLogger(Callback):
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

# The difference is the backend used for image logging and visualization.
# SaxiImageLogger logs images directly within the PyTorch Lightning logging system.
# SaxiImageLoggerNeptune sends images to the Neptune platform for more advanced experiment tracking and visualization. 



######################################################################### SEGMENTATION PART #########################################################################################


class TeethNetImageLogger(Callback):
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



######################################################################### ICOCONV PART #########################################################################################


class ImageLogger(Callback):
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
                numpy_grid_images = self.grid_images(images)
                trainer.logger.experiment.add_image('Image features', numpy_grid_images, pl_module.global_step)

                images_noiseM = pl_module.noise(images)
                numpy_grid_images_noiseM = self.grid_images(images_noiseM)
                trainer.logger.experiment.add_image('Image + noise M ', numpy_grid_images_noiseM, pl_module.global_step)    
    

