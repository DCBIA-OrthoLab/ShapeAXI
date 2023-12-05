import argparse
import subprocess
import math
import os
import sys
import pandas as pd
import numpy as np 
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
import monai
import nibabel as nib
from sklearn.utils import class_weight
from pl_bolts.models.self_supervised import Moco_v2


from .saxi_dataset import SaxiDataModule, SaxiDataset, BrainIBISDataModule
from .saxi_transforms import TrainTransform, EvalTransform, RandomRemoveTeethTransform, UnitSurfTransform, RandomRotationTransform,ApplyRotationTransform, GaussianNoisePointTransform, NormalizePointTransform, CenterTransform
from .saxi_nets import MonaiUNet, SaxiIcoClassification
from .saxi_logger import SaxiImageLogger, TeethNetImageLogger, ImageLogger

# Training machine learning models

def SaxiClassification_SaxiRegression_train(args, checkpoint_callback, mount_point, df_train, df_val, df_test):
    #Initialize the dataset and corresponding data loader for training and validation
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=args.patience, verbose=True, mode="min")
    callbacks = [early_stop_callback, checkpoint_callback]
    saxi_args = vars(args)

    saxi_data = SaxiDataModule(df_train, df_val, df_test,
                        mount_point = mount_point,
                        batch_size = args.batch_size,
                        num_workers = args.num_workers,
                        model = args.nn,
                        surf_column = args.surf_column,
                        class_column = args.class_column,
                        train_transform = TrainTransform(scale_factor=args.scale_factor),
                        valid_transform = EvalTransform(scale_factor=args.scale_factor),
                        test_transform = EvalTransform(scale_factor=args.scale_factor))
    
    logger=None
    if args.tb_dir:
        logger = TensorBoardLogger(save_dir=args.tb_dir, name=args.tb_name)
        callbacks.append(SaxiImageLogger())

    elif args.neptune_project:
        logger = NeptuneLogger(
            project=args.neptune_project,
            tags=args.neptune_tags,
            api_key=os.environ['NEPTUNE_API_TOKEN']
        )
        image_logger = SaxiImageLoggerNeptune(num_images=args.num_images)

    if args.nn == "SaxiClassification":
        unique_classes = np.sort(np.unique(df_train[args.class_column]))
        unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=df_train[args.class_column]))    
        class_weights = unique_class_weights
        saxi_args['class_weights'] = class_weights
        saxi_args['out_classes'] = len(class_weights)

    elif args.nn =="SaxiRegression":
        saxi_args['out_features'] = 1

    SAXINETS = getattr(saxi_nets, args.nn)
    model = SAXINETS(**saxi_args)

    trainer = Trainer(
        logger=logger,
        max_epochs=args.epochs,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=callbacks,
        devices=torch.cuda.device_count(), 
        accelerator="gpu", 
        strategy=DDPStrategy(find_unused_parameters=False),
        num_sanity_val_steps=0,
        profiler=args.profiler
    )

    trainer.fit(model, datamodule=saxi_data, ckpt_path=args.model)


def SaxiSegmentation_train(args, checkpoint_callback, mount_point, df_train, df_val, df_test):
     # The dataset and corresponding data loader are initialized for training and validation
    class_weights = None

    saxi_data = SaxiDataModule(df_train, df_val, df_test,
                        mount_point = mount_point,
                        batch_size = args.batch_size,
                        num_workers = args.num_workers,
                        model = args.nn,
                        surf_column = 'surf',
                        surf_property = 'UniversalID',
                        #train_transform = RandomRemoveTeethTransform(surf_property="UniversalID", random_rotation=True),
                        train_transform = UnitSurfTransform(),
                        valid_transform = UnitSurfTransform(),
                        test_transform = UnitSurfTransform())

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=args.patience, verbose=True, mode="min")
    logger=None
    
    if args.tb_dir:
        logger = TensorBoardLogger(save_dir=args.tb_dir, name=args.tb_name)    

    image_logger = TeethNetImageLogger()
    model = MonaiUNet(args, out_channels = 34, class_weights=class_weights, image_size=320, train_sphere_samples=args.train_sphere_samples)

    trainer = Trainer(logger=logger,max_epochs=args.epochs,log_every_n_steps=args.log_every_n_steps,
        callbacks=[early_stop_callback, checkpoint_callback, image_logger],
        devices=torch.cuda.device_count(), 
        accelerator="gpu", 
        strategy=DDPStrategy(find_unused_parameters=False, process_group_backend="nccl"),
        num_sanity_val_steps=0,
        profiler=args.profiler
    )

    trainer.fit(model, datamodule=saxi_data, ckpt_path=args.model)
    trainer.test(datamodule=saxi_data)


def SaxiIcoClassification_train(args):
    list_path_ico = [args.path_ico_left,args.path_ico_right]

    #Demographics
    list_demographic = ['Gender','MRI_Age','AmygdalaLeft','HippocampusLeft','LatVentsLeft','ICV','Crbm_totTissLeft','Cblm_totTissLeft','AmygdalaRight','HippocampusRight','LatVentsRight','Crbm_totTissRight','Cblm_totTissRight'] #MLR

    #Transformation
    list_train_transform = [] 
    list_train_transform.append(CenterTransform())
    list_train_transform.append(NormalizePointTransform())
    list_train_transform.append(RandomRotationTransform())        
    list_train_transform.append(GaussianNoisePointTransform(args.mean,args.std)) #Do not use this transformation if your object is not a sphere
    list_train_transform.append(NormalizePointTransform()) #Do not use this transformation if your object is not a sphere
    train_transform = monai.transforms.Compose(list_train_transform)

    list_val_and_test_transform = []    
    list_val_and_test_transform.append(CenterTransform())
    list_val_and_test_transform.append(NormalizePointTransform())
    val_and_test_transform = monai.transforms.Compose(list_val_and_test_transform)

    #Get number of images
    list_nb_verts_ico = [12, 42, 162, 642, 2562, 10242, 40962, 163842]
    nb_images = list_nb_verts_ico[args.ico_lvl-1]
    
    #Creation of Dataset
    brain_data = BrainIBISDataModule(args.batch_size,list_demographic,args.csv_train,args.csv_valid,args.csv_test,list_path_ico,train_transform = train_transform,val_and_test_transform=val_and_test_transform,num_workers=args.num_workers)#MLR
    nbr_features = brain_data.get_features()
    weights = brain_data.get_weigths()
    nbr_demographic = brain_data.get_nbr_demographic()

    if args.ico_lvl == 1:
        radius = 1.76 
    elif args.ico_lvl == 2:
        radius = 1

    saxi_args = vars(args)
    saxi_args['nbr_features'] = nbr_features
    saxi_args['nbr_demographic'] = nbr_demographic
    saxi_args['weights'] = weights
    saxi_args['radius'] = radius

    #Creation of our model
    SAXINETS = getattr(saxi_nets, args.nn)
    model = SAXINETS(**saxi_args)

    #Creation of Checkpoint (if we want to save best models)
    checkpoint_callback_loss = ModelCheckpoint(dirpath='Checkpoint/'+args.name,filename='{epoch}-{val_loss:.2f}',save_top_k=10,monitor='val_loss',)

    #Logger (Useful if we use Tensorboard)
    logger = TensorBoardLogger(save_dir="test_tensorboard", name="my_model")

    #Early Stopping
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=args.min_delta_early_stopping, patience=args.patience_early_stopping, verbose=True, mode="min")

    #Image Logger (Useful if we use Tensorboard)
    image_logger = ImageLogger(num_features = nbr_features,num_images = nb_images,mean = 0,std=args.noise_lvl)

    #Trainer
    trainer = Trainer(log_every_n_steps=10,reload_dataloaders_every_n_epochs=True,logger=logger,max_epochs=args.epochs,callbacks=[early_stop_callback,checkpoint_callback_loss,image_logger],accelerator="gpu") #,accelerator="gpu"
    trainer.fit(model,datamodule=brain_data)
    trainer.test(model, datamodule=brain_data)
    print('Number of features : ',nbr_features)

    return model, brain_data, trainer



def main(args):

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss'
    )

    mount_point = args.mount_point  
    df_train = pd.read_csv(os.path.join(mount_point, args.csv_train))
    df_val = pd.read_csv(os.path.join(mount_point, args.csv_valid))
    df_test = pd.read_csv(os.path.join(mount_point, args.csv_test))

    if args.nn == "SaxiClassification" or args.nn == "SaxiRegression":
        SaxiClassification_SaxiRegression_train(args, checkpoint_callback, mount_point, df_train, df_val, df_test)

    elif args.nn == "SaxiSegmentation":
        SaxiSegmentation_train(args, checkpoint_callback, mount_point, df_train, df_val, df_test)
    
    elif args.nn == "SaxiIcoClassification":
        SaxiIcoClassification_train(args)
    
    else:
        raise ValueError ("Unknown neural network name: {}, choose between SaxiClassification, SaxiRegression, SaxiSegmentation, SaxiIcoClassification".format(args.nn))




def get_argparse():
    #This function defines the arguments that can be passed to the script
    parser = argparse.ArgumentParser(description='Shape Analysis Explainability and Interpretability')

    ##Input
    in_group = parser.add_argument_group('Input')
    in_group.add_argument('--csv_train', help='CSV with column surf', type=str, required=True)    
    in_group.add_argument('--csv_valid', help='CSV with column surf', type=str)
    in_group.add_argument('--csv_test', help='CSV with column surf', type=str, required=True)        
    in_group.add_argument('--surf_column', help='Surface column name', type=str, default="surf")
    in_group.add_argument('--class_column', help='Class column name', type=str, default=None)
    in_group.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    in_group.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    in_group.add_argument('--path_ico_left', type=str, default='./3DObject/sphere_f327680_v163842.vtk', help='Path to ico left (default: ../3DObject/sphere_f327680_v163842.vtk)')
    in_group.add_argument('--path_ico_right', type=str, default='./3DObject/sphere_f327680_v163842.vtk', help='Path to ico right (default: ../3DObject/sphere_f327680_v163842.vtk)')

    ##Model
    model_group = parser.add_argument_group('Input model')
    model_group.add_argument('--model', help='Model to continue training', type=str, default= None)

    ##Hyperparameters
    hyper_group = parser.add_argument_group('Hyperparameters')
    hyper_group.add_argument('--nn', help='Neural network name : SaxiClassification, SaxiRegression, SaxiSegmentation, SaxiIcoClassification', type=str, default='SaxiClassification')
    hyper_group.add_argument('--base_encoder', help='Base encoder for the feature extraction', type=str, default='resnet18')
    hyper_group.add_argument('--base_encoder_params', help='Base encoder parameters that are passed to build the feature extraction', type=str, default='pretrained=False,spatial_dims=2,n_input_channels=4,num_classes=512')
    hyper_group.add_argument('--hidden_dim', help='Hidden dimension for features output. Should match with output of base_encoder. Default value is 512', type=int, default=512)
    hyper_group.add_argument('--radius', help='Radius of icosphere', type=float, default=1.35)    
    hyper_group.add_argument('--subdivision_level', help='Subdivision level for icosahedron', type=int, default=1)
    hyper_group.add_argument('--image_size', help='Image resolution size', type=int, default=256)
    hyper_group.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    hyper_group.add_argument('--epochs', help='Max number of epochs', type=int, default=200)   
    hyper_group.add_argument('--batch_size', help='Batch size', type=int, default=3)    
    hyper_group.add_argument('--train_sphere_samples', help='Number of training sphere samples or views used during training and validation', type=int, default=4)  
    hyper_group.add_argument('--patience', help='Patience for early stopping', type=int, default=30)
    hyper_group.add_argument('--scale_factor', help='Scale factor to rescale the shapes', type=float, default=1.0) 
    hyper_group.add_argument('--noise_lvl', type=float, default=0.01, help='Noise level (default: 0.01)')
    hyper_group.add_argument('--ico_lvl', type=int, default=2, help='Ico level, minimum level is 1 (default: 2)')
    hyper_group.add_argument('--pretrained', type=bool, default=False, help='Pretrained (default: False)')
    hyper_group.add_argument('--dropout_lvl', type=float, default=0.2, help='Dropout level (default: 0.2)')

    ##Gaussian Filter
    gaussian_group = parser.add_argument_group('Gaussian filter')
    gaussian_group.add_argument('--mean', type=float, default=0, help='Mean (default: 0)')
    gaussian_group.add_argument('--std', type=float, default=0.005, help='Standard deviation (default: 0.005)')

    ##Early Stopping
    early_stopping_group = parser.add_argument_group('Early stopping')
    early_stopping_group.add_argument('--min_delta_early_stopping', type=float, default=0.00, help='Minimum delta (default: 0.00)')
    early_stopping_group.add_argument('--patience_early_stopping', type=int, default=100, help='Patience (default: 100)')
    
    ##Name and layer
    name_group = parser.add_argument_group('Name and layer')
    name_group.add_argument('--layer', type=str, default='IcoConv2D', help="Layer, choose between 'Att','IcoConv2D','IcoConv1D','IcoLinear' (default: IcoConv2D)")
    name_group.add_argument('--name', type=str, default='Experiment0', help='Name of your experiment (default: Experiment0)')

    ##Logger
    logger_group = parser.add_argument_group('Logger')
    logger_group.add_argument('--log_every_n_steps', help='Log every n steps', type=int, default=10)    
    logger_group.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default=None)
    logger_group.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default=None)
    logger_group.add_argument('--neptune_project', help='Neptune project', type=str, default=None)
    logger_group.add_argument('--neptune_tags', help='Neptune tags', type=str, default=None)

    ##Output
    out_group = parser.add_argument_group('Output')
    out_group.add_argument('--out', help='Output', type=str, default="./")

    ##Debug
    debug_group = parser.add_argument_group('Debug')
    debug_group.add_argument('--profiler', help='Use a profiler', type=str, default=None)

    return parser


if __name__ == '__main__':
    parser = get_argparse()
    args = parser.parse_args()
    main(args)