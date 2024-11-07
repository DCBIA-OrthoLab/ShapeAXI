import argparse
import torch

from shapeaxi import saxi_dataset 
from shapeaxi import saxi_nets
from shapeaxi.saxi_transforms import *
from shapeaxi import saxi_nets_lightning

from lightning import Trainer

def main(args):

    DATAMODULE = getattr(saxi_dataset, args.data_module)

    args_d = vars(args)

    SAXINETS = getattr(saxi_nets_lightning, args.nn)
    model = SAXINETS(**args_d)
    model = SAXINETS.load_from_checkpoint(args.model, strict=False)
    model.eval()
    model.cuda()

    model.hparams.csv_test = args.csv_test
    # model.hparams.mount_point = args.mount_point
    model.hparams.model = args.model
    model.hparams.out = args.out

    valid_transform = EvalTransform(scale_factor=model.hparams.scale_factor)
    args_d['test_transform'] = valid_transform

    data = DATAMODULE(**args_d)
    data.setup()


    trainer = Trainer(devices=torch.cuda.device_count(), 
                      accelerator="gpu",
                      )
    
    trainer.test(model, datamodule=data)


def get_argparse():
    parser = argparse.ArgumentParser(description='Shape Analysis Explainaiblity and Interpretability predict script', conflict_handler='resolve')
    input_group = parser.add_argument_group('Input')
    
    input_group.add_argument('--nn', help='Neural network name', required=True, type=str, 
                             choices=['SaxiClassification', 'SaxiRegression', 'SaxiSegmentation', 'SaxiIcoClassification',
                                       'SaxiIcoClassification_fs', 'SaxiRing', 'SaxiRingClassification', 'SaxiRingMT', 
                                       'SaxiMHA', 'SaxiMHAFBClassification', 'SaxiMHAFBRegression', 'SaxiOctree', 'SaxiMHAClassification', 
                                       'SaxiMHAFBRegression_V', 'SaxiPointTransformer', 'SaxiMHAFBClassification_V','SaxiIdxAE', 'SaxiDenoiseUnet', 'SaxiDDPMUnet'])

    input_group.add_argument('--data_module', help='Data module type', required=True, type=str, default=None)
    input_group.add_argument('--model', help='Model for prediction', type=str, required=True)
    input_group.add_argument('--out', help='Output directory', type=str, default="./")
    
    return parser

if __name__ == '__main__':

    parser = get_argparse()
    
    initial_args, unknownargs = parser.parse_known_args()

    model_args = getattr(saxi_nets_lightning, initial_args.nn)
    parser = model_args.add_model_specific_args(parser)

    data_module = getattr(saxi_dataset, initial_args.data_module)
    parser = data_module.add_data_specific_args(parser)

    args = parser.parse_args()
    main(args)