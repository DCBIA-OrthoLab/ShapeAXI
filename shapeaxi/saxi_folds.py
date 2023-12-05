import os
import sys
import glob 
import subprocess
import argparse
from argparse import Namespace
import pandas as pd
import numpy as np
import pickle 


from . import compute_min_scale, split_train_eval, saxi_eval, saxi_predict, saxi_train, saxi_gradcam

class bcolors:
    HEADER = '\033[95m'
    #blue
    PROC = '\033[94m'
    #CYAN
    INFO = '\033[96m'
    #green
    SUCCESS = '\033[92m'
    #yellow
    WARNING = '\033[93m'
    #red
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def get_last_checkpoint(checkpoint_dir):
    # Get the last checkpoint
    checkpoint_paths = os.path.join(checkpoint_dir, '*.ckpt')
    checkpoint_paths = sorted(glob.iglob(checkpoint_paths), key=os.path.getctime, reverse=True) 
    if len(checkpoint_paths) > 0:
        return checkpoint_paths[0]
    return None

def get_val_loss(checkpoint_path):
    # Get the last checkpoint
    checkpoint_fn = os.path.basename(checkpoint_path)
    checkpoint_dict = eval('dict(%s)' % checkpoint_fn.replace('-',',').replace(' ','').replace('.ckpt', ''))
    return checkpoint_dict['val_loss']

def get_best_checkpoint(checkpoint_dir):
    # Get the last checkpoint
    checkpoint_paths = os.path.join(checkpoint_dir, '*.ckpt')
    checkpoint_paths = sorted(glob.iglob(checkpoint_paths), key=get_val_loss) 
    if len(checkpoint_paths) > 0:
        return checkpoint_paths[0]
    return None

def get_argparse_dict(parser):
    # Get the default arguments from the parser
    default = {}
    for action in parser._actions:
        if action.dest != "help":
            default[action.dest] = action.default
    return default

def replace_extension(fname, new_extension):
    return os.path.splitext(fname)[0] + new_extension

def get_output_filename(fname, suffix):
    return replace_extension(fname, f'_{suffix}')



def main(args, arg_groups):
    # Main function
    create_folds = False
    scale_factor = None

    if not os.path.exists(args.out):
        os.makedirs(args.out)
    
    # Kill the program if the number of folds is 0
    if args.folds == 0:
        sys.exit("The value of nn is 0. You must specify a value greater than 0.")

    # Kill the program if the split is negative
    if args.valid_split < 0:
        sys.exit("The value of split is negative. You must specify a value greater than 0.")

################################################################################## SCALE FACTOR PART #####################################################################################################
    if args.compute_scale_factor:
        # Compute the scale factor for the dataset
        compute_min_scale_args_out = os.path.basename(args.csv)

        ext = os.path.splitext(compute_min_scale_args_out)[1]        
        compute_min_scale_args_out = compute_min_scale_args_out.replace(ext, '_scale_factor' + ext)
        compute_min_scale_args_out = os.path.join(args.out, compute_min_scale_args_out)

        if os.path.exists(compute_min_scale_args_out):
            # If the scale factor has already been computed, load it
            df_scale = pd.read_csv(compute_min_scale_args_out)
            scale_factor = np.min(df_scale['surf_scale'])

        else:
            # Compute the scale factor
            compute_min_scale_args = get_argparse_dict(compute_min_scale.get_argparse())
            compute_min_scale_args['csv'] = args.csv
            compute_min_scale_args['surf_column'] = args.surf_column
            compute_min_scale_args['out'] = compute_min_scale_args_out

            compute_min_scale_args = Namespace(**compute_min_scale_args)
            scale_factor = compute_min_scale.main(compute_min_scale_args)

################################################################################## SPLIT PART #####################################################################################################

    for f in range(args.folds):
        #Check if the folds already exist and if there are train, val and test dataset inside
        ext = os.path.splitext(args.csv)[1]
        csv_train = get_output_filename(args.csv, f'fold{f}_train_train.csv')
        csv_valid = get_output_filename(args.csv, f'fold{f}_train_test.csv')
        csv_test = get_output_filename(args.csv, f'fold{f}_test.csv')

        if not os.path.exists(csv_train) or not os.path.exists(csv_valid) or not os.path.exists(csv_test):
            create_folds = True
            break


    if create_folds:

        # First split of the data to use only the train set for the split into the different folds 
        print(bcolors.INFO, "Start spliting data", bcolors.ENDC)
        split_csv_args = get_argparse_dict(split_train_eval.get_argparse())
        split_csv_args['csv'] = args.csv
        split_csv_args['split'] = args.valid_split
        split_csv_args = Namespace(**split_csv_args)
        split_train_eval.first_split_data(split_csv_args)
        print(bcolors.SUCCESS, "End spliting data", bcolors.ENDC)

        # Creation of test and train dataset for each fold
        print(f"{bcolors.INFO}Start creating {args.folds} folds{bcolors.ENDC}")
        csv_train = get_output_filename(args.csv, 'train.csv')
        split_csv_args = get_argparse_dict(split_train_eval.get_argparse())
        for k in arg_groups['Split']:
            split_csv_args[k] = arg_groups['Split'][k]
        split_csv_args['csv'] = csv_train
        split_csv_args = Namespace(**split_csv_args)
        split_train_eval.split_data_folds_test_train(split_csv_args)

        # Creation of the val dataset
        for f in range(args.folds):
            # Use of split_train_eval to split the train into train and validation
            csv_train = get_output_filename(args.csv, f'train_fold{f}_train.csv')
            split_csv_args = get_argparse_dict(split_train_eval.get_argparse())

            for k in arg_groups['Split']:
                split_csv_args[k] = arg_groups['Split'][k]

            split_csv_args['csv'] = csv_train

            split_csv_args = Namespace(**split_csv_args)
            split_train_eval.split_data_folds_train_eval(split_csv_args)

        print(f"{bcolors.SUCCESS}End of creating the {args.folds} folds {bcolors.ENDC}")

#################################################################################### TRAIN PART #####################################################################################################

    for f in range(0, args.folds):
        #Train the model for each fold
        print(bcolors.INFO, "Start training for fold {f}".format(f=f), bcolors.ENDC)
        csv_train = args.csv.replace(ext, '_train_fold{f}_train_train.csv').format(f=f)
        csv_valid = args.csv.replace(ext, '_train_fold{f}_train_test.csv').format(f=f)
        csv_test = args.csv.replace(ext, '_train_fold{f}_test.csv').format(f=f)
        saxi_train_args = get_argparse_dict(saxi_train.get_argparse())

        for k in arg_groups['Train']:
            if k in saxi_train_args:        
                saxi_train_args[k] = arg_groups['Train'][k]

        saxi_train_args['csv_train'] = csv_train
        saxi_train_args['csv_test'] = csv_test
        saxi_train_args['csv_valid'] = csv_valid
        saxi_train_args['out'] = os.path.join(args.out, 'train', 'fold{f}'.format(f=f))

        last_checkpoint = get_last_checkpoint(saxi_train_args['out'])
        
        if last_checkpoint is None:
            command = [sys.executable, os.path.join(os.path.dirname(__file__), 'saxi_train.py')]

            for k in saxi_train_args:
                if saxi_train_args[k]:
                    command.append('--' + str(k))
                    command.append(str(saxi_train_args[k]))
            subprocess.run(command)

        print(bcolors.SUCCESS, "End training for fold {f}".format(f=f), bcolors.ENDC)

#################################################################################### TEST PART #####################################################################################################
    #Declaration of the storage variables for choose the best model and its weighted F1 score
    best_weighted_f1_score = 0.0
    best_model_fold = ""

    for f in range(0, args.folds):
        #Test the model for each fold
        print(bcolors.INFO, "Start test for fold {f}".format(f=f), bcolors.ENDC)
        ext = os.path.splitext(args.csv)[1]
        csv_test = args.csv.replace(ext, '_train_fold{f}_test.csv').format(f=f)
        saxi_train_args_out = os.path.join(args.out, 'train', 'fold{f}'.format(f=f))
        best_model_path = get_best_checkpoint(saxi_train_args_out)
        saxi_predict_args = get_argparse_dict(saxi_predict.get_argparse())
        
        saxi_predict_args['csv'] = csv_test
        saxi_predict_args['model'] = best_model_path
        saxi_predict_args['surf_column'] = args.surf_column
        saxi_predict_args['class_column'] = args.class_column
        saxi_predict_args['mount_point'] = args.mount_point
        saxi_predict_args['nn'] = args.nn
        saxi_predict_args['out'] = os.path.join(args.out, 'test', 'fold{f}'.format(f=f))

        saxi_predict_args = Namespace(**saxi_predict_args)
        fname = os.path.basename(csv_test)
        out_prediction = os.path.join(saxi_predict_args.out, os.path.basename(best_model_path), fname.replace(ext, "_prediction" + ext))

        if not os.path.exists(out_prediction):
            saxi_predict.main(saxi_predict_args)

        print(bcolors.SUCCESS, "End test prediction for fold {f}".format(f=f), bcolors.ENDC)

################################################################################# EVALUATION PART #####################################################################################################

        #Run the evaluation for the prediction
        print(bcolors.INFO, "Start evaluation for fold {f}".format(f=f), bcolors.ENDC)
        saxi_eval_args = get_argparse_dict(saxi_eval.get_argparse())

        saxi_eval_args['csv'] = out_prediction
        saxi_eval_args['csv_true_column'] = args.class_column
        saxi_eval_args['nn'] = args.nn
        saxi_eval_args = Namespace(**saxi_eval_args)

        current_weighted_f1_score = saxi_eval.main(saxi_eval_args)

        if current_weighted_f1_score > best_weighted_f1_score:
            best_weighted_f1_score = current_weighted_f1_score
            best_model_fold = f'{f}'

        print(bcolors.SUCCESS, "End evaluation prediction for fold {f}".format(f=f), bcolors.ENDC)

################################################################################## AGGREGATE PART #####################################################################################################

    print(bcolors.INFO, "Start aggregate for all folds".format(f=f), bcolors.ENDC)
    if args.nn == "SaxiClassification":
        # Create a single dataframe and prob array
        out_prediction_agg = []
        out_prediction_probs_agg = []
        for f in range(0, args.folds):
            
            ext = os.path.splitext(args.csv)[1]
            csv_test = args.csv.replace(ext, '_train_fold{f}_test.csv').format(f=f)

            saxi_train_args_out = os.path.join(args.out, 'train', 'fold{f}'.format(f=f))
            best_model_path = get_best_checkpoint(saxi_train_args_out)

            fname = os.path.basename(csv_test)
            out_prediction_fn = os.path.join(args.out, 'test', 'fold{f}'.format(f=f), os.path.basename(best_model_path), fname.replace(ext, "_prediction" + ext))

            out_prediction_agg.append(pd.read_csv(out_prediction_fn))

            probs_fn = out_prediction_fn.replace("_prediction.csv", "_probs.pickle")
            out_prediction_probs_agg.append(pickle.load(open(probs_fn, 'rb')))
            
        # Concatenate all datragrames and probs
        out_prediction_agg = pd.concat(out_prediction_agg)
        fname = os.path.basename(args.csv)
        ext = os.path.splitext(args.csv)[1]
        out_prediction_agg_fn = os.path.join(args.out, 'test', fname.replace(ext, "_aggregate_prediction" + ext))
        out_prediction_agg.to_csv(out_prediction_agg_fn, index=False)

        out_prediction_probs_agg = np.concatenate(out_prediction_probs_agg)
        out_prediction_probs_agg_fn = out_prediction_agg_fn.replace("_prediction.csv", "_probs.pickle")
        pickle.dump(out_prediction_probs_agg, open(out_prediction_probs_agg_fn, 'wb'))

        


    elif args.nn == "SaxiRegression":
        # Create a single dataframe and prob array
        out_prediction_agg = []
        
        for f in range(0, args.folds):
            
            ext = os.path.splitext(args.csv)[1]
            csv_test = args.csv.replace(ext, '_train_fold{f}_test.csv').format(f=f)

            saxi_train_args_out = os.path.join(args.out, 'train', 'fold{f}'.format(f=f))
            best_model_path = get_best_checkpoint(saxi_train_args_out)

            fname = os.path.basename(csv_test)
            out_prediction_fn = os.path.join(args.out, 'test', 'fold{f}'.format(f=f), os.path.basename(best_model_path), fname.replace(ext, "_prediction" + ext))

            out_prediction_agg.append(pd.read_csv(out_prediction_fn))
            
        # Concatenate all datragrames and probs
        out_prediction_agg = pd.concat(out_prediction_agg)
        fname = os.path.basename(args.csv)
        ext = os.path.splitext(args.csv)[1]
        out_prediction_agg_fn = os.path.join(args.out, 'test', fname.replace(ext, "_aggregate_prediction" + ext))
        out_prediction_agg.to_csv(out_prediction_agg_fn, index=False)
    
    
    if args.nn == "SaxiClassification" or args.nn == "SaxiRegression":
        #Run the evaluation for the aggregate
        saxi_eval_args = get_argparse_dict(saxi_eval.get_argparse())
        saxi_eval_args['csv'] = out_prediction_agg_fn
        saxi_eval_args['csv_true_column'] = args.class_column
        saxi_eval_args['nn'] = args.nn
        saxi_eval_args = Namespace(**saxi_eval_args)    
        saxi_eval.main(saxi_eval_args)

    # Print the best model and its weighted F1 score
    print("Best model fold :", best_model_fold)
    print("Best Weighted F1 score:", best_weighted_f1_score)


    print(bcolors.SUCCESS, "END aggregate prediction for ALL folds", bcolors.ENDC)


################################################################################ EXPLAINABILITY PART ##############################################################################################################      

    for f in range(0, args.folds):
        print(bcolors.INFO, "Start explainability for fold {f}".format(f=f), bcolors.ENDC)
        ext = os.path.splitext(args.csv)[1]
        csv_test = args.csv.replace(ext, '_train_fold{f}_test.csv').format(f=f)
        fname = os.path.basename(csv_test)
        df_test = pd.read_csv(csv_test)
        saxi_train_args_out = os.path.join(args.out, 'train', 'fold{f}'.format(f=f))    
        best_model_path = get_best_checkpoint(saxi_train_args_out)
        saxi_predict_args_out = os.path.join(args.out, 'test', 'fold{f}'.format(f=f))
        out_prediction = os.path.join(saxi_predict_args_out, os.path.basename(best_model_path), fname.replace(ext, "_prediction" + ext))

        if args.nn == "SaxiClassification":
            for target_class in df_test[args.class_column].unique():
                saxi_gradcam_args = get_argparse_dict(saxi_gradcam.get_argparse())
                saxi_gradcam_args['nn'] = args.nn
                saxi_gradcam_args['csv_test'] = out_prediction
                saxi_gradcam_args['surf_column'] = args.surf_column
                saxi_gradcam_args['class_column'] = args.class_column
                saxi_gradcam_args['num_workers'] = args.num_workers
                saxi_gradcam_args['model'] = best_model_path
                saxi_gradcam_args['target_layer'] = args.target_layer
                saxi_gradcam_args['target_class'] = target_class
                saxi_gradcam_args['mount_point'] = args.mount_point
                saxi_gradcam_args['fps'] = args.fps
                saxi_gradcam_args = Namespace(**saxi_gradcam_args)
                saxi_gradcam.main(saxi_gradcam_args)

        elif args.nn == "SaxiRegression":
            saxi_gradcam_args = get_argparse_dict(saxi_gradcam.get_argparse())
            saxi_gradcam_args['nn'] = args.nn
            saxi_gradcam_args['csv_test'] = out_prediction
            saxi_gradcam_args['surf_column'] = args.surf_column
            saxi_gradcam_args['class_column'] = args.class_column
            saxi_gradcam_args['num_workers'] = args.num_workers
            saxi_gradcam_args['model'] = best_model_path
            saxi_gradcam_args['nn'] = args.nn
            saxi_gradcam_args['target_layer'] = args.target_layer
            saxi_gradcam_args['target_class'] = None
            saxi_gradcam_args['mount_point'] = args.mount_point
            saxi_gradcam_args['fps'] = args.fps
            saxi_gradcam_args = Namespace(**saxi_gradcam_args)
            saxi_gradcam.main(saxi_gradcam_args)
        
        elif args.nn == "IcoConv":
            saxi_gradcam_args = get_argparse_dict(saxi_gradcam.get_argparse())
            saxi_gradcam_args['csv_test'] = 'test'
            saxi_gradcam_args['nn'] = args.nn

        print(bcolors.SUCCESS, "End explainability for fold {f}".format(f=f), bcolors.ENDC)


############################################################## TEST + EVALUATION OF THE BEST MODEL ######################################################################################################################

    ext = os.path.splitext(args.csv)[1]
    csv_test = args.csv.replace(ext, '_test.csv')
    saxi_train_args_out = os.path.join(args.out, 'train', 'fold{f}'.format(f=best_model_fold))

    print(bcolors.INFO, "Start testing of the best model (fold {f})".format(f=best_model_fold), bcolors.ENDC)
    best_model_path = get_best_checkpoint(saxi_train_args_out)
    saxi_predict_args = get_argparse_dict(saxi_predict.get_argparse())

    saxi_predict_args['csv'] = csv_test
    saxi_predict_args['model'] = best_model_path
    saxi_predict_args['surf_column'] = args.surf_column
    saxi_predict_args['class_column'] = args.class_column
    saxi_predict_args['mount_point'] = args.mount_point
    saxi_predict_args['nn'] = args.nn
    saxi_predict_args['out'] = os.path.join(args.out, 'best_test_fold{f}'.format(f=best_model_fold))

    saxi_predict_args = Namespace(**saxi_predict_args)
    fname = os.path.basename(csv_test)
    out_prediction = os.path.join(saxi_predict_args.out, os.path.basename(best_model_path), fname.replace(ext, "_prediction" + ext))

    if not os.path.exists(out_prediction):
        saxi_predict.main(saxi_predict_args)

    print(bcolors.SUCCESS, "End testing of the best model", bcolors.ENDC)

    #Run the evaluation for the prediction
    print(bcolors.INFO, "Start evaluation of the best model")
    saxi_eval_args = get_argparse_dict(saxi_eval.get_argparse())

    saxi_eval_args['csv'] = out_prediction
    saxi_eval_args['csv_true_column'] = args.class_column
    saxi_eval_args['nn'] = args.nn
    saxi_eval_args = Namespace(**saxi_eval_args)

    saxi_eval.main(saxi_eval_args)

    print(bcolors.SUCCESS, "End evaluation of the best model".format(f=f), bcolors.ENDC)


#####################################################################################################################################################################################################

def cml():
    # Command line interface
    parser = argparse.ArgumentParser(description='Automatically train and evaluate a N fold cross-validation model for Shape Analysis Explainability and Interpretability')

    # Arguments used for split the data into the different folds
    split_group = parser.add_argument_group('Split')
    split_group.add_argument('--csv', help='CSV with columns surf,class', type=str, required=True)
    split_group.add_argument('--folds', help='Number of folds', type=int, default=5)
    split_group.add_argument('--valid_split', help='Split float [0-1]', type=float, default=0.2)
    split_group.add_argument('--group_by', help='GroupBy criteria in the CSV. For example, SubjectID in case the same subjects has multiple timepoints/data points and the subject must belong to the same data split', type=str, default=None)

    # Arguments used for training
    train_group = parser.add_argument_group('Train')
    train_group.add_argument('--nn', help='Neural network name : SaxiClassification, SaxiRegression, SaxiSegmentation, SaxiIcoClassification', type=str, required=True)
    train_group.add_argument('--csv_train', help='CSV with column surf', type=str)
    train_group.add_argument('--csv_valid', help='CSV with column surf', type=str)
    train_group.add_argument('--csv_test', help='CSV with column surf', type=str)
    train_group.add_argument('--model', help='Model to continue training', type=str, default= None)
    train_group.add_argument('--train_sphere_samples', help='Number of samples for the training sphere', type=int, default=10000)
    train_group.add_argument('--surf_column', help='Surface column name', type=str, default="surf")
    train_group.add_argument('--class_column', help='Class column name', type=str, default="class")
    train_group.add_argument('--scale_factor', help='Scale factor for the shapes', type=float,default=1.0)
    train_group.add_argument('--profiler', help='Profiler', type=str, default=None)
    train_group.add_argument('--compute_scale_factor', help='Compute a global scale factor for all shapes in the population.', type=int, default=0)
    train_group.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    train_group.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    train_group.add_argument('--base_encoder', help='Base encoder for the feature extraction', type=str, default='resnet18')
    train_group.add_argument('--base_encoder_params', help='Base encoder parameters that are passed to build the feature extraction', type=str, default='pretrained=False,spatial_dims=2,n_input_channels=4,num_classes=512')
    train_group.add_argument('--hidden_dim', help='Hidden dimension for features output. Should match with output of base_encoder. Default value is 512', type=int, default=512)
    train_group.add_argument('--radius', help='Radius of icosphere', type=float, default=1.35)    
    train_group.add_argument('--subdivision_level', help='Subdivision level for icosahedron', type=int, default=1)
    train_group.add_argument('--image_size', help='Image resolution size', type=float, default=256)
    train_group.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    train_group.add_argument('--epochs', help='Max number of epochs', type=int, default=200)   
    train_group.add_argument('--batch_size', help='Batch size', type=int, default=3)    
    train_group.add_argument('--patience', help='Patience for early stopping', type=int, default=30)
    train_group.add_argument('--log_every_n_steps', help='Log every n steps', type=int, default=10)    
    train_group.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default=None)
    train_group.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default=None)
    train_group.add_argument('--neptune_project', help='Neptune project', type=str, default=None)
    train_group.add_argument('--neptune_tags', help='Neptune tags', type=str, default=None)

    # Arguments used for explainability
    explain_group = parser.add_argument_group('Explainability group')
    explain_group.add_argument('--target_layer', help='Target layer for explainability', type=str, default='layer4')
    explain_group.add_argument('--fps', help='Frames per second', type=int, default=24) 

    # Arguments used for evaluation
    out_group = parser.add_argument_group('Output')
    out_group.add_argument('--out', help='Output', type=str, default="./")

    args = parser.parse_args()

    arg_groups = {}
    for group in parser._action_groups:
        arg_groups[group.title] = {a.dest:getattr(args,a.dest,None) for a in group._group_actions}

    main(args, arg_groups)


if __name__ == '__main__':
    cml()