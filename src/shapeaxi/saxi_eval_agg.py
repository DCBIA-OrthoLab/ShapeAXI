import argparse
from argparse import Namespace
import os
import glob
import pandas as pd
import pickle 
import numpy as np 

def get_val_loss(checkpoint_path):
    checkpoint_fn = os.path.basename(checkpoint_path)
    checkpoint_dict = eval('dict(%s)' % checkpoint_fn.replace('-',',').replace(' ','').replace('.ckpt', ''))
    return checkpoint_dict['val_loss']

def get_best_checkpoint(checkpoint_dir):
    checkpoint_paths = os.path.join(checkpoint_dir, '*.ckpt')
    checkpoint_paths = sorted(glob.iglob(checkpoint_paths), key=get_val_loss) 
    if len(checkpoint_paths) > 0:
        return checkpoint_paths[0]
    return None

args = {"folds": 5,
        "csv": "Deg_classification_aggregate_long_exists.csv",
        "out": "train_output"}
        
args = Namespace(**args)



out_prediction_agg = []
out_prediction_probs_agg = []
for f in range(0, args.folds):
    
    ext = os.path.splitext(args.csv)[1]
    csv_test = args.csv.replace(ext, 'fold{f}_test.csv').format(f=f)

    saxi_train_args_out = os.path.join(args.out, 'train', 'fold{f}'.format(f=f))
    best_model_path = get_best_checkpoint(saxi_train_args_out)

    fname = os.path.basename(csv_test)
    out_prediction_fn = os.path.join(args.out, 'test', 'fold{f}'.format(f=f), os.path.basename(best_model_path), fname.replace(ext, "_prediction" + ext))

    out_prediction_agg.append(pd.read_csv(out_prediction_fn))

    probs_fn = out_prediction_fn.replace("_prediction.csv", "_probs.pickle")
    out_prediction_probs_agg.append(pickle.load(open(probs_fn, 'rb')))

out_prediction_agg = pd.concat(out_prediction_agg)
fname = os.path.basename(args.csv)
ext = os.path.splitext(args.csv)[1]
out_prediction_agg_fn = os.path.join(args.out, 'test', fname.replace(ext, "_aggregate_prediction" + ext))
out_prediction_agg.to_csv(out_prediction_agg_fn, index=False)

out_prediction_probs_agg = np.concatenate(out_prediction_probs_agg)
out_prediction_probs_agg_fn = out_prediction_agg_fn.replace("_prediction.csv", "_probs.pickle")
pickle.dump(out_prediction_probs_agg, open(out_prediction_probs_agg_fn, 'wb'))