import numpy as np
import argparse
import importlib
import os
from datetime import datetime
import json
import glob
import sys
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import classification_report
import pandas as pd
from vtk.util.numpy_support import vtk_to_numpy
from sklearn.metrics import jaccard_score
import seaborn as sns
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import itertools
from scipy import interp
import pickle 
import plotly.graph_objects as go
import plotly.express as px

from . import utils
from .saxi_train import SaxiIcoClassification_train

# This file is used to evaluate the results of a classification or segmentation task (after the model has been trained and predictions have been made)

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    #This function prints and plots the confusion matrix. Normalization can be applied by setting `normalize=True`.
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix, avg:", np.trace(cm)/len(classes))
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()

    return cm
  

def SaxiIcoClassification_eval():
    print("Not implemented yet")


def SaxiClassification_eval(df, args, y_true_arr, y_pred_arr):
    # For the classification, evaluating a classification model, generating classification metrics, creating confusion matrix visualizations
    # It also responsible for plotting ROC curves, aggregating and reporting classification metrics in a structured format
    if(args.csv_tag_column):
      class_names = df[[args.csv_tag_column, args.csv_prediction_column]].drop_duplicates()[args.csv_tag_column]
      class_names.sort()
    else:
      class_names = pd.unique(df[args.csv_prediction_column])
      class_names.sort()

    for idx, row in df.iterrows():
      y_true_arr.append(row[args.csv_true_column])
      y_pred_arr.append(row[args.csv_prediction_column])

    report = classification_report(y_true_arr, y_pred_arr, output_dict=True)
    print(json.dumps(report, indent=2))

    cnf_matrix = confusion_matrix(y_true_arr, y_pred_arr)
    np.set_printoptions(precision=3)

    # Plot non-normalized confusion matrix
    fig = plt.figure(figsize=args.figsize)

    plot_confusion_matrix(cnf_matrix, classes=class_names, title=args.title)
    confusion_filename = os.path.splitext(args.csv)[0] + "_confusion.png"
    fig.savefig(confusion_filename)


    # Plot normalized confusion matrix
    fig2 = plt.figure(figsize=args.figsize)
    cm = plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title=args.title + ' - normalized')

    norm_confusion_filename = os.path.splitext(args.csv)[0] + "_norm_confusion.png"
    fig2.savefig(norm_confusion_filename)


    probs_fn = args.csv.replace("_prediction.csv", "_probs.pickle")
    # print(probs_fn, os.path.splitext(probs_fn)[1])
    if os.path.exists(probs_fn) and os.path.splitext(probs_fn)[1] == ".pickle":
      
      # print("Reading:", probs_fn)

      with open(probs_fn, 'rb') as f:
        y_scores = pickle.load(f)

      y_onehot = pd.get_dummies(y_true_arr)


      # Create an empty figure, and iteratively add new lines
      # every time we compute a new class
      fig = go.Figure()
      fig.add_shape(
          type='line', line=dict(dash='dash'),
          x0=0, x1=1, y0=0, y1=1
      )

      for i in range(y_scores.shape[1]):

          y_true = y_onehot.iloc[:, i]
          y_score = y_scores[:, i]

          fpr, tpr, _ = roc_curve(y_true, y_score)
          auc_score = roc_auc_score(y_true, y_score)
          report[str(i)]["auc"] = auc_score

          name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
          fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

      fig.update_layout(
          xaxis_title='False Positive Rate',
          yaxis_title='True Positive Rate',
          yaxis=dict(scaleanchor="x", scaleratio=1),
          xaxis=dict(constrain='domain'),
          width=700, height=500
      )

      roc_filename = os.path.splitext(args.csv)[0] + "_roc.png"

      fig.write_image(roc_filename)

      support = []
      auc = []
      for i in range(y_scores.shape[1]):
          support.append(report[str(i)]["support"])
          auc.append(report[str(i)]["auc"])

      support = np.array(support)
      auc = np.array(auc)

      report["macro avg"]["auc"] = np.average(auc) 
      report["weighted avg"]["auc"] = np.average(auc, weights=support) 
          
      df_report = pd.DataFrame(report).transpose()
      report_filename = os.path.splitext(args.csv)[0] + "_classification_report.csv"
      df_report.to_csv(report_filename)
      
      # Calculate F1 score
      weighted_f1_score = report["weighted avg"]["f1-score"]

      # Print or store F1 score
      print("Weighted F1 Score:", weighted_f1_score)

      # Return the F1 score or any other relevant metric
      return weighted_f1_score


def SaxiSegmentation_eval(df, args, y_true_arr, y_pred_arr):
   # For the segmentation, evaluating a segmentation model, generating segmentation metrics, creating confusion matrix visualizations
    dice_arr = []
    df = pd.read_csv(args.csv)

    for idx, row in df.iterrows():

      print("Reading:", row["surf"])
      surf = utils.ReadSurf(row["surf"])
      print("Reading:", row["pred"])
      pred = utils.ReadSurf(row["pred"])
      surf_features_np = vtk_to_numpy(surf.GetPointData().GetScalars(args.surf_id))
      pred_features_np = vtk_to_numpy(pred.GetPointData().GetScalars(args.pred_id))
      pred_features_np[pred_features_np==-1] = 1
      surf_features_np = np.reshape(surf_features_np, -1)
      pred_features_np = np.reshape(pred_features_np, -1)
      unique_v = np.unique(np.union1d(surf_features_np, pred_features_np))

      for v in range(1, 34):
        if v not in unique_v:
          surf_features_np = np.concatenate([surf_features_np, np.repeat([v], 95)])
          surf_features_np = np.concatenate([surf_features_np, np.repeat([v + 1], 5)])
          pred_features_np = np.concatenate([pred_features_np, np.repeat([v], 95)])
          pred_features_np = np.concatenate([pred_features_np, np.repeat([v], 5)])

      jaccard = jaccard_score(surf_features_np, pred_features_np, average=None)
      dice = 2.0*jaccard/(1.0 + jaccard)
      dice_arr.append(dice)
      y_true_arr.extend(surf_features_np)
      y_pred_arr.extend(pred_features_np)

    dice_arr = np.array(dice_arr)
    cnf_matrix = confusion_matrix(y_true_arr, y_pred_arr)
    fig = plt.figure(figsize=(20, 20))
    # plot_confusion_matrix(cnf_matrix, classes=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16","33"], title="Confusion Matrix Segmentation")
    plot_confusion_matrix(cnf_matrix, classes=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16","17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32","33"], title="Confusion Matrix Segmentation")
    confusion_filename = os.path.splitext(args.csv)[0] + "_confusion.png"
    fig.savefig(confusion_filename)
    cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    print(cnf_matrix)
    FP = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)  
    FN = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    F1 = 2 * (PPV * TPR)/(PPV + TPR)
    print("True positive rate, sensitivity or recall:", TPR)
    print("True negative rate or specificity:", TNR)
    print("Positive predictive value or precision:", PPV)
    print("Negative predictive value:", NPV)
    print("False positive rate or fall out", FPR)
    print("False negative rate:", FNR)
    print("False discovery rate:", FDR)
    print("Overall accuracy:", ACC)
    print("F1 score:", F1)
    report = classification_report(y_true_arr, y_pred_arr, output_dict=True)
    print(report)
    jaccard = jaccard_score(y_true_arr, y_pred_arr, average=None)
    print("jaccard score:", jaccard)
    l_dice = 2.0*jaccard/(1.0 + jaccard)
    print("dice:", 2.0*jaccard/(1.0 + jaccard))
    print ('dice ', l_dice)
    print(f'average dice : {sum(l_dice)/len(l_dice)}')
    # Plot normalized confusion matrix
    fig2 = plt.figure(figsize=(20, 20))
    # cm = plot_confusion_matrix(cnf_matrix, classes=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16","33"], normalize=True, title="Confusion Matrix Segmentation - normalized")
    cm = plot_confusion_matrix(cnf_matrix, classes=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16","17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32","33"], normalize=True, title="Confusion Matrix Segmentation - normalized")
    norm_confusion_filename = os.path.splitext(args.csv)[0] + "_norm_confusion.png"
    #plt.show()
    fig2.savefig(norm_confusion_filename)
    fig3 = plt.figure() 
    # Creating plot
    print("Dice coefficient shape : ", dice_arr.shape)
    #dice_del = np.delete(dice_arr,[0,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],1) # upper
    dice_del = np.delete(dice_arr,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,31,32],1)
    #print(dice_del.shape)
    s = sns.violinplot(data=dice_del, scale='count',cut=0)
    #plt.xticks([0, 1, 2, 3, 4, 5, 6,7,8], ["1", "2", "3", "4", "5", "6", "7","8", "9"])
    plt.xticks([0,1, 2, 3, 4, 5, 6,7,8,9,10,11,12,13], ["18", "19", "20", "21", "22", "23","24","25","26","27","28","29","30","31"]) # lower
    #plt.xticks([0,1a, 2, 3, 4, 5, 6,7,8,9,10,11,12,13], ["2", "3", "4", "5", "6", "7","8","9","10","11","12","13","14","15"]) # upper
    #plt.xticks([0,1, 2, 3, 4, 5, 6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32], ["1","2", "3", "4", "5", "6", "7","8","9","10","11","12","13","14","15","16","17", "18", "19", "20", "21", "22","23","24","25","26","27","28","29","30","31","32","33"]) # lower
    s.set_title('Dice coefficients: lower jaw')
    box_plot_filename = os.path.splitext(args.csv)[0] + "_violin_plot.png"
    ax = plt.gca()
    ax.set_ylim([0.75, 1.005])
    plt.show()
    fig3.savefig(box_plot_filename)
    # Access the weighted f1-score
    weighted_f1_score = report['weighted avg']['f1-score']
    # Print or store F1 score
    print("Weighted F1 Score:", weighted_f1_score)
    # Return the F1 score or any other relevant metric
    return weighted_f1_score


def SaxiRegression_eval(df, args, y_true_arr, y_pred_arr):
   #Visualization of the distribution of absolute errors and prediction errors
    y_true_arr = [] 
    y_pred_arr = []

    if(os.path.splitext(args.csv)[1] == ".csv"):        
        df = pd.read_csv(args.csv)
    else:        
        df = pd.read_parquet(args.csv)

    y_true_arr = df[args.csv_true_column]
    y_pred_arr = df[args.csv_prediction_column]
    df['abs'] = np.abs(y_true_arr - y_pred_arr)
    df['error'] = y_true_arr - y_pred_arr
    fig = px.violin(df, y="abs")
    abs_filename = os.path.splitext(args.csv)[0] + "_abs.png"
    fig.write_image(abs_filename)

    fig = px.violin(df, y="error")
    error_filename = os.path.splitext(args.csv)[0] + "_error.png"
    fig.write_image(error_filename)


def main(args):
    y_true_arr = [] 
    y_pred_arr = []

    if(os.path.splitext(args.csv)[1] == ".csv"):        
        df = pd.read_csv(args.csv)
    else:        
        df = pd.read_parquet(args.csv)

    if args.nn == "SaxiClassification":
      f1_score = SaxiClassification_eval(df, args, y_true_arr, y_pred_arr)

    elif args.nn == "SaxiSegmentation":
      f1_score = SaxiSegmentation_eval(df, args, y_true_arr, y_pred_arr)

    elif args.nn == "SaxiRegression":
      f1_score = SaxiRegression_eval(df, args, y_true_arr, y_pred_arr)
    
    elif args.nn == "SaxiIcoClassification":
      SaxiIcoClasssification_eval()

    else:
      raise NotImplementedError(f"Neural network {args.nn} is not implemented")

    return f1_score



def get_argparse():
  # Function to parse arguments for the evaluation script
  parser = argparse.ArgumentParser(description='Evaluate classification result', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('--csv', type=str, help='CSV file', required=True)
  parser.add_argument('--csv_true_column', type=str, help='Which column to do the stats on', default="class")
  parser.add_argument('--csv_tag_column', type=str, help='Which column has the actual names', default=None)
  parser.add_argument('--csv_prediction_column', type=str, help='csv true class', default='pred')
  parser.add_argument('--nn', help='Neural network name : SaxiClassification, SaxiRegression, SaxiSegmentation, SaxiIcoClassification', type=str, default='SaxiClassification')
  parser.add_argument('--title', type=str, help='Title for the image', default="Confusion matrix")
  parser.add_argument('--figsize', type=float, nargs='+', help='Figure size', default=(6.4, 4.8))
  parser.add_argument('--surf_id', type=str, help='Name of array in point data for the labels', default="UniversalID")
  parser.add_argument('--pred_id', type=str, help='Name of array in point data for the predicted labels', default="PredictedID")

  return parser


if __name__ == "__main__": 
  parser = get_argparse() 
  args = parser.parse_args()
  main(args)





