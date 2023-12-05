# DentalModelSeg
---

Welcome to the official documentation for **DentalModelSeg**. 

---

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

---

## Introduction

---

**DentalModelSeg** is an executable which is installed when you install **ShapeAXI**. This executable is used to run the prediction of your data using a pretrained model already set up in this tool or your own model.

---

## Installation

--- 

To install this executable, you have to use the same procedure than the installation of **ShapeAXI**. If is not already installed, you can follow this procedure : 
[Installation](README.md#Installation) 

---

## Usage

---

```
usage: dentalmodelseg [-h] [--model MODEL] [--vtk VTK] [--csv CSV] [--out OUT] [--mount_point MOUNT_POINT] [--num_workers NUM_WORKERS] [--segmentation_crown SEGMENTATION_CROWN] [--array_name ARRAY_NAME] [--fdi FDI]

Evaluate classification result

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Path to the model
  --vtk VTK             Path to your vtk file
  --csv CSV             Path to your csv file
  --out OUT             Output directory
  --mount_point MOUNT_POINT
                        Mount point for the dataset
  --num_workers NUM_WORKERS
                        Number of workers for loading
  --segmentation_crown SEGMENTATION_CROWN
                        Isolation of each different tooth in a specific vtk file
  --array_name ARRAY_NAME
                        Predicted ID array name for output vtk
  --fdi FDI             numbering system. 0: universal numbering; 1: FDI world dental Federation notation
  --overwrite OVERWRITE
                        Overwrite the input vtk file
```

## Input and type of data

To use this tool, you can have two different inpute files, **vtk** or **csv** but **NOT** both of them at the same time.    
You specify your input by adding **--vtk** if you want to use vtk input and **--csv** if you want csv.  
Identify the mount point of your dataset (the folder of your vtk files for example) with the **--mount_point** argument.

## Which model

There is already a model loaded in the package thaht you can find here :  
*https://github.com/DCBIA-OrthoLab/Fly-by-CNN/releases/download/3.0/07-21-22_val-loss0.169.pth*  
If you want to load your own model, your can by specify **--model** and adding the path to your model. 

## Output

You can add the folder of your output where the data will be stored and if this folder does not exist, it will be create.  
You can identify it by adding **--out**.  
The prediction files will be created in this out folder with the name of your input with *_pred*. Otherwise, if you use vtk input, if you set **--overwrite** to *True*, it will overwrite your vtk input by the predicted file with the same name.
Moreover, using **--segmentation_crown**, it will create a file for each teeth depending on their label in the output directory.

---

## License

**DentalModelSeg** is under the [APACHE 2.0](LICENSE) license.
