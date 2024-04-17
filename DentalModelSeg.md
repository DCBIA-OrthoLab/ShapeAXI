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
usage: dentalmodelseg [-h] [--vtk VTK] [--csv CSV] [--stl STL] [--model MODEL] [--suffix SUFFIX] [--out OUT] [--num_workers NUM_WORKERS] [--crown_segmentation CROWN_SEGMENTATION]
                      [--array_name ARRAY_NAME] [--fdi FDI] [--overwrite OVERWRITE] [--device DEVICE] [--vtk_folder VTK_FOLDER]

Evaluate classification result

optional arguments:
  -h, --help            show this help message and exit
  --vtk VTK             Path to your vtk file
  --csv CSV             Path to your csv file
  --stl STL             Path to your stl file
  --model MODEL         Path to the model
  --suffix SUFFIX       Suffix of the prediction
  --out OUT             Output directory
  --num_workers NUM_WORKERS
                        Number of workers for loading
  --crown_segmentation CROWN_SEGMENTATION
                        Isolation of each different tooth in a specific vtk file
  --array_name ARRAY_NAME
                        Predicted ID array name for output vtk
  --fdi FDI             numbering system. 0: universal numbering; 1: FDI world dental Federation notation
  --overwrite OVERWRITE
                        Overwrite the input vtk file
  --device DEVICE       Device to use for inference
  --vtk_folder VTK_FOLDER
                        Path to tronquate your input path
```

## Input and type of data

To use this tool, you can have three different inputs files, **vtk**, **stl** or **csv** but **NOT** all three at once.    
You specify your input by adding **--vtk** if you want to use vtk, **--stl** for stl and **--csv** for csv.  

## Which model

There is already a model loaded in the package that you can find here :  
*https://github.com/DCBIA-OrthoLab/Fly-by-CNN/releases/download/3.0/07-21-22_val-loss0.169.pth*  
If you want to load your own model, your can by specify **--model** and adding the path to your model. 

## Output

You can add the folder of your output where the data will be stored and if this folder does not exist, it will be create.   
You can identify it by adding **--out**.    
The prediction files will be created in this out folder with the name of your input with *_pred*. Otherwise, if you use vtk input, if you set **--overwrite** to *True*, it will overwrite your input by the predicted file.  
Moreover, using **--segmentation_crown**, will create a file for each teeth depending on their label in the output directory.<br>

Finally, if you use a **csv** input, you can use another argumment **--vtk_folder**.    
- For example the path to your csv is : /home/data/filecsv.csv
- Your csv file contains lines like these :  
 */home/data/vtk_nonsegmented/test/file1.vtk*  
 */home/data/vtk_nonsegmented/file2.vtk*  
 */home/data/vtk_nonsegmented/file3.vtk*  
 *...*
- Your output is : /home/data/vtk_segmented
- Your vtk_folder is : /home/data/vtk_non_segmented
- Your suffix is : pred

It will store your vtk files in this path : /output + /filecsv_pred + /path_to_vtk_file - /vtk_folder  
In this example, it will sotre the results here : /home/data/vtk_segmented/filecsv_pred/test/file1.vtk (for the first vtk file)  
Instead of here : /home/data/vtk_segmented/filecsv_pred/home/data/vtk_nonsegmented/test/file1.vtk if you do not specify vtk_folder  

---

## License

**DentalModelSeg** is under the [APACHE 2.0](LICENSE) license.
