# DentalModelSeg
---

Welcome to the official documentation for **DentalModelSeg**. 

---

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Experiments & Results](#experiments--results)
- [Explainability](#explainability)
- [Contribute](#contribute)
- [FAQs](#faqs)
- [License](#license)

---

## Introduction

---

**DentalModelSeg** is an executable which is installed when you install **ShapeAxi**. This executable is used to run the prediction of your data using a pretrained model already set up in this tool or your own model.

---

## Installation

--- 

To install this executable, you have to use the same procedure than the installation of **Shapeaxi**. If is not already installed, you can follow this procedure : 
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
```

## Type of data

To use this tool, you can have two different inpute files, **vtk** or **csv** but **NOT** both of them at the same time.  
You specify your input by adding **--vtk** if you want to use vtk input and **--csv** if you want csv.

---

## Experiments & Results

---

## Explainability

--- 

## Contribute

We welcome community contributions to **DentalModelSeg**. For those keen on enhancing this tool, please adhere to the steps below:

1. **Fork** the repository.
2. Create your **feature branch** (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a **pull request**.

For a comprehensive understanding of our contribution process, consult our [Contribution Guidelines](path/to/contribution_guidelines.md).

--- 

## FAQs

---

## License

**DentalModelSeg** is under the [APACHE 2.0](LICENSE) license.
