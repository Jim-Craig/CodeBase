# GTMS Defect Detection and Segmentation

## Overview

This repository contains the complete pipeline for detecting and analyzing defects in GTMS components using a combination of:

- YOLO-based Region of Interest (ROI) Detection
- U-Net Semantic Segmentation
- U-Net++ Semantic Segmentation
- U-Net++ with Focal Loss
- Automated defect reporting with defect size estimation in pixels

---

## Repository Structure

```
├── ROI_Detector/
├── UNET/
├── UNetPP/
├── UNetPP_FL/
├── utils/
├── Component_Extraction.py
├── Data_Prep.py
├── Generate_train_val_split.py
├── Image_Extraction.py
├── Predict.py
└── ISRO_Training.ipynb
```

---

## Folder Description

### `ROI_Detector`

Contains the trained YOLO model weights used for detecting the **LGP** and **SGP** Regions of Interest (ROI) from input images.

The ROI detector serves as the first stage of the pipeline by localizing the relevant component clusters before segmentation and defect analysis.

### `UNET`

Implementation and training code for the **U-Net** segmentation model.

Contents include:
- `train.py` — Training script for U-Net
- Model checkpoints and related files (if included)

### `UNetPP`

Implementation and training code for the **U-Net++** segmentation model.

Contents include:
- `train.py` — Training script for U-Net++

### `UNetPP_FL`

Implementation and training code for **U-Net++ with Focal Loss**.

Contents include:
- `train.py` — Training script for U-Net++ using Focal Loss

### `utils`

Utility functions used throughout training, preprocessing, inference, and evaluation.

---

## Scripts

### `Component_Extraction.py`

Extracts individual GTMS components from pre-processed images containing the complete component cluster. Used to isolate individual components before defect analysis.

### `Data_Prep.py`

Performs data preparation and augmentation on binary segmentation masks before training. Typical operations include:

- Mask augmentation
- Image-mask consistency checks
- Dataset organization

### `Generate_train_val_split.py`

Generates training and validation splits for segmentation model training, ensuring a reproducible dataset partitioning process.

### `Image_Extraction.py`

Extracts images from the folder structure provided by VSSC.

> **Note:** The directory structure may vary across datasets. Modify the extraction paths and logic as required for your environment.

### `Predict.py`

Main inference pipeline for defect detection and reporting. The script can process:

1. Pre-processed component cluster images
2. Individual component images

Outputs include:
- LGP defect detection
- SGP defect detection
- Segmentation masks
- Defect size estimation (in pixels)
- Detailed defect report

### `ISRO_Training.ipynb`

Jupyter notebook containing training experiments, model development, and evaluation workflows.

---

## Workflow

```
Raw Images
      │
      ▼
Image_Extraction.py
      │
      ▼
ROI Detector (YOLO)
      │
      ▼
Component_Extraction.py
      │
      ▼
Segmentation Models
(U-Net / U-Net++ / U-Net++ + Focal Loss)
      │
      ▼
Predict.py
      │
      ▼
Defect Report
(Size, Location, Type)
```

---

## Training

### Data Preparation

```bash
python Data_Prep.py
python Generate_train_val_split.py
```

### Train U-Net

```bash
cd UNET
python train.py
```

### Train U-Net++

```bash
cd UNetPP
python train.py
```

### Train U-Net++ with Focal Loss

```bash
cd UNetPP_FL
python train.py
```

---

## Inference

Run the complete defect detection and reporting pipeline:

```bash
python Predict.py
```

The script accepts either pre-processed component cluster images or individual component images, and generates defect reports for both **LGP** and **SGP** regions.

---

## Features

- Automated ROI detection using YOLO
- Multiple segmentation architectures for comparison:
  - U-Net
  - U-Net++
  - U-Net++ with Focal Loss
- Defect localization and segmentation
- Pixel-level defect size estimation
- End-to-end inference pipeline
- Modular and extensible design

---

## Notes

- ROI detector weights are stored in the `ROI_Detector` directory.
- Update dataset paths according to your local environment.
- Modify `Image_Extraction.py` if the source image directory structure differs from the VSSC format.

---

## Acknowledgements

This work was developed for automated defect detection and analysis of GTMS components using deep learning-based object detection and semantic segmentation techniques.
