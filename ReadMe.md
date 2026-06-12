\documentclass[11pt]{article}

\usepackage[a4paper,margin=1in]{geometry}
\usepackage{enumitem}
\usepackage{hyperref}
\usepackage{listings}
\usepackage{xcolor}

\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    urlcolor=blue
}

\title{GTMS Defect Detection and Segmentation}
\author{}
\date{}

\begin{document}

\maketitle

\section{Overview}

This repository contains the complete pipeline for detecting and analyzing defects in GTMS components using a combination of:

\begin{itemize}
    \item YOLO-based Region of Interest (ROI) Detection
    \item U-Net Semantic Segmentation
    \item U-Net++ Semantic Segmentation
    \item U-Net++ with Focal Loss
    \item Automated defect reporting with defect size estimation in pixels
\end{itemize}

\section{Repository Structure}

\begin{verbatim}
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
\end{verbatim}

\section{Folder Description}

\subsection{ROI\_Detector}

Contains the trained YOLO model weights used for detecting the \textbf{LGP} and \textbf{SGP} Regions of Interest (ROI) from input images.

The ROI detector serves as the first stage of the pipeline by localizing the relevant component clusters before segmentation and defect analysis.

\subsection{UNET}

Implementation and training code for the \textbf{U-Net} segmentation model.

Contents include:

\begin{itemize}
    \item \texttt{train.py} -- Training script for U-Net
    \item Model checkpoints and related files (if included)
\end{itemize}

\subsection{UNetPP}

Implementation and training code for the \textbf{U-Net++} segmentation model.

Contents include:

\begin{itemize}
    \item \texttt{train.py} -- Training script for U-Net++
\end{itemize}

\subsection{UNetPP\_FL}

Implementation and training code for \textbf{U-Net++ with Focal Loss}.

Contents include:

\begin{itemize}
    \item \texttt{train.py} -- Training script for U-Net++ using Focal Loss
\end{itemize}

\subsection{utils}

Utility functions used throughout training, preprocessing, inference, and evaluation.

\section{Scripts}

\subsection{Component\_Extraction.py}

Extracts individual GTMS components from pre-processed images containing the complete component cluster.

This script is used to isolate individual components before defect analysis.

\subsection{Data\_Prep.py}

Performs data preparation and augmentation on binary segmentation masks before training.

Typical operations may include:

\begin{itemize}
    \item Mask augmentation
    \item Image-mask consistency checks
    \item Dataset organization
\end{itemize}

\subsection{Generate\_train\_val\_split.py}

Generates training and validation splits for segmentation model training.

This script ensures a reproducible dataset partitioning process.

\subsection{Image\_Extraction.py}

Extracts images from the folder structure provided by VSSC.

\textbf{Note:} The directory structure may vary across datasets. Modify the extraction paths and logic as required for your environment.

\subsection{Predict.py}

Main inference pipeline for defect detection and reporting.

The script can process:

\begin{enumerate}
    \item Pre-processed component cluster images
    \item Individual component images
\end{enumerate}

Outputs include:

\begin{itemize}
    \item LGP defect detection
    \item SGP defect detection
    \item Segmentation masks
    \item Defect size estimation (in pixels)
    \item Detailed defect report
\end{itemize}

\subsection{ISRO\_Training.ipynb}

Jupyter notebook containing training experiments, model development, and evaluation workflows.

\section{Workflow}

\begin{verbatim}
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
\end{verbatim}

\section{Training}

\subsection{Data Preparation}

\begin{verbatim}
python Data_Prep.py
python Generate_train_val_split.py
\end{verbatim}

\subsection{Train U-Net}

\begin{verbatim}
cd UNET
python train.py
\end{verbatim}

\subsection{Train U-Net++}

\begin{verbatim}
cd UNetPP
python train.py
\end{verbatim}

\subsection{Train U-Net++ with Focal Loss}

\begin{verbatim}
cd UNetPP_FL
python train.py
\end{verbatim}

\section{Inference}

Run the complete defect detection and reporting pipeline:

\begin{verbatim}
python Predict.py
\end{verbatim}

The script accepts either:

\begin{itemize}
    \item Pre-processed component cluster images
    \item Individual component images
\end{itemize}

and generates defect reports for both \textbf{LGP} and \textbf{SGP} regions.

\section{Features}

\begin{itemize}
    \item Automated ROI detection using YOLO
    \item Multiple segmentation architectures for comparison
    \item U-Net
    \item U-Net++
    \item U-Net++ with Focal Loss
    \item Defect localization and segmentation
    \item Pixel-level defect size estimation
    \item End-to-end inference pipeline
    \item Modular and extensible design
\end{itemize}

\section{Notes}

\begin{itemize}
    \item ROI detector weights are stored in the \texttt{ROI\_Detector} directory.
    \item Update dataset paths according to your local environment.
    \item Modify \texttt{Image\_Extraction.py} if the source image directory structure differs from the VSSC format.
\end{itemize}

\section{Acknowledgements}

This work was developed for automated defect detection and analysis of GTMS components using deep learning-based object detection and semantic segmentation techniques.

\end{document}