# Denoising Diffusion Probabilistic Model (DDPM) Implementation

This repository contains a Jupyter notebook (DDPM.ipynb) that implements a Denoising Diffusion Probabilistic Model (DDPM) from scratch, including unconditional generation and classifier-free diffusion guidance. It uses the MNIST dataset to train the model for generating digit images from noise. The notebook covers the forward diffusion process, backward denoising, model architecture (U-Net), training, sampling, and evaluation using FID scores.

The notebook is designed for educational purposes, providing a step-by-step guide to understanding and implementing diffusion models in PyTorch, based on the original DDPM paper and classifier-free guidance concepts.

## Table of Contents

*   Project Overview
*   Dataset
*   Features
*   Requirements
*   Installation
*   Usage
*   Results and Analysis
*   Contributing
*   License

## Project Overview

*   **Unconditional DDPM (Part 1)**: Implement the forward noise addition process and train a U-Net to reverse it, generating random MNIST digits from noise.
*   **Conditional DDPM with Classifier-Free Guidance (Part 2)**: Extend the model to condition on class labels (MNIST digits 0-9) without a separate classifier, allowing guided generation.
*   Key concepts: Diffusion schedules (beta, alpha), noise prediction, sampling algorithms, and evaluation metrics.
*   Training: Uses MSE loss for noise prediction; includes visualization of training progress.
*   Sampling: Generate images unconditionally or conditioned on labels, with optional guidance strength.
*   Evaluation: Compute FID scores for generated vs. real images, both overall and per class.

The notebook includes mathematical derivations, code implementations, and visualizations to illustrate the processes.

## Dataset

*   **MNIST**: 60,000 training and 10,000 test images of handwritten digits (28x28 grayscale).
*   Automatically downloaded and loaded via torchvision.datasets.MNIST.
*   Preprocessing: Normalization to \[-1, 1\] range for diffusion compatibility.

## Features

*   Custom DDPM class for forward and backward processes.
*   U-Net architecture with time and class embeddings for conditional generation.
*   Linear and cosine beta schedules for noise addition.
*   Training loop with progress tracking and sample visualizations.
*   Sampling functions for unconditional and conditional generation.
*   Classifier-free guidance for improved conditional sampling.
*   FID score computation using pre-trained Inception-v3 for evaluation.
*   Visualizations: Noisy images, generated samples, and FID results.

## Requirements

*   Python 3.11+
*   PyTorch
*   Torchvision
*   Matplotlib
*   NumPy
*   Tqdm (for progress bars)
*   FID Score library (installed via pip in the notebook)

See the notebook's import section for the full list:
```python
import os
import math
from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
```

## Installation

1.  Clone the repository:
    ```sh
    git clone https://github.com/yourusername/ddpm-implementation.git
    cd ddpm-implementation
    ```
    
2.  Install dependencies:
    ```sh
    pip install torch torchvision matplotlib numpy tqdm
    ```
    
3.  (Optional) Use a virtual environment:
    ```sh
    python -m venv env
    source env/bin/activate  # On Linux/Mac
    .\env\Scripts\activate   # On Windows
    pip install -r requirements.txt  # Create this file with the above libraries
    ```
    

## Usage

1.  Open the Jupyter notebook:
    ```
    jupyter notebook DDPM.ipynb
    ```
    
2.  Run the cells sequentially:
    *   Import packages and set up device (GPU recommended).
    *   Load and preprocess MNIST dataset.
    *   Define DDPM components (beta schedule, forward/backward processes).
    *   Implement U-Net model.
    *   Train the unconditional model.
    *   Sample and visualize unconditional generations.
    *   Extend to conditional model with guidance.
    *   Train, sample, and evaluate with FID.

Note: Training requires a GPU for efficiency (set device = 'cuda' if available). The notebook includes code to check for CUDA. Adjust hyperparameters like epochs, batch size, and timesteps as needed.

## Results and Analysis

*   **Unconditional Generation**: Produces random MNIST-like digits; visualizations show evolution from noise to clear images.
*   **Conditional Generation**: Generates specific digits (0-9) with classifier-free guidance; improves sample quality and adherence to labels.
*   **FID Scores**: Example results show per-class FID (e.g., ~140-150) and average (~147), indicating reasonable generation quality for a from-scratch implementation.
*   **Analysis**: Discusses noise schedules' impact, guidance strength trade-offs (quality vs. diversity), and potential improvements like advanced schedulers or larger models.
*   Refer to the notebook for plots (e.g., training samples, FID computations) and detailed outputs.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for improvements, bug fixes, or additional features like alternative datasets or advanced diffusion variants.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
