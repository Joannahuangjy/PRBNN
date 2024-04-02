# PRBNN Implementation in PyTorch with GPU Support

This repository contains the implementation of Probabilistic Radial Basis Function Neural Networks (PRBNN) using PyTorch, optimized for performance with GPU support. It showcases PRBNN's application in theoretical simulations and real-world scenarios, demonstrating its predictive capabilities and efficiency in handling uncertainty.

## Overview

The PRBNN framework, implemented in PyTorch, leverages the power of GPUs to perform complex predictive modeling tasks efficiently. This implementation is designed to serve as a comprehensive guide for applying PRBNN in various domains, from simple 1D regression problems to complex, real-world applications like solar energy forecasting.

### Implementation Details

Our PRBNN implementation can be explored in detail here: [LINK_TO_PRBNN_IMPLEMENTATION]. This section of the repository delves into the neural network's architecture, showcasing how probabilistic predictions and uncertainty estimations are achieved at scale.

### Project Structure

- **`1d_regression`**: Includes simulation examples detailed in our research paper, demonstrating PRBNN's application to regression tasks and its effectiveness in uncertainty quantification in 1D data scenarios.

- **`solar_case`**: Focuses on applying PRBNN to forecast solar plant energy output. This part of the repository contains all necessary code, except for the dataset, which is proprietary.

## Prerequisites

- **Python 3.6+**: Ensure Python is installed on your machine. You can download it from [Python's official website](https://www.python.org/downloads/).

- **PyTorch with GPU Support**: This implementation requires [PyTorch](https://pytorch.org/get-started/locally/) with GPU support to run efficiently. Install PyTorch by following the official instructions, ensuring you select the CUDA version compatible with your GPU.

- **Additional Python Packages**: Some additional packages are required to run the examples. These can be installed via pip:
  ```bash
  pip install -r requirements.txt
