# PRBNN Implementation in PyTorch with GPU Support

This repository contains the implementation of Probabilistic Radial Basis Function Neural Networks (PRBNN) using PyTorch, optimized for performance with GPU support. It showcases PRBNN's application in theoretical simulations and real-world scenarios, demonstrating its predictive capabilities and efficiency in handling uncertainty.

## Overview

The PRBNN framework, implemented in PyTorch, leverages the power of GPUs to perform complex predictive modeling tasks efficiently. This implementation is designed to serve as a comprehensive guide for applying PRBNN in various domains, from simple 1D regression problems to complex, real-world applications like solar energy forecasting.

### Implementation Details

Our PRBNN implementation can be explored in detail here: [(https://arxiv.org/pdf/2210.08608.pdf)]. 

### Project Structure

- **`PRBNN_1d_regression`**: Includes simulation examples detailed in our research paper, demonstrating PRBNN's application to regression tasks and its effectiveness in uncertainty quantification in 1D data scenarios.

- **`PRBNN_solar_case_study`**: Focuses on applying PRBNN to forecast solar plant energy output. This part of the repository contains all necessary code, except for the dataset, which is proprietary.

## Prerequisites

- **PyTorch with GPU Support**: This implementation requires [PyTorch](https://pytorch.org/get-started/locally/) with GPU support to run efficiently. Install PyTorch by following the official instructions, ensuring you select the CUDA version compatible with your GPU.

- **Additional Python Packages**: Some additional packages are required to run the examples. These can be installed via pip:
  ```bash
  pip install -r requirement.txt
