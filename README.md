# torch-ddp

This project is a step-by-step implementation of PyTorch's DistributedDataParallel (DDP), built from scratch. It serves as a learning exercise to understand the core concepts of distributed training in PyTorch.

## Overview

The project is broken down into four main scripts, each representing a stage in the development of a custom DDP implementation:

*   **`ddp1.py`**: A basic, manual implementation of DDP. This script demonstrates the fundamental concepts of broadcasting model parameters and manually reducing gradients across multiple processes.
*   **`ddp2.py`**: Introduces a `DDP` wrapper class that automates the process of broadcasting and gradient reduction using hooks.
*   **`ddp3.py`**: Improves upon the `DDP` wrapper by using asynchronous operations for gradient reduction, overlapping communication with computation.
*   **`ddp4.py`**: The most advanced implementation, which introduces the concept of gradient bucketing to further optimize the communication of gradients.

## Model

The `mlp.py` script defines a simple Multi-Layer Perceptron (MLP) that is used as the model for all DDP implementations.

## Usage

To run any of the DDP scripts, you can use `torchrun`. For example, to run `ddp1.py` with two processes:

```bash
torchrun --nproc_per_node=4 ddp1.py
```