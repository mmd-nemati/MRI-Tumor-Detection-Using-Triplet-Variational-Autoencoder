
# MRI Tumor Detection Using Triplet Variational Autoencoder

This repository contains the implementation of a **Triplet Variational Autoencoder (Tri-VAE)** designed to detect anomalies in brain MRI scans. The method is inspired by the CVPR 2024 paper: *"[Triplet Variational Autoencoder for Unsupervised Anomaly Detection in Brain MRI](https://openaccess.thecvf.com/content/CVPR2024W/VAND/papers/Wijanarko_Tri-VAE_Triplet_Variational_Autoencoder_for_Unsupervised_Anomaly_Detection_in_Brain_CVPRW_2024_paper.pdf)"*.

Tri-VAE is trained to model the distribution of healthy brain MRIs and detect abnormal regions, such as tumors, by analyzing reconstruction errors. It combines variational inference with metric learning through a triplet loss to enhance the model's discriminative capacity for anomaly localization.

---

## Table of Contents

* [Overview](#overview)
* [Model Architecture](#model-architecture)
* [Dataset](#dataset)
* [Implementation](#implementation)
* [Evaluation](#evaluation)
* [Visual Results](#visual-results)
* [Limitations and Future Work](#limitations-and-future-work)


## Overview

The Tri-VAE approach learns a compact latent space representation of healthy brain MRIs. It utilizes:

* **Anchor**: A clean, healthy MRI slice.
* **Positive**: Another healthy slice (noise-free).
* **Negative**: A healthy slice with added coarse noise or Simplex Noise.

By minimizing the distance between anchor and positive while maximizing the distance to the negative, the model learns to distinguish structural anomalies.

**Core Components:**

* Coarse and full-scale reconstruction branches
* Triplet loss
* L1 reconstruction loss
* KL divergence loss
* Structural Similarity Index (SSIM) loss

The trained model is evaluated on the BraTS dataset by comparing reconstruction errors. Anomalies (tumor regions) are expected to yield higher reconstruction errors.


## Model Architecture

The Tri-VAE architecture includes the following components:

* Encoder
* Coarse Decoder
* Full Decoder
* Triplet loss branch
* SSIM evaluation module



## Dataset

### Training Data

* **IXI Dataset**: Used for training on healthy MRI slices.

<img src="https://github.com/mmd-nemati/MRI-Tumor-Detection-Using-Triplet-Variational-Autoencoder/blob/main/assets/ixi.png" width="520">

### Testing Data

* **BraTS Dataset**: Used for evaluating tumor detection.

<img src="https://github.com/mmd-nemati/MRI-Tumor-Detection-Using-Triplet-Variational-Autoencoder/blob/main/assets/bra.png" width="520">

Data preprocessing includes:

* Skull stripping
* Normalization
* Resizing MRI slices



## Implementation

The model is implemented using **PyTorch** and structured as follows:

### 1. **Triplet Setup**

Each input is grouped as:

* **Anchor (A)**: A healthy slice from IXI
* **Positive (P)**: Another healthy slice without noise
* **Negative (N)**: A healthy slice with added artificial noise

These are passed through a **shared encoder** to produce latent embeddings, and reconstruction is carried out at two scales:

* **Coarse Decoder**: Provides low-resolution reconstruction
* **Full Decoder**: Produces detailed output from coarse output + latent features


### 2. **Loss Functions**

The training is guided by a combination of:

* **Coarse Reconstruction L1 Loss for All Images** – Pixel-wise reconstruction accuracy for all images
* **Full Reconstruction L1 Loss for Negative Images** – Pixel-wise reconstruction accuracy for negative samples
* **KL Divergence** – Regularization for the VAE
* **Triplet Loss** – Metric learning signal for embedding separation
* **SSIM Loss** – Structural similarity preservation

These losses are balanced to enforce latent structure learning and faithful reconstructions.

## Evaluation

Evaluation focuses on anomaly detection performance:

1. **Reconstruction Error Maps**: Input vs. reconstructed MRI slices are subtracted to produce error maps.
2. **Dice Score**: Computed against ground truth tumor masks (BraTS) to evaluate detection accuracy.
3. **Visualizations**: Tumor regions typically appear as high-error areas in the residuals.


## Visual Results

Below are examples of anomaly localization via reconstruction error:

Normal VAE:

<img src="https://github.com/mmd-nemati/MRI-Tumor-Detection-Using-Triplet-Variational-Autoencoder/blob/main/assets/vae.png" width="520">

Triplet VAE:

<img src="https://github.com/mmd-nemati/MRI-Tumor-Detection-Using-Triplet-Variational-Autoencoder/blob/main/assets/tri_vae.png" width="520">


## Limitations and Future Work

* The model is currently limited to 2D slice-based analysis.
* No domain adaptation is performed between IXI and BraTS.
* Future extensions could include attention mechanisms, 3D volumetric models, and unsupervised domain adaptation.
