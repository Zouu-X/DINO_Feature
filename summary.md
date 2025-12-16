# Makeup Similarity Pipeline Summary

This document outlines the pipeline for extracting facial makeup features and computing similarity between makeup styles. The process involves inputting dataset images, extracting features using a pre-trained DINOv3 model masked by specific facial regions, and finally ranking similar images using Gram Matrix similarity.

## Pipeline Overview

1.  **Data Preparation**: Prepare input face images and corresponding region masks (e.g., cheek, lips).
2.  **Feature Extraction (`DINO_v3.py`)**: Extract localized features from images using DINOv3 and apply spatial masks.
3.  **Similarity Calculation (`makeup_similarity.py`)**: Compute similarity scores between a target image and a database of images using Gram Matrices.

---

## 1. Data Preparation

Before running the feature extraction, the dataset must be prepared with paired images and masks.

*   **Input Images**: RGB images of faces (aligned/cropped is recommended).
*   **Masks**: Grayscale masks corresponding to the specific facial region of interest (e.g., cheeks for blush, lips for lipstick).
    *   **Resolution**: The pipeline expects masks to be processable into tensors. The current script handles resizing, but consistent 256x256 or 512x512 is standard.
    *   **Format**: `L` mode (grayscale), where pixel values indicate the intensity/attention of the mask.

---

## 2. Feature Extraction

**Script:** `DINO_v3.py`

This step uses a specific vision transformer (DINOv3) to extract high-level semantic features from the images.

### Key Components:
*   **Model**: `facebook/dinov3-vits16-pretrain-lvd1689m` (ViT-Small).
*   **Layer Aggregation**: The script extracts hidden states from layers **9, 10, and 11**.
    *   Formula: `feature = layer_11 + layer_9 + layer_10`
*   **Spatial Masking**:
    1.  The transformer output (sequence of patches) is reshaped into a spatial grid (e.g., 14x14).
    2.  The input mask is downsampled to match this spatial grid (14x14).
    3.  **Weighted Feature**: The feature map is element-wise multiplied by the downsampled mask.
        `weighted_features = feature_spatial * masks_down`

### Usage:
The script supports Distributed Data Parallel (DDP) for multi-GPU efficiency.

```bash
torchrun --nproc_per_node=8 DINO_v3.py \
  --token <HF_TOKEN> \
  --input_dir /path/to/images \
  --mask_dir /path/to/masks \
  --output_dir /path/to/save/features \
  --batch_size 32
```

**Output**: `.pt` files containing the weighted feature tensors for each image.

---

## 3. Similarity Calculation

**Script:** `makeup_similarity.py`

This step ranks images from the database based on how similar their makeup style is to a target image.

### Algorithm:
1.  **Gram Matrix**: For every feature tensor $F$ (shape $C \times H \times W$), the Gram Matrix $G$ is computed to capture texture/style information.
    *   $G = F_{flat} \times F_{flat}^T$ (Shape $C \times C$)
    *   This discards spatial arrangement and focuses on feature correlations (style).
2.  **Cosine Similarity**: The similarity between Target ($T$) and Candidate ($C$) is calculated using the Cosine Similarity of their flattened Gram Matrices.
    *   $Score = \text{CosineSimilarity}(G_T.flatten(), G_C.flatten())$

### Usage:
The script also supports DDP to parallelize the search over a large database.

```bash
torchrun --nproc_per_node=8 makeup_similarity.py \
  --target_feature /path/to/target.pt \
  --feature_dir /path/to/database_features \
  --top_k 5
```

**Output**: A list of the top `K` filenames and their similarity scores.
