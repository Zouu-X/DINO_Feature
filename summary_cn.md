# 妆容相似度流程总结

本文档概述了提取面部妆容特征并计算妆容风格相似度的流程。该过程包括输入数据集图像，使用特定面部区域掩码（mask）的预训练 DINOv3 模型提取特征，最后使用 Gram 矩阵相似度对相似图像进行排序。

## 流程概览

1.  **数据准备**：准备输入的人脸图像和相应的区域掩码（例如：脸颊、嘴唇）。
2.  **特征提取 (`DINO_v3.py`)**：使用 DINOv3 从图像中提取局部特征并应用空间掩码。
3.  **相似度计算 (`makeup_similarity.py`)**：使用 Gram 矩阵计算目标图像与数据库图像之间的相似度得分。

---

## 1. 数据准备

在运行特征提取之前，必须准备好成对的图像和掩码数据集。

*   **输入图像**：人脸的 RGB 图像（建议进行对齐/裁剪）。
*   **掩码 (Masks)**：对应特定面部区域（如腮红对应的脸颊，口红对应的嘴唇）的灰度掩码。
    *   **Mask获取**：在face-parsing.PyTorch中的`new_infer.py`脚本里执行
    *   **分辨率**：流程期望掩码可以处理为张量。目前的脚本可以处理缩放，但建议保持一致的 256x256 或 512x512 分辨率。
    *   **格式**：`L` 模式（灰度），像素值表示掩码的强度/关注度。

---

## 2. 特征提取

**脚本：** `DINO_v3.py`

此步骤使用特定的视觉 Transformer (DINOv3) 从图像中提取高级语义特征。

### 关键组件：
*   **模型**：`facebook/dinov3-vits16-pretrain-lvd1689m` (ViT-Small)。(**注意**：该模型对Transformers库的版本需求为`4.57.3`及以上,至撰写本文档时间（2025年12月15日），pip install安装Transformers库版本仍停留在4.41.2，需要手动升级`pip install --upgrade transformers`，更详细请参考官方HuggingFace页面的QA)
*   **层聚合**：脚本提取 **特定层级** 的隐藏状态，根据妆容关注点不同，选择低层（偏向颜色）或高层（偏向结构）的特征聚合`feature`。
    *   **眼妆**：主要关注结构，因此采用高层聚合，公式：`feature = layer_11 + layer_9 + layer_10`
    *   **口妆/腮红**：主要关注颜色纹理，因此采用低层聚合，公式：`feature = layer_3 + layer_4 + layer_5`
*   **空间掩码处理**：
    1.  Transformer 输出（Patch 序列）被重塑为空间网格（Patch size: 14x14）。
    2.  输入掩码被**下采样**以匹配此空间网格 (14x14)。
    3.  **加权特征**：特征图与下采样后的掩码进行逐元素相乘,得到Mask部分的特征结果。
        `weighted_features = feature_spatial * masks_down`

### 用法：
```bash
python DINO_v3.py \
  --token <HF_TOKEN> \
  --input_dir /path/to/images \
  --mask_dir /path/to/masks \
  --output_dir /path/to/save/features \
  --batch_size 32
```
该脚本支持分布式数据并行 (DDP) 以实现多 GPU 高效处理。

```bash
torchrun --nproc_per_node=8 DINO_v3.py \
  --token <HF_TOKEN> \
  --input_dir /path/to/images \
  --mask_dir /path/to/masks \
  --output_dir /path/to/save/features \
  --batch_size 32
```

**输出**：包含每张图像加权特征张量的 `.pt` 文件。

---

## 3. 相似度计算

**脚本：** `makeup_similarity.py`

此步骤根据妆容风格与目标图像的相似程度对数据库中的图像进行排序。

### 算法：
1.  **Gram 矩阵**：对于每个特征张量 $F$（形状 $C \times H \times W$），计算 Gram 矩阵 $G$ 以捕获纹理/风格信息。
    *   $G = F_{flat} \times F_{flat}^T$（形状 $C \times C$）
    *   这一步舍弃了空间排列信息，专注于特征的相关性（风格）。
2.  **余弦相似度**：目标 ($T$) 和候选 ($C$) 之间的相似度是使用其扁平化 Gram 矩阵的余弦相似度计算的。
    *   $Score = \text{CosineSimilarity}(G_T.flatten(), G_C.flatten())$

### 用法：
```bash
python makeup_similarity.py \
  --target_feature /path/to/target.pt \
  --feature_dir /path/to/database_features \
  --top_k 5
```
该脚本也支持 DDP，以便在大规模数据库上并行搜索。

```bash
torchrun --nproc_per_node=8 makeup_similarity.py \
  --target_feature /path/to/target.pt \
  --feature_dir /path/to/database_features \
  --top_k 5
```

**输出**：前 `K` 个文件名及其相似度得分的列表。
