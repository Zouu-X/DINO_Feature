import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
from huggingface_hub import login
import argparse
import os
from PIL import Image
import torch.nn.functional as F

image_path = "/db-mnt/mnt/efs-mount/home/xiangxzou/beauty_bank/BMS_subset/images/train/image_200_make it mod makeup_Bronze.png"
mask_path = "/db-mnt/mnt/efs-mount/home/xiangxzou/beauty_bank/BMS_subset/masks/train/mouth/image_200_make it mod makeup_Bronze.png"
image = Image.open(image_path).convert("RGB")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, required=True)
    # parser.add_argument("--input_dir", type=str, required=True)
    # parser.add_argument("--output_dir", type=str, required=True)
    # parser.add_argument("--mask_dir", type=str, required=True)
    return parser.parse_args()

args = parse_args()
print("------")
login(token=args.token)
pretrained_model = "facebook/dinov3-vitb16-pretrain-lvd1689m"
processor = AutoImageProcessor.from_pretrained(pretrained_model)
model = AutoModel.from_pretrained(
    pretrained_model,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
# 图像处理：Processor turns img into 244*244. No center crop
inputs = processor(images=image, return_tensors='pt').to(model.device)

with torch.inference_mode():
    outputs = model(**inputs, output_hidden_states=True)
config = model.config
hidden_states = outputs.hidden_states
print(f"Total layers captured: {len(hidden_states)}")

layer_3 = hidden_states[3]
layer_last = hidden_states[-1]

feat_map_3 = layer_3[:, 1+config.num_register_tokens:, :]
feat_map_last = layer_last[:, 1+config.num_register_tokens:, :]
feature = feat_map_3 + feat_map_last
print("Feature shape: ", feature.shape)

last_hidden_state = outputs.last_hidden_state
patches_flat = last_hidden_state[:, 1+config.num_register_tokens:, :]

B, N, D = patches_flat.shape #Batch, Number of patches, Dimension
H_grid = W_grid = int(N**0.5) # RES= sqrt196 = 14

spatial_map = patches_flat.reshape(B, H_grid, W_grid, D)
spatial_map = spatial_map.permute(0, 3, 1, 2)



# Mask downsample
def mask_down(mask_path):
    mask = Image.open(mask_path).convert("L") #256*256
    print("Mask shape:", mask.size)

mask_down(mask_path=mask_path)