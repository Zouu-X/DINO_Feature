import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
from huggingface_hub import login

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image(url)

pretrained_model = "facebook/dinov3-vitb16-pretrain-lvd1689m"
processor = AutoImageProcessor.from_pretrained(pretrained_model)
model = AutoModel.from_pretrained(
    pretrained_model,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

inputs = processor(images=image, return_tensors='pt').to(model.device)
with torch.inference_mode():
    outputs = model(**inputs)
config = model.config


last_hidden_state = outputs.last_hidden_state
patches_flat = last_hidden_state[:, 1+config.num_register_tokens:, :]
print("Patches", patches_flat.shape)

B, N, D = patches_flat.shape #Batch, Number of patches, Dimension
H_grid = W_grid = int(N**0.5) # sqrt196 = 14

spatial_map = patches_flat.reshape(B, H_grid, W_grid, D)
spatial_map = spatial_map.permute(0, 3, 1, 2)
print("Spatial map", spatial_map.shape)
