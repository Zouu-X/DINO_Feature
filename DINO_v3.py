import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoImageProcessor, AutoModel
from huggingface_hub import login
import argparse
import os
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm

def setup_ddp():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    dist.destroy_process_group()

class MakeupDataset(Dataset):
    def __init__(self, input_dir, mask_dir, processor):
        self.input_dir = input_dir
        self.mask_dir = mask_dir
        self.processor = processor
        self.image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        # Filter to ensure corresponding masks exist
        self.valid_files = []
        for f in self.image_files:
            if os.path.exists(os.path.join(mask_dir, f)):
                self.valid_files.append(f)
            else:
                print(f"Warning: Mask not found for {f}, skipping.")
        
    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        filename = self.valid_files[idx]
        image_path = os.path.join(self.input_dir, filename)
        mask_path = os.path.join(self.mask_dir, filename)

        image = Image.open(image_path).convert("RGB") # 512x512
        mask = Image.open(mask_path).convert("L")     # 256x256

        # Processor handles resizing and normalization for the model
        # DINOv3/ViT usually expects 224x224
        inputs = self.processor(images=image, return_tensors='pt')
        pixel_values = inputs['pixel_values'].squeeze(0)

        # Convert mask to tensor
        mask_tensor = torch.from_numpy(np.array(mask)).float() / 255.0
        mask_tensor = mask_tensor.unsqueeze(0) # (1, H, W)

        return pixel_values, mask_tensor, filename

import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, required=True, help="HuggingFace token")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing makeup face images")
    parser.add_argument("--mask_dir", type=str, required=True, help="Directory containing mask images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save features")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    args = parser.parse_args()

    setup_ddp()
    local_rank = int(os.environ["LOCAL_RANK"])
    
    if local_rank == 0:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)
        print("------ DINOv3 Feature Extraction ------")
        login(token=args.token)

    # Ensure login happens on all processes or just main? 
    # Usually better to do it on all if they need to download, but here we assume cached or do it once.
    # To be safe, let's just let all do it or assume pre-downloaded. 
    # If we run on multiple nodes, each node needs it. 
    # If we run on one node with multiple GPUs, one login is enough if home dir is shared.
    # We'll keep it simple.
    
    pretrained_model = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    processor = AutoImageProcessor.from_pretrained(pretrained_model)
    model = AutoModel.from_pretrained(
        pretrained_model,
        torch_dtype=torch.bfloat16,
        # device_map="auto" # Do not use device_map="auto" with DDP
    )
    
    model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    model.eval()

    dataset = MakeupDataset(args.input_dir, args.mask_dir, processor)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        sampler=sampler, 
        num_workers=args.num_workers, 
        pin_memory=True
    )

    config = model.module.config if hasattr(model, 'module') else model.config

    if local_rank == 0:
        print(f"Processing {len(dataset)} images on {dist.get_world_size()} GPUs...")

    for pixel_values, masks, filenames in tqdm(dataloader, disable=(local_rank != 0)):
        pixel_values = pixel_values.to(local_rank, dtype=torch.bfloat16)
        masks = masks.to(local_rank, dtype=torch.bfloat16)

        with torch.no_grad():
            outputs = model(pixel_values, output_hidden_states=True)
        
        hidden_states = outputs.hidden_states
        # layer_3 = hidden_states[3]
        # layer_last = hidden_states[-1]
        
        # Note: hidden_states is a tuple.
        # We need to be careful about the index if the model output format changes, 
        # but usually it includes embeddings at 0.
        # The original code used [3] and [-1].
        
        layer_3 = hidden_states[3]
        layer_last = hidden_states[-1]

        # Extract patches, skipping register tokens
        # Original code: feat_map_3 = layer_3[:, 1+config.num_register_tokens:, :]
        # Check if num_register_tokens exists in config, otherwise default to 0 or check model type.
        # DINOv3 might have register tokens.
        
        num_registers = getattr(config, "num_register_tokens", 0)
        # Also need to skip CLS token (index 0)
        # The original code: layer_3[:, 1+config.num_register_tokens:, :]
        # This implies index 0 is CLS, then registers, then patches.
        
        feat_map_3 = layer_3[:, 1+num_registers:, :]
        feat_map_last = layer_last[:, 1+num_registers:, :]
        
        feature = feat_map_3 + feat_map_last # (B, N, D)
        
        # Reshape to spatial map
        B, N, D = feature.shape
        H_grid = W_grid = int(N**0.5) # Should be 14 for ViT-B/16 with 224x224 input
        
        # Reshape feature to (B, D, H, W) for easier multiplication with mask
        feature_spatial = feature.transpose(1, 2).reshape(B, D, H_grid, W_grid)
        
        # Downsample mask to 14x14
        # masks is (B, 1, 256, 256) -> resize to (14, 14)
        masks_down = F.interpolate(masks, size=(H_grid, W_grid), mode='nearest')
        
        # Compute weighted features
        # feature_spatial: (B, D, 14, 14)
        # masks_down: (B, 1, 14, 14)
        weighted_features = feature_spatial * masks_down
        
        # Flatten back if needed, or save as is. 
        # "Use the feature and mask to get the weighted features"
        # Usually for similarity we might want a vector, e.g., global average pooling or just the map.
        # The user said "Save the results. I will need a feature sapce to compute cosine similarity later".
        # Keeping it as spatial map or flattened patches is safer. 
        # Let's save the weighted feature map.
        
        # Save results
        for i in range(B):
            fname = filenames[i]
            save_path = os.path.join(args.output_dir, os.path.splitext(fname)[0] + '.pt')
            torch.save(weighted_features[i].cpu(), save_path)

    cleanup_ddp()

if __name__ == "__main__":
    main()