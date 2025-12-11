import torch
import torch.nn.functional as F
import torch.distributed as dist
import argparse
import os
import sys
from tqdm import tqdm

def setup_ddp():
    if "LOCAL_RANK" not in os.environ:
        # Fallback for single GPU/CPU run or if not run with torchrun
        os.environ["LOCAL_RANK"] = "0"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, torch.device(f"cuda:{local_rank}")

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def compute_gram_matrix(feature):
    """
    Compute the Gram Matrix for a given feature tensor.
    Input shape: (C, H, W)
    Output shape: (C, C)
    """
    if feature.dim() == 4:
        feature = feature.squeeze(0)
    
    C, H, W = feature.shape
    N = H * W
    
    # Reshape to (C, N)
    feature_flat = feature.view(C, N)
    
    # Compute Gram Matrix: G = F @ F.T
    gram = torch.matmul(feature_flat, feature_flat.t())
    
    # Normalize
    gram = gram / N
    
    return gram

def main():
    parser = argparse.ArgumentParser(description="Find top similar makeups using Gram Matrix similarity.")
    parser.add_argument("--target_feature", type=str, required=True, help="Path to the target feature file (.pt)")
    parser.add_argument("--feature_dir", type=str, required=True, help="Directory containing database of feature files (.pt)")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top results to return")
    args = parser.parse_args()

    # Setup DDP
    rank, world_size, device = setup_ddp()

    # Load target feature (all ranks need this)
    if not os.path.exists(args.target_feature):
        if rank == 0:
            print(f"Target file not found: {args.target_feature}")
        cleanup_ddp()
        sys.exit(1)
    
    if rank == 0:
        print(f"Loading target feature from: {args.target_feature}")
    
    # Load to CPU first, then move to device if needed, but Gram calculation is fast enough on CPU usually?
    # Better to do on GPU for speed and because user asked to use GPUs.
    target_feature = torch.load(args.target_feature, map_location=device)
    target_gram = compute_gram_matrix(target_feature)
    target_gram_flat = target_gram.flatten().unsqueeze(0) # (1, C*C)

    # Get file list (Rank 0 reads and broadcasts, or all read and sort)
    # Safest is all read and sort to ensure consistency
    files = sorted([f for f in os.listdir(args.feature_dir) if f.endswith('.pt')])
    
    # Shard files
    my_files = files[rank::world_size]
    
    if rank == 0:
        print(f"Scanning {len(files)} features using {world_size} GPUs.")
    
    local_results = []
    
    # Process local shard
    # Use tqdm only on rank 0 usually to avoid log spam, or minimal on others
    iterator = tqdm(my_files, desc=f"Rank {rank}") if rank == 0 else my_files
    
    for fname in iterator:
        fpath = os.path.join(args.feature_dir, fname)
        
        if os.path.abspath(fpath) == os.path.abspath(args.target_feature):
            continue
            
        try:
            cand_feature = torch.load(fpath, map_location=device)
            cand_gram = compute_gram_matrix(cand_feature)
            cand_gram_flat = cand_gram.flatten().unsqueeze(0)
            
            similarity = F.cosine_similarity(target_gram_flat, cand_gram_flat).item()
            local_results.append((fname, similarity))
            
        except Exception as e:
            # print(f"Rank {rank} Error processing {fname}: {e}")
            pass

    # Gather results
    if world_size > 1:
        # We need to gather lists of tuples. dist.all_gather_object is simplest for this.
        all_results_lists = [None for _ in range(world_size)]
        dist.all_gather_object(all_results_lists, local_results)
    else:
        all_results_lists = [local_results]
    
    if rank == 0:
        # Flatten list of lists
        final_results = []
        for res_list in all_results_lists:
            final_results.extend(res_list)
            
        # Sort
        final_results.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nTop {args.top_k} Similar Makeups:")
        print(f"{'Rank':<5} {'Filename':<40} {'Similarity':<10}")
        print("-" * 60)
        
        for i in range(min(args.top_k, len(final_results))):
            fname, score = final_results[i]
            print(f"{i+1:<5} {fname:<40} {score:.4f}")

    cleanup_ddp()

if __name__ == "__main__":
    main()
