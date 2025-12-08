import torch
import torch.nn.functional as F
import argparse
import os
from tqdm import tqdm

def compute_gram_matrix(feature):
    """
    Compute the Gram Matrix for a given feature tensor.
    Input shape: (C, H, W)
    Output shape: (C, C)
    """
    if feature.dim() == 4:
        # If input is (1, C, H, W) or (B, C, H, W), handle accordingly.
        # Assuming single feature (C, H, W) or (1, C, H, W)
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

    # Load target feature
    if not os.path.exists(args.target_feature):
        raise FileNotFoundError(f"Target file not found: {args.target_feature}")
    
    print(f"Loading target feature from: {args.target_feature}")
    target_feature = torch.load(args.target_feature, map_location='cpu')
    target_gram = compute_gram_matrix(target_feature)
    # Flatten Gram Matrix for cosine similarity
    target_gram_flat = target_gram.flatten().unsqueeze(0) # (1, C*C)

    results = []
    
    print(f"Scanning features in: {args.feature_dir}")
    files = [f for f in os.listdir(args.feature_dir) if f.endswith('.pt')]
    
    for fname in tqdm(files):
        fpath = os.path.join(args.feature_dir, fname)
        
        # Skip if it's the target itself (optional, but good practice)
        if os.path.abspath(fpath) == os.path.abspath(args.target_feature):
            continue
            
        try:
            cand_feature = torch.load(fpath, map_location='cpu')
            cand_gram = compute_gram_matrix(cand_feature)
            cand_gram_flat = cand_gram.flatten().unsqueeze(0) # (1, C*C)
            
            # Compute Cosine Similarity
            # Cosine Sim ranges [-1, 1]. Higher is more similar.
            similarity = F.cosine_similarity(target_gram_flat, cand_gram_flat).item()
            
            results.append((fname, similarity))
            
        except Exception as e:
            print(f"Error processing {fname}: {e}")

    # Sort by similarity (descending)
    results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop {args.top_k} Similar Makeups:")
    print(f"{'Rank':<5} {'Filename':<40} {'Similarity':<10}")
    print("-" * 60)
    
    for i in range(min(args.top_k, len(results))):
        fname, score = results[i]
        print(f"{i+1:<5} {fname:<40} {score:.4f}")

if __name__ == "__main__":
    main()
