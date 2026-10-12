"""
build_index.py
Build k-means centroids on document token embeddings and assign each token to a centroid.
Outputs:
  outputs/centroids.pt -> centroids (K, dim)
  outputs/codes.pt     -> codes (N_tokens,) long
"""
import argparse
from pathlib import Path

import torch
from sklearn.cluster import KMeans
from tqdm import tqdm


def sample_embeddings(D, n_samples=200000, seed=42):
    """Sample a subset of token embeddings for k-means training."""
    N = D.shape[0]
    if N <= n_samples:
        return D
    rng = torch.Generator().manual_seed(seed)
    idx = torch.randperm(N, generator=rng)[:n_samples]
    return D[idx]


def build_centroids(D, K=8192, n_samples=200000, seed=42, verbose=True):
    """Run k-means on sampled token embeddings, then L2-normalize centroids."""
    sample = sample_embeddings(D, n_samples, seed)
    print(f'K-means sample: {sample.shape} -> K={K}')
    sample_np = sample.numpy()
    kmeans = KMeans(n_clusters=K, random_state=seed, n_init=1, max_iter=50, verbose=verbose)
    kmeans.fit(sample_np)
    centroids = torch.from_numpy(kmeans.cluster_centers_).float()
    centroids = torch.nn.functional.normalize(centroids, p=2, dim=-1)
    return centroids


def assign_codes(D, centroids, batch_size=65536):
    """Assign each token embedding to nearest centroid via max inner product.
    Both D and centroids are assumed L2-normalized (cosine similarity).
    """
    codes = []
    for i in tqdm(range(0, D.shape[0], batch_size), desc='Assigning codes'):
        batch = D[i:i+batch_size]
        scores = batch @ centroids.T  # (batch, K)
        codes.append(scores.argmax(dim=1).cpu())
    return torch.cat(codes, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_path', default='outputs/doc_embeddings.pt')
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--K', type=int, default=8192)
    parser.add_argument('--n_samples', type=int, default=200000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    data = torch.load(args.emb_path, map_location='cpu')
    D = data['D']

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    centroids = build_centroids(D, K=args.K, n_samples=args.n_samples, seed=args.seed)
    torch.save(centroids, out_dir / 'centroids.pt')
    print(f'Saved centroids: {centroids.shape}')

    codes = assign_codes(D, centroids)
    torch.save(codes, out_dir / 'codes.pt')
    print(f'Saved codes: {codes.shape}')


if __name__ == '__main__':
    main()
