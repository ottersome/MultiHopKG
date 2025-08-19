#!/usr/bin/env python3
"""
Script to analyze distances and norms between embedding vectors.

This script loads embeddings from a numpy file and computes comprehensive
statistics on pairwise distances and vector norms.
"""

import argparse
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')


def load_embeddings(embedding_path: str) -> np.ndarray:
    """Load embeddings from numpy file."""
    print(f"Loading embeddings from: {embedding_path}")
    embeddings = np.load(embedding_path)
    print(f"Loaded embeddings shape: {embeddings.shape}")
    return embeddings


def compute_norms(embeddings: np.ndarray) -> Dict[str, Any]:
    """Compute various norms for each embedding vector."""
    print("Computing vector norms...")
    
    # L1 norm (Manhattan)
    l1_norms = np.linalg.norm(embeddings, ord=1, axis=1)
    
    # L2 norm (Euclidean)
    l2_norms = np.linalg.norm(embeddings, ord=2, axis=1)
    
    # L-infinity norm (Maximum)
    linf_norms = np.linalg.norm(embeddings, ord=np.inf, axis=1)
    
    # Frobenius norm (same as L2 for vectors)
    frob_norms = np.linalg.norm(embeddings, ord='fro', axis=1)
    
    norm_stats = {
        'l1': {
            'values': l1_norms,
            'mean': np.mean(l1_norms),
            'std': np.std(l1_norms),
            'min': np.min(l1_norms),
            'max': np.max(l1_norms),
            'median': np.median(l1_norms),
            'q25': np.percentile(l1_norms, 25),
            'q75': np.percentile(l1_norms, 75),
            'skewness': stats.skew(l1_norms),
            'kurtosis': stats.kurtosis(l1_norms)
        },
        'l2': {
            'values': l2_norms,
            'mean': np.mean(l2_norms),
            'std': np.std(l2_norms),
            'min': np.min(l2_norms),
            'max': np.max(l2_norms),
            'median': np.median(l2_norms),
            'q25': np.percentile(l2_norms, 25),
            'q75': np.percentile(l2_norms, 75),
            'skewness': stats.skew(l2_norms),
            'kurtosis': stats.kurtosis(l2_norms)
        },
        'linf': {
            'values': linf_norms,
            'mean': np.mean(linf_norms),
            'std': np.std(linf_norms),
            'min': np.min(linf_norms),
            'max': np.max(linf_norms),
            'median': np.median(linf_norms),
            'q25': np.percentile(linf_norms, 25),
            'q75': np.percentile(linf_norms, 75),
            'skewness': stats.skew(linf_norms),
            'kurtosis': stats.kurtosis(linf_norms)
        }
    }
    
    return norm_stats


def compute_pairwise_distances(embeddings: np.ndarray, max_samples: int = 10000) -> Dict[str, Any]:
    """Compute pairwise distances between embeddings."""
    n_embeddings = embeddings.shape[0]
    
    # Sample embeddings if too many to avoid memory issues
    if n_embeddings > max_samples:
        print(f"Sampling {max_samples} embeddings from {n_embeddings} for pairwise distance computation")
        indices = np.random.choice(n_embeddings, max_samples, replace=False)
        sample_embeddings = embeddings[indices]
    else:
        sample_embeddings = embeddings
        
    print(f"Computing pairwise distances for {sample_embeddings.shape[0]} embeddings...")
    
    # Compute different distance metrics
    distance_metrics = ['euclidean', 'manhattan', 'cosine', 'chebyshev']
    distance_stats = {}
    
    for metric in distance_metrics:
        print(f"  Computing {metric} distances...")
        distances = pdist(sample_embeddings, metric=metric)
        
        distance_stats[metric] = {
            'values': distances,
            'mean': np.mean(distances),
            'std': np.std(distances),
            'min': np.min(distances),
            'max': np.max(distances),
            'median': np.median(distances),
            'q25': np.percentile(distances, 25),
            'q75': np.percentile(distances, 75),
            'skewness': stats.skew(distances),
            'kurtosis': stats.kurtosis(distances),
            'n_pairs': len(distances)
        }
    
    return distance_stats


def compute_nearest_neighbor_stats(embeddings: np.ndarray, k: int = 5, max_samples: int = 5000) -> Dict[str, Any]:
    """Compute k-nearest neighbor statistics."""
    n_embeddings = embeddings.shape[0]
    
    if n_embeddings > max_samples:
        print(f"Sampling {max_samples} embeddings for k-NN analysis")
        indices = np.random.choice(n_embeddings, max_samples, replace=False)
        sample_embeddings = embeddings[indices]
    else:
        sample_embeddings = embeddings
    
    print(f"Computing {k}-nearest neighbor statistics...")
    
    # Compute pairwise distances
    distances = squareform(pdist(sample_embeddings, metric='euclidean'))
    
    # For each point, find k nearest neighbors (excluding itself)
    knn_distances = []
    for i in range(len(sample_embeddings)):
        # Sort distances and take k+1 (excluding self at index 0)
        sorted_distances = np.sort(distances[i])
        knn_distances.append(sorted_distances[1:k+1])  # Exclude self-distance (0)
    
    knn_distances = np.array(knn_distances)
    
    knn_stats = {}
    for j in range(k):
        neighbor_distances = knn_distances[:, j]
        knn_stats[f'{j+1}_nearest'] = {
            'mean': np.mean(neighbor_distances),
            'std': np.std(neighbor_distances),
            'min': np.min(neighbor_distances),
            'max': np.max(neighbor_distances),
            'median': np.median(neighbor_distances)
        }
    
    # Average k-NN distance
    avg_knn_distances = np.mean(knn_distances, axis=1)
    knn_stats['avg_knn'] = {
        'mean': np.mean(avg_knn_distances),
        'std': np.std(avg_knn_distances),
        'min': np.min(avg_knn_distances),
        'max': np.max(avg_knn_distances),
        'median': np.median(avg_knn_distances)
    }
    
    return knn_stats


def compute_embedding_statistics(embeddings: np.ndarray) -> Dict[str, Any]:
    """Compute general statistics about the embedding space."""
    print("Computing embedding space statistics...")
    
    # Dimension-wise statistics
    dim_means = np.mean(embeddings, axis=0)
    dim_stds = np.std(embeddings, axis=0)
    dim_mins = np.min(embeddings, axis=0)
    dim_maxs = np.max(embeddings, axis=0)
    
    # Overall statistics
    overall_mean = np.mean(embeddings)
    overall_std = np.std(embeddings)
    overall_min = np.min(embeddings)
    overall_max = np.max(embeddings)
    
    # Sparsity (percentage of near-zero values)
    threshold = 1e-6
    sparsity = np.mean(np.abs(embeddings) < threshold)
    
    embedding_stats = {
        'shape': embeddings.shape,
        'overall': {
            'mean': overall_mean,
            'std': overall_std,
            'min': overall_min,
            'max': overall_max,
            'sparsity': sparsity
        },
        'dimension_wise': {
            'mean_of_means': np.mean(dim_means),
            'std_of_means': np.std(dim_means),
            'mean_of_stds': np.mean(dim_stds),
            'std_of_stds': np.std(dim_stds),
            'mean_range': np.mean(dim_maxs - dim_mins),
            'std_range': np.std(dim_maxs - dim_mins)
        }
    }
    
    return embedding_stats


def create_visualizations(norm_stats: Dict, distance_stats: Dict, output_dir: str):
    """Create visualization plots."""
    print("Creating visualizations...")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # Plot norm distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Embedding Norm Distributions', fontsize=16)
    
    norm_types = ['l1', 'l2', 'linf']
    for i, norm_type in enumerate(norm_types):
        ax = axes[i//2, i%2]
        values = norm_stats[norm_type]['values']
        ax.hist(values, bins=50, alpha=0.7, density=True)
        ax.set_title(f'{norm_type.upper()} Norm Distribution')
        ax.set_xlabel(f'{norm_type.upper()} Norm')
        ax.set_ylabel('Density')
        ax.axvline(norm_stats[norm_type]['mean'], color='red', linestyle='--', 
                  label=f"Mean: {norm_stats[norm_type]['mean']:.3f}")
        ax.legend()
    
    # Remove empty subplot
    axes[1, 1].remove()
    
    plt.tight_layout()
    plt.savefig(output_path / 'norm_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot distance distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Pairwise Distance Distributions', fontsize=16)
    
    distance_types = ['euclidean', 'manhattan', 'cosine', 'chebyshev']
    for i, dist_type in enumerate(distance_types):
        ax = axes[i//2, i%2]
        values = distance_stats[dist_type]['values']
        # Sample for plotting if too many points
        if len(values) > 10000:
            sample_values = np.random.choice(values, 10000, replace=False)
        else:
            sample_values = values
            
        ax.hist(sample_values, bins=50, alpha=0.7, density=True)
        ax.set_title(f'{dist_type.capitalize()} Distance Distribution')
        ax.set_xlabel(f'{dist_type.capitalize()} Distance')
        ax.set_ylabel('Density')
        ax.axvline(distance_stats[dist_type]['mean'], color='red', linestyle='--',
                  label=f"Mean: {distance_stats[dist_type]['mean']:.3f}")
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / 'distance_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_results(results: Dict, output_path: str):
    """Save results to JSON file."""
    print(f"Saving results to: {output_path}")
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    # Remove large arrays to keep JSON file manageable
    results_copy = convert_numpy(results)
    for norm_type in results_copy.get('norms', {}):
        if 'values' in results_copy['norms'][norm_type]:
            del results_copy['norms'][norm_type]['values']
    
    for dist_type in results_copy.get('distances', {}):
        if 'values' in results_copy['distances'][dist_type]:
            del results_copy['distances'][dist_type]['values']
    
    with open(output_path, 'w') as f:
        json.dump(results_copy, f, indent=2)


def print_summary(results: Dict):
    """Print a summary of the analysis."""
    print("\n" + "="*80)
    print("EMBEDDING ANALYSIS SUMMARY")
    print("="*80)
    
    # Embedding info
    shape = results['embedding_stats']['shape']
    print(f"Embedding Shape: {shape[0]} entities × {shape[1]} dimensions")
    
    # Overall statistics
    overall = results['embedding_stats']['overall']
    print(f"\nOverall Statistics:")
    print(f"  Mean: {overall['mean']:.6f}")
    print(f"  Std:  {overall['std']:.6f}")
    print(f"  Min:  {overall['min']:.6f}")
    print(f"  Max:  {overall['max']:.6f}")
    print(f"  Sparsity: {overall['sparsity']:.2%}")
    
    # Norm statistics
    print(f"\nNorm Statistics:")
    for norm_type in ['l1', 'l2', 'linf']:
        stats = results['norms'][norm_type]
        print(f"  {norm_type.upper()} Norm - Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
    
    # Distance statistics
    print(f"\nPairwise Distance Statistics:")
    for dist_type in ['euclidean', 'manhattan', 'cosine', 'chebyshev']:
        stats = results['distances'][dist_type]
        print(f"  {dist_type.capitalize()} - Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
    
    # k-NN statistics
    if 'knn_stats' in results:
        print(f"\nk-Nearest Neighbor Statistics:")
        knn = results['knn_stats']
        print(f"  1st NN distance - Mean: {knn['1_nearest']['mean']:.3f}")
        print(f"  Average k-NN distance - Mean: {knn['avg_knn']['mean']:.3f}")


def main():
    parser = argparse.ArgumentParser(description='Analyze embedding distances and statistics')
    parser.add_argument('embedding_path', type=str, help='Path to numpy embedding file', default="./models/graph_embeddings/transE_mquake_dim500/entity_embedding.npy")
    parser.add_argument('--output_dir', type=str, default='embedding_analysis/', 
                       help='Output directory for results')
    parser.add_argument('--max_samples', type=int, default=10000,
                       help='Maximum samples for pairwise distance computation')
    parser.add_argument('--k_neighbors', type=int, default=5,
                       help='Number of nearest neighbors to analyze')
    parser.add_argument('--no_plots', action='store_true',
                       help='Skip generating plots')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load embeddings
    embeddings = load_embeddings(args.embedding_path)
    
    # Compute all statistics
    results = {}
    
    # Basic embedding statistics
    results['embedding_stats'] = compute_embedding_statistics(embeddings)
    
    # Norm statistics
    results['norms'] = compute_norms(embeddings)
    
    # Pairwise distance statistics
    results['distances'] = compute_pairwise_distances(embeddings, args.max_samples)
    
    # k-NN statistics
    results['knn_stats'] = compute_nearest_neighbor_stats(embeddings, args.k_neighbors, args.max_samples)
    
    # Create visualizations
    if not args.no_plots:
        create_visualizations(results['norms'], results['distances'], args.output_dir)
    
    # Save results
    save_results(results, output_dir / 'analysis_results.json')
    
    # Print summary
    print_summary(results)
    
    print(f"\nAnalysis complete! Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
