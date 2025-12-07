"""
analyze_universality.py
Quantifying universality: Firing Entropy + Co-Firing Proportion
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from app.usae_ import UniversalSAE
from app.models import device

# Configuration
CONFIG = {
    'activation_dir': './activations',
    'checkpoint_dir': './checkpoints',
    'log_dir': './logs',
    'batch_size': 128,
    'hidden_dim': 2042,
    'k': 64,
}

def collate_fn(batch):
    resnet_batch = torch.stack([b[0] for b in batch])
    vit_batch = torch.stack([b[1] for b in batch])
    return [resnet_batch, vit_batch]

def extract_sparse_codes(usae, dataloader, device):
    """
    Extract sparse codes Z from both models for all samples
    
    Returns:
        z_codes: dict with keys 'model_0', 'model_1'
                 each is [N, hidden_dim] tensor of sparse codes
    """
    usae.eval()
    
    z_codes = {'model_0': [], 'model_1': []}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting sparse codes"):
            batch = [b.to(device) for b in batch]
            
            # Encode from each model
            for model_idx in range(2):
                z = usae.encode(batch[model_idx], model_idx)
                z_codes[f'model_{model_idx}'].append(z.cpu())
    
    # Concatenate
    z_codes['model_0'] = torch.cat(z_codes['model_0'], dim=0)  # [N, hidden_dim]
    z_codes['model_1'] = torch.cat(z_codes['model_1'], dim=0)
    
    return z_codes

def compute_firing_entropy(z_codes, M=2):
    """
    Compute Normalized Firing Entropy for each feature k
    
    FE_k = -1/log(M) * Σ p_k^(i) * log(p_k^(i))
    
    where p_k^(i) = proportion of activations for feature k from model i
    
    Args:
        z_codes: dict with 'model_0', 'model_1' sparse codes [N, hidden_dim]
        M: number of models (2)
    
    Returns:
        firing_entropies: [hidden_dim] array of FE values
        feature_counts: [hidden_dim, M] array of activation counts per model
    """
    hidden_dim = z_codes['model_0'].shape[1]
    
    # Count activations per feature per model
    feature_counts = np.zeros((hidden_dim, M))
    
    for model_idx in range(M):
        z = z_codes[f'model_{model_idx}']
        # Feature k fires when z[:, k] != 0
        activations = (z != 0).float().sum(dim=0).numpy()  # [hidden_dim]
        feature_counts[:, model_idx] = activations
    
    # Total activations per feature
    total_counts = feature_counts.sum(axis=1)  # [hidden_dim]
    
    # Compute proportions p_k^(i)
    # Avoid division by zero for features that never fire
    proportions = np.zeros_like(feature_counts)
    mask = total_counts > 0
    proportions[mask] = feature_counts[mask] / total_counts[mask, np.newaxis]
    
    # Compute firing entropy
    firing_entropies = np.zeros(hidden_dim)
    
    for k in range(hidden_dim):
        if total_counts[k] == 0:
            firing_entropies[k] = 0.0  # Never fires
            continue
        
        # FE_k = -1/log(M) * Σ p_k^(i) * log(p_k^(i))
        entropy = 0.0
        for i in range(M):
            p = proportions[k, i]
            if p > 0:
                entropy += p * np.log(p)
        
        firing_entropies[k] = -entropy / np.log(M)
    
    return firing_entropies, feature_counts

def compute_cofiring_proportion(z_codes):
    """
    For each feature k, compute co-firing proportion:
    
    When feature k fires for Model A, how often does it fire for Model B?
    
    Returns:
        cofiring_props: [hidden_dim] array of co-firing proportions
    """
    z0 = z_codes['model_0']  # [N, hidden_dim]
    z1 = z_codes['model_1']
    
    hidden_dim = z0.shape[1]
    cofiring_props = np.zeros(hidden_dim)
    
    for k in range(hidden_dim):
        # When does feature k fire in model 0?
        fires_m0 = (z0[:, k] != 0)
        
        # When does feature k fire in model 1?
        fires_m1 = (z1[:, k] != 0)
        
        # Co-firing: fires in BOTH models for the same sample
        cofires = fires_m0 & fires_m1
        
        # Proportion: out of times it fires in m0, how often does it also fire in m1?
        n_fires_m0 = fires_m0.sum().item()
        n_cofires = cofires.sum().item()
        
        if n_fires_m0 > 0:
            cofiring_props[k] = n_cofires / n_fires_m0
        else:
            cofiring_props[k] = 0.0
    
    return cofiring_props

def plot_firing_entropy_histogram(firing_entropies, save_path):
    """Plot histogram of firing entropies"""
    plt.figure(figsize=(10, 6))
    
    # Filter out features that never fire
    active_features = firing_entropies[firing_entropies > 0]
    
    plt.hist(active_features, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Normalized Firing Entropy', fontsize=12)
    plt.ylabel('Number of Features', fontsize=12)
    plt.title(f'Firing Entropy Distribution\n(Active features: {len(active_features)}/{len(firing_entropies)})', 
              fontsize=14)
    plt.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Midpoint (0.5)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"✓ Saved firing entropy histogram to {save_path}")

def analyze_universality():
    """Main analysis function"""
    
    print("\n" + "="*80)
    print("UNIVERSALITY ANALYSIS")
    print("="*80)
    
    # Load activations
    print("\nLoading activations...")
    resnet_acts = torch.load(Path(CONFIG['activation_dir']) / 'resnet_activations.pt')
    vit_acts = torch.load(Path(CONFIG['activation_dir']) / 'vit_activations.pt')
    print(f"  ResNet: {resnet_acts.shape}")
    print(f"  ViT: {vit_acts.shape}")
    
    # Create dataset (use all data, not just val split)
    dataset = TensorDataset(resnet_acts, vit_acts)
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], 
                           shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    # Load trained USAE
    print("\nLoading trained USAE...")
    model_dims = [resnet_acts.shape[1], vit_acts.shape[1]]
    usae = UniversalSAE(model_dims, CONFIG['hidden_dim'], CONFIG['k']).to(device)
    
    checkpoint = torch.load(Path(CONFIG['checkpoint_dir']) / 'usae_best.pt')
    usae.load_state_dict(checkpoint['model_state_dict'])
    print("  ✓ Loaded best model")
    
    # Extract sparse codes
    print("\nExtracting sparse codes from both models...")
    z_codes = extract_sparse_codes(usae, dataloader, device)
    print(f"  Model 0 (ResNet): {z_codes['model_0'].shape}")
    print(f"  Model 1 (ViT): {z_codes['model_1'].shape}")
    
    # Compute Firing Entropy
    print("\nComputing Firing Entropy...")
    firing_entropies, feature_counts = compute_firing_entropy(z_codes)
    
    # Statistics
    active_features = (feature_counts.sum(axis=1) > 0).sum()
    print(f"  Active features: {active_features}/{CONFIG['hidden_dim']}")
    print(f"  FE mean: {firing_entropies[firing_entropies > 0].mean():.4f}")
    print(f"  FE std: {firing_entropies[firing_entropies > 0].std():.4f}")
    print(f"  FE range: [{firing_entropies.min():.4f}, {firing_entropies.max():.4f}]")
    
    # Categorize features
    universal = (firing_entropies > 0.8).sum()
    model_specific = (firing_entropies < 0.2).sum()
    mixed = ((firing_entropies >= 0.2) & (firing_entropies <= 0.8)).sum()
    
    print(f"\n  Universal (FE > 0.8): {universal}")
    print(f"  Model-specific (FE < 0.2): {model_specific}")
    print(f"  Mixed (0.2 <= FE <= 0.8): {mixed}")
    
    # Check for bimodal distribution
    print("\n  Bimodal distribution check:")
    low_fe = (firing_entropies < 0.3).sum()
    high_fe = (firing_entropies > 0.7).sum()
    mid_fe = ((firing_entropies >= 0.3) & (firing_entropies <= 0.7)).sum()
    
    if (low_fe + high_fe) > mid_fe:
        print("    ✓ Suggests bimodal distribution (peaks at extremes)")
    else:
        print("    ✗ No clear bimodal distribution (peak in middle)")
    
    # Compute Co-Firing Proportion
    print("\nComputing Co-Firing Proportion...")
    cofiring_props = compute_cofiring_proportion(z_codes)
    
    active_cofiring = cofiring_props[cofiring_props > 0]
    print(f"  Mean co-firing: {active_cofiring.mean():.4f}")
    print(f"  Std co-firing: {active_cofiring.std():.4f}")
    
    # Plot histogram
    print("\nGenerating plots...")
    plot_firing_entropy_histogram(firing_entropies, Path(CONFIG['log_dir']) / 'firing_entropy_histogram.png')
    
    # Save results
    results = {
        'firing_entropies': firing_entropies.tolist(),
        'feature_counts': feature_counts.tolist(),
        'cofiring_proportions': cofiring_props.tolist(),
        'statistics': {
            'active_features': int(active_features),
            'total_features': CONFIG['hidden_dim'],
            'fe_mean': float(firing_entropies[firing_entropies > 0].mean()),
            'fe_std': float(firing_entropies[firing_entropies > 0].std()),
            'universal_features': int(universal),
            'model_specific_features': int(model_specific),
            'mixed_features': int(mixed),
            'cofiring_mean': float(active_cofiring.mean()),
            'cofiring_std': float(active_cofiring.std()),
        }
    }
    
    import json
    with open(Path(CONFIG['log_dir']) / 'universality_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Saved results to {CONFIG['log_dir']}/universality_analysis.json")
    
    print("\n" + "="*80)
    print("✓ UNIVERSALITY ANALYSIS COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    analyze_universality()