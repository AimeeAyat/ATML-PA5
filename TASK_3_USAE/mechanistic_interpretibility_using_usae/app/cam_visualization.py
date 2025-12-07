"""
cam_visualization.py
Coordinated Activation Maximization - Visualize what universal features represent
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm

from app.usae_ import UniversalSAE
from app.models import resnet, vit, device

# Configuration
CONFIG = {
    'checkpoint_dir': './checkpoints',
    'log_dir': './logs',
    'hidden_dim': 2042,
    'k': 64,
    'n_features_to_visualize': 3,  # Number of features to visualize
    
    # Optimization params
    'n_iterations': 500,
    'lr': 0.1,
    'tv_weight': 1e-4,  # Total variation regularization
    'l2_weight': 1e-5,  # L2 pixel regularization
}

def load_universal_features():
    """Load firing entropy results and select top universal features"""
    results_path = Path(CONFIG['log_dir']) / 'universality_analysis.json'
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    firing_entropies = np.array(results['firing_entropies'])
    
    # Select features with high firing entropy (universal)
    # Filter active features first
    active_mask = firing_entropies > 0
    active_fe = firing_entropies[active_mask]
    active_indices = np.where(active_mask)[0]
    
    # Get top N universal features
    top_indices = np.argsort(active_fe)[-CONFIG['n_features_to_visualize']:][::-1]
    feature_indices = active_indices[top_indices]
    feature_entropies = active_fe[top_indices]
    
    print(f"Selected {len(feature_indices)} universal features:")
    for idx, (feat_idx, fe) in enumerate(zip(feature_indices, feature_entropies)):
        print(f"  {idx+1}. Feature {feat_idx}: FE = {fe:.4f}")
    
    return feature_indices, feature_entropies

def total_variation_loss(img):
    """
    Total variation regularization - encourages spatial smoothness
    Penalizes large pixel differences between neighbors
    """
    tv_h = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).sum()
    tv_w = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).sum()
    return tv_h + tv_w

def optimize_image_for_feature(model, encoder, feature_idx, model_name, seed=42):
    """
    Optimize input image to maximize activation of feature_idx
    
    Args:
        model: ResNet or ViT model
        encoder: USAE encoder for this model
        feature_idx: Which feature in Z to maximize
        model_name: 'ResNet' or 'ViT'
        seed: Random seed for initialization
    
    Returns:
        optimized_img: [3, 224, 224] tensor
        activations: list of Z[feature_idx] values during optimization
    """
    torch.manual_seed(seed)
    
    # Initialize random noise image [1, 3, 224, 224]
    # Start with small values around 0
    img = torch.randn(1, 3, 224, 224, device=device) * 0.1
    img.requires_grad = True
    
    # Optimizer
    optimizer = torch.optim.Adam([img], lr=CONFIG['lr'])
    
    # ImageNet normalization stats
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    
    activations = []
    
    model.eval()
    encoder.eval()
    
    for iteration in tqdm(range(CONFIG['n_iterations']), desc=f"Optimizing {model_name}", leave=False):
        optimizer.zero_grad()
        
        # Normalize image for model input
        img_normalized = (img - mean) / std
        
        # Clamp to valid range after normalization
        img_normalized = torch.clamp(img_normalized, -3, 3)
        
        # Forward through model to get activations
        if 'ResNet' in model_name:
            # ResNet: extract layer4, then pool
            activation_extractor = {}
            def hook_fn(module, input, output):
                activation_extractor['layer4'] = output
            
            hook = model.layer4.register_forward_hook(hook_fn)
            _ = model(img_normalized)
            hook.remove()
            
            # Pool: [1, 512, 7, 7] → [1, 512]
            model_activation = activation_extractor['layer4'].mean(dim=[2, 3])
        
        else:  # ViT
            # ViT: extract block 11, then CLS token
            activation_extractor = {}
            def hook_fn(module, input, output):
                activation_extractor['block11'] = output
            
            hook = model.blocks[11].register_forward_hook(hook_fn)
            _ = model(img_normalized)
            hook.remove()
            
            # CLS token: [1, 197, 768] → [1, 768]
            model_activation = activation_extractor['block11'][:, 0, :]
        
        # Encode to sparse code Z
        z = encoder(model_activation)  # [1, hidden_dim]
        
        # Target: maximize Z[feature_idx]
        feature_activation = z[0, feature_idx]
        
        # Loss: negative activation (we want to maximize, so minimize negative)
        loss = -feature_activation
        
        # Regularization
        tv_loss = total_variation_loss(img)
        l2_loss = torch.norm(img)
        
        total_loss = loss + CONFIG['tv_weight'] * tv_loss + CONFIG['l2_weight'] * l2_loss
        
        # Backward
        total_loss.backward()
        optimizer.step()
        
        # Clamp pixel values to reasonable range
        with torch.no_grad():
            img.clamp_(-2, 2)
        
        # Log
        activations.append(feature_activation.item())
    
    # Denormalize for visualization
    with torch.no_grad():
        img_vis = img.clone()
        # Rescale to [0, 1] for display
        img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min())
    
    return img_vis[0].cpu(), activations

def visualize_feature_comparison(feature_idx, fe_value, resnet_img, vit_img, 
                                 resnet_acts, vit_acts, save_path):
    """
    Visualize optimized images from both models side-by-side
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Top row: optimized images
    axes[0, 0].imshow(resnet_img.permute(1, 2, 0))
    axes[0, 0].set_title(f'ResNet Optimized\nFeature {feature_idx}', fontsize=14)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(vit_img.permute(1, 2, 0))
    axes[0, 1].set_title(f'ViT Optimized\nFeature {feature_idx}', fontsize=14)
    axes[0, 1].axis('off')
    
    # Bottom row: activation curves
    axes[1, 0].plot(resnet_acts, linewidth=2)
    axes[1, 0].set_xlabel('Iteration', fontsize=12)
    axes[1, 0].set_ylabel('Feature Activation', fontsize=12)
    axes[1, 0].set_title('ResNet Optimization Curve', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(vit_acts, linewidth=2, color='orange')
    axes[1, 1].set_xlabel('Iteration', fontsize=12)
    axes[1, 1].set_ylabel('Feature Activation', fontsize=12)
    axes[1, 1].set_title('ViT Optimization Curve', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Feature {feature_idx} (Firing Entropy: {fe_value:.3f})\n' +
                 f'Final Activations: ResNet={resnet_acts[-1]:.2f}, ViT={vit_acts[-1]:.2f}',
                 fontsize=16, y=0.98)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved visualization to {save_path}")

def cam_analysis():
    """Main CAM visualization function"""
    
    print("\n" + "="*80)
    print("COORDINATED ACTIVATION MAXIMIZATION (CAM)")
    print("="*80)
    
    # Load universal features
    print("\nSelecting universal features...")
    feature_indices, feature_entropies = load_universal_features()
    
    # Load trained USAE
    print("\nLoading trained USAE...")
    model_dims = [512, 768]  # ResNet, ViT
    usae = UniversalSAE(model_dims, CONFIG['hidden_dim'], CONFIG['k']).to(device)
    
    checkpoint = torch.load(Path(CONFIG['checkpoint_dir']) / 'usae_best.pt')
    usae.load_state_dict(checkpoint['model_state_dict'])
    print("  ✓ Loaded best model")
    
    # Get encoders
    resnet_encoder = usae.encoders[0]
    vit_encoder = usae.encoders[1]
    
    # Visualize each feature
    print("\nGenerating CAM visualizations...")
    
    for idx, (feature_idx, fe_value) in enumerate(zip(feature_indices, feature_entropies)):
        print(f"\n[{idx+1}/{len(feature_indices)}] Feature {feature_idx} (FE={fe_value:.4f})")
        
        # Optimize through ResNet
        print("  Optimizing via ResNet...")
        resnet_img, resnet_acts = optimize_image_for_feature(
            resnet, resnet_encoder, feature_idx, 'ResNet', seed=42+idx
        )
        
        # Optimize through ViT
        print("  Optimizing via ViT...")
        vit_img, vit_acts = optimize_image_for_feature(
            vit, vit_encoder, feature_idx, 'ViT', seed=100+idx
        )
        
        # Visualize comparison
        save_path = Path(CONFIG['log_dir']) / f'cam_feature_{feature_idx}.png'
        visualize_feature_comparison(
            feature_idx, fe_value, resnet_img, vit_img,
            resnet_acts, vit_acts, save_path
        )
        
        # Check convergence
        final_resnet = resnet_acts[-1]
        final_vit = vit_acts[-1]
        
        if abs(final_resnet - final_vit) < 0.5 * (final_resnet + final_vit):
            print(f"    ✓ Models converge (similar activation levels)")
        else:
            print(f"    ⚠ Models diverge (different activation levels)")
    
    print("\n" + "="*80)
    print("✓ CAM VISUALIZATION COMPLETE!")
    print("="*80)
    print("\nInterpretation Guide:")
    print("  - Similar images → Models see same concept (convergence)")
    print("  - Different images → Models disagree on concept (divergence)")
    print("  - Check activation curves for optimization quality")

if __name__ == "__main__":
    cam_analysis()