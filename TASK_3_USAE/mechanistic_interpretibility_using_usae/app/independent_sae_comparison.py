"""
independent_sae_comparison.py
Task 4: Compare USAE with Independent SAEs to measure alignment tax
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

from app.usae_ import UniversalSAE, USAEEncoder, USAEDecoder, TopKActivation
from app.models import device

# Configuration
CONFIG = {
    'activation_dir': './activations',
    'checkpoint_dir': './checkpoints',
    'log_dir': './logs',
    'hidden_dim': 2042,
    'k': 64,
    'batch_size': 128,
    'epochs': 100,
    'lr': 1e-3,
}

class IndependentSAE(nn.Module):
    """Standard SAE - single model, no universality constraint"""
    def __init__(self, input_dim, hidden_dim, k):
        super().__init__()
        self.encoder = USAEEncoder(input_dim, hidden_dim, k)
        self.decoder = USAEDecoder(hidden_dim, input_dim)
    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return z, x_recon
    
    def compute_loss(self, x, x_recon):
        return nn.functional.mse_loss(x_recon, x)

def train_independent_sae(model_name, activations, device):
    """Train independent SAE for one model"""
    print(f"\nTraining Independent SAE for {model_name}...")
    
    input_dim = activations.shape[1]
    sae = IndependentSAE(input_dim, CONFIG['hidden_dim'], CONFIG['k']).to(device)
    
    # Split train/val
    n_train = int(0.9 * len(activations))
    train_acts = activations[:n_train]
    val_acts = activations[n_train:]
    
    train_loader = DataLoader(train_acts, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_acts, batch_size=CONFIG['batch_size'], shuffle=False)
    
    optimizer = optim.Adam(sae.parameters(), lr=CONFIG['lr'])
    
    best_loss = float('inf')
    train_losses = []
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        # Train
        sae.train()
        epoch_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            
            z, x_recon = sae(batch)
            loss = sae.compute_loss(batch, x_recon)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}/{CONFIG['epochs']}: Loss = {avg_loss:.4f}")
        
        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': sae.state_dict(),
                'epoch': epoch,
                'loss': best_loss
            }, Path(CONFIG['checkpoint_dir']) / f'independent_sae_{model_name}.pt')
    
    print(f"  ✓ Training complete. Best loss: {best_loss:.4f}")
    
    return sae, train_losses

def compute_self_reconstruction_r2(sae, activations, device):
    """Compute R² for self-reconstruction (A → Z → A)"""
    sae.eval()
    
    all_true = []
    all_recon = []
    
    loader = DataLoader(activations, batch_size=CONFIG['batch_size'], shuffle=False)
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, x_recon = sae(batch)
            
            all_true.append(batch.cpu())
            all_recon.append(x_recon.cpu())
    
    all_true = torch.cat(all_true, dim=0)
    all_recon = torch.cat(all_recon, dim=0)
    
    # R² = 1 - SS_res / SS_tot
    ss_res = ((all_true - all_recon) ** 2).sum()
    ss_tot = ((all_true - all_true.mean(dim=0)) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot)
    
    return r2.item()

def analyze_alignment_tax():
    """Compare USAE vs Independent SAEs"""
    print("\n" + "="*80)
    print("ALIGNMENT TAX ANALYSIS")
    print("="*80)
    
    # Load activations
    print("\nLoading activations...")
    resnet_acts = torch.load(Path(CONFIG['activation_dir']) / 'resnet_activations.pt')
    vit_acts = torch.load(Path(CONFIG['activation_dir']) / 'vit_activations.pt')
    
    # Train independent SAEs
    print("\n" + "="*80)
    print("Training Independent SAEs")
    print("="*80)
    
    resnet_sae, resnet_losses = train_independent_sae('resnet', resnet_acts, device)
    vit_sae, vit_losses = train_independent_sae('vit', vit_acts, device)
    
    # Compute R² for independent SAEs
    print("\nComputing Independent SAE R² scores...")
    r2_ind_resnet = compute_self_reconstruction_r2(resnet_sae, resnet_acts, device)
    r2_ind_vit = compute_self_reconstruction_r2(vit_sae, vit_acts, device)
    
    print(f"  Independent ResNet R²: {r2_ind_resnet:.4f}")
    print(f"  Independent ViT R²: {r2_ind_vit:.4f}")
    
    # Load USAE and compute R²
    print("\nLoading USAE...")
    usae = UniversalSAE([512, 768], CONFIG['hidden_dim'], CONFIG['k']).to(device)
    checkpoint = torch.load(Path(CONFIG['checkpoint_dir']) / 'usae_best.pt')
    usae.load_state_dict(checkpoint['model_state_dict'])
    
    # Compute USAE self-reconstruction R²
    print("Computing USAE R² scores...")
    
    # USAE ResNet: encode ResNet → decode ResNet
    usae.eval()
    resnet_loader = DataLoader(resnet_acts, batch_size=CONFIG['batch_size'], shuffle=False)
    
    all_true = []
    all_recon = []
    
    with torch.no_grad():
        for batch in resnet_loader:
            batch = batch.to(device)
            z = usae.encode(batch, 0)  # Encode from ResNet
            recon = usae.decode(z, 0)  # Decode to ResNet
            all_true.append(batch.cpu())
            all_recon.append(recon.cpu())
    
    all_true = torch.cat(all_true, dim=0)
    all_recon = torch.cat(all_recon, dim=0)
    ss_res = ((all_true - all_recon) ** 2).sum()
    ss_tot = ((all_true - all_true.mean(dim=0)) ** 2).sum()
    r2_usae_resnet = (1 - ss_res / ss_tot).item()
    
    # USAE ViT
    vit_loader = DataLoader(vit_acts, batch_size=CONFIG['batch_size'], shuffle=False)
    
    all_true = []
    all_recon = []
    
    with torch.no_grad():
        for batch in vit_loader:
            batch = batch.to(device)
            z = usae.encode(batch, 1)
            recon = usae.decode(z, 1)
            all_true.append(batch.cpu())
            all_recon.append(recon.cpu())
    
    all_true = torch.cat(all_true, dim=0)
    all_recon = torch.cat(all_recon, dim=0)
    ss_res = ((all_true - all_recon) ** 2).sum()
    ss_tot = ((all_true - all_true.mean(dim=0)) ** 2).sum()
    r2_usae_vit = (1 - ss_res / ss_tot).item()
    
    print(f"  USAE ResNet R²: {r2_usae_resnet:.4f}")
    print(f"  USAE ViT R²: {r2_usae_vit:.4f}")
    
    # Compute alignment tax
    print("\n" + "="*80)
    print("ALIGNMENT TAX RESULTS")
    print("="*80)
    
    tax_resnet = r2_ind_resnet - r2_usae_resnet
    tax_vit = r2_ind_vit - r2_usae_vit
    tax_avg = (tax_resnet + tax_vit) / 2
    
    print(f"\nSelf-Reconstruction R² Comparison:")
    print(f"  ResNet: Independent={r2_ind_resnet:.4f}, USAE={r2_usae_resnet:.4f}, Tax={tax_resnet:.4f}")
    print(f"  ViT:    Independent={r2_ind_vit:.4f}, USAE={r2_usae_vit:.4f}, Tax={tax_vit:.4f}")
    print(f"  Average Alignment Tax: {tax_avg:.4f}")
    
    if tax_avg > 0:
        print(f"\n  ⚠ USAE pays {tax_avg*100:.2f}% alignment tax")
        print(f"    (Self-reconstruction is worse due to universality constraint)")
    else:
        print(f"\n  ✓ No alignment tax detected (USAE performs comparably)")
    
    # Interpretation
    print("\nInterpretation:")
    if tax_avg > 0.05:
        print("  - Significant alignment tax: Forcing shared features degrades quality")
        print("  - Trade-off: Better interpretability vs worse reconstruction")
    elif tax_avg > 0:
        print("  - Moderate alignment tax: Acceptable trade-off")
    else:
        print("  - No tax: Universal features don't harm performance")
    
    # Save results
    results = {
        'independent_sae': {
            'resnet_r2': float(r2_ind_resnet),
            'vit_r2': float(r2_ind_vit),
        },
        'usae': {
            'resnet_r2': float(r2_usae_resnet),
            'vit_r2': float(r2_usae_vit),
        },
        'alignment_tax': {
            'resnet': float(tax_resnet),
            'vit': float(tax_vit),
            'average': float(tax_avg),
        }
    }
    
    with open(Path(CONFIG['log_dir']) / 'alignment_tax_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved results to {CONFIG['log_dir']}/alignment_tax_analysis.json")
    
    # Plot comparison
    plot_alignment_tax_comparison(results)

def plot_alignment_tax_comparison(results):
    """Plot R² comparison"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    models = ['ResNet', 'ViT']
    ind_r2 = [results['independent_sae']['resnet_r2'], results['independent_sae']['vit_r2']]
    usae_r2 = [results['usae']['resnet_r2'], results['usae']['vit_r2']]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, ind_r2, width, label='Independent SAE', color='steelblue')
    bars2 = ax.bar(x + width/2, usae_r2, width, label='USAE', color='coral')
    
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('Self-Reconstruction R²: Independent SAE vs USAE\n(Alignment Tax Analysis)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(Path(CONFIG['log_dir']) / 'alignment_tax_comparison.png', dpi=150)
    plt.close()
    
    print(f"✓ Saved plot to {CONFIG['log_dir']}/alignment_tax_comparison.png")

if __name__ == "__main__":
    analyze_alignment_tax()