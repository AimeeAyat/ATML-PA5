"""
USAE Training Script
Trains Universal Sparse Autoencoder on pre-extracted activations
with full logging, checkpointing, and evaluation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from tqdm import tqdm
import os

from .usae_ import UniversalSAE

class USAETrainer:
    """Handles USAE training with logging and checkpointing"""
    
    def __init__(self, usae, config, device):
        """
        Args:
            usae: UniversalSAE model
            config: dictionary with training hyperparameters
            device: cuda or cpu
        """
        self.usae = usae.to(device)
        self.config = config
        self.device = device
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.usae.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
        
        # Setup learning rate scheduler (optional)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs']
        )
        
        # Create directories for saving
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(config['log_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self.train_losses = []  # Total loss per epoch
        self.model_losses = {i: [] for i in range(usae.num_models)}  # Per-model losses
        self.best_loss = float('inf')
        
        print(f"✓ Trainer initialized")
        print(f"  Optimizer: Adam (lr={config['lr']})")
        print(f"  Device: {device}")
        print(f"  Checkpoint dir: {self.checkpoint_dir}")
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch with random source selection
        
        Args:
            train_loader: DataLoader yielding batches of activations
        Returns:
            avg_loss: average loss for the epoch
            avg_model_losses: list of per-model average losses
        """
        self.usae.train()
        
        epoch_loss = 0.0
        epoch_model_losses = [0.0] * self.usae.num_models
        num_batches = 0
        
        for batch in tqdm(train_loader, desc="Training", leave=False):
            # batch is a list of activation tensors [resnet_batch, vit_batch]
            batch = [b.to(self.device) for b in batch]
            
            # Randomly select source model for this batch
            # This implements the USAE training procedure
            source_idx = np.random.randint(0, self.usae.num_models)
            
            # Forward pass: encode from source, decode to all
            z, reconstructions = self.usae(batch, source_idx)
            
            # Compute loss: sum of reconstruction errors across all models
            loss, individual_losses = self.usae.compute_loss(batch, reconstructions)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.usae.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            epoch_loss += loss.item()
            for i, ind_loss in enumerate(individual_losses):
                epoch_model_losses[i] += ind_loss.item()
            num_batches += 1
        
        # Average losses over batches
        avg_loss = epoch_loss / num_batches
        avg_model_losses = [l / num_batches for l in epoch_model_losses]
        
        return avg_loss, avg_model_losses
    
    def evaluate(self, val_loader):
        """
        Evaluate on validation set
        
        Returns:
            avg_loss: validation loss
            avg_model_losses: per-model validation losses
        """
        self.usae.eval()
        
        val_loss = 0.0
        val_model_losses = [0.0] * self.usae.num_models
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                batch = [b.to(self.device) for b in batch]
                
                # Test with both sources and average
                losses_per_source = []
                
                for source_idx in range(self.usae.num_models):
                    z, reconstructions = self.usae(batch, source_idx)
                    loss, individual_losses = self.usae.compute_loss(batch, reconstructions)
                    losses_per_source.append((loss.item(), [l.item() for l in individual_losses]))
                
                # Average over sources
                avg_batch_loss = np.mean([l[0] for l in losses_per_source])
                avg_batch_model_losses = np.mean([[l[1][i] for l in losses_per_source] 
                                                  for i in range(self.usae.num_models)], axis=1)
                
                val_loss += avg_batch_loss
                for i, ml in enumerate(avg_batch_model_losses):
                    val_model_losses[i] += ml
                num_batches += 1
        
        avg_val_loss = val_loss / num_batches
        avg_val_model_losses = [l / num_batches for l in val_model_losses]
        
        return avg_val_loss, avg_val_model_losses
    
    def save_checkpoint(self, epoch, is_best=False):
        """
        Save model checkpoint
        
        Args:
            epoch: current epoch number
            is_best: whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.usae.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'model_losses': self.model_losses,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'usae_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model separately
        if is_best:
            best_path = self.checkpoint_dir / 'usae_best.pt'
            torch.save(checkpoint, best_path)
            print(f"  ✓ Saved best model (epoch {epoch}, loss {self.train_losses[-1]:.4f})")
    
    def save_losses(self):
        """Save losses to JSON file for later analysis"""
        losses_dict = {
            'train_losses': self.train_losses,
            'model_losses': {f'model_{i}': losses for i, losses in self.model_losses.items()},
            'config': self.config
        }
        
        loss_path = self.log_dir / 'losses.json'
        with open(loss_path, 'w') as f:
            json.dump(losses_dict, f, indent=2)
        
        print(f"  ✓ Saved losses to {loss_path}")
    
    def plot_losses(self):
        """Plot and save training curves"""
        epochs = range(1, len(self.train_losses) + 1)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Total loss
        axes[0].plot(epochs, self.train_losses, 'b-', linewidth=2, label='Total Loss')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('USAE Training Loss', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Plot 2: Per-model losses
        for i, losses in self.model_losses.items():
            axes[1].plot(epochs, losses, linewidth=2, label=f'Model {i}')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].set_title('Per-Model Reconstruction Loss', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        plot_path = self.log_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved training curves to {plot_path}")
    
    def train(self, train_loader, val_loader, epochs, start_epoch=0):
        """
        Main training loop
        
        Args:
            train_loader: training data loader
            val_loader: validation data loader
            epochs: number of epochs to train
        """
        print("\n" + "="*80)
        print("Starting USAE Training")
        print("="*80)
        
        for epoch in range(start_epoch + 1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            print("-" * 80)
            
            # Train for one epoch
            train_loss, train_model_losses = self.train_epoch(train_loader)
            
            # Evaluate on validation set
            val_loss, val_model_losses = self.evaluate(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Log losses
            self.train_losses.append(train_loss)
            for i, ml in enumerate(train_model_losses):
                self.model_losses[i].append(ml)
            
            # Print statistics
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Model Losses: {[f'{ml:.4f}' for ml in train_model_losses]}")
            print(f"  LR: {current_lr:.6f}")
            
            # Save checkpoint
            is_best = train_loss < self.best_loss
            if is_best:
                self.best_loss = train_loss
            
            if epoch % self.config['save_every'] == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Save losses every epoch (in case of crash)
            self.save_losses()
        
        # Final plotting
        self.plot_losses()
        
        print("\n" + "="*80)
        print("Training Complete!")
        print(f"Best Loss: {self.best_loss:.4f}")
        print("="*80)

def compute_r2_matrix(usae, dataloader, device):
    """
    Compute R² reconstruction matrix
    
    R²[i,j] = how well codes from model i reconstruct model j
    
    Returns:
        r2_matrix: [M, M] numpy array
    """
    usae.eval()
    
    M = usae.num_models
    r2_matrix = np.zeros((M, M))
    
    print("\n" + "="*80)
    print("Computing R² Reconstruction Matrix")
    print("="*80)
    
    with torch.no_grad():
        # Collect all activations
        all_activations = [[] for _ in range(M)]
        all_reconstructions = [[[] for _ in range(M)] for _ in range(M)]
        
        for batch in tqdm(dataloader, desc="R² computation"):
            batch = [b.to(device) for b in batch]
            
            # For each source model
            for source_idx in range(M):
                # Encode from source
                z = usae.encode(batch[source_idx], source_idx)
                
                # Decode to all models
                for target_idx in range(M):
                    recon = usae.decode(z, target_idx)
                    all_reconstructions[source_idx][target_idx].append(recon.cpu())
            
            # Store true activations
            for i, b in enumerate(batch):
                all_activations[i].append(b.cpu())
        
        # Concatenate batches
        all_activations = [torch.cat(acts, dim=0) for acts in all_activations]
        all_reconstructions = [[torch.cat(recons, dim=0) for recons in row] 
                               for row in all_reconstructions]
        
        # Compute R² for each (source, target) pair
        for i in range(M):  # Source
            for j in range(M):  # Target
                true = all_activations[j]
                pred = all_reconstructions[i][j]
                
                # R² = 1 - SS_res / SS_tot
                ss_res = ((true - pred) ** 2).sum()
                ss_tot = ((true - true.mean(dim=0)) ** 2).sum()
                r2 = 1 - (ss_res / ss_tot)
                
                r2_matrix[i, j] = r2.item()
    
    return r2_matrix

def plot_r2_matrix(r2_matrix, save_path):
    """Plot R² confusion matrix"""
    M = r2_matrix.shape[0]
    model_names = [f'Model {i}' for i in range(M)]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(r2_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                xticklabels=model_names, yticklabels=model_names,
                vmin=-0.5, vmax=1.0, center=0.5)
    plt.xlabel('Target Model (Reconstruction)', fontsize=12)
    plt.ylabel('Source Model (Encoding)', fontsize=12)
    plt.title('R² Reconstruction Matrix\n(Positive off-diagonal = Universality)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved R² matrix to {save_path}")

def main():
    """Main training pipeline"""
    
    # =========================================================================
    # Configuration
    # =========================================================================
    config = {
        # Data
        'activation_dir': './activations',
        'batch_size': 128,
        'train_split': 0.9,  # 90% train, 10% val
        
        # Model
        'hidden_dim': 8192,  # Shared feature space dimension
        'k': 32,  # TopK sparsity
        
        # Training
        'epochs': 100,
        'lr': 1e-3,
        'weight_decay': 1e-5,
        
        # Logging
        'checkpoint_dir': './checkpoints',
        'log_dir': './logs',
        'save_every': 10,  # Save checkpoint every N epochs
    }
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # =========================================================================
    # Load Activations
    # =========================================================================
    print("\nLoading activations...")
    resnet_acts = torch.load(Path(config['activation_dir']) / 'resnet_activations.pt')
    vit_acts = torch.load(Path(config['activation_dir']) / 'vit_activations.pt')
    
    print(f"  ResNet: {resnet_acts.shape}")
    print(f"  ViT: {vit_acts.shape}")
    
    # Split into train/val
    n_samples = resnet_acts.shape[0]
    n_train = int(n_samples * config['train_split'])
    
    train_resnet = resnet_acts[:n_train]
    train_vit = vit_acts[:n_train]
    val_resnet = resnet_acts[n_train:]
    val_vit = vit_acts[n_train:]
    
    print(f"  Train: {n_train}, Val: {n_samples - n_train}")
    
    # Create datasets
    # Custom collate function to return list of tensors (one per model)
    def collate_fn(batch):
        # batch is list of tuples [(resnet, vit), ...]
        resnet_batch = torch.stack([b[0] for b in batch])
        vit_batch = torch.stack([b[1] for b in batch])
        return [resnet_batch, vit_batch]
    
    train_dataset = TensorDataset(train_resnet, train_vit)
    val_dataset = TensorDataset(val_resnet, val_vit)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                             shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                           shuffle=False, collate_fn=collate_fn, num_workers=2)
    
    # =========================================================================
    # Create USAE
    # =========================================================================
    print("\nInitializing USAE...")
    model_dims = [resnet_acts.shape[1], vit_acts.shape[1]]  # [512, 768]
    usae = UniversalSAE(model_dims, config['hidden_dim'], config['k'])
    
    # =========================================================================
    # Train
    # =========================================================================
    trainer = USAETrainer(usae, config, device)
    trainer.train(train_loader, val_loader, config['epochs'])
    
    # =========================================================================
    # Evaluate: Compute R² Matrix
    # =========================================================================
    print("\n" + "="*80)
    print("Final Evaluation")
    print("="*80)
    
    # Load best model
    best_checkpoint = torch.load(Path(config['checkpoint_dir']) / 'usae_best.pt')
    usae.load_state_dict(best_checkpoint['model_state_dict'])
    
    # Compute R² matrix
    r2_matrix = compute_r2_matrix(usae, val_loader, device)
    
    print("\nR² Reconstruction Matrix:")
    print(r2_matrix)
    print("\nInterpretation:")
    print(f"  Diagonal (self-reconstruction): {np.diag(r2_matrix)}")
    print(f"  Off-diagonal (cross-reconstruction): {r2_matrix[0, 1]:.3f}, {r2_matrix[1, 0]:.3f}")
    
    if r2_matrix[0, 1] > 0 and r2_matrix[1, 0] > 0:
        print("  ✓ Positive off-diagonal values → Evidence of universality!")
    else:
        print("  ✗ Negative off-diagonal values → Limited universality")
    
    # Plot and save
    plot_r2_matrix(r2_matrix, Path(config['log_dir']) / 'r2_matrix.png')
    
    # Save R² matrix
    np.save(Path(config['log_dir']) / 'r2_matrix.npy', r2_matrix)
    
    print("\n" + "="*80)
    print("✓ USAE training and evaluation complete!")
    print("="*80)

# if __name__ == "__main__":
#     main()