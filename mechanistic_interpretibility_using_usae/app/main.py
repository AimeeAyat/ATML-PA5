"""
Main USAE Pipeline - Orchestrator
Imports and calls functions from existing modules
"""

import torch
from pathlib import Path

# Import from your existing modules
from .models import resnet, vit, device
from .extract_activations import extract_resnet_activations, extract_vit_activations
from .usae_ import UniversalSAE
from .train_usae import USAETrainer, compute_r2_matrix, plot_r2_matrix

import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import json
import os
# ============================================================================
# Configuration
# ============================================================================
CONFIG = {
    # Data
    'data_root': './data',
    'batch_size': 128,
    'num_samples': 10000,
    
    # USAE
    'hidden_dim': 2042,
    'k': 64,
    
    # Training
    'batch_size_train': 128,
    'train_split': 0.9,
    'epochs': 100,
    'lr': 1e-3,
    'weight_decay': 1e-5,
    
    # Directories
    'activation_dir': './activations',
    'checkpoint_dir': './checkpoints',
    'log_dir': './logs',
    'save_every': 10,
}

def load_dataset():
    """Load CIFAR-10 dataset"""
    print("="*80)
    print("Loading Dataset")
    print("="*80)
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.CIFAR10(
        root=CONFIG['data_root'],
        train=True,
        download=False,  # Use existing data
        transform=transform
    )
    
    # Subset
    if len(dataset) > CONFIG['num_samples']:
        indices = torch.randperm(len(dataset))[:CONFIG['num_samples']]
        dataset = torch.utils.data.Subset(dataset, indices)
    
    print(f"✓ Loaded {len(dataset)} images")
    
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    
    return dataloader


# Custom collate
def collate_fn(batch):
    resnet_batch = torch.stack([b[0] for b in batch])
    vit_batch = torch.stack([b[1] for b in batch])
    return [resnet_batch, vit_batch]


def main():
    """Main pipeline"""
    
    print("\n" + "="*80)
    print("USAE TRAINING PIPELINE")
    print("="*80)
    
    # Create directories
    Path(CONFIG['activation_dir']).mkdir(parents=True, exist_ok=True)
    Path(CONFIG['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(CONFIG['log_dir']).mkdir(parents=True, exist_ok=True)
    
    print(f"\nDevice: {device}")
    print(f"ResNet on: {next(resnet.parameters()).device}")
    print(f"ViT on: {next(vit.parameters()).device}")
    
    # Step 1: Load dataset
    dataloader = load_dataset()
    
    # Step 2: Extract activations (use functions from extract_activations.py)
    print("\n" + "="*80)
    print("Extracting Activations")
    print("="*80)
    
    if os.path.exists(os.path.join('activations', 'resnet_activations.pt')):
        print("  ✓ Loading cached activations")
        resnet_acts = torch.load(os.path.join('activations', 'resnet_activations.pt'))
        vit_acts = torch.load(os.path.join('activations', 'vit_activations.pt'))
    else:
        resnet_acts = extract_resnet_activations(resnet, dataloader, device)
        vit_acts = extract_vit_activations(vit, dataloader, device)
        torch.save(resnet_acts, Path(CONFIG['activation_dir']) / 'resnet_activations.pt')
        torch.save(vit_acts, Path(CONFIG['activation_dir']) / 'vit_activations.pt')
        print(f"✓ Saved activations to {CONFIG['activation_dir']}/")
    
    
    # Step 3: Prepare training data
    print("\n" + "="*80)
    print("Preparing Training Data")
    print("="*80)
    
    n_samples = resnet_acts.shape[0]
    n_train = int(n_samples * CONFIG['train_split'])
    
    train_resnet = resnet_acts[:n_train]
    train_vit = vit_acts[:n_train]
    val_resnet = resnet_acts[n_train:]
    val_vit = vit_acts[n_train:]
    
    print(f"Train: {n_train}, Val: {n_samples - n_train}")
    

    
    train_dataset = TensorDataset(train_resnet, train_vit)
    val_dataset = TensorDataset(val_resnet, val_vit)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size_train'],
                             shuffle=True, collate_fn=collate_fn, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size_train'],
                           shuffle=False, collate_fn=collate_fn, num_workers=8)
    
    # Step 4: Create USAE (use class from usae.py)
    print("\n" + "="*80)
    print("Initializing USAE")
    print("="*80)
    
    model_dims = [resnet_acts.shape[1], vit_acts.shape[1]]
    usae = UniversalSAE(model_dims, CONFIG['hidden_dim'], CONFIG['k'])

    start_epoch = 0
    checkpoint_path = Path(CONFIG['checkpoint_dir']) / 'usae_best.pt'
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path)
        usae.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"✓ Loaded best model from {checkpoint_path}")
    else:
        print(f"✗ No best model found at {checkpoint_path}")
    
    trainer = USAETrainer(usae, CONFIG, device)

    # Load optimizer state if resuming
    if start_epoch > 0:
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.train_losses = checkpoint['train_losses']
        trainer.model_losses = checkpoint['model_losses']

        for _ in range(start_epoch):
            trainer.scheduler.step()



    # Step 5: Train (use trainer from train_usae.py)

    trainer.train(train_loader, val_loader, CONFIG['epochs'], start_epoch=start_epoch)
    
    # Step 6: Evaluate (use functions from train_usae.py)
    print("\n" + "="*80)
    print("Final Evaluation")
    print("="*80)
    
    best_checkpoint = torch.load(Path(CONFIG['checkpoint_dir']) / 'usae_best.pt')
    usae.load_state_dict(best_checkpoint['model_state_dict'])
    
    r2_matrix = compute_r2_matrix(usae, val_loader, device)


    

    print("\nR² Matrix:")
    print(r2_matrix)


    np.save(Path(CONFIG['log_dir']) / 'r2_matrix.npy', r2_matrix)

    r2_dict = {
        'r2_matrix': r2_matrix.tolist(),
        'diagonal': [r2_matrix[i, i] for i in range(len(r2_matrix))],
        'off_diagonal': {
            'resnet_to_vit': float(r2_matrix[0, 1]),
            'vit_to_resnet': float(r2_matrix[1, 0])
        },
        'universality': bool(r2_matrix[0, 1] > 0 and r2_matrix[1, 0] > 0)
    }

    with open(Path(CONFIG['log_dir']) / 'r2_scores.json', 'w') as f:
        json.dump(r2_dict, f, indent=2)


    plot_r2_matrix(r2_matrix, Path(CONFIG['log_dir']) / 'r2_matrix.png')
    print(f"\n✓ Saved R² matrix to {CONFIG['log_dir']}/r2_matrix.npy")
    print(f"✓ Saved R² scores to {CONFIG['log_dir']}/r2_scores.json")
    print(f"✓ Saved R² plot to {CONFIG['log_dir']}/r2_matrix.png")
    
    
    
    print("\n" + "="*80)
    print("✓ PIPELINE COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()