"""
Activation Extraction Script
Extracts intermediate activations from ResNet-18 (layer4) and ViT-B (block 11)
using PyTorch forward hooks and saves them for USAE training
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models
import timm
import numpy as np
from tqdm import tqdm
import os

class ActivationExtractor:
    """Extracts activations from specified layers using forward hooks"""
    
    def __init__(self):
        self.activations = {}
    
    def get_activation(self, name):
        """Creates a hook function to capture layer output"""
        def hook(model, input, output):
            # Detach to prevent gradient tracking
            self.activations[name] = output.detach()
        return hook
    
    def register_hooks(self, model, layer_names):
        """Register forward hooks on specified layers"""
        hooks = []
        for name, layer in model.named_modules():
            if name in layer_names:
                hook = layer.register_forward_hook(self.get_activation(name))
                hooks.append(hook)
                print(f"  ✓ Registered hook on: {name}")
        return hooks
    
    def remove_hooks(self, hooks):
        """Remove all registered hooks"""
        for hook in hooks:
            hook.remove()

def extract_resnet_activations(model, dataloader, device):
    """
    Extract activations from ResNet-18 layer4
    
    Args:
        model: ResNet-18 model
        dataloader: DataLoader for images
        device: cuda or cpu
    
    Returns:
        activations: [N, 512] tensor
    """
    print("\nExtracting ResNet-18 activations...")
    
    model.eval()
    extractor = ActivationExtractor()
    
    # Register hook on layer4 (before avgpool)
    hooks = extractor.register_hooks(model, ['layer4'])
    
    all_activations = []
    
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="ResNet-18"):
            images = images.to(device)
            
            # Forward pass
            _ = model(images)
            
            # Get layer4 output: [B, 512, 7, 7]
            layer4_out = extractor.activations['layer4']
            
            # Global average pooling: [B, 512, 7, 7] → [B, 512]
            pooled = layer4_out.mean(dim=[2, 3])
            
            all_activations.append(pooled.cpu())
    
    # Remove hooks
    extractor.remove_hooks(hooks)
    
    # Concatenate all batches: [N, 512]
    activations = torch.cat(all_activations, dim=0)
    print(f"  ✓ Extracted shape: {activations.shape}")
    
    return activations

def extract_vit_activations(model, dataloader, device):
    """
    Extract activations from ViT-B block 11 (CLS token)
    
    Args:
        model: ViT-B model
        dataloader: DataLoader for images
        device: cuda or cpu
    
    Returns:
        activations: [N, 768] tensor
    """
    print("\nExtracting ViT-B activations...")
    
    model.eval()
    extractor = ActivationExtractor()
    
    # Register hook on block 11 (last transformer block)
    hooks = extractor.register_hooks(model, ['blocks.11'])
    
    all_activations = []
    
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="ViT-B"):
            images = images.to(device)
            
            # Forward pass
            _ = model(images)
            
            # Get block 11 output: [B, 197, 768]
            # 197 tokens = 1 CLS + 196 patches
            block11_out = extractor.activations['blocks.11']
            
            # Extract CLS token (first token): [B, 197, 768] → [B, 768]
            cls_token = block11_out[:, 0, :]
            
            all_activations.append(cls_token.cpu())
    
    # Remove hooks
    extractor.remove_hooks(hooks)
    
    # Concatenate all batches: [N, 768]
    activations = torch.cat(all_activations, dim=0)
    print(f"  ✓ Extracted shape: {activations.shape}")
    
    return activations

def main():
    """Main extraction pipeline"""
    
    # Configuration
    DATA_ROOT = './data'  # Change this to your dataset path
    BATCH_SIZE = 128
    NUM_SAMPLES = 5000  # Number of images to use (adjust based on compute)
    SAVE_DIR = './activations'
    
    # Create save directory
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms (ImageNet normalization)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset (using ImageNet validation set or your custom dataset)
    # For demo, using a subset of ImageNet
    print("\nLoading dataset...")
    try:
        # Try ImageNet first
        dataset = datasets.ImageNet(DATA_ROOT, split='val', transform=transform)
    except:
        # Fallback to CIFAR-10 for testing (upscale to 224x224)
        print("  ImageNet not found, using CIFAR-10 for demo")
        # dataset = datasets.CIFAR10(DATA_ROOT, train=True, download=True, transform=transform)
    
    # Subset for faster extraction
    if len(dataset) > NUM_SAMPLES:
        indices = torch.randperm(len(dataset))[:NUM_SAMPLES]
        dataset = torch.utils.data.Subset(dataset, indices)
    
    print(f"  ✓ Using {len(dataset)} images")
    
    # DataLoader
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, num_workers=8, pin_memory=True)
    
    # Load pre-trained models
    print("\nLoading models...")
    from ..app.models import resnet, vit
    print("  ✓ Models loaded")
    
    # Extract activations
    resnet_acts = extract_resnet_activations(resnet, dataloader, device)
    vit_acts = extract_vit_activations(vit, dataloader, device)
    
    # Save activations
    print("\nSaving activations...")
    torch.save(resnet_acts, os.path.join(SAVE_DIR, 'resnet_activations.pt'))
    torch.save(vit_acts, os.path.join(SAVE_DIR, 'vit_activations.pt'))
    print(f"  ✓ Saved to {SAVE_DIR}/")
    
    # Print statistics
    print("\n" + "="*80)
    print("Extraction Summary")
    print("="*80)
    print(f"ResNet-18 activations: {resnet_acts.shape}")
    print(f"  Mean: {resnet_acts.mean().item():.4f}, Std: {resnet_acts.std().item():.4f}")
    print(f"ViT-B activations: {vit_acts.shape}")
    print(f"  Mean: {vit_acts.mean().item():.4f}, Std: {vit_acts.std().item():.4f}")
    print("="*80)
    print("\n✓ Activation extraction complete. Ready for USAE training.")

# if __name__ == "__main__":
#     main()