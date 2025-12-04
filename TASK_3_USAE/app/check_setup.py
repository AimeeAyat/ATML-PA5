"""
Setup Verification Script
Checks if all dependencies and GPU are properly configured
"""

import sys

def check_imports():
    """Check if all required packages are installed"""
    print("="*80)
    print("Checking Dependencies")
    print("="*80)
    
    packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'timm': 'Timm (Vision Transformers)',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'tqdm': 'TQDM',
    }
    
    all_good = True
    
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT INSTALLED")
            all_good = False
    
    return all_good

def check_gpu():
    """Check GPU availability and CUDA version"""
    print("\n" + "="*80)
    print("GPU Configuration")
    print("="*80)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"  ✓ CUDA version: {torch.version.cuda}")
            print(f"  ✓ Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"\n  GPU {i}: {props.name}")
                print(f"    Memory: {props.total_memory / 1024**3:.2f} GB")
                print(f"    Compute Capability: {props.major}.{props.minor}")
            
            # Test GPU with small tensor
            x = torch.randn(100, 100).cuda()
            y = x @ x.T
            print(f"\n  ✓ GPU test passed (matrix multiply)")
            
        else:
            print("  ✗ CUDA not available")
            print("  → Training will run on CPU (slow)")
            print("  → Install CUDA-enabled PyTorch:")
            print("     uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
    
    except Exception as e:
        print(f"  ✗ Error checking GPU: {e}")

def check_models():
    """Check if models can be downloaded"""
    print("\n" + "="*80)
    print("Model Download Test")
    print("="*80)
    
    try:
        import torch
        import torchvision.models as models
        import timm
        
        print("  Testing ResNet-18 download...")
        resnet = models.resnet18(weights='IMAGENET1K_V1')
        print(f"    ✓ ResNet-18 loaded")
        
        print("  Testing ViT-B download...")
        vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        print(f"    ✓ ViT-B loaded")
        
        # Test forward pass
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            _ = resnet(dummy)
            _ = vit(dummy)
        print(f"    ✓ Forward pass successful")
        
    except Exception as e:
        print(f"  ✗ Error loading models: {e}")

def check_directories():
    """Check if required directories exist or can be created"""
    print("\n" + "="*80)
    print("Directory Setup")
    print("="*80)
    
    from pathlib import Path
    
    dirs = {
        'activations': './activations',
        'checkpoints': './checkpoints',
        'logs': './logs',
    }
    
    for name, path in dirs.items():
        p = Path(path)
        if p.exists():
            print(f"  ✓ {name}: {path} (exists)")
        else:
            try:
                p.mkdir(parents=True, exist_ok=True)
                print(f"  ✓ {name}: {path} (created)")
            except Exception as e:
                print(f"  ✗ {name}: {path} (failed to create)")

def estimate_memory():
    """Estimate memory requirements"""
    print("\n" + "="*80)
    print("Memory Estimates")
    print("="*80)
    
    # Activation storage
    n_samples = 10000
    resnet_dim = 512
    vit_dim = 768
    bytes_per_float = 4
    
    resnet_mb = (n_samples * resnet_dim * bytes_per_float) / (1024**2)
    vit_mb = (n_samples * vit_dim * bytes_per_float) / (1024**2)
    
    print(f"  Activation storage ({n_samples} samples):")
    print(f"    ResNet: {resnet_mb:.1f} MB")
    print(f"    ViT: {vit_mb:.1f} MB")
    print(f"    Total: {resnet_mb + vit_mb:.1f} MB")
    
    # USAE model size
    hidden_dim = 8192
    model_1_params = (resnet_dim * hidden_dim) + (hidden_dim * resnet_dim)
    model_2_params = (vit_dim * hidden_dim) + (hidden_dim * vit_dim)
    total_params = model_1_params + model_2_params
    model_mb = (total_params * bytes_per_float) / (1024**2)
    
    print(f"\n  USAE model size:")
    print(f"    Parameters: {total_params / 1e6:.1f}M")
    print(f"    Memory: {model_mb:.1f} MB")
    
    # Training memory (rough estimate)
    batch_size = 128
    training_mb = (batch_size * (resnet_dim + vit_dim + 2*hidden_dim) * bytes_per_float) / (1024**2)
    
    print(f"\n  Training (batch_size={batch_size}):")
    print(f"    Per batch: ~{training_mb:.1f} MB")
    print(f"    Recommended GPU: >= 4 GB")

def main():
    """Run all checks"""
    print("\n" + "="*80)
    print("USAE Setup Verification")
    print("="*80 + "\n")
    
    # Run checks
    deps_ok = check_imports()
    check_gpu()
    
    if deps_ok:
        check_models()
    
    check_directories()
    estimate_memory()
    
    # Final summary
    print("\n" + "="*80)
    print("Setup Summary")
    print("="*80)
    
    if deps_ok:
        print("  ✓ All dependencies installed")
    else:
        print("  ✗ Missing dependencies - install them first")
    
    import torch
    if torch.cuda.is_available():
        print("  ✓ GPU available for training")
    else:
        print("  ⚠ No GPU - training will be slow")
    
    print("\n  Next steps:")
    print("    1. python analyze_models.py")
    print("    2. python extract_activations.py")
    print("    3. python train_usae.py")
    print("\n" + "="*80)

# if __name__ == "__main__":
#     main()