"""
Model Architecture Analysis Script
Analyzes ResNet-18 and ViT-B to understand layer structure and dimensions
"""

import torch
import torchvision.models as models
import timm

def analyze_resnet():
    """Analyze ResNet-18 architecture"""
    print("="*80)
    print("ResNet-18 Architecture Analysis")
    print("="*80)
    
    resnet = models.resnet18(weights='IMAGENET1K_V1')
    resnet.eval()
    
    # Print all named modules
    print("\nAll layers:")
    for name, module in resnet.named_modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            print(f"  {name}: {module.__class__.__name__}")
    
    # Test activation shapes at different layers
    print("\n" + "-"*80)
    print("Activation shapes at key layers:")
    print("-"*80)
    
    dummy_input = torch.randn(1, 3, 224, 224)
    
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.shape
        return hook
    
    # Register hooks at key layers
    resnet.layer1.register_forward_hook(get_activation('layer1'))
    resnet.layer2.register_forward_hook(get_activation('layer2'))
    resnet.layer3.register_forward_hook(get_activation('layer3'))
    resnet.layer4.register_forward_hook(get_activation('layer4'))
    resnet.avgpool.register_forward_hook(get_activation('avgpool'))
    
    with torch.no_grad():
        _ = resnet(dummy_input)
    
    for name, shape in activations.items():
        print(f"  {name}: {shape}")
    
    # Recommended layer for USAE
    print("\n" + "="*80)
    print("RECOMMENDED: Use layer4 output (before avgpool)")
    print(f"  Shape: {activations['layer4']} → After avgpool: {activations['avgpool']}")
    print(f"  Flattened dimension: {activations['avgpool'][1]}")
    print("="*80)

def analyze_vit():
    """Analyze ViT-B/16 architecture"""
    print("\n\n" + "="*80)
    print("ViT-B/16 Architecture Analysis")
    print("="*80)
    
    vit = timm.create_model('vit_base_patch16_224', pretrained=True)
    vit.eval()
    
    # Print transformer blocks
    print("\nTransformer Blocks:")
    for i, block in enumerate(vit.blocks):
        print(f"  Block {i}: {block.__class__.__name__}")
    
    # Test activation shapes
    print("\n" + "-"*80)
    print("Activation shapes at key blocks:")
    print("-"*80)
    
    dummy_input = torch.randn(1, 3, 224, 224)
    
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.shape
        return hook
    
    # Register hooks at different blocks
    vit.blocks[5].register_forward_hook(get_activation('block_5'))
    vit.blocks[8].register_forward_hook(get_activation('block_8'))
    vit.blocks[11].register_forward_hook(get_activation('block_11'))
    vit.norm.register_forward_hook(get_activation('norm_final'))
    
    with torch.no_grad():
        _ = vit(dummy_input)
    
    for name, shape in activations.items():
        print(f"  {name}: {shape}")
    
    # Explain ViT output structure
    print("\n" + "="*80)
    print("ViT Output Structure:")
    print(f"  Shape: [batch, num_tokens, embed_dim]")
    print(f"  num_tokens = 197 (1 CLS token + 196 patch tokens for 224x224 image)")
    print(f"  embed_dim = 768 (ViT-B)")
    print("\nRECOMMENDED: Use block 11 output, extract CLS token ([:, 0, :])")
    print(f"  This gives dimension: 768")
    print("="*80)

def compare_dimensions():
    """Compare dimensions for USAE setup"""
    print("\n\n" + "="*80)
    print("USAE Setup Recommendations")
    print("="*80)
    
    print("\nModel Activation Dimensions:")
    print("  ResNet-18 (layer4 → avgpool): 512")
    print("  ViT-B (block 11, CLS token): 768")
    
    print("\nShared Feature Space Z:")
    print("  Recommended dimension: m = 4096 or 8192")
    print("  Must satisfy: m >> max(512, 768)")
    print("  TopK sparsity: k = 32 (as per paper)")
    
    print("\nUSAE Architecture:")
    print("  Encoder_ResNet: 512 → 8192")
    print("  Encoder_ViT: 768 → 8192")
    print("  Shared Z: 8192-dim (sparse, ~32 active)")
    print("  Decoder_ResNet: 8192 → 512")
    print("  Decoder_ViT: 8192 → 768")
    print("="*80)

# if __name__ == "__main__":
#     # Analyze both models
#     analyze_resnet()
#     analyze_vit()
#     compare_dimensions()
    
#     print("\n\n✓ Analysis complete. Ready to extract activations.")