"""
Universal Sparse Autoencoder (USAE) Architecture
Implements multi-encoder, shared bottleneck, multi-decoder structure
with TopK sparsity for universal feature learning across models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKActivation(nn.Module):
    """
    TopK sparsity activation: keeps only top k values, zeros out rest
    This is more aggressive than L1 penalty - hard sparsity constraint
    """
    def __init__(self, k):
        super().__init__()
        self.k = k
    
    def forward(self, x):
        """
        Args:
            x: [batch, hidden_dim] activation tensor
        Returns:
            Sparse tensor with only top k values per sample
        """
        # Get top k values and indices for each sample
        topk_values, topk_indices = torch.topk(x, self.k, dim=-1)
        
        # Create sparse tensor: start with zeros
        sparse_x = torch.zeros_like(x)
        
        # Scatter top k values back to their positions
        # This creates a sparse representation with exactly k non-zero elements
        sparse_x.scatter_(-1, topk_indices, topk_values)
        
        return sparse_x

class USAEEncoder(nn.Module):
    """
    Encoder: Maps model-specific activation A^(i) to shared code Z
    
    Architecture:
        A^(i) (d_i dims) → pre-bias → linear → TopK → Z (m dims, k-sparse)
    """
    def __init__(self, input_dim, hidden_dim, k=32):
        """
        Args:
            input_dim: Dimension of model activation (512 for ResNet, 768 for ViT)
            hidden_dim: Dimension of shared feature space Z (e.g., 8192)
            k: TopK sparsity (number of active features)
        """
        super().__init__()
        
        # Pre-bias: learnable centering (b_pre in the paper)
        # Subtracting this helps normalize different model activations
        self.pre_bias = nn.Parameter(torch.zeros(input_dim))
        
        # Encoder weight: projects to shared space
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=False)
        
        # TopK activation for hard sparsity
        self.topk = TopKActivation(k)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.encoder.weight)
    
    def forward(self, x):
        """
        Args:
            x: [batch, input_dim] - model activation
        Returns:
            z: [batch, hidden_dim] - sparse shared code (k non-zero)
        """
        # Step 1: Center the activation
        x_centered = x - self.pre_bias
        
        # Step 2: Project to shared space
        z_dense = self.encoder(x_centered)
        
        # Step 3: Apply TopK sparsity
        # This implements: Z = TopK(W_enc * (A - b_pre))
        z_sparse = self.topk(z_dense)
        
        return z_sparse

class USAEDecoder(nn.Module):
    """
    Decoder: Maps shared code Z back to model-specific activation Â^(j)
    
    Architecture:
        Z (m dims, sparse) → linear → post-bias → Â^(j) (d_j dims)
    """
    def __init__(self, hidden_dim, output_dim):
        """
        Args:
            hidden_dim: Dimension of shared feature space Z
            output_dim: Dimension of target model activation
        """
        super().__init__()
        
        # Decoder weight: projects from shared space back to model space
        self.decoder = nn.Linear(hidden_dim, output_dim, bias=False)
        
        # Post-bias: learnable offset (b_dec in the paper)
        self.post_bias = nn.Parameter(torch.zeros(output_dim))
        
        # Initialize weights
        nn.init.xavier_uniform_(self.decoder.weight)
    
    def forward(self, z):
        """
        Args:
            z: [batch, hidden_dim] - sparse shared code
        Returns:
            x_recon: [batch, output_dim] - reconstructed activation
        """
        # Step 1: Project back to model space
        x_recon = self.decoder(z)
        
        # Step 2: Add post-bias
        # This implements: Â = Z * D + b_dec
        x_recon = x_recon + self.post_bias
        
        return x_recon

class UniversalSAE(nn.Module):
    """
    Universal Sparse Autoencoder
    
    Architecture:
        M encoders (model-specific) → Shared Z (sparse) → M decoders (model-specific)
    
    Key property: Any encoder can produce Z that all decoders can use
    This forces Z to capture universal features across models
    """
    def __init__(self, model_dims, hidden_dim=8192, k=32):
        """
        Args:
            model_dims: List of activation dimensions [d1, d2, ...] for M models
            hidden_dim: Dimension of shared feature space Z
            k: TopK sparsity
        """
        super().__init__()
        
        self.num_models = len(model_dims)
        self.model_dims = model_dims
        self.hidden_dim = hidden_dim
        self.k = k
        
        # Create M encoders (one per model)
        self.encoders = nn.ModuleList([
            USAEEncoder(dim, hidden_dim, k) for dim in model_dims
        ])
        
        # Create M decoders (one per model)
        self.decoders = nn.ModuleList([
            USAEDecoder(hidden_dim, dim) for dim in model_dims
        ])
        
        print(f"✓ USAE initialized:")
        print(f"  Models: {self.num_models}")
        print(f"  Input dims: {model_dims}")
        print(f"  Shared space: {hidden_dim}")
        print(f"  TopK sparsity: {k}")
    
    def encode(self, x, model_idx):
        """
        Encode activation from model_idx to shared code Z
        
        Args:
            x: [batch, d_i] - activation from model i
            model_idx: which model's encoder to use (0 to M-1)
        Returns:
            z: [batch, hidden_dim] - sparse shared code
        """
        return self.encoders[model_idx](x)
    
    def decode(self, z, model_idx):
        """
        Decode shared code Z to model_idx's activation space
        
        Args:
            z: [batch, hidden_dim] - sparse shared code
            model_idx: which model's decoder to use (0 to M-1)
        Returns:
            x_recon: [batch, d_i] - reconstructed activation
        """
        return self.decoders[model_idx](z)
    
    def forward(self, activations, source_idx):
        """
        USAE forward pass: Universal reconstruction
        
        Args:
            activations: List of [batch, d_i] tensors (one per model)
            source_idx: which model to encode from (random in training)
        
        Returns:
            z: shared code
            reconstructions: List of reconstructed activations (one per model)
        """
        # Step 1: Encode from source model
        z = self.encode(activations[source_idx], source_idx)
        
        # Step 2: Decode to ALL models (universal reconstruction)
        # This is the key: same Z must reconstruct all models' activations
        reconstructions = [
            self.decode(z, i) for i in range(self.num_models)
        ]
        
        return z, reconstructions
    
    def compute_loss(self, activations, reconstructions):
        """
        Compute aggregate reconstruction loss across all models
        
        L = Σ_j ||A^(j) - Â^(j)||²_F
        
        Args:
            activations: List of ground truth activations
            reconstructions: List of reconstructed activations
        Returns:
            total_loss: scalar
            individual_losses: List of per-model losses (for monitoring)
        """
        individual_losses = []
        
        for act_true, act_recon in zip(activations, reconstructions):
            # Frobenius norm (L2) for each model
            loss = F.mse_loss(act_recon, act_true, reduction='mean')
            individual_losses.append(loss)
        
        # Sum across all models
        total_loss = sum(individual_losses)
        
        return total_loss, individual_losses
    
    def get_sparsity(self, z):
        """
        Measure actual sparsity of Z
        
        Returns:
            avg_active: average number of active (non-zero) features
            sparsity_ratio: fraction of active features
        """
        # Count non-zero elements per sample
        active = (z != 0).float().sum(dim=-1).mean().item()
        ratio = active / self.hidden_dim
        
        return active, ratio

# if __name__ == "__main__":
#     """Test USAE architecture"""
    
#     # Configuration
#     batch_size = 32
#     model_dims = [512, 768]  # ResNet, ViT
#     hidden_dim = 8192
#     k = 32
    
#     # Create USAE
#     usae = UniversalSAE(model_dims, hidden_dim, k)
    
#     # Dummy activations
#     resnet_act = torch.randn(batch_size, 512)
#     vit_act = torch.randn(batch_size, 768)
#     activations = [resnet_act, vit_act]
    
#     # Forward pass (encode from ResNet)
#     print("\n" + "="*80)
#     print("Testing forward pass...")
#     z, recons = usae(activations, source_idx=0)
    
#     print(f"  Z shape: {z.shape}")
#     active, ratio = usae.get_sparsity(z)
#     print(f"  Sparsity: {active:.1f} active / {hidden_dim} total ({ratio*100:.2f}%)")
#     print(f"  Reconstructions: {[r.shape for r in recons]}")
    
#     # Compute loss
#     loss, individual_losses = usae.compute_loss(activations, recons)
#     print(f"  Total loss: {loss.item():.4f}")
#     print(f"  Individual losses: {[l.item() for l in individual_losses]}")
#     print("="*80)
#     print("\n✓ USAE architecture test passed!")