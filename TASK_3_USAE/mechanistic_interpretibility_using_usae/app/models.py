import torch
import torchvision.models as models
import timm

resnet = models.resnet18(weights='IMAGENET1K_V1')
resnet.eval()

# ViT-B/16 pre-trained on ImageNet-1k  
vit = timm.create_model('vit_base_patch16_224', pretrained=True)
vit.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = resnet.to(device)
vit = vit.to(device)

print("Models downloaded and ready!")
print(f"ResNet on: {next(resnet.parameters()).device}")
print(f"ViT on: {next(vit.parameters()).device}")
