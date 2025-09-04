import numpy as np
import sys
import torch
from torch import nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'

latent_dim = 100

# Pixel Norm
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-10)

# function to generate normalized random noise
def rand_noise(batch = 1):
    noise = torch.randn(batch, latent_dim, device=device)
    noise = noise / (noise.norm(dim=1, keepdim=True) + 1e-10)
    return noise

# Generator
class UpBlockOne(nn.Module):
    """
    Simple upsampling block:
      - Bilinear upsampling ( scale ×2 )
      - Convolution layer ( To extract meaningful features )
      - PReLU activation
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.PReLU()
        )

    def forward(self, x):
        return self.block(x)  # Direct bilinear upsample followed by convolution and PReLU activation


class UpBlockTwo(nn.Module):
    """
    More complex upsampling block composed of two parallel paths:
      - Path 1: Single ConvTranspose2d (stride=2) -> upsample
      - Path 2: A convolution with stride = 1 to reduce channels followed by a convolution transpose with stride 2x to up-sample
      - Outputs from both paths are concatenated along channels,
        then fused with a Conv2d followed by a PReLU activation.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.p1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.PReLU(),
        )
        self.p2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, (out_channels + in_channels) // 2, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d((out_channels + in_channels) // 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.PReLU(),
        )
        self.b1 = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
        )

    def forward(self, x):
        y1 = self.p1(x)                 # Path 1
        y2 = self.p2(x)                 # Path 2
        y = torch.cat((y1, y2), dim=1)  # channel-wise concat
        return self.b1(y)               # fuse both paths


class Generator(nn.Module):
    """
    Generator network composed of:
      1. Latent → feature projection (fully connected + PixelNorm + activations)
      2. Progressive upsampling via transposed convolutions and custom blocks
      3. Parallel DCGAN-style path (dcgan1) and bilinear+conv path (dcgan2)
      4. Final convolution to RGB (3 channels), scaled with Tanh
    """
    def __init__(self):
        super().__init__()
        
        # === Latent projection & initial feature map ===
        self.layerStart = nn.Sequential(
            nn.Flatten(),                                    # Flatten latent vector

            # Mapping the latent space ( inspired from styleGAN architecture )
            nn.Linear(latent_dim, 100),
            PixelNorm(),
            nn.LeakyReLU(0.2),
            
            nn.Linear(100, 100),
            PixelNorm(),
            nn.LeakyReLU(0.2),
            
            nn.Linear(100, 512),                             # Project to 512 channels
            PixelNorm(),
            nn.LeakyReLU(0.2),
            
            nn.Unflatten(1, (512, 1, 1)),                    # Reshape -> (512, 1, 1)
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=1, padding=0),  # Upsample -> (512, 4, 4)
            nn.LeakyReLU(0.2),
        )
        
        # --- Path 1: DCGAN-style transposed convolutions + custom block --- #
        self.dcgan1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # -> (256, 8, 8)
            nn.PReLU(),

            UpBlockTwo(256, 128),                                              # -> (128, 16, 16)

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # -> (64, 32, 32)
            nn.PReLU(),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # -> (32, 64, 64)
            nn.PReLU()
        )
        
        # === Path 2: Bilinear upsampling + convolution blocks ===
        self.dcgan2 = nn.Sequential(
            UpBlockOne(512, 256),   # -> (256, 8, 8)
            UpBlockOne(256, 128),   # -> (128, 16, 16)
            UpBlockOne(128, 64),    # -> (64, 32, 32)
            UpBlockOne(64, 32)      # -> (32, 64, 64)
        )
        
        # === Final fusion to RGB output ===
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 3, 3, 1, 1),  # Obtaining a 3 channel RGB image
            nn.Tanh()                   # Output normalized to [-1, 1]
        )

    def forward(self, x):
        x = self.layerStart(x)          # latent vector to (512, 4, 4)
        y1 = self.dcgan1(x)             # Path 1: ConvTranspose2d-based
        y2 = self.dcgan2(x)             # Path 2: Upsample+Conv-based
        y = torch.cat((y1, y2), dim=1)  # Concatenate features (channel axis)
        return self.conv1(y)            # Final RGB image output


# Wasserstein Loss
def wloss(D, real, fake, lambda_gp=10):
    eps = torch.rand(real.size(0), 1, 1, 1, device=real.device, requires_grad=True)
    mixed = eps * real + (1 - eps) * fake
    mixed_pred = D(mixed)
    grads = torch.autograd.grad(mixed_pred.sum(), mixed, create_graph=True)[0]
    gp = ((grads.view(grads.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()
    real_pred = D(real)
    fake_pred = D(fake)
    return fake_pred.mean() - real_pred.mean() + lambda_gp * gp

# Critic definition
class Block(nn.Module):
    """
    Residual-style block with:
      - Conv2d (inp → out), LeakyReLU
      - Conv2d (out → out), LeakyReLU
      - Skip connection: output of first conv (b1) + second conv (b2)
      - PReLU activation applied after skip addition
      - MaxPool2d for downsampling (factor 2)

    Purpose:
      - Learns richer features by allowing gradient flow through skip
        connection (residual learning).
      - Reduces spatial resolution by half after feature extraction.
    """
    def __init__(self, inp, out):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(inp,out,3,1,1),    # First convolution block
            nn.LeakyReLU(0.2),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(out,out,3,1,1),    # Second Convolution block
            nn.LeakyReLU(0.2),
        )
        self.prelu = nn.PReLU()
        self.pool = nn.MaxPool2d(2,2)

    def forward(self,x):
        x = self.b1(x)                   # First conv path
        return self.pool(self.prelu(x + self.b2(x))) # Second conv path with skip connection

class Discriminator(nn.Module):
    """
    Discriminator for GAN:
      - Input: RGB image (3×64×64)
      - Series of conv + residual blocks with downsampling
      - Global average pooling followed by fully connected layers
      - Final output: single scalar ( critic score )
    """
    def __init__(self):
        super().__init__()
        # Input size --> (*, 3, 64, 64)
        self.l1 = nn.Sequential(
            nn.Conv2d(3,64,3,1,1),        # Basic feature extraction
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2,2),            # Downsample to (64, 32, 32)

            Block(64, 128),               # Residual block → (128, 16, 16)
            Block(128, 256),              # Residual block → (256, 8, 8)
            Block(256, 384),              # Residual block → (384, 4, 4)
            nn.AdaptiveAvgPool2d((1,1)),  # Global average pooling → (384, 1, 1)
            nn.Flatten(),                 # → (384,)

            # Fully-connected layers
            nn.Linear(384, 192), # -> (192,)
            nn.PReLU(),
            nn.Linear(192, 192), # -> (192,)
            nn.PReLU(),
            # Final output scalar
            nn.Linear(192,1) # -> (1,)
        )
        
    def forward(self,x):
        return self.l1(x)
    
# Paths to the model
generator_path = 'models/generator31.pth'
filter_path = 'models/filter.pth'

# Register alias so torch.load finds it
sys.modules['__main__'].Generator = Generator
sys.modules['__main__'].Discriminator = Discriminator
sys.modules['__main__'].PixelNorm = PixelNorm
sys.modules['__main__'].Block = Block
sys.modules['__main__'].UpBlockOne = UpBlockOne
sys.modules['__main__'].UpBlockTwo = UpBlockTwo

# Actual models
generator = torch.load(generator_path, map_location=device, weights_only=False).to(device)
filter = torch.load(filter_path, map_location=device, weights_only=False).to(device)

generator.eval()
filter.eval()

def generate():
    temp = -100
    temp_img = generator(rand_noise())  # done to prevent NoneType error
    for _ in range(16):
        img = generator(rand_noise())
        sc = filter(img).item()
        if temp < sc:
            temp = sc
            temp_img = img

    arr = temp_img.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

    # If generator outputs [-1,1] from tanh, rescale → [0,1]
    arr = (arr + 1) / 2 if arr.min() < 0 else arr  

    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)

    return arr