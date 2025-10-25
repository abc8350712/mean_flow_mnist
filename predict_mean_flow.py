import torch
import torch.nn as nn
from torchvision.utils import save_image
import math
import os
import argparse
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SHAPE = (1, 28, 28)
IMAGE_DIM = 784
DEFAULT_MODEL_PATH = "meanflow_mnist_unet.pth"

class PositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        t = t.view(-1, 1) 
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t * embeddings
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.conv_res = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels)
        )
        self.activation = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.net(x)
        time_cond = self.time_mlp(t_emb)
        h = h + time_cond.unsqueeze(-1).unsqueeze(-1)
        return self.activation(h + self.conv_res(x))

class SimpleUNet(nn.Module):
    def __init__(self, image_channels=1, time_emb_dim=128):
        super().__init__()
        
        self.time_emb = PositionalEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim * 2, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        self.conv_in = nn.Conv2d(image_channels, 64, kernel_size=3, padding=1)
        self.down1 = UNetBlock(64, 128, time_emb_dim)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = UNetBlock(128, 256, time_emb_dim)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = UNetBlock(256, 256, time_emb_dim)

        self.up1_deconv = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up1_block = UNetBlock(128 + 256, 128, time_emb_dim)
        
        self.up2_deconv = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up2_block = UNetBlock(64 + 128, 64, time_emb_dim)

        self.conv_out = nn.Conv2d(64, image_channels, kernel_size=1)

    def forward(self, z, r, t):
        z_img = z.view(-1, *IMAGE_SHAPE)
        
        r_emb = self.time_emb(r)
        t_emb = self.time_emb(t)
        time_input = torch.cat([r_emb, t_emb], dim=1)
        time_cond = self.time_mlp(time_input)

        h1 = self.conv_in(z_img)
        h2 = self.down1(h1, time_cond)
        h3 = self.pool1(h2)
        h4 = self.down2(h3, time_cond)
        h5 = self.pool2(h4)
        
        h_bottle = self.bottleneck(h5, time_cond)
        
        up1 = self.up1_deconv(h_bottle)
        up1_cat = torch.cat([up1, h4], dim=1)
        up1_out = self.up1_block(up1_cat, time_cond)
        
        up2 = self.up2_deconv(up1_out)
        up2_cat = torch.cat([up2, h2], dim=1)
        up2_out = self.up2_block(up2_cat, time_cond)
        
        output_img = self.conv_out(up2_out)
        
        return output_img.view(z.size(0), -1)


def generate_samples(model, n_samples=64, n_steps=1, output_file="generated_samples.png"):
    print(f"Generating {n_samples} samples in {n_steps} step(s)...")
    model.eval()
    
    with torch.no_grad():
        z = torch.randn(n_samples, IMAGE_DIM, device=DEVICE)
        
        times = torch.linspace(1.0, 0.0, n_steps + 1, device=DEVICE)

        for i in tqdm(range(n_steps), desc="Generating"):
            t_start = times[i]
            t_end = times[i+1]
            
            t_tensor = torch.full((n_samples, 1), t_start, device=DEVICE)
            r_tensor = torch.full((n_samples, 1), t_end, device=DEVICE)
            
            u_pred = model(z, r_tensor, t_tensor)
            
            z = z - u_pred
            
        generated_images = z.view(n_samples, *IMAGE_SHAPE)
        
        generated_images = (generated_images + 1) / 2
        
        save_image(
            generated_images.clamp(0.0, 1.0),
            output_file,
            nrow=int(math.sqrt(n_samples))
        )
    print(f"Samples saved to '{output_file}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate samples from the Mean Flow (UNet) model")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=DEFAULT_MODEL_PATH, 
        help=f"Path to the trained .pth model (default: {DEFAULT_MODEL_PATH})"
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=2, 
        help="Number of inference steps (integration steps). 1 is the paper's default."
    )
    parser.add_argument(
        "--n_samples", 
        type=int, 
        default=64, 
        help="Number of samples to generate (preferably a perfect square, e.g., 64)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="generated_steps_{steps}.png", 
        help="Output file. Use {steps} to include the number of steps."
    )
    
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at '{args.model_path}'")
        print("Please train the model first by running 'mean_flow_mnist_unet_fixed.py'")
        exit(1)

    print(f"Using device: {DEVICE}")

    try:
        model = SimpleUNet().to(DEVICE)
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
        print(f"Model loaded successfully from '{args.model_path}'")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure the model architecture in this script matches the saved .pth file.")
        exit(1)

    output_filename = args.output.format(steps=args.steps)
    os.makedirs(os.path.dirname(output_filename) or '.', exist_ok=True)

    generate_samples(
        model, 
        n_samples=args.n_samples, 
        n_steps=args.steps, 
        output_file=output_filename
    )