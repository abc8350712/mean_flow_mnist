# mean_flow_mnist_unet_fixed.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd.functional import jvp
from tqdm import tqdm
import math
import os

# --- 1. 配置参数 (Hyperparameters) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 512
EPOCHS = 25
LR = 1e-4
IMAGE_SHAPE = (1, 28, 28)
IMAGE_DIM = 784
MODEL_PATH = "meanflow_mnist_unet.pth"
SAMPLES_DIR = "samples_unet"
os.makedirs(SAMPLES_DIR, exist_ok=True)


# --- 2. 模型定义 (Model Definition) ---

class PositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=DEVICE) * -embeddings)
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

        # Downsampling path
        self.conv_in = nn.Conv2d(image_channels, 64, kernel_size=3, padding=1)
        self.down1 = UNetBlock(64, 128, time_emb_dim)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = UNetBlock(128, 256, time_emb_dim)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = UNetBlock(256, 256, time_emb_dim)

        # Upsampling path
        self.up1_deconv = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # FIX: Input channels = 128 (from deconv) + 256 (from skip connection h4)
        self.up1_block = UNetBlock(128 + 256, 128, time_emb_dim)
        
        self.up2_deconv = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # FIX: Input channels = 64 (from deconv) + 128 (from skip connection h2)
        self.up2_block = UNetBlock(64 + 128, 64, time_emb_dim)

        # Output
        self.conv_out = nn.Conv2d(64, image_channels, kernel_size=1)

    def forward(self, z, r, t):
        z_img = z.view(-1, *IMAGE_SHAPE)
        
        r_emb = self.time_emb(r)
        t_emb = self.time_emb(t)
        time_input = torch.cat([r_emb, t_emb], dim=1)
        time_cond = self.time_mlp(time_input)

        # Downsampling
        h1 = self.conv_in(z_img)
        h2 = self.down1(h1, time_cond)
        h3 = self.pool1(h2)
        h4 = self.down2(h3, time_cond)
        h5 = self.pool2(h4)
        
        # Bottleneck
        h_bottle = self.bottleneck(h5, time_cond)
        
        # Upsampling
        up1 = self.up1_deconv(h_bottle)
        up1_cat = torch.cat([up1, h4], dim=1)
        up1_out = self.up1_block(up1_cat, time_cond)
        
        up2 = self.up2_deconv(up1_out)
        up2_cat = torch.cat([up2, h2], dim=1)
        up2_out = self.up2_block(up2_cat, time_cond)
        
        output_img = self.conv_out(up2_out)
        
        return output_img.view(z.size(0), -1)

# --- 3. 训练逻辑 (Training Logic) ---
def train(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    func_to_jvp = lambda z, r, t: model(z, r, t)
    progress_bar = tqdm(dataloader, desc="Training")
    for batch_idx, (x, _) in enumerate(progress_bar):
        optimizer.zero_grad()
        x = x.to(DEVICE).view(x.size(0), -1)
        epsilon = torch.randn_like(x)
        t_samples = torch.rand(x.size(0), 1, device=DEVICE)
        r_samples = torch.rand(x.size(0), 1, device=DEVICE)
        t, r = torch.max(t_samples, r_samples), torch.min(t_samples, r_samples)
        z_t = (1 - t) * x + t * epsilon
        v_t = epsilon - x
        tangents_z = v_t
        tangents_r = torch.zeros_like(r)
        tangents_t = torch.ones_like(t)
        u_pred_jvp, dudt = jvp(
            func_to_jvp,                                                                                                                                                                                                                                                                                                                                                               
            (z_t, r, t),
            (tangents_z, tangents_r, tangents_t)
        )
        u_tgt = v_t - (t - r) * dudt
        u_pred = model(z_t, r, t)
        loss = nn.functional.mse_loss(u_pred, u_tgt.detach())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch completed. Average Training Loss: {avg_loss:.4f}")

# --- 4. 推理逻辑 (Inference Logic) ---
def sample(model, epoch, n_samples=64):
    model.eval()
    with torch.no_grad():
        z_1 = torch.randn(n_samples, IMAGE_DIM, device=DEVICE)
        r = torch.zeros(n_samples, 1, device=DEVICE)
        t = torch.ones(n_samples, 1, device=DEVICE)
        u_full_path = model(z_1, r, t)
        z_0 = z_1 - u_full_path
        generated_images = z_0.view(n_samples, *IMAGE_SHAPE)
        generated_images = (generated_images + 1) / 2
        save_image(
            generated_images.clamp(0.0, 1.0),
            os.path.join(SAMPLES_DIR, f"sample_epoch_{epoch:03d}.png"),
            nrow=8
        )
    print(f"Saved {n_samples} samples for epoch {epoch}.")

# --- 5. 主执行函数 (Main Execution) ---
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    num_workers = 0 if os.name == 'nt' else 4
    dataset = MNIST(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    model = SimpleUNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    print("Starting training with SimpleUNet model...")
    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{EPOCHS} ---")
        train(model, dataloader, optimizer)
        sample(model, epoch)
        torch.save(model.state_dict(), MODEL_PATH)
    print("\nTraining finished.")
    print(f"Model saved to {MODEL_PATH}")
    print(f"Generated samples are in the '{SAMPLES_DIR}' directory.")