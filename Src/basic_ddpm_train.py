import torch
import deepinv as dinv
print(dinv.__version__)
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

device = 'cuda'
batch_size = 32
image_size = 32

transform = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.0,), (1.0,))
    ]
)

train_dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(root="./datas", train=True, download=True, transform=transform),
    shuffle=True # shuffle 表示随机打乱顺序
)

# 创建日志目录
log_dir = "ddpm_train_log"
os.makedirs(log_dir, exist_ok=True)

# 时间戳用于区分不同的训练
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
current_log_dir = os.path.join(log_dir, timestamp)
os.makedirs(current_log_dir, exist_ok=True)

lr = 1e-4
epochs = 100

model = dinv.models.DiffUNet(in_channels=1, out_channels=1, pretrained=None).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = dinv.loss.MSE()

# 设置扩散模型参数
beta_start = 1e-4
beta_end = 0.02
timesteps = 1000

betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)  # 对α进行连乘积
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)  # 根号α
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)  # 根号1-α

# 记录每个epoch的损失
epoch_loss_list = []

# 训练循环
model.train()
for epoch in tqdm(range(epochs), desc="Epoch", total=epochs):
    epoch_loss = 0.0
    batch_count = 0
    
    batch_bar = tqdm(
        train_dataloader,
        desc=f"Train {epoch + 1}/{epochs}",
        leave=False,
        total=len(train_dataloader)
    )

    for data, _ in batch_bar:
        imgs = data.to(device)
        noise = torch.randn_like(imgs)
        t = torch.randint(0, timesteps, (imgs.size(0),), device=device)

        noised_imgs = (
            sqrt_alphas_cumprod[t, None, None, None] * imgs +
            sqrt_one_minus_alphas_cumprod[t, None, None, None] * noise
        )

        optimizer.zero_grad()
        estimate_noise = model(noised_imgs, t, type_t="timestep")  # 预测第t步的噪声
        loss = loss_fn(estimate_noise, noise)  # 对比噪声
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        batch_count += 1
        batch_bar.set_postfix(loss=f"{loss.item():.6f}")
    
    # 计算平均损失
    avg_epoch_loss = epoch_loss / batch_count
    epoch_loss_list.append(avg_epoch_loss)
    
    # 每10个epoch打印一次
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Average Loss: {avg_epoch_loss:.6f}")

# 保存模型
model_path = os.path.join(current_log_dir, "trained_basic_diffusion_model.pth")
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), epoch_loss_list, linewidth=2, marker='o', markersize=4)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Average Loss', fontsize=12)
plt.title('Training Loss Curve', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 保存损失曲线
loss_curve_path = os.path.join(current_log_dir, "loss_curve.png")
plt.savefig(loss_curve_path, dpi=100)
print(f"Loss curve saved to {loss_curve_path}")
plt.close()

# 保存数值日志
log_file = os.path.join(current_log_dir, "training_log.txt")
with open(log_file, 'w') as f:
    f.write(f"Training Configuration\n")
    f.write(f"{'='*50}\n")
    f.write(f"Device: {device}\n")
    f.write(f"Batch Size: {batch_size}\n")
    f.write(f"Image Size: {image_size}\n")
    f.write(f"Learning Rate: {lr}\n")
    f.write(f"Epochs: {epochs}\n")
    f.write(f"Timesteps: {timesteps}\n")
    f.write(f"\n")
    f.write(f"Training Results\n")
    f.write(f"{'='*50}\n")
    for epoch_idx, avg_loss in enumerate(epoch_loss_list):
        f.write(f"Epoch {epoch_idx + 1}: Loss = {avg_loss:.6f}\n")
    
    f.write(f"\n")
    f.write(f"Final Statistics\n")
    f.write(f"{'='*50}\n")
    f.write(f"Min Loss: {min(epoch_loss_list):.6f}\n")
    f.write(f"Max Loss: {max(epoch_loss_list):.6f}\n")
    f.write(f"Final Loss: {epoch_loss_list[-1]:.6f}\n")

print(f"Training log saved to {log_file}")
print(f"\nAll results saved to directory: {current_log_dir}")