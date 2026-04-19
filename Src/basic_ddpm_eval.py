import random
import torch
import deepinv as dinv
import matplotlib.pyplot as plt
from pathlib import Path

device = 'cuda'
image_size = 32

checkpoint_path = "DDPM/checkpoints/trained_basic_diffusion_model.pth"
model = dinv.models.DiffUNet(in_channels=1, out_channels=1, pretrained=Path(checkpoint_path)
                             ).to(device)

# 设置扩散模型参数，必须和训练时一样，不然出现模型不匹配
beta_start = 1e-4
beta_end = 0.02
timesteps = 1000

betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)  # 对α进行连乘积
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)  # 根号α
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)  # 根号1-α

model.eval()

n_samples = 10

# 随机选3个样本 index 来展示
selected_idx = random.sample(range(n_samples), 3)

# 用来存储每200 step的结果，并确保包含 t=0
save_steps = sorted(set(list(range(0, timesteps, 200)) + [0]), reverse=True)

def eval():
    with torch.no_grad():
        x = torch.randn(n_samples, 1, image_size, image_size).to(device)

        results = {idx: [] for idx in selected_idx}

        for t in reversed(range(timesteps)):
            t_tensor = torch.ones(n_samples, device=device).long() * t

            predicted_noise = model(x, t_tensor, type_t="timestep")

            alpha = alphas[t]
            alpha_cumprod = alphas_cumprod[t]
            # sqrt_alpha_cumprod = sqrt_alphas_cumprod[t]
            beta = betas[t]

            if t>0:
                noise = torch.randn_like(x)
            else:
                noise = 0

            x = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
            ) + torch.sqrt(beta) * noise

            # 记录中间结果（强制包含 t=0）
            if t in save_steps or t == 0:
                for idx in selected_idx:
                    results[idx].append(x[idx, 0].cpu())

    return results

def visulization(results):
    fig, axes = plt.subplots(3, len(save_steps), figsize=(15, 6))

    for row, idx in enumerate(selected_idx):
        for col, img in enumerate(results[idx]):
            axes[row, col].imshow(img, cmap='gray')
            axes[row, col].axis('off')

            if row == 0:
                axes[row, col].set_title(f"t={save_steps[col]}")

    plt.tight_layout()
    plt.show()

def main():
    # 推理阶段
    eval_results = eval()

    # 可视化结果
    visulization(eval_results)

if __name__ == "__main__":
    main()