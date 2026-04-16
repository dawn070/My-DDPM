import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

def fit_function(t, x0, beta):
    x_t_1 = x0.copy()
    history = [x0.copy()]

    for i in range(t):
        epsilon = np.random.rand(*x_t_1.shape)
        beta_t = beta[i]
        x_t = np.sqrt(1 - beta_t) * x_t_1 + beta_t * epsilon
        x_t_1 = x_t.copy()
        history.append(x_t.copy())
    return history

def main():
    # 定义步长和噪声强度β,β要在0-1之间，不然不会收敛
    t = 100
    # 使用线性增长的β值
    beta = np.linspace(0.001, 0.1, t)

    # 定义一个均匀分布
    n_sample = 10000
    u0 = np.random.uniform(low=0, high=1, size=n_sample)

    # 使用 fit_function 进行标准正态分布的逼近
    u_history = fit_function(t, u0, beta)

    # 可视化
    plt.figure(figsize=(12, 8))

    for i, step in enumerate([0, 10, 30, 50, 70, 100]):
        plt.subplot(2, 3, i+1)
        plt.hist(u_history[step], bins=50, density=True)
        plt.title(f"t = {step}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()