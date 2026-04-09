"""一个适合初学者的 SVD 降噪示例：秩 2 信号 + 小噪声。

运行示例：
    python svd2.py
    python svd2.py --save-dir figures
    python svd2.py --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

np.set_printoptions(precision=3, suppress=True)


def build_clean_matrix(n: int = 8) -> dict[str, np.ndarray]:
    """构造与 svd1.py 同源的精确秩 2 矩阵。"""

    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    X_grid, Y_grid = np.meshgrid(x, y)

    X_clean = X_grid + Y_grid + np.sin(3.0 * X_grid)
    g = x + np.sin(3.0 * x)
    ones = np.ones(n)
    X_factored = np.outer(y, ones) + np.outer(ones, g)

    if not np.allclose(X_clean, X_factored):
        raise ValueError("矩阵分解检查失败：X_clean 与 y1^T + 1g^T 不一致。")

    return {
        "x": x,
        "y": y,
        "g": g,
        "X_clean": X_clean,
    }


def add_noise(X_clean: np.ndarray, noise_level: float, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """在干净的低秩矩阵上叠加高斯噪声。"""

    rng = np.random.default_rng(seed)
    noise = noise_level * rng.standard_normal(size=X_clean.shape)
    X_noisy = X_clean + noise
    return X_noisy, noise


def truncated_svd(X: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回 thin SVD 以及 rank-r 截断重构。"""

    U, S, VT = np.linalg.svd(X, full_matrices=False)
    X_r = U[:, :r] @ np.diag(S[:r]) @ VT[:r, :]
    return U, S, VT, X_r


def print_report(
    data: dict[str, np.ndarray],
    X_noisy: np.ndarray,
    noise: np.ndarray,
    U: np.ndarray,
    S: np.ndarray,
    VT: np.ndarray,
    X_r: np.ndarray,
    r_compare: int,
) -> None:
    """打印说明。"""

    del U, VT  # 这里不直接打印向量，只使用奇异值和重构结果。

    X_clean = data["X_clean"]
    clean_rank = np.linalg.matrix_rank(X_clean)
    noisy_rank = np.linalg.matrix_rank(X_noisy)
    total_energy = np.sum(S**2)
    energy_each = S**2 / total_energy
    energy_cum = np.cumsum(S**2) / total_energy

    error_to_noisy = np.linalg.norm(X_noisy - X_r, ord="fro") / np.linalg.norm(X_noisy, ord="fro")
    error_to_clean = np.linalg.norm(X_clean - X_r, ord="fro") / np.linalg.norm(X_clean, ord="fro")
    noisy_vs_clean = np.linalg.norm(X_clean - X_noisy, ord="fro") / np.linalg.norm(X_clean, ord="fro")

    print("========== 干净矩阵的结构 ==========")
    print("X_clean[i, j] = x_j + y_i + sin(3 x_j) = y_i + g_j")
    print("因此 X_clean = y 1^T + 1 g^T，是两个秩 1 矩阵之和。")
    print(f"数值秩 rank(X_clean) = {clean_rank}")
    print("这说明干净信号本身依然是精确秩 2。")

    print("\n========== 加噪后的矩阵 ==========")
    print(f"噪声样本标准差 = {np.std(noise):.4f}")
    print(f"数值秩 rank(X_noisy) = {noisy_rank}")
    print("加入随机噪声后，矩阵通常会变成满秩；但若噪声较小，前几个奇异值仍然代表主结构。")

    print("\n========== 干净矩阵 X_clean ==========")
    print(np.round(X_clean, 3))

    print("\n========== 带噪矩阵 X_noisy ==========")
    print(np.round(X_noisy, 3))

    print("\n========== 奇异值与能量 ==========")
    for i, sigma in enumerate(S, start=1):
        print(
            f"第 {i:>2} 个奇异值: {sigma:>8.4f} | "
            f"单项能量占比: {energy_each[i - 1] * 100:>6.2f}% | "
            f"累计能量占比: {energy_cum[i - 1] * 100:>6.2f}%"
        )

    print(f"\n========== rank-{r_compare} 截断 SVD 重构 ==========")
    print(f"相对于带噪矩阵的相对误差: {error_to_noisy * 100:.2f}%")
    print(f"相对于干净矩阵的相对误差: {error_to_clean * 100:.2f}%")
    print(f"原始带噪矩阵相对于干净矩阵的误差: {noisy_vs_clean * 100:.2f}%")
    if error_to_clean < noisy_vs_clean:
        print("由于 rank-r 重构比原始带噪矩阵更接近 X_clean，所以这里确实出现了降噪效果。")
    else:
        print("这里没有明显优于原始带噪矩阵，说明当前噪声强度或截断秩还可以继续调整。")
    print("\n重构矩阵 X_r：")
    print(np.round(X_r, 3))


def render_plots(
    X_clean: np.ndarray,
    X_noisy: np.ndarray,
    X_r: np.ndarray,
    S_noisy: np.ndarray,
    S_clean: np.ndarray,
    show: bool,
    save_dir: Path | None,
) -> None:
    """可选图形化输出。"""

    if not show and save_dir is None:
        return

    import matplotlib

    if not show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    energy_cum = np.cumsum(S_noisy**2) / np.sum(S_noisy**2)
    residual = X_noisy - X_r

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)

    im0 = axes[0, 0].imshow(X_clean, cmap="viridis")
    axes[0, 0].set_title("Clean Low-Rank Matrix")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(X_noisy, cmap="viridis")
    axes[0, 1].set_title("Noisy Matrix")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[0, 2].imshow(X_r, cmap="viridis")
    axes[0, 2].set_title("Truncated SVD Reconstruction")
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    im3 = axes[1, 0].imshow(residual, cmap="viridis")
    axes[1, 0].set_title("Residual: X_noisy - X_r")
    fig.colorbar(im3, ax=axes[1, 0], fraction=0.046)

    axes[1, 1].semilogy(np.arange(1, len(S_clean) + 1), S_clean, marker="o", label="clean")
    axes[1, 1].semilogy(np.arange(1, len(S_noisy) + 1), S_noisy, marker="s", label="noisy")
    axes[1, 1].set_title("Singular Value Spectrum")
    axes[1, 1].set_xlabel("Index")
    axes[1, 1].set_ylabel("sigma_i")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    axes[1, 2].plot(np.arange(1, len(S_noisy) + 1), energy_cum, marker="o")
    axes[1, 2].axhline(0.99, color="gray", linestyle="--", linewidth=1, label="99% energy")
    axes[1, 2].set_title("Cumulative Energy of Noisy Matrix")
    axes[1, 2].set_xlabel("Retained Rank")
    axes[1, 2].set_ylabel("Energy Ratio")
    axes[1, 2].set_ylim(0.0, 1.05)
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend()

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        output_path = save_dir / "svd2_summary.png"
        fig.savefig(output_path, dpi=180)
        print(f"\n图形已保存到：{output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="低秩矩阵 + 噪声的截断 SVD 示例")
    parser.add_argument("--n", type=int, default=8, help="矩阵边长，默认 8")
    parser.add_argument("--noise-level", type=float, default=0.08, help="高斯噪声标准差，默认 0.08")
    parser.add_argument("--rank", type=int, default=2, help="截断 SVD 使用的秩，默认 2")
    parser.add_argument("--show", action="store_true", help="显示图形窗口")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="保存图形的目录，例如 --save-dir figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.n < 2:
        raise ValueError("n 必须至少为 2，否则无法构造秩 2 示例。")
    if args.rank < 1 or args.rank > args.n:
        raise ValueError("截断秩 rank 必须满足 1 <= rank <= n。")

    data = build_clean_matrix(n=args.n)
    X_clean = data["X_clean"]
    X_noisy, noise = add_noise(X_clean, noise_level=args.noise_level, seed=42)

    U_noisy, S_noisy, VT_noisy, X_r = truncated_svd(X_noisy, r=args.rank)
    _, S_clean, _, _ = truncated_svd(X_clean, r=min(2, args.n))

    print_report(data, X_noisy, noise, U_noisy, S_noisy, VT_noisy, X_r, r_compare=args.rank)
    render_plots(
        X_clean=X_clean,
        X_noisy=X_noisy,
        X_r=X_r,
        S_noisy=S_noisy,
        S_clean=S_clean,
        show=args.show,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()

'''
========== 干净矩阵的结构 ==========
X_clean[i, j] = x_j + y_i + sin(3 x_j) = y_i + g_j
因此 X_clean = y 1^T + 1 g^T, 是两个秩 1 矩阵之和。
数值秩 rank(X_clean) = 2
这说明干净信号本身依然是精确秩 2。

========== 加噪后的矩阵 ==========
噪声样本标准差 = 0.0631
数值秩 rank(X_noisy) = 8
加入随机噪声后，矩阵通常会变成满秩；但若噪声较小，前几个奇异值仍然代表主结构。

========== 干净矩阵 X_clean ==========
[[0.    0.558 1.042 1.388 1.561 1.555 1.397 1.141]
 [0.143 0.701 1.185 1.531 1.704 1.698 1.54  1.284]
 [0.286 0.844 1.327 1.674 1.847 1.841 1.683 1.427]
 [0.429 0.987 1.47  1.817 1.99  1.984 1.825 1.57 ]
 [0.571 1.13  1.613 1.96  2.133 2.127 1.968 1.713]
 [0.714 1.273 1.756 2.102 2.275 2.269 2.111 1.855]
 [0.857 1.416 1.899 2.245 2.418 2.412 2.254 1.998]
 [1.    1.558 2.042 2.388 2.561 2.555 2.397 2.141]]

========== 带噪矩阵 X_noisy ==========
[[0.024 0.475 1.102 1.463 1.405 1.451 1.407 1.116]
 [0.142 0.633 1.255 1.593 1.709 1.788 1.577 1.215]
 [0.315 0.767 1.398 1.67  1.832 1.786 1.78  1.414]
 [0.394 0.959 1.513 1.846 2.023 2.018 1.997 1.537]
 [0.53  1.065 1.662 2.05  2.123 2.059 1.902 1.765]
 [0.774 1.316 1.703 2.121 2.285 2.287 2.181 1.873]
 [0.911 1.421 1.922 2.296 2.302 2.387 2.216 1.947]
 [0.978 1.678 1.972 2.466 2.427 2.528 2.41  2.188]]

========== 奇异值与能量 ==========
第  1 个奇异值:  13.5448 | 单项能量占比:  99.50% | 累计能量占比:  99.50%
第  2 个奇异值:   0.9008 | 单项能量占比:   0.44% | 累计能量占比:  99.94%
第  3 个奇异值:   0.2194 | 单项能量占比:   0.03% | 累计能量占比:  99.97%
第  4 个奇异值:   0.1492 | 单项能量占比:   0.01% | 累计能量占比:  99.98%
第  5 个奇异值:   0.1386 | 单项能量占比:   0.01% | 累计能量占比:  99.99%
第  6 个奇异值:   0.1098 | 单项能量占比:   0.01% | 累计能量占比: 100.00%
第  7 个奇异值:   0.0459 | 单项能量占比:   0.00% | 累计能量占比: 100.00%
第  8 个奇异值:   0.0187 | 单项能量占比:   0.00% | 累计能量占比: 100.00%

========== rank-2 截断 SVD 重构 ==========
相对于带噪矩阵的相对误差: 2.38%
相对于干净矩阵的相对误差: 2.87%
原始带噪矩阵相对于干净矩阵的误差: 3.73%
由于 rank-r 重构比原始带噪矩阵更接近 X_clean, 所以这里确实出现了降噪效果。

重构矩阵 X_r: 
[[0.051 0.475 1.095 1.382 1.482 1.482 1.41  1.074]
 [0.137 0.616 1.271 1.598 1.702 1.706 1.622 1.264]
 [0.296 0.79  1.379 1.72  1.81  1.823 1.732 1.404]
 [0.393 0.932 1.536 1.91  2.001 2.019 1.917 1.577]
 [0.541 1.089 1.624 2.008 2.086 2.111 2.003 1.695]
 [0.751 1.32  1.768 2.171 2.232 2.269 2.15  1.881]
 [0.875 1.458 1.856 2.272 2.323 2.367 2.242 1.994]
 [1.025 1.637 1.996 2.436 2.479 2.53  2.396 2.163]]
'''