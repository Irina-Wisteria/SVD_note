"""一个适合初学者的 SVD 最小示例：精确秩 2 矩阵。

运行示例：
    python svd1.py
    python svd1.py --save-dir figures
    python svd1.py --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

np.set_printoptions(precision=3, suppress=True)


def build_exact_rank_2_matrix(n: int = 5) -> dict[str, np.ndarray]:
    """构造一个精确秩 2 的矩阵。
    """

    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    X_grid, Y_grid = np.meshgrid(x, y)

    X = X_grid + Y_grid + np.sin(3.0 * X_grid)
    g = x + np.sin(3.0 * x)
    ones = np.ones(n)
    X_factored = np.outer(y, ones) + np.outer(ones, g)

    if not np.allclose(X, X_factored):
        raise ValueError("矩阵分解检查失败: X 与 y1^T + 1g^T 不一致。")

    return {
        "x": x,
        "y": y,
        "g": g,
        "ones": ones,
        "X": X,
    }


def compute_svd_story(X: np.ndarray) -> dict[str, np.ndarray]:
    """计算 SVD、逐层外积展开及其误差。"""

    U, S, VT = np.linalg.svd(X, full_matrices=False)
    layers = np.array([S[i] * np.outer(U[:, i], VT[i, :]) for i in range(len(S))])
    reconstructions = np.cumsum(layers, axis=0)

    total_energy = np.sum(S**2)
    cumulative_energy = np.cumsum(S**2) / total_energy
    relative_errors = np.array(
        [
            np.linalg.norm(X - reconstructions[i], ord="fro") / np.linalg.norm(X, ord="fro")
            for i in range(len(S))
        ]
    )

    return {
        "U": U,
        "S": S,
        "VT": VT,
        "layers": layers,
        "reconstructions": reconstructions,
        "cumulative_energy": cumulative_energy,
        "relative_errors": relative_errors,
    }


def print_report(data: dict[str, np.ndarray], svd_story: dict[str, np.ndarray]) -> None:
    """打印说明。"""

    X = data["X"]
    x = data["x"]
    y = data["y"]
    g = data["g"]

    S = svd_story["S"]
    cumulative_energy = svd_story["cumulative_energy"]
    relative_errors = svd_story["relative_errors"]
    reconstructions = svd_story["reconstructions"]

    exact_rank = np.linalg.matrix_rank(X)

    print("========== 矩阵构造 ==========")
    print("原始定义: X[i, j] = x_j + y_i + sin(3 x_j)")
    print("重写后：  X[i, j] = y_i + g_j, 其中 g_j = x_j + sin(3 x_j)")
    print("因此：    X = y 1^T + 1 g^T")
    print("这说明 X 是两个秩 1 矩阵的和，所以 rank(X) <= 2。")
    print()
    print(f"x = {np.round(x, 3)}")
    print(f"y = {np.round(y, 3)}")
    print(f"g = x + sin(3x) = {np.round(g, 3)}")
    print(f"数值秩 rank(X) = {exact_rank}")
    print("由于 y 不是常向量，而 g 也不是常向量，因此列空间确实由 {1, y} 张成, rank(X) = 2。")

    print("\n========== 原始矩阵 X ==========")
    print(np.round(X, 3))

    print("\n========== 奇异值 ==========")
    print(np.round(S, 6))
    print("可以看到只有前两个奇异值显著非零，这与 rank(X) = 2 完全一致。")

    print("\n========== 外积逐层展开 ==========")
    for i in range(min(3, len(S))):
        print("-" * 50)
        print(f"加入第 {i + 1} 层: sigma = {S[i]:.6f}")
        print(f"累计能量占比：{cumulative_energy[i] * 100:.4f}%")
        print(f"相对重构误差：{relative_errors[i] * 100:.4f}%")
        print("当前重构矩阵：")
        print(np.round(reconstructions[i], 3))



def render_plots(
    data: dict[str, np.ndarray],
    svd_story: dict[str, np.ndarray],
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

    X = data["X"]
    S = svd_story["S"]
    layers = svd_story["layers"]
    reconstructions = svd_story["reconstructions"]
    cumulative_energy = svd_story["cumulative_energy"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)

    im0 = axes[0, 0].imshow(X, cmap="viridis")
    axes[0, 0].set_title("Original Matrix X")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    axes[0, 1].plot(np.arange(1, len(S) + 1), S, marker="o")
    axes[0, 1].set_title("Singular Values")
    axes[0, 1].set_xlabel("Index")
    axes[0, 1].set_ylabel("sigma_i")
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(np.arange(1, len(S) + 1), cumulative_energy, marker="o")
    axes[0, 2].axhline(1.0, color="gray", linestyle="--", linewidth=1)
    axes[0, 2].set_title("Cumulative Energy")
    axes[0, 2].set_xlabel("Retained Rank")
    axes[0, 2].set_ylabel("Energy Ratio")
    axes[0, 2].set_ylim(0.0, 1.05)
    axes[0, 2].grid(True, alpha=0.3)

    im1 = axes[1, 0].imshow(layers[0], cmap="viridis")
    axes[1, 0].set_title("First Rank-1 Layer")
    fig.colorbar(im1, ax=axes[1, 0], fraction=0.046)

    im2 = axes[1, 1].imshow(layers[1], cmap="viridis")
    axes[1, 1].set_title("Second Rank-1 Layer")
    fig.colorbar(im2, ax=axes[1, 1], fraction=0.046)

    rank_2_error = np.linalg.norm(X - reconstructions[1], ord="fro")
    im3 = axes[1, 2].imshow(reconstructions[1], cmap="viridis")
    axes[1, 2].set_title(f"Rank-2 Reconstruction\nFrobenius error = {rank_2_error:.2e}")
    fig.colorbar(im3, ax=axes[1, 2], fraction=0.046)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        output_path = save_dir / "svd1_summary.png"
        fig.savefig(output_path, dpi=180)
        print(f"\n图形已保存到: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="精确秩 2 矩阵的 SVD 示例")
    parser.add_argument("--n", type=int, default=5, help="矩阵边长，默认 5")
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
        raise ValueError("n 必须至少为 2, 否则无法构造秩 2 示例。")
    data = build_exact_rank_2_matrix(n=args.n)
    svd_story = compute_svd_story(data["X"])
    print_report(data, svd_story)
    render_plots(data, svd_story, show=args.show, save_dir=args.save_dir)


if __name__ == "__main__":
    main()

'''
========== 矩阵构造 ==========
原始定义: X[i, j] = x_j + y_i + sin(3 x_j)
重写后：  X[i, j] = y_i + g_j, 其中 g_j = x_j + sin(3 x_j)
因此：    X = y 1^T + 1 g^T
这说明 X 是两个秩 1 矩阵的和，所以 rank(X) <= 2。

x = [0.   0.25 0.5  0.75 1.  ]
y = [0.   0.25 0.5  0.75 1.  ]
g = x + sin(3x) = [0.    0.932 1.497 1.528 1.141]
数值秩 rank(X) = 2
由于 y 不是常向量，而 g 也不是常向量，因此列空间确实由 {1, y} 张成, rank(X) = 2。

========== 原始矩阵 X ==========
[[0.    0.932 1.497 1.528 1.141]
 [0.25  1.182 1.747 1.778 1.391]
 [0.5   1.432 1.997 2.028 1.641]
 [0.75  1.682 2.247 2.278 1.891]
 [1.    1.932 2.497 2.528 2.141]]

========== 奇异值 ==========
[8.261 0.596 0.    0.    0.   ]
可以看到只有前两个奇异值显著非零，这与 rank(X) = 2 完全一致。

========== 外积逐层展开 ==========
--------------------------------------------------
加入第 1 层: sigma = 8.261358
累计能量占比：99.4831%
相对重构误差：7.1897%
当前重构矩阵：
[[0.389 1.022 1.407 1.427 1.165]
 [0.469 1.233 1.696 1.721 1.404]
 [0.55  1.443 1.986 2.015 1.644]
 [0.63  1.654 2.276 2.309 1.884]
 [0.71  1.864 2.565 2.603 2.124]]
--------------------------------------------------
加入第 2 层: sigma = 0.595507
累计能量占比：100.0000%
相对重构误差：0.0000%
当前重构矩阵：
[[0.    0.932 1.497 1.528 1.141]
 [0.25  1.182 1.747 1.778 1.391]
 [0.5   1.432 1.997 2.028 1.641]
 [0.75  1.682 2.247 2.278 1.891]
 [1.    1.932 2.497 2.528 2.141]]
--------------------------------------------------
加入第 3 层: sigma = 0.000000
累计能量占比：100.0000%
相对重构误差：0.0000%
当前重构矩阵：
[[0.    0.932 1.497 1.528 1.141]
 [0.25  1.182 1.747 1.778 1.391]
 [0.5   1.432 1.997 2.028 1.641]
 [0.75  1.682 2.247 2.278 1.891]
 [1.    1.932 2.497 2.528 2.141]]
'''