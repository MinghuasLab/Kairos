# plot_workload_distribution.py
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def load_batch_csv(path):
    df = pd.read_csv(path)
    # 统一列名，去除可能的空格
    df.columns = [c.strip() for c in df.columns]
    return df


def compute_cdf(x):
    """从样本计算经验 CDF。"""
    xs = np.sort(np.asarray(x))
    ys = np.arange(1, len(xs) + 1) / len(xs)
    return xs, ys


def plot_workload(df, out_prefix="workload", log_x=True):
    tokens = df["batch_num_tokens"].to_numpy()
    prefill = df["batch_num_prefill_tokens"].to_numpy()
    decode = df["batch_num_decode_tokens"].to_numpy()
    batch_size = df["batch_size"].to_numpy()
    exec_time = df["batch_execution_time"].to_numpy()
    batch_id = df["Batch Id"].to_numpy()

    # 分类：纯 prefill、纯 decode、混合
    is_pure_prefill = (prefill > 0) & (decode == 0)
    is_pure_decode = (prefill == 0) & (decode > 0)
    is_mixed = (prefill > 0) & (decode > 0)

    # ============================================================
    # 图 1：batch_num_tokens 的直方图（对数横轴）
    # ============================================================
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.logspace(0, np.log10(max(tokens.max(), 1)), 40)
    ax.hist(tokens, bins=bins, color="#4C72B0", edgecolor="white")
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel("batch_num_tokens (log scale)")
    ax.set_ylabel("Number of iterations")
    ax.set_title("Distribution of total tokens per iteration")
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_hist_tokens.png", dpi=150)
    plt.close(fig)

    # ============================================================
    # 图 2：batch_num_tokens 的 CDF
    # ============================================================
    xs, ys = compute_cdf(tokens)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, color="#C44E52", lw=2, label="All iterations")
    ax.set_xlabel("batch_num_tokens")
    ax.set_ylabel("CDF")
    ax.set_title("CDF of tokens per iteration")
    if log_x:
        ax.set_xscale("log")
    # 标注关键百分位
    for p in [50, 90, 95, 99]:
        v = np.percentile(tokens, p)
        ax.axvline(v, ls="--", lw=0.8, color="gray")
        ax.text(v, 0.02, f"P{p}={int(v)}", rotation=90,
                va="bottom", ha="right", fontsize=8, color="gray")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_cdf_tokens.png", dpi=150)
    plt.close(fig)

    # ============================================================
    # 图 3：纯 prefill / 纯 decode / 混合 的 CDF 对比
    # ============================================================
    fig, ax = plt.subplots(figsize=(6, 4))
    for mask, label, color in [
        (is_pure_prefill, "Pure prefill", "#4C72B0"),
        (is_pure_decode, "Pure decode", "#55A868"),
        (is_mixed, "Mixed", "#C44E52"),
    ]:
        if mask.sum() == 0:
            continue
        xs, ys = compute_cdf(tokens[mask])
        ax.plot(xs, ys, lw=2, label=f"{label} (n={mask.sum()})", color=color)
    ax.set_xlabel("batch_num_tokens")
    ax.set_ylabel("CDF")
    ax.set_title("CDF by iteration type")
    if log_x:
        ax.set_xscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_cdf_by_type.png", dpi=150)
    plt.close(fig)

    # ============================================================
    # 图 4：随时间变化的 batch_num_tokens（展示动态性）
    # ============================================================
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(batch_id, tokens, lw=0.8, color="#4C72B0")
    ax.fill_between(batch_id, 0, tokens, alpha=0.2, color="#4C72B0")
    ax.set_xlabel("Iteration index")
    ax.set_ylabel("batch_num_tokens")
    ax.set_title("Temporal evolution of tokens per iteration")
    if log_x:
        ax.set_yscale("log")
    # 用不同颜色标记 iteration 类型
    ax.scatter(batch_id[is_pure_prefill], tokens[is_pure_prefill],
               s=4, color="#4C72B0", label="Pure prefill")
    ax.scatter(batch_id[is_pure_decode], tokens[is_pure_decode],
               s=4, color="#55A868", label="Pure decode")
    ax.scatter(batch_id[is_mixed], tokens[is_mixed],
               s=4, color="#C44E52", label="Mixed")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_temporal_tokens.png", dpi=150)
    plt.close(fig)

    # ============================================================
    # 图 5：batch_size vs batch_num_tokens 散点（展示 M 的双重来源）
    # ============================================================
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sc = ax.scatter(batch_size[is_pure_prefill], tokens[is_pure_prefill],
                    s=8, alpha=0.5, color="#4C72B0", label="Pure prefill")
    ax.scatter(batch_size[is_pure_decode], tokens[is_pure_decode],
               s=8, alpha=0.5, color="#55A868", label="Pure decode")
    ax.scatter(batch_size[is_mixed], tokens[is_mixed],
               s=8, alpha=0.5, color="#C44E52", label="Mixed")
    ax.set_xlabel("batch_size")
    ax.set_ylabel("batch_num_tokens")
    ax.set_title("batch_size vs batch_num_tokens")
    if log_x:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_scatter_bs_tokens.png", dpi=150)
    plt.close(fig)

    # ============================================================
    # 图 6：decode 阶段 batch_size 的 CDF（用于静态基线选择）
    # ============================================================
    decode_bs = batch_size[is_pure_decode]
    if len(decode_bs) > 0:
        xs, ys = compute_cdf(decode_bs)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, ys, color="#55A868", lw=2)
        ax.set_xlabel("batch_size (decode)")
        ax.set_ylabel("CDF")
        ax.set_title("CDF of decode-phase batch size")
        if log_x:
            ax.set_xscale("log")
        for p in [50, 90, 95, 99]:
            v = np.percentile(decode_bs, p)
            ax.axvline(v, ls="--", lw=0.8, color="gray")
            ax.text(v, 0.02, f"P{p}={int(v)}", rotation=90,
                    va="bottom", ha="right", fontsize=8, color="gray")
        fig.tight_layout()
        fig.savefig(f"{out_prefix}_cdf_decode_bs.png", dpi=150)
        plt.close(fig)

    # ============================================================
    # 打印统计信息（可直接引用到论文）
    # ============================================================
    print("=" * 60)
    print(f"Total iterations: {len(tokens)}")
    print(f"Pure prefill: {is_pure_prefill.sum()} "
          f"({100*is_pure_prefill.mean():.1f}%)")
    print(f"Pure decode : {is_pure_decode.sum()} "
          f"({100*is_pure_decode.mean():.1f}%)")
    print(f"Mixed       : {is_mixed.sum()} "
          f"({100*is_mixed.mean():.1f}%)")
    print("-" * 60)
    print("batch_num_tokens percentiles:")
    for p in [1, 5, 25, 50, 75, 90, 95, 99, 100]:
        print(f"  P{p:>3}: {int(np.percentile(tokens, p))}")
    print("-" * 60)
    print("batch_size percentiles (all iterations):")
    for p in [50, 90, 95, 99]:
        print(f"  P{p:>3}: {int(np.percentile(batch_size, p))}")
    if len(decode_bs) > 0:
        print("-" * 60)
        print("decode batch_size percentiles:")
        for p in [50, 90, 95, 99]:
            print(f"  P{p:>3}: {int(np.percentile(decode_bs, p))}")
    print("=" * 60)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_prefix", default="images/workload/workload")
    ap.add_argument("--linear_x", action="store_true",
                    help="Use linear x-axis instead of log")
    args = ap.parse_args()
    path = "/data/xiezx/workspace/vidur/simulator_output/Orca_qps6_128_code/batch_metrics.csv"
    df = load_batch_csv(path)
    plot_workload(df, out_prefix=args.out_prefix, log_x=not args.linear_x)


if __name__ == "__main__":
    main()