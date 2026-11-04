# plot_workload_distribution.py
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

def load_batch_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def make_bins(tokens, log_x, n_bins=40):
    """根据 X 轴刻度类型生成合适的 bins。"""
    x_max = max(float(tokens.max()), 1.0)
    x_min = max(float(tokens.min()), 1.0)
    if log_x:
        return np.logspace(np.log10(max(x_min, 1.0)), np.log10(x_max), n_bins)
    else:
        return np.linspace(0, x_max, n_bins)

def format_stats_text(tokens, batch_size, is_pure_prefill,
                      is_pure_decode, is_mixed):
    """把统计信息格式化为多行字符串，用于在图中显示。"""
    pcts = [50, 75, 90, 95, 99]
    token_pcts = {p: int(np.percentile(tokens, p)) for p in pcts}
    bs_pcts = {p: int(np.percentile(batch_size, p)) for p in pcts}

    lines = [
        f"Pure prefill: {100 * is_pure_prefill.mean():3.1f}%",
        f"Pure decode : {100 * is_pure_decode.mean():3.1f}%",
        f"Mixed case  : {100 * is_mixed.mean():3.1f}%",
        "",
        "Tokens percentiles:",
        f"    P50 : {token_pcts[50]}",
        f"    P75 : {token_pcts[75]}",
        f"    P90 : {token_pcts[90]}",
        f"    P95 : {token_pcts[95]}",
        f"    P99 : {token_pcts[99]}",
        # "Batch size P50/P90/P95/P99:",
        # f"  {bs_pcts[50]} / {bs_pcts[90]} / "
        # f"{bs_pcts[95]} / {bs_pcts[99]}",
    ]
    return "\n".join(lines)

def plot_workload(df, out_prefix="workload", log_x=True,
                  break_y=False, y_break_low=None, y_break_high=None,
                  log_y=False):
    tokens = df["batch_num_tokens"].to_numpy()
    prefill = df["batch_num_prefill_tokens"].to_numpy()
    decode = df["batch_num_decode_tokens"].to_numpy()
    batch_size = df["batch_size"].to_numpy()

    is_pure_prefill = (prefill > 0) & (decode == 0)
    is_pure_decode = (prefill == 0) & (decode > 0)
    is_mixed = (prefill > 0) & (decode > 0)

    stats_text = format_stats_text(tokens, batch_size,
                                   is_pure_prefill, is_pure_decode, is_mixed)

    bins = make_bins(tokens, log_x=log_x)

    # ============================================================
    # 图 1：batch_num_tokens 直方图
    # ============================================================
    if break_y:
        # -------- broken y-axis 版本 --------
        counts, _ = np.histogram(tokens, bins=bins)
        max_count = counts.max()

        if y_break_low is None:
            y_break_low = max_count * 0.05
        if y_break_high is None:
            y_break_high = max_count * 0.5

        fig = plt.figure(figsize=(5, 4.7))
        gs = fig.add_gridspec(2, 1, hspace=0.13, height_ratios=[1, 1])
        ax_top = fig.add_subplot(gs[0, 0])
        ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)

        for ax in [ax_top, ax_bot]:
            ax.hist(tokens, bins=bins, color="#4C72B0", edgecolor="white")
            if log_x:
                ax.set_xscale("log")
            else:
                ax.set_xscale("linear")

        ax_top.set_ylim(y_break_high, max_count+5)
        ax_bot.set_ylim(0, y_break_low)

        ax_top.spines['bottom'].set_visible(False)
        ax_bot.spines['top'].set_visible(False)
        ax_top.tick_params(axis='x', which='both',
                           bottom=False, labelbottom=False)
        ax_bot.tick_params(axis='x', which='both', top=False)

        d = .015
        kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
        ax_top.plot((-d, +d), (-d, +d), **kwargs)
        ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        kwargs.update(transform=ax_bot.transAxes)
        ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

        ax_bot.set_xlabel("batch_num_tokens")
                        #   + (" (log scale)" if log_x else " (linear scale)"))
        ax_top.set_ylabel("Number of iterations")
        # ax_top.set_title("Distribution of total tokens per iteration "
                        #  "(broken y-axis)")

        # ax_bot.text(
        #     0.45, 1.8, stats_text,
        #     transform=ax_bot.transAxes,
        #     ha="left", va="top",
        #     fontsize=12, family="monospace",
        #     bbox=dict(boxstyle="round,pad=0.35",
        #               facecolor="white", edgecolor="gray", alpha=0.85),
        # )

        fig.tight_layout()
        fig.savefig(f"{out_prefix}_hist_tokens.png", dpi=150)
        fig.savefig(f"{out_prefix}_hist_tokens.pdf", dpi=150)
        plt.close(fig)

    else:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(tokens, bins=bins, color="#4C72B0", edgecolor="white")
        if log_x:
            ax.set_xscale("log")
        else:
            ax.set_xscale("linear")
        if log_y:
            ax.set_yscale("log")
            ax.set_ylabel("Number of iterations (log scale)")
        else:
            ax.set_ylabel("Number of iterations")
        ax.set_xlabel("batch_num_tokens")
                    #   + (" (log scale)" if log_x else " (linear scale)"))
        ax.set_title("Distribution of total tokens per iteration")

        ax.text(
            0.98, 0.95, stats_text,
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=8, family="monospace",
            bbox=dict(boxstyle="round,pad=0.35",
                      facecolor="white", edgecolor="gray", alpha=0.85),
        )

        fig.tight_layout()
        fig.savefig(f"{out_prefix}_hist_tokens.png", dpi=150)
        plt.close(fig)

    print("=" * 60)
    print(f"Total iterations: {len(tokens)}")
    print(f"Pure prefill: {is_pure_prefill.sum()} "
          f"({100 * is_pure_prefill.mean():.1f}%)")
    print(f"Pure decode : {is_pure_decode.sum()} "
          f"({100 * is_pure_decode.mean():.1f}%)")
    print(f"Mixed       : {is_mixed.sum()} "
          f"({100 * is_mixed.mean():.1f}%)")
    print("-" * 60)
    print("batch_num_tokens percentiles:")
    for p in [1, 5, 25, 50, 75, 90, 95, 99, 100]:
        print(f"  P{p:>3}: {int(np.percentile(tokens, p))}")
    print("-" * 60)
    print("batch_size percentiles (all iterations):")
    for p in [50, 90, 95, 99]:
        print(f"  P{p:>3}: {int(np.percentile(batch_size, p))}")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_prefix", default="images/ablation/workload")
    ap.add_argument("--linear_x", action="store_true",
                    help="Use linear x-axis instead of log")
    ap.add_argument("--break_y", action="store_true",
                    help="Break the y-axis to separate high-frequency "
                         "small-M region from low-frequency large-M region.")
    ap.add_argument("--y_break_low", type=float, default=None,
                    help="Lower bound of the upper y subplot (data units).")
    ap.add_argument("--y_break_high", type=float, default=None,
                    help="Upper bound of the lower y subplot (data units).")
    ap.add_argument("--log_y", action="store_true",
                    help="Use log scale on the y-axis (non-broken mode).")
    args = ap.parse_args()

    path = "/data/xiezx/workspace/vidur/simulator_output/Orca_qps3_128_code/batch_metrics.csv"
    df = load_batch_csv(path)
    plot_workload(
        df,
        out_prefix=args.out_prefix,
        log_x=args.linear_x,
        break_y=True,
        # break_y=args.break_y,
        # y_break_low=args.y_break_low,
        # y_break_high=args.y_break_high,
        y_break_low = 25,
        y_break_high = 50,
        log_y=args.log_y,
    )


if __name__ == "__main__":
    main()
    # python eval/draw.py --break_y --y_break_low 10 --y_break_high 960 --linear_x