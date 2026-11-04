import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from kairos.quantization import *
from utils.command_parser import parser
from utils.perf_eval import timer
from utils.color_print import *

from thirdparty.SmoothQuant.w8a8_linear import W8A8Linear

from thirdparty.AWQ.w4a16_linear import W4A16Linear_AWQ
from thirdparty.AutoGPTQ.w4a16_linear import W4A16Linear_Marlin

from thirdparty.Qserve.w4a8_linear import W4A8Linear
from thirdparty.QQQ.w4a8_linear import W4A8Linear_QQQ

from kairos.dynamic_linear_v2 import DynamicLinear
from kairos.perf_profiler_v2 import Profiler

path = "/data/xiezx/workspace/vidur/simulator_output/Orca_qps3_128_code/batch_metrics.csv"
df = pd.read_csv(path)
df.columns = [c.strip() for c in df.columns]

# 每个 iteration 的 M
M_seq = df["batch_num_tokens"].astype(int).tolist()

# 迭代类型标记（用于背景着色）
iter_type = []
for _, row in df.iterrows():
    if row["batch_num_prefill_tokens"] > 0 and row["batch_num_decode_tokens"] == 0:
        iter_type.append("prefill")
    elif row["batch_num_prefill_tokens"] == 0 and row["batch_num_decode_tokens"] > 0:
        iter_type.append("decode")
    else:
        iter_type.append("mixed")

M_seq = [1,2,4,8,16,1024, 2048, 4096]


save_path = 'llama_production.pt'
prof_path = './prof_llama/'
shape_list = [(1024, 4096), (4096, 4096), (4096, 14336), (14336, 4096)] # Llama-3 8B

from kairos.dynamic_linear_v2 import DynamicLinear
from kairos.perf_profiler_v2 import Profiler
DynamicLinear.prof = Profiler(shape_list)
# DynamicLinear.prof.profiling()
# DynamicLinear.prof.save(prof_path)
DynamicLinear.prof.load(prof_path)
comp_type_list = ['W4A16 GEMM', 'W4A16 GEMV', 'W8A8 GEMM', 'w8a16_gemm', 'w8a16_gemv', 'FP16 GEMM']

total_lat_kairos = []
total_lat_static_layer = []

for shape_idx in range(4):
    shape = shape_list[shape_idx]
    N = shape[0]; K = shape[1]; bias = False
    torchLinear   = torch.nn.Linear(in_features=K, out_features=N, bias=bias, dtype=torch.float16).cuda()

    kairos_linear = DynamicLinear.from_module(torchLinear, 'cuda')
    if shape_idx < 2:
        best_static_linear = W4A16Linear_AWQ.from_module(torchLinear, 'cuda')
    else:
        best_static_linear = W4A16Linear_Marlin.from_module(torchLinear, 'cuda')


    # qlinear_w4a8_qserve  = W4A8Linear.from_module(torchLinear, 'cuda')
    # qlinear_w4a8_qqq     = W4A8Linear_QQQ.from_module(torchLinear)
    # qlinear_w4a16_awq    = W4A16Linear_AWQ.from_module(torchLinear, 'cuda')
    # qlinear_w4a16_marlin = W4A16Linear_Marlin.from_module(torchLinear)
    # qlinear_w8a8_awq     = W8A8Linear.from_module(torchLinear, 'cuda', backend='awq')
    # dlinear  = DynamicLinear.from_module(torchLinear, 'cuda')

    def benchmark_config(config, M):
        """对给定 M，测一次线性层 GEMM 延迟（含 quant/conversion/dispatch）。"""
        # config: "kairos" / "static_layer"
        # 内部根据 config 决定 kernel、activation 路径、是否转换 weight
        # 返回 latency in ms
        n = 300; preheat_time = 100
        recordList = []; print_flag = True
        if config == "kairos      ":
            linear = kairos_linear
            if print_flag:
                print(f'M: {M} ')
        else:  linear = best_static_linear
        input_tensor = torch.randn((1,M, K), dtype=torch.float16, device = 'cuda')
        torch.cuda.empty_cache()
        # Preheat Kernels
        for _ in range(preheat_time):
            out = linear(input_tensor)
        with timer(config, n=n, recordList=recordList, print_flag=print_flag):
            for _ in range(n):
                out = linear(input_tensor)
        return recordList[-1]
    lat_kairos = []
    lat_static_layer = []
    for M in M_seq:
        lat_kairos.append(benchmark_config(      "kairos      ", M))
        lat_static_layer.append(benchmark_config("static_layer", M))
    total_lat_kairos.append(lat_kairos)
    total_lat_static_layer.append(lat_static_layer)

lat_kairos = list(total_lat_kairos[0])
lat_static_layer = list(total_lat_static_layer[0])

for i in range(len(lat_kairos)):
    lat_kairos[i] = total_lat_kairos[0][i] * 2 + total_lat_kairos[1][i]* 2 + total_lat_kairos[2][i] * 2 + total_lat_kairos[3][i]
    lat_static_layer[i] = total_lat_static_layer[0][i] * 2 + total_lat_static_layer[1][i]* 2 + total_lat_static_layer[2][i] * 2 + total_lat_static_layer[3][i]

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_fig16(M_seq, iter_type, lat_kairos, lat_static_layer, out_prefix="./images/production/fig16"):
    it = np.arange(len(M_seq))

    # ============================================================
    # 面板 (a)：逐 iteration 延迟时间线
    # ============================================================
    fig, ax = plt.subplots(figsize=(10, 3.8))

    # 背景着色
    type_color = {"decode": "#D6E4F0", "prefill": "#FDE2C8", "mixed": "#E5D4EF"}
    start = 0
    for i in range(1, len(iter_type) + 1):
        if i == len(iter_type) or iter_type[i] != iter_type[start]:
            ax.axvspan(start, i, color=type_color[iter_type[start]],
                       alpha=0.6, lw=0)
            start = i

    # 三条曲线
    ax.plot(it, lat_static_layer, lw=1.0, color="#4C72B0",
            label="Static-Best-Layer", alpha=0.85)
    # ax.plot(it, lat_static_shape, lw=1.0, color="#55A868",
    #         label="Static-Best-Shape", alpha=0.85)
    ax.plot(it, lat_kairos, lw=1.2, color="#C44E52",
            label="Kairos (dynamic)", alpha=1.0)

    ax.set_xlabel("Iteration index")
    ax.set_ylabel("GEMM latency (ms)")
    ax.set_yscale("log")
    ax.set_title("Per-iteration GEMM latency under fixed-$M$ replay (QPS=3)")
    ax.legend(loc="upper right", fontsize=8, ncol=3)

    # 背景图例
    handles = [
        mpatches.Patch(color=type_color["decode"], alpha=0.6, label="Pure decode"),
        mpatches.Patch(color=type_color["prefill"], alpha=0.6, label="Pure prefill"),
        mpatches.Patch(color=type_color["mixed"], alpha=0.6, label="Mixed"),
    ]
    ax.legend(handles=handles + ax.get_legend_handles_labels()[0],
              loc="upper right", fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_a.pdf", dpi=300)
    fig.savefig(f"{out_prefix}_a.png", dpi=300)
    plt.close(fig)

    # ============================================================
    # 面板 (b)：按 M 分箱的平均延迟
    # ============================================================
    bins = [1, 4, 16, 64, 256, 1024, 4096, 16384]
    labels = ["1–4", "5–16", "17–64", "65–256",
              "257–1024", "1025–4096", "4097–16384"]

    M_arr = np.array(M_seq)
    lat_k = np.array(lat_kairos)
    lat_sl = np.array(lat_static_layer)
    lat_ss = np.array(lat_static_shape)

    mean_k, mean_sl, mean_ss = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (M_arr >= lo) & (M_arr <= hi)
        if mask.sum() == 0:
            mean_k.append(0); mean_sl.append(0); mean_ss.append(0)
        else:
            mean_k.append(lat_k[mask].mean())
            mean_sl.append(lat_sl[mask].mean())
            mean_ss.append(lat_ss[mask].mean())

    x = np.arange(len(labels))
    width = 0.27

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(x - width, mean_sl, width, label="Static-Best-Layer", color="#4C72B0")
    # ax.bar(x,         mean_ss, width, label="Static-Best-Shape", color="#55A868")
    ax.bar(x + width, mean_k,  width, label="Kairos (dynamic)", color="#C44E52")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_xlabel("$M$ range (batch_num_tokens)")
    ax.set_ylabel("Avg GEMM latency (ms)")
    ax.set_yscale("log")
    ax.set_title("Average GEMM latency by $M$ bin")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_b.pdf", dpi=300)
    fig.savefig(f"{out_prefix}_b.png", dpi=300)
    plt.close(fig)

torch.save(lat_kairos, './tmp/lat_kairos.pt')
torch.save(lat_static_layer, './tmp/lat_static_layer.pt')
plot_fig16(M_seq, iter_type, lat_kairos, lat_static_layer)
