from utils.util import load_json
import matplotlib.pyplot as plt
import numpy as np
from utils.color_print import color_text
from matplotlib.ticker import MaxNLocator

model_type = ['llama']
# model_type = ['qwen']
# model_type = ['llama', 'qwen']
model_name = ['LLaMa3 8B']
# model_name = ['Qwen2.5 7B']
# model_name = ['LLaMa3 8B', 'Qwen2.5 7B']

# 原来 3 列，现在增加一列 LUTSelection（数据先复制 Oracle 的）
# quant_type = ['DynamicLinear', 'Best-Static', 'SwitchLinear', 'LUTSelection']
# quant_labels = ['Kairos 4-bit', 'Static-Best', 'Oracle Dynamic', 'LUT Selection']

quant_type = ['Best-Static', 'SwitchLinear', 'LUTSelection', 'DynamicLinear']
quant_labels = ['Static-Best', 'Oracle Dynamic', 'LUT Selection','Kairos 4-bit']
# 蓝色调配色：浅蓝 -> 深蓝
quant_colors = ['#7EC8E3', '#3B9CD9', '#1D6FA5','#0B3C5D']

# 设置全局字体和样式
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'text.usetex': False,     # 如果系统支持 LaTeX 可改为 True
    'figure.dpi': 150,
})

for idx in range(len(model_type)):
    fig = plt.figure(figsize=(8, 4.8))   # 稍微加高，给顶部图例留空间
    font = {'style': 'normal', 'weight': 'bold', 'size': 12}

    model = model_type[idx]
    result_list = [[] for _ in range(len(quant_type))]
    TTFT = [[] for _ in range(len(quant_type))]
    TPOT = [[] for _ in range(len(quant_type))]
    TTL  = [[] for _ in range(len(quant_type))]
    VRAM = [[] for _ in range(len(quant_type))]
    TMP  = [[] for _ in range(len(quant_type))]
    TOTAL = [[] for _ in range(len(quant_type))]

    input_len = load_json('./result/end2end_batch/test_batch.json')

    for i in range(len(quant_type)):
        if quant_type[i] == 'LUTSelection':
            result_list[i] = list(result_list[1])
        else:
            result_list[i] = load_json('./result/end2end_batch/' + model + '_' + quant_type[i] + '.json')

        for res in result_list[i]:
            TTFT[i].append(res[0])
            TPOT[i].append(res[1])
            TTL[i].append(res[2])
            VRAM[i].append(res[3])
            TMP[i].append(0)
            TOTAL[i].append(0)

    categories = input_len[0:4]
    x = np.arange(len(categories))

    # 柱状图宽度与偏移（4 列时自动适配）
    n_bars = len(quant_type)
    BAR_WIDTH = 0.8 / n_bars
    BAR_OFFSETS = np.arange(n_bars) * BAR_WIDTH - (n_bars - 1) * BAR_WIDTH / 2

    # ---------------- (1) TTFT ----------------
    plt.subplot(2, 2, 1)
    ax1 = plt.gca()
    plt.xlabel('Batch Size')
    plt.ylabel('TTFT (ms)')
    for i in range(n_bars):
        ax1.bar(x + BAR_OFFSETS[i], TTFT[i][0:4], width=BAR_WIDTH,
                color=quant_colors[i], edgecolor='k', linewidth=0.8,
                label=quant_labels[i])
    plt.xticks(x, categories)
    plt.yticks()

    # ---------------- (2) TPOT ----------------
    plt.subplot(2, 2, 2)
    plt.xlabel('Batch Size')
    plt.ylabel('TPOT (ms)')
    for i in range(n_bars):
        plt.bar(x + BAR_OFFSETS[i], TPOT[i][0:4], width=BAR_WIDTH,
                color=quant_colors[i], edgecolor='k', linewidth=0.8,
                label=quant_labels[i])
    plt.xticks(x, categories)
    plt.yticks()

    # ---------------- (3) Throughput ----------------
    plt.subplot(2, 2, 3)
    plt.xlabel('Batch Size')
    plt.ylabel('Throughput')
    for i in range(n_bars):
        plt.bar(x + BAR_OFFSETS[i], TTL[i][0:4], width=BAR_WIDTH,
                color=quant_colors[i], edgecolor='k', linewidth=0.8,
                label=quant_labels[i])
    plt.xticks(x, categories, fontsize=12)
    plt.yticks(fontsize=12)

    # ---------------- (4) Peak Memory ----------------
    plt.subplot(2, 2, 4)
    plt.xlabel('Input Sequence Length')
    plt.ylabel('Peak Memory (GB)')
    for i in range(n_bars):
        plt.bar(x + BAR_OFFSETS[i], VRAM[i][0:4], width=BAR_WIDTH,
                color=quant_colors[i], edgecolor='k', linewidth=0.8,
                label=quant_labels[i])
    plt.xticks(x, categories)
    plt.yticks()
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    # ---------------- 统一图例（放在整张图顶部） ----------------
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='upper center',
               ncol=n_bars,
               bbox_to_anchor=(0.5, 1.0),
               frameon=False,
               fontsize=12,
               handletextpad=0.4,
               columnspacing=1.2)

    # 为顶部图例留出空间
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig('./tmp.png', dpi=300, bbox_inches='tight')