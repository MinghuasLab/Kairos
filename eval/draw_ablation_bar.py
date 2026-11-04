import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

# ================= 数据 =================
configs = [
    'PyTorch\nFP16', 'Qserve\nW4A8', 'QQQ\nW4A8', 'AWQ\nW4A16',
    'Marlin\nW4A16', 'SmoothQ\nW8A8', 'Static-\nBest',
    'Kairos', 'Oracle', 'LUT\nSelection', 'Simple-\nPolicy'
]

exec_time_ms = [
    980.4545, 1021.9594, 1078.6632, 1015.9525,
    977.3217, 856.8418, 833.8139,
    685.9190, 636.0888, 647.2531, 670.4524
]

scale = 32 / 1000
exec_time = [v * scale for v in exec_time_ms]

kairos_conv_overhead_ms = 36.8494
kairos_conv_overhead = kairos_conv_overhead_ms * scale
kairos_base = exec_time[7] - kairos_conv_overhead

n_static = 7
n_dynamic = 4
x = np.arange(len(configs))
bar_colors = [
    '#0B3C5D','#1D6FA5','#219EBC', '#3B9CD9',
    '#7EC8E3','#AED9E0', '#8ECAE6',
    '#90EE90','#32CD32','#228B22','#006400'
]

fig, ax = plt.subplots(figsize=(6.5, 1.5))

for i in range(len(configs)):
    if i == 7:  # Kairos 单独堆叠
        continue
    ax.bar(x[i], exec_time[i], width=0.6,
           color=bar_colors[i], edgecolor='black', linewidth=0.5, zorder=3)

# Kairos 堆叠：底部基础时间 + 顶部转换开销
ax.bar(x[7], kairos_base, width=0.6, color=bar_colors[7],
       edgecolor='black', linewidth=0.5, zorder=3, label='Kairos GEMM Time')
ax.bar(x[7], kairos_conv_overhead, width=0.6, bottom=kairos_base,
       color='#FFF700', edgecolor='black', linewidth=0.5, zorder=3,
       label='Weight Conversion Overhead')

# ================= 在每个柱子顶部标注数值 =================
for i in range(len(configs)):
    if i == 7:
        # Kairos 总高度
        total = exec_time[7]
        ax.text(x[i], total + 0.5, f'{total:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    else:
        ax.text(x[i], exec_time[i] + 0.5, f'{exec_time[i]:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

# ================= 轴设置 =================
ax.set_ylabel('Total Time (s)', fontsize=8)
ax.set_ylim(15, 40)
ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=8)
ax.grid(axis='y', linestyle='--', alpha=0.3)
ax.set_axisbelow(True)

# 分隔线
ax.axvline(x=n_static - 0.5, color='gray', linestyle='--', linewidth=2, alpha=0.8)

# # 分组文字
# ax.text(n_static/2 - 0.5, 37, 'Static Frameworks',
#         ha='center', fontsize=9, color='#333333', style='italic', weight='bold')
# ax.text(n_static + n_dynamic/2 - 0.5, 37, 'Dynamic Ablation',
#         ha='center', fontsize=9, color='#1D3557', style='italic', weight='bold')

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=bar_colors[7], edgecolor='black', label='Kairos GEMM Time'),
    Patch(facecolor='#FFF700', edgecolor='black', label='Weight Conversion Overhead'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=8, frameon=False)

plt.tight_layout()
plt.savefig('./tmp/ablation_bar.png', bbox_inches='tight')
plt.savefig('./images/ablation/ablation_bar.png', bbox_inches='tight')
plt.show()