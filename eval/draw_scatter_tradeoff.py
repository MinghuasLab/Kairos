import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

# ================= 数据 =================
configs = [
    'PyTorch FP16', 'Qserve W4A8', 'QQQ W4A8', 'AWQ W4A16',
    'Marlin W4A16', 'SmoothQ W8A8', 'Static-Best',
    'Kairos', 'Oracle', 'LUT Selection', 'Simple-Policy'
]

exec_time = [
    980.4545, 1021.9594, 1078.6632, 1015.9525,
    977.3217, 856.8418, 833.8139,
    685.9190, 636.0888, 647.2531, 670.4524
]
for i in range(len(exec_time)):
    exec_time[i] = exec_time[i]*32/1000

memory = [
    13.00, 3.35, 7.36, 3.45,
    3.35, 6.50, 6.38,
    3.53, 31.90, 23.14, 9.96
]

# ================= 样式 =================
markers = ['o', 'X', '<', 'P', 'D','p', 'h',  '*', 'h', 'p', 'P']
colors = [
    '#A8DADC', '#457B9D', '#1D3557', '#6A994E',
    '#BC4749', '#9B5DE5', '#F4A261',
    '#E63946', '#2A9D8F', '#E9C46A', '#264653'
]

# ================= 绘图 =================
fig, ax = plt.subplots(figsize=(4.2, 3.7))

for i, cfg in enumerate(configs):
    ax.scatter(
        memory[i], exec_time[i],
        marker=markers[i], color=colors[i],
        s=140, edgecolor='black', linewidth=0.6,
        label=cfg, zorder=3
    )

ax.annotate('Kairos', xy=(memory[7], exec_time[7]+0.2),
            xytext=(memory[7], exec_time[7] + 2),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
            fontsize=12, ha='center')
ax.annotate('Oracle', xy=(memory[8]-0.3, exec_time[8]+0.2),
            xytext=(memory[8]-4, exec_time[8]+1),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
            fontsize=12, ha='center')

ax.set_xlabel('Peak HBM Usage (GB)', fontsize=12)
ax.set_ylabel('Total Time (s)', fontsize=12)
# ax.set_title('Time--Memory Trade-off Across Configurations', fontsize=12)
ax.grid(linestyle='--', alpha=0.4)
ax.set_xlim(-1, 35)
# ax.set_ylim(600, 1120)

# 图例分两列
ax.legend(loc='upper right', ncol=1, fontsize=9, frameon=True, framealpha=0.9, handletextpad=0.3, labelspacing=0.6)

plt.tight_layout()
# plt.savefig('scatter_hbm_time.pdf', bbox_inches='tight')
# plt.show()
plt.savefig('./tmp/ablation_bar.png', bbox_inches='tight')
plt.savefig('./images/ablation/scatter_hbm_time.png', bbox_inches='tight')
# plt.show()