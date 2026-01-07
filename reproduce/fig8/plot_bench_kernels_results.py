import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gmean
from functools import partial

ALL_K_N = [
    # Qwen3-8B
    (4096, 6144), # qkv
    (4096, 4096), # o
    (4096, 24576), # ug
    (12288, 4096), # d
    # Qwen3-14B
    (5120, 7168), # qkv
    (5120, 5120), # o
    (5120, 34816), # ug
    (17408, 5120), # d
    # Qwen3-32B
    (5120, 10240), # qkv
    (8192, 5120), # o
    (5120, 51200), # ug
    (25600, 5120), # d
    # Llama-3.1-70B
    (8192, 10240), # qkv
    (8192, 8192), # o
    (8192, 57344), # ug
    (28672, 8192), # d
]

KERNEL_DICT = {
    'BF16': 'baseline_half',
    'INT8': 'baseline_int8_perchannel',  # INT8放在BF16右边
    'Atom': 'baseline_atom',
    # 'Torch': 'baseline_torch',
    'Quarot': 'baseline_quarot',
    'RoMeo': 'baseline_mixed_precision_multistream',
}

BAR_COLOR = [
    "#d6dce5",  # BF16
    "#e377c2",  # INT8 - 粉紫色
    "#d6d680",  # Atom
    # "#97c6e2",  # Torch
    "#9cc97d",  # Quarot
    "#f3a875",  # RoMeo
]

BAR_HATCH = [
    "",     # BF16
    "++",   # INT8
    "..",   # Atom
    # "oo",   # Torch
    "**",   # Quarot
    "xx",   # RoMeo
]

DATA = None

def query(m, n, k, kernel):
    global DATA
    for entry in DATA:
        if entry['M'] == m and entry['N'] == n and entry['K'] == k and entry['kernel'] == kernel:
            return entry['us']
    raise ValueError(f"No data found for M={m}, N={n}, K={k}, kernel={kernel}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True, help='Input JSON file path (without .json extension)')
    parser.add_argument('--m', type=int, default=4096)
    args = parser.parse_args()

    global DATA
    with open(f'{args.file}.json', 'r') as f:
        DATA = json.load(f)

    kernels = [
        'BF16',
        'INT8',  # INT8放在BF16右边
        'Atom',
        # 'Torch',
        'Quarot',
        'RoMeo',
    ]

    group_labels = [
        'Qwen3-8B',
        'Qwen3-14B',
        'Qwen3-32B',
        'Llama-3.1-70B',
        'GeoMean'
    ]

    group_members = [
        'QKV',
        'O',
        'UG',
        'D',
    ]

    speedup_data = {}
    for k, n in ALL_K_N:
        key = (args.m, n, k)
        value = {}
        for ker in kernels:
            entry = query(args.m, n, k, KERNEL_DICT[ker])
            base_time = query(args.m, n, k, 'baseline_half')
            value[ker] = base_time / entry
        speedup_data[key] = value

    geo_means = {}
    for ker in kernels:
        all_speeds = [speedup_data[key][ker] for key in speedup_data]
        geo_means[ker] = gmean(all_speeds)
    
    # Plotting

    plt.figure(figsize=(8.5, 1.5))  # 稍微增加宽度以容纳额外的列
    plt.rcParams.update({'font.size': 8})

    bar_width = 0.17  # 稍微减小条宽以容纳额外的列
    group_spacing = 0.05

    x_indices = []
    current_position = 0
    num_groups = len(speedup_data) // len(group_members) + 1

    for i in range(num_groups):
        if i < num_groups - 1:
            group_indices = [current_position + j for j in range(len(group_members))]
            x_indices.extend(group_indices)
            current_position += len(group_members) + group_spacing
        else:  # GeoMean group
            x_indices.append(current_position + 0.3)
            current_position += 1 + group_spacing

    for i, ker in enumerate(kernels):
        speeds = [speedup_data[key][ker] for key in speedup_data]
        speeds.append(geo_means[ker])
        
        offset = (i - len(kernels) / 2 + 0.5) * bar_width
        plt.bar(
            np.array(x_indices) + offset, 
            speeds,
            width=bar_width,
            label=ker,
            color=BAR_COLOR[i],
            hatch=BAR_HATCH[i],
            edgecolor="black",
            zorder=10,
        )

    group_positions = []
    current_idx = 0
    for i in range(num_groups):
        if i < num_groups - 1:
            group_start = x_indices[current_idx]
            group_end = x_indices[current_idx + 3]
            group_center = (group_start + group_end) / 2
            group_positions.append(group_center)
            current_idx += 4
        else:  # GeoMean group
            group_center = x_indices[-1]
            group_positions.append(group_center)

    plt_vline = partial(
        plt.axvline,
        ymin=-0.32,  # Extend below chart bottom
        ymax=0,  # Extend to chart top
        color='black',
        linestyle='-',
        alpha=1,
        linewidth=0.5,
        clip_on=False  # Allow drawing outside chart area
    )

    # Add vertical separators between model groups
    for i in range(1, num_groups):
        sep_position = group_positions[i - 1] + (group_positions[1] - group_positions[0]) / 2
        plt_vline(x=sep_position)

    # Left and Rightborder line
    plt_vline(x=min(x_indices) - 0.7)
    plt_vline(x=max(x_indices) + 0.7 + 0.15)
    
    chart_height = plt.ylim()[1] - plt.ylim()[0]
    member_label_y = plt.ylim()[0] - chart_height * 0.15
    model_label_y = plt.ylim()[0] - chart_height * 0.30

    # Add member labels
    for group_idx in range(num_groups - 1):  # Only for the first 4 groups
        for member_idx, member_label in enumerate(group_members):
            pos = x_indices[group_idx * 4 + member_idx]
            plt.text(
                pos, 
                member_label_y,
                member_label, 
                ha='center', 
                zorder=10
            )

    # Add group labels
    for i, label in enumerate(group_labels):
        plt.text(
            group_positions[i], 
            0.5 * (model_label_y + member_label_y) if i == len(group_labels) - 1 else model_label_y,
            label, 
            ha='center', 
            fontweight='bold',
            zorder=10  # Ensure labels are above lines
        )

    fig = plt.gcf()
    fig.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, 1),
        ncol=len(kernels),
        columnspacing=4,
        frameon=True,
        fontsize=7
    )

    plt.ylabel('Norm. Speedup')
    plt.xticks([])
    plt.xlim(min(x_indices) - 0.7, max(x_indices) + 0.7 + 0.15)
    plt.yticks([0, 2, 4, 6], [0, 2, 4, 6])
    plt.grid(True, axis='y', linestyle='--', alpha=1, color='grey')

    plt.tight_layout(rect=[0, 0, 1, 0.90])

    plt.savefig(f'{args.file}.pdf')
    print(f"Image saved as '{args.file}.pdf'")
    print(geo_means)


if __name__ == '__main__':
    main()
