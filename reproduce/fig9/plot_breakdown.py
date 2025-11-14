from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import re
import argparse


def read_data_from_file(filename):
    data = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    non_empty_lines = [line.strip() for line in lines if line.strip()]
    last_six_lines = non_empty_lines[-6:]
    
    for line in last_six_lines:
        numbers = re.findall(r'\d+\.\d+', line)
        if len(numbers) >= 6:
            row = [float(num) for num in numbers[:6]]
            other = row[0] - sum(row[1:])
            row.append(other)
            data.append(row)
    
    return np.array(data)

def plot(ax, f):
    data = read_data_from_file(f)
    labels = np.array(['BF16', 'U-ker', '+ Pipe.', 'S-ker.', '+ Pipe.', '+ Async.'])
    components = ['Attention', 'Hadamard', 'Quantization', 'Gemm', 'Post_matmul', 'Others']

    component_data = np.column_stack((
        data[:, 1],  # attn
        data[:, 5],  # hadamard
        data[:, 3],  # quant
        data[:, 2],  # gemm
        data[:, 4],  # post_mul
        data[:, 6]   # other
    ))

    colors = ['#d62728', '#ef7f51', "#478fb8", '#9cc97d', '#9355b0', '#d6dce5']
    hatchs = ['xxx', 'ooo', '***', '...', '***', '+++']

    left = np.zeros(len(labels))
    position = 5 - np.arange(6)
    for i in range(len(components)):
        ax.barh(position, component_data[:, i], left=left, color=colors[i], label=components[i], height=0.6, hatch=hatchs[i], edgecolor='black', linewidth=0.8)
        left += component_data[:, i]
    
    for i, total in enumerate(left):
        ax.text(
            total + 0.02 * left[0],
            position[i],
            f"{total:.2f} ms",
            va='center_baseline',
            ha='left',
            fontsize=6
        )
    
    plt_hline = lambda y: partial(ax.axhline,
        xmin=0, xmax=1,
        color='black', linestyle='--',
        linewidth=0.8, clip_on=False
    )(y) and partial(ax.axhline,
        xmin=-0.12, xmax=0,
        color='black', linestyle='-',
        linewidth=0.5, clip_on=False
    )(y)

    plt_hline(4.5)
    plt_hline(2.5)

    ax.set_yticks(position, labels)
    ax.set_xlim(0, left[0] * 1.17)
    ax.tick_params(axis='x', labelsize=7)

    def extract_batch(filename):
        match = re.search(r'b(\d+)', filename)
        return match.group(1) if match else "unknown"
    ax.text(
        0.97, 0.1,
        f'Batch={extract_batch(f)}',
        transform=ax.transAxes,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "lightgray",
            "edgecolor": 'black',
            "linewidth": 0.8
        }
    )


def main(args):
    plt.rcParams.update({'font.size': 8})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.5, 3.2), sharex=False)
    plot(ax1, args.file1)
    plot(ax2, args.file2)

    ax2.set_xlabel('Latency (ms)')

    idx = [0, 3, 1, 4, 2, 5]
    components = np.array(['Attention', 'Hadamard', 'Quantization', 'Gemm', 'Post_matmul', 'Others'])
    colors = np.array(['#d62728', '#ef7f51', "#478fb8", '#9cc97d', '#9355b0', '#d6dce5'])
    hatchs = np.array(['xxx', 'ooo', '***', '...', '***', '+++'])

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[idx][i], edgecolor='black', hatch=hatchs[idx][i], label=components[idx][i])
        for i in range(len(components))
    ]

    fig.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, 1),
        ncol=3,
        fontsize=6.5,
        columnspacing=4,
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.subplots_adjust(hspace=0.2)
    print(f"Image saved as 'plot_breakdown.pdf'")
    plt.savefig(f"plot_breakdown.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a stacked bar chart from benchmark data')
    parser.add_argument('--file1', required=True, help='Path to the first benchmark data file')
    parser.add_argument('--file2', required=True, help='Path to the second benchmark data file')
    args = parser.parse_args()
    main(args)
