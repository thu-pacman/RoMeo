import numpy as np
import matplotlib.pyplot as plt
import re

models = ['Qwen3-8B', 'Qwen3-14B', 'Qwen3-32B']
batch_sizes = [16, 64, 256]
methods = ['BF16', 'QuaRot', 'RoMeo']

def read_bf16_romeo_data(filename):
    bf16_data = []
    romeo_data = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    table_data = []
    for line in lines:
        if re.match(r'^\s*[\d\.]+\s+[\d\.]', line):
            numbers = [float(x) for x in line.split()]
            table_data.append(numbers)
    
    if len(table_data) >= 6:
        for i in range(3):
            bf16_data.append(table_data[i][0])
        
        for i in range(3):
            romeo_data.append(table_data[i+3][0])
    
    return bf16_data, romeo_data

def read_quarot_data(filename):
    quarot_data = []
    
    with open(filename, 'r') as f:
        content = f.read()
    
    layer_matches = re.findall(r'Layer\s+([\d\.]+)\s+ms', content)
    
    if len(layer_matches) >= 9:
        quarot_8b = [float(layer_matches[i]) for i in range(3)]
        quarot_14b = [float(layer_matches[i]) for i in range(3, 6)]
        quarot_32b = [float(layer_matches[i]) for i in range(6, 9)]
        
        quarot_data = [quarot_8b, quarot_14b, quarot_32b]
    
    return quarot_data

def read_all_data():
    bf16 = [[0, 0, 0] for _ in range(3)]    # [8B, 14B, 32B]
    quarot = [[0, 0, 0] for _ in range(3)]  # [8B, 14B, 32B]
    bitweaver = [[0, 0, 0] for _ in range(3)] # [8B, 14B, 32B]
    
    try:
        bf16_8b, romeo_8b = read_bf16_romeo_data('bench_e2e_Qwen3-8B.log')
        bf16[0] = bf16_8b
        bitweaver[0] = romeo_8b
        print(f"8B: BF16={bf16_8b}, RoMeo={romeo_8b}")
    except Exception as e:
        print(f"Error reading 8B data: {e}")
    try:
        bf16_14b, romeo_14b = read_bf16_romeo_data('bench_e2e_Qwen3-14B.log')
        bf16[1] = bf16_14b
        bitweaver[1] = romeo_14b
        print(f"14B: BF16={bf16_14b}, RoMeo={romeo_14b}")
    except Exception as e:
        print(f"Error reading 14B data: {e}")
    try:
        bf16_32b, romeo_32b = read_bf16_romeo_data('bench_e2e_Qwen3-32B.log')
        bf16[2] = bf16_32b
        bitweaver[2] = romeo_32b
        print(f"32B: BF16={bf16_32b}, RoMeo={romeo_32b}")
    except Exception as e:
        print(f"Error reading 32B data: {e}")
    try:
        quarot_data = read_quarot_data('bench_e2e_quarot.log')
        if len(quarot_data) == 3:
            quarot = quarot_data
            print(f"QuaRot: 8B={quarot[0]}, 14B={quarot[1]}, 32B={quarot[2]}")
    except Exception as e:
        print(f"Error reading QuaRot data: {e}")
    
    return bf16, quarot, bitweaver

def main():
    bf16, quarot, bitweaver = read_all_data()
    
    print("\nFinal data:")
    print(f"BF16: {bf16}")
    print(f"QuaRot: {quarot}")
    print(f"RoMeo: {bitweaver}")
    norm_func = lambda x: [[v / b for v, b in zip(row, bf_row)] for row, bf_row in zip(x, bf16)]
    bf16_normalized = norm_func(bf16)
    quarot_normalized = norm_func(quarot)
    bitweaver_normalized = norm_func(bitweaver)

    plt.rcParams.update({'font.size': 8})
    fig, axes = plt.subplots(1, 3, figsize=(4.5, 1.8))

    method_colors = ["#afafaf", "#8eb275", "#d1956d"]
    hatches = ['', '///', '\\\\\\']

    for i, model in enumerate(models):
        ax = axes[i]
        
        x = np.arange(len(batch_sizes))
        width = 0.27
        
        for j in range(len(batch_sizes)):
            ax.bar(x[j] - width, bf16_normalized[i][j], width,
                        color=method_colors[0], hatch=hatches[0], edgecolor='black', linewidth=1)
            ax.bar(x[j], quarot_normalized[i][j], width,
                        color=method_colors[1], hatch=hatches[1], edgecolor='black', linewidth=1)
            ax.bar(x[j] + width, bitweaver_normalized[i][j], width,
                        color=method_colors[2], hatch=hatches[2], edgecolor='black', linewidth=1)
        
        ax.set_title(model, fontsize=8, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(batch_sizes)
        ax.set_xlabel('Batch Size')
        if i == 0:
            ax.set_ylabel('Norm. Layer Latency')
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        else:
            ax.set_yticks([])
        ax.set_xlim(-0.7, 2.7)
        ax.set_ylim(0, 1.2)
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=0.5, zorder=-10)

        for j in range(len(batch_sizes)):
            value = bf16[i][j]
            ax.annotate(f'({value:.1f})', 
                        xy=(x[j] - width, bf16_normalized[i][j]), 
                        xytext=(0, 1),
                        textcoords='offset points',
                        ha='center', va='bottom', fontsize=5, color='black', rotation=10)

    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor=method_colors[0], hatch=hatches[0], edgecolor='black'),
        plt.Rectangle((0,0),1,1, facecolor=method_colors[1], hatch=hatches[1], edgecolor='black'),
        plt.Rectangle((0,0),1,1, facecolor=method_colors[2], hatch=hatches[2], edgecolor='black')
    ]

    fig.legend(legend_elements, methods, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.98), fontsize=7)
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    print(f"Image saved as 'layer_latency.pdf'")
    
    plt.savefig('layer_latency.pdf')


if __name__ == '__main__':
    main()