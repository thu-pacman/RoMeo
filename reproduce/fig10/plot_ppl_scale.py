import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import re

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def extract_baseline_ppl(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    QuaRot_match = re.search(r'TASK:\s*Quarot.*?Perplexity:\s*([0-9.]+)', content, re.DOTALL)
    if QuaRot_match:
        return float(QuaRot_match.group(1))
    
    first_ppl_match = re.search(r'Perplexity:\s*([0-9.]+)', content)
    if first_ppl_match:
        return float(first_ppl_match.group(1))
    
    raise ValueError(f"No baseline PPL found in {file_path}")

def extract_scale_data(file_path):
    data = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    current_percent = None
    current_ppl = None
    
    for line in lines:
        percent_match = re.search(r'percent\s*=\s*([0-9.]+)', line)
        if percent_match:
            if current_percent is not None and current_ppl is not None:
                data.append([current_percent, current_ppl])
            
            current_percent = float(percent_match.group(1))
            current_ppl = None
        
        ppl_match = re.search(r'Perplexity:\s*([0-9.]+)', line)
        if ppl_match and current_percent is not None:
            current_ppl = float(ppl_match.group(1))
    
    if current_percent is not None and current_ppl is not None:
        data.append([current_percent, current_ppl])
    
    return data

def main():
    qwen_baseline_ppl = extract_baseline_ppl('perplexity_Qwen3-8B.log')
    llama_baseline_ppl = extract_baseline_ppl('perplexity_Llama-3.1-8B.log')
    
    print(f"Qwen3-8B Baseline PPL (QuaRot): {qwen_baseline_ppl}")
    print(f"Llama-3.1-8B Baseline PPL (QuaRot): {llama_baseline_ppl}")
    
    qwen_scale_data = extract_scale_data('ppl_scale_Qwen3-8B.log')
    llama_scale_data = extract_scale_data('ppl_scale_Llama-3.1-8B.log')
    
    qwen_data = [[0, qwen_baseline_ppl]] + qwen_scale_data
    llama_data = [[0, llama_baseline_ppl]] + llama_scale_data
    
    qwen_data.sort(key=lambda x: x[0])
    llama_data.sort(key=lambda x: x[0])
    
    print("Qwen data:", qwen_data)
    print("Llama data:", llama_data)
    
    plt.rcParams.update({'font.size': 8})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4.5, 1.6))

    qwen_percent = np.array([d[0] for d in qwen_data])
    qwen_ppl = np.array([d[1] for d in qwen_data])
    llama_percent = np.array([d[0] for d in llama_data])
    llama_ppl = np.array([d[1] for d in llama_data])

    qwen_baseline = qwen_data[0][1]
    llama_baseline = llama_data[0][1]

    color = "#f08036"
    quarot_color = "#528a2c"

    ax1.plot(qwen_percent, qwen_ppl, linestyle='--', color=color, linewidth=1)
    ax1.scatter(qwen_percent[0], qwen_ppl[0], color=quarot_color, s=50, marker='*', zorder=3, edgecolor='black', linewidth=0.5)
    ax1.scatter(qwen_percent[1:], qwen_ppl[1:], color=color, s=50, marker='*', zorder=3, edgecolor='black', linewidth=0.5)
    ax1.axhline(y=qwen_baseline, color=quarot_color, linestyle='--', linewidth=1)
    ax1.set_title('Qwen3-8B', fontweight='bold')
    ax1.set_xlabel('Outliers (%)')
    ax1.set_ylabel('PPL', fontsize=8.5)

    ax1.set_ylim(10.65, 11.75)
    ax1.set_yticks([10.7, 10.9, 11.1, 11.3, 11.5, 11.7])
    ax1.text(
        0.97, (qwen_baseline - ax1.get_ylim()[0]) / (ax1.get_ylim()[1] - ax1.get_ylim()[0]) + 0.03,
        'Quarot', ha='right', va='bottom', color=quarot_color, transform=ax1.transAxes, fontweight='bold'
    )
    ax1.text(
        0.97, (qwen_baseline - ax1.get_ylim()[0]) / (ax1.get_ylim()[1] - ax1.get_ylim()[0]) - 0.05,
        'RoMeo', ha='right', va='top', color=color, transform=ax1.transAxes, fontweight='bold'
    )

    ax2.plot(llama_percent, llama_ppl, linestyle='--', color=color, linewidth=1)
    ax2.scatter(llama_percent[0], llama_ppl[0], color=quarot_color, s=50, marker='*', zorder=3, edgecolor='black', linewidth=0.5)
    ax2.scatter(llama_percent[1:], llama_ppl[1:], color=color, s=50, marker='*', zorder=3, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=llama_baseline, color=quarot_color, linestyle='--', linewidth=1)
    ax2.set_title('Llama-3.1-8B', fontweight='bold')
    ax2.set_xlabel('Outliers (%)')

    ax2.set_ylim(7.65, 8.65)
    ax2.set_yticks([7.7, 7.9, 8.1, 8.3, 8.5])

    ax2.text(
        0.97, (llama_baseline - ax2.get_ylim()[0]) / (ax2.get_ylim()[1] - ax2.get_ylim()[0]) + 0.03,
        'Quarot', ha='right', va='bottom', color=quarot_color, transform=ax2.transAxes, fontweight='bold'
    )
    ax2.text(
        0.97, (llama_baseline - ax2.get_ylim()[0]) / (ax2.get_ylim()[1] - ax2.get_ylim()[0]) - 0.05,
        'RoMeo', ha='right', va='top', color=color, transform=ax2.transAxes, fontweight='bold'
    )

    x_ticks = np.array([0, 0.025, 0.05, 0.075, 0.10, 0.125])
    x_tick_labels = [str(int(xt * 100)) if round(xt * 100) == xt * 100 else str(xt * 100) for xt in x_ticks]
    ax1.set_xticks(x_ticks, x_tick_labels)
    ax2.set_xticks(x_ticks, x_tick_labels)

    ax1.set_xlim(-0.01, max(x_ticks) + 0.015)
    ax2.set_xlim(-0.01, max(x_ticks) + 0.015)

    plt.tight_layout()
    print(f"Image saved as 'percent-ppl.pdf'")
    plt.savefig('percent-ppl.pdf')


if __name__ == '__main__':
    main()
