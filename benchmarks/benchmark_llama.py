import re
import torch
import matplotlib.pyplot as plt

from qfactory.profile import profile_latency
from qfactory.kernels.gemm_w4a4 import gemm_int4_int4_nt, gemm_int4_int4_nt_cutlass

BUFFER = ""
original_print = print

def print(line):
    global BUFFER
    BUFFER += line + "\n"
    original_print(line)

def benchmark(
    m: int,
    n: int,
    k: int,
    gemm,
) -> float:
    x = torch.randint(0, 256, (m, k // 2), dtype=torch.uint8, device='cuda')
    w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device='cuda')
    out = torch.empty(m, n, dtype=torch.int32, device='cuda')

    def run():
        return gemm(x, w, out)
    
    latency = profile_latency(run)
    flops = 2 * m * n * k
    tflops = flops / latency / 1e6

    print(f"batch_size={m}, in_features={k}, out_features={n}, latency={latency:.2f} us, TFLOPS={tflops:.2f}")

def benchmark_all(name: str, in_features: int, out_features: int):
    print(name + "-qfactory")
    for batch_size in [2 ** i for i in range(8, 15)]:
        benchmark(batch_size, out_features, in_features, gemm_int4_int4_nt)
    print(name + "-cutlass")
    for batch_size in [2 ** i for i in range(8, 15)]:
        benchmark(batch_size, out_features, in_features, gemm_int4_int4_nt_cutlass)

def parse_log():
    global BUFFER
    pattern = r"batch_size=(\d+).*?TFLOPS=([\d.]+)"
    data = []
    all_data = []
    labels = []
    
    for line in BUFFER.split('\n'):
        if "batch_size" in line and "TFLOPS" in line:
            matches = re.findall(pattern, line)
            if matches:
                batch_size, tflops = matches[0]
                data.append((int(batch_size), float(tflops)))
        else:
            if data:
                all_data.append(data)
                labels.append(name)
                data = []
            name = line.strip()
    if data:
        all_data.append(data)
        labels.append(name)
    
    for i in range(len(all_data)):
        all_data[i].sort(key=lambda x: x[0])
        all_data[i] = zip(*all_data[i])

    return all_data, labels

def log2(x):
    import numpy as np
    x = np.array(x)
    return np.log2(x)

def plot():
    all_data, labels = parse_log()
    plt.figure(figsize=(9, 5))
    for (batch_sizes, tflops_values), label in zip(all_data, labels):
        if label.endswith('cutlass'):
            plt.plot(log2(batch_sizes), tflops_values, 'o--', label=label, alpha=0.5)
        else:
            plt.plot(log2(batch_sizes), tflops_values, 'o-', label=label)

    plt.xticks(log2(batch_sizes), batch_sizes)
    plt.xlabel('Batch Size')
    plt.ylabel('TFLOPS')

    plt.legend()
    plt.tight_layout()

    plt.savefig(f'llama.png')


if __name__ == '__main__':
    # Llama-2-7B
    benchmark_all(name="Llama-2-7B-gate-up", in_features=4096, out_features=22016)
    benchmark_all(name="Llama-2-7B-down", in_features=11008, out_features=4096)

    # Llama-2-70B
    benchmark_all(name="Llama-2-70B-gate-up", in_features=8192, out_features=57344)
    benchmark_all(name="Llama-2-70B-down", in_features=28672, out_features=8192)

    plot()
