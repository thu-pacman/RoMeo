#!/usr/bin/env python3
import os
import re
import glob
from dataclasses import dataclass
from typing import List, Dict, Tuple
import argparse

import matplotlib.pyplot as plt

# Expected log filename patterns:
#   <model>_tp<TP>_b<B>_in<IL>_romeo.log
#   <model>_tp<TP>_b<B>_in<IL>_fp16.log
# Or unquantized may be named as 'unquantized.log'
# We'll detect quantization by suffix: 'romeo' vs 'fp16' or 'unquantized'

FILENAME_RE = re.compile(r"^(?P<model>[^_]+)_tp(?P<tp>\d+)_b(?P<batch>\d+)_in(?P<input>\d+)_(?P<quant>[^.]+)\.log$")

# In log content, we look for lines that include 'Benchmark' (not warmup) and a 'Prefill throughput'
# Example lines may vary; We'll search the last 'Benchmark' block and then extract 'Prefill throughput' numeric value.
BENCHMARK_HEADER_RE = re.compile(r"Benchmark(?!.*warmup)", re.IGNORECASE)
# Match lines like:
#   "Prefill. latency: 1.19626 s, throughput:   6848.02 token/s"
# or  "Prefill throughput: 6848.02"
PREFILL_LINE_RE = re.compile(
    r"^\s*Prefill.*?throughput\s*[:=]\s*(?P<val>[0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)

@dataclass
class Record:
    model: str
    tp: int
    batch: int
    input_len: int
    quant: str
    throughput: float
    path: str


def parse_filename(path: str) -> Tuple[str, int, int, int, str]:
    name = os.path.basename(path)
    m = FILENAME_RE.match(name)
    if not m:
        raise ValueError(f"Unexpected filename format: {name}")
    model = m.group('model')
    tp = int(m.group('tp'))
    batch = int(m.group('batch'))
    input_len = int(m.group('input'))
    quant = m.group('quant')
    return model, tp, batch, input_len, quant


def extract_prefill_throughput(text: str) -> float:
    # Strategy: find all occurrences of 'Benchmark' headers; after the last occurrence,
    # search forward for the first 'Prefill throughput' value.
    # If no header, fallback to the last occurrence of 'Prefill throughput'.
    last_bench_idx = -1
    for m in re.finditer(r"Benchmark", text, re.IGNORECASE):
        # Skip occurrences that include 'warmup' nearby
        span_text = text[m.start(): m.end()+100].lower()
        if 'warmup' in span_text:
            continue
        last_bench_idx = m.start()
    search_zone = text[last_bench_idx:] if last_bench_idx != -1 else text
    for line in search_zone.splitlines():
        pm = PREFILL_LINE_RE.search(line)
        if pm:
            return float(pm.group('val'))
    # Fallback: search entire text for last occurrence
    matches = list(PREFILL_LINE_RE.finditer(text))
    if matches:
        return float(matches[-1].group('val'))
    raise ValueError("Prefill throughput not found in log")


def scan_logs(log_root: str) -> List[Record]:
    paths = []
    # Search both time-stamped subdirs and flat logs root
    if os.path.isdir(log_root):
        # all .log files recursively
        paths.extend(glob.glob(os.path.join(log_root, '**', '*.log'), recursive=True))
    else:
        raise FileNotFoundError(f"Logs directory not found: {log_root}")

    records: List[Record] = []
    for p in paths:
        model, tp, batch, input_len, quant = parse_filename(p)
        try:
            with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            thr = extract_prefill_throughput(text)
        except Exception:
            # Skip logs that fail to parse
            continue
        records.append(Record(model, tp, batch, input_len, quant, thr, p))
    return records


def group_by_model(records: List[Record]) -> Dict[str, List[Record]]:
    groups: Dict[str, List[Record]] = {}
    for r in records:
        key = f"{r.model}_tp{r.tp}"
        groups.setdefault(key, []).append(r)
    # Sort each group by batch size
    for k in groups:
        groups[k] = sorted(groups[k], key=lambda x: x.batch)
    return groups


def aggregate_records(records: List[Record]) -> List[Dict[str, object]]:
    """Aggregate to rows keyed by (model,tp,batch,input_len) with both quant values."""
    agg: Dict[Tuple[str, int, int, int], Dict[str, object]] = {}
    for r in sorted(records, key=lambda x: (x.model, x.tp, x.batch, x.input_len, x.quant)):
        key = (r.model, r.tp, r.batch, r.input_len)
        row = agg.setdefault(key, {
            'model': r.model,
            'tp': r.tp,
            'batch': r.batch,
            'input_len': r.input_len,
            'fp16': None,
            'romeo': None,
        })
        q = r.quant.lower()
        if q in ('fp16', 'bf16', 'unquantized'):
            row['fp16'] = r.throughput
        elif q == 'romeo':
            row['romeo'] = r.throughput
    # flatten
    rows = []
    for (_, _, _, _), row in agg.items():
        fp = row['fp16'] if row['fp16'] is not None else 0.0
        ro = row['romeo'] if row['romeo'] is not None else 0.0
        speedup = (ro / fp) if (fp and ro) else None
        out = dict(row)
        out['speedup'] = speedup
        rows.append(out)
    # sort rows for stability: model->tp->batch
    rows.sort(key=lambda r: (r['model'], r['tp'], r['batch']))
    return rows


def write_tsv(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\t'.join(['model', 'tp', 'batch', 'input_len', 'fp16', 'romeo', 'speedup']) + '\n')
        for r in rows:
            fp = '' if r['fp16'] is None else f"{r['fp16']:.4f}"
            ro = '' if r['romeo'] is None else f"{r['romeo']:.4f}"
            sp = '' if r['speedup'] is None else f"{r['speedup']:.4f}"
            f.write('\t'.join([
                str(r['model']), str(r['tp']), str(r['batch']), str(r['input_len']), fp, ro, sp
            ]) + '\n')


def build_pretty_table(rows: List[Dict[str, object]]) -> str:
    """Build a formatted table text grouped by model with columns Batch | FP16 | RoMeo | Speedup."""
    # group by model+tp
    groups: Dict[str, List[Dict[str, object]]] = {}
    for r in rows:
        key = f"{r['model']}_tp{r['tp']}"
        groups.setdefault(key, []).append(r)
    # order
    ordered_keys = sorted(groups.keys(), key=lambda k: (
        0 if '8B' in k else 1 if '14B' in k else 2 if '32B' in k else 99, k
    ))
    lines: List[str] = []
    for key in ordered_keys:
        grp = sorted(groups[key], key=lambda r: r['batch'])
        title = key.replace('_tp', ' (TP') + ')'
        lines.append(title)
        lines.append('-' * max(24, len(title)))
        header = f"{'Batch':>6} | {'FP16':>10} | {'RoMeo':>10} | {'Speedup':>8}"
        lines.append(header)
        lines.append('-' * len(header))
        for r in grp:
            fp = '' if r['fp16'] is None else f"{r['fp16']:.2f}"
            ro = '' if r['romeo'] is None else f"{r['romeo']:.2f}"
            sp = '' if r['speedup'] is None else f"{r['speedup']:.2f}x"
            lines.append(f"{r['batch']:>6} | {fp:>10} | {ro:>10} | {sp:>8}")
        lines.append('')
    return '\n'.join(lines)

def plot(groups: Dict[str, List[Record]], out_path: str | None = None):
    # Three subplots sharing Y axis, ordered 8B,14B,32B. Titles/X labels mimic example style.
    keys = sorted(groups.keys(), key=lambda k: (
        0 if '8B' in k else 1 if '14B' in k else 2 if '32B' in k else 99,
        k
    ))
    plt.rcParams.update({'font.size': 8})
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(4.5, 1.8), sharey=True)
    if n == 1:
        axes = [axes]

    width = 0.38
    color_fp16 = '#afafaf'
    color_romeo = '#d1956d'

    legend_handles = None
    legend_labels = None

    for ax, key in zip(axes, keys):
        recs = groups[key]
        batches = sorted(set(r.batch for r in recs))
        romeo_vals = []
        fp16_vals = []
        for b in batches:
            rb = [r for r in recs if r.batch == b and r.quant.lower() in ('romeo',)]
            fb = [r for r in recs if r.batch == b and r.quant.lower() in ('fp16','unquantized')]
            romeo_vals.append(rb[-1].throughput if rb else 0.0)
            fp16_vals.append(fb[-1].throughput if fb else 0.0)

        x = list(range(len(batches)))
        bars_fp = ax.bar([xi - width/2 for xi in x], fp16_vals, width=width, label='BF16', color=color_fp16, edgecolor='black', linewidth=1)
        bars_ro = ax.bar([xi + width/2 for xi in x], romeo_vals, width=width, label='RoMeo', color=color_romeo, edgecolor='black', linewidth=1, hatch='//')

        if legend_handles is None:
            legend_handles = [bars_fp, bars_ro]
            legend_labels = ['BF16', 'RoMeo']

        title = key.split('_tp')[0] + f" (TP{key.split('_tp')[1]})"
        ax.set_title(title, fontsize=8, fontweight='bold')
        ax.set_xlabel('Batch Size')
        ax.set_xticks(x)
        ax.set_xticklabels([str(b) for b in batches])
        ax.set_yticks([0, 5000, 10000, 15000, 20000])

    axes[0].set_ylabel('Throughput')
    fig.legend(legend_handles, legend_labels, loc='upper center', ncol=2, frameon=True)
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig(out_path)


def main():
    parser = argparse.ArgumentParser(description='Analyze Prefill throughput from logs and plot comparison.')
    parser.add_argument('--logs', required=True, help='Logs root directory')
    parser.add_argument('--out', default="throughput.pdf", help='Output image path (e.g., PNG)')
    parser.add_argument('--tsv', default=None, help='Output TSV path (tab-delimited summary)')
    parser.add_argument('--table', default=None, help='Output formatted table text path')
    args = parser.parse_args()

    records = scan_logs(args.logs)
    if not records:
        print('No valid logs found.')
        return
    groups = group_by_model(records)
    if not groups:
        print('No groups constructed.')
        return
    # Aggregates and tabular outputs
    rows = aggregate_records(records)
    out_dir = os.path.dirname(args.out) or '.'
    tsv_path = args.tsv or os.path.join(out_dir, 'prefill_throughput.tsv')
    table_path = args.table or os.path.join(out_dir, 'prefill_throughput.txt')
    write_tsv(tsv_path, rows)
    pretty = build_pretty_table(rows)
    with open(table_path, 'w', encoding='utf-8') as f:
        f.write(pretty + '\n')
    # Still produce the chart
    plot(groups, args.out)

if __name__ == '__main__':
    main()
