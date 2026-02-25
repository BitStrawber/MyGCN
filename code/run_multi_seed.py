import argparse
import ast
import json
import os
import re
import statistics
import subprocess
import sys
from datetime import datetime


METRIC_PATTERN = {
    'precision': re.compile(r"'precision': array\(\[([^\]]+)\]\)"),
    'recall': re.compile(r"'recall': array\(\[([^\]]+)\]\)"),
    'ndcg': re.compile(r"'ndcg': array\(\[([^\]]+)\]\)"),
}


def parse_list_from_array_text(text):
    parts = [p.strip() for p in text.split(',') if p.strip()]
    return [float(p) for p in parts]


def parse_testbest_metrics(line):
    result = {}
    for key, pattern in METRIC_PATTERN.items():
        match = pattern.search(line)
        if not match:
            return None
        result[key] = parse_list_from_array_text(match.group(1))
    return result


def run_one_seed(seed, args):
    cmd = [
        sys.executable,
        'main.py',
        f'--dataset={args.dataset}',
        f'--model={args.model}',
        f'--alpha={args.alpha}',
        f'--beta={args.beta}',
        f'--gamma={args.gamma}',
        f'--tau={args.tau}',
        f'--bpr_batch={args.bpr_batch}',
        f'--recdim={args.recdim}',
        f'--layer={args.layer}',
        f'--lr={args.lr}',
        f'--decay={args.decay}',
        f'--testbatch={args.testbatch}',
        f'--epochs={args.epochs}',
        f'--topks={args.topks}',
        f'--tensorboard={args.tensorboard}',
        f'--seed={seed}',
    ]

    print(f"\n===== Running seed {seed} =====")
    print(' '.join(cmd))

    process = subprocess.Popen(
        cmd,
        cwd=os.path.dirname(os.path.abspath(__file__)),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    seen_test_best = False
    metric_line = None

    for line in process.stdout:
        print(line, end='')
        if '[TEST-BEST]' in line:
            seen_test_best = True
            continue
        if seen_test_best and "{'precision':" in line and "'recall':" in line and "'ndcg':" in line:
            metric_line = line.strip()

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f'Seed {seed} failed with exit code {return_code}')

    if metric_line is None:
        raise RuntimeError(f'Seed {seed} finished but no [TEST-BEST] metrics were found.')

    metrics = parse_testbest_metrics(metric_line)
    if metrics is None:
        raise RuntimeError(f'Seed {seed} metric parsing failed. Raw line: {metric_line}')

    return metrics


def summarize(all_seed_metrics, topk_index):
    recalls = [m['recall'][topk_index] for m in all_seed_metrics]
    ndcgs = [m['ndcg'][topk_index] for m in all_seed_metrics]

    summary = {
        'recall_mean': statistics.mean(recalls),
        'recall_std': statistics.pstdev(recalls) if len(recalls) > 1 else 0.0,
        'ndcg_mean': statistics.mean(ndcgs),
        'ndcg_std': statistics.pstdev(ndcgs) if len(ndcgs) > 1 else 0.0,
    }
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description='Run PCSRec with multiple seeds and summarize TEST-BEST results.')
    parser.add_argument('--dataset', type=str, default='amazon_instrument')
    parser.add_argument('--model', type=str, default='pcsrec')
    parser.add_argument('--seeds', type=str, default='2020,2021,2022', help='Comma-separated seeds')

    parser.add_argument('--alpha', type=float, default=-0.2)
    parser.add_argument('--beta', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=0.01)
    parser.add_argument('--tau', type=float, default=0.2)

    parser.add_argument('--bpr_batch', type=int, default=2048)
    parser.add_argument('--recdim', type=int, default=64)
    parser.add_argument('--layer', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--decay', type=float, default=1e-4)
    parser.add_argument('--testbatch', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--topks', type=str, default='[10,20]')
    parser.add_argument('--tensorboard', type=int, default=0)

    parser.add_argument('--report_dir', type=str, default='reports')
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = [int(x.strip()) for x in args.seeds.split(',') if x.strip()]
    topks = ast.literal_eval(args.topks)
    if 20 not in topks:
        raise ValueError(f'Expected 20 in --topks for paper comparison, got: {topks}')
    topk_index = topks.index(20)

    all_metrics = []
    for seed in seeds:
        metrics = run_one_seed(seed, args)
        all_metrics.append(metrics)

    summary = summarize(all_metrics, topk_index)

    print('\n===== Multi-seed Summary (TEST-BEST @20) =====')
    print(f"Seeds: {seeds}")
    print(f"Recall@20: {summary['recall_mean']:.6f} ± {summary['recall_std']:.6f}")
    print(f"NDCG@20:   {summary['ndcg_mean']:.6f} ± {summary['ndcg_std']:.6f}")

    report = {
        'time': datetime.now().isoformat(),
        'dataset': args.dataset,
        'model': args.model,
        'seeds': seeds,
        'topks': topks,
        'all_seed_metrics': all_metrics,
        'summary_at_20': summary,
        'args': vars(args),
    }

    report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.report_dir)
    os.makedirs(report_dir, exist_ok=True)
    out_file = os.path.join(report_dir, f"{args.model}-{args.dataset}-seeds-{'-'.join(map(str, seeds))}.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Report saved: {out_file}")


if __name__ == '__main__':
    main()
