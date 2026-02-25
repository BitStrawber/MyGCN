import argparse
import ast
import csv
import json
import os
import re
import statistics
import subprocess
import sys
from datetime import datetime


PAPER_RESULTS = {
    'amazon_video': {'ndcg20': 0.06744, 'recall20': 0.11000},
    'amazon_instrument': {'ndcg20': 0.03014, 'recall20': 0.04984},
    'amazon_crafts': {'ndcg20': 0.03388, 'recall20': 0.05418},
}

# 可按论文/经验继续微调
DATASET_HPARAMS = {
    'amazon_video': {'alpha': 0.5, 'beta': 1.0, 'gamma': 0.01, 'tau': 0.2},
    'amazon_instrument': {'alpha': -0.2, 'beta': 0.2, 'gamma': 0.01, 'tau': 0.2},
    'amazon_crafts': {'alpha': 0.5, 'beta': 1.0, 'gamma': 0.01, 'tau': 0.2},
}

METRIC_PATTERNS = {
    'precision': re.compile(r"'precision': array\(\[([^\]]+)\]\)"),
    'recall': re.compile(r"'recall': array\(\[([^\]]+)\]\)"),
    'ndcg': re.compile(r"'ndcg': array\(\[([^\]]+)\]\)"),
}


def parse_array_values(text):
    values = [v.strip() for v in text.split(',') if v.strip()]
    return [float(v) for v in values]


def parse_metric_line(line):
    parsed = {}
    for key, pattern in METRIC_PATTERNS.items():
        m = pattern.search(line)
        if not m:
            return None
        parsed[key] = parse_array_values(m.group(1))
    return parsed


def run_and_stream(cmd, cwd):
    env = os.environ.copy()
    omp_val = env.get('OMP_NUM_THREADS', '').strip()
    if (not omp_val.isdigit()) or int(omp_val) <= 0:
        env['OMP_NUM_THREADS'] = '1'
    env['PYTHONUNBUFFERED'] = '1'

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    lines = []
    for line in proc.stdout:
        print(line, end='')
        lines.append(line.rstrip('\n'))

    code = proc.wait()
    if code != 0:
        raise RuntimeError(f'Command failed({code}): {" ".join(cmd)}')
    return lines


def run_preprocess(repo_root, dataset):
    print(f"\n===== Preprocess: {dataset} =====")
    cmd = [
        sys.executable,
        '-u',
        'build_instrument.py',
        '--dataset',
        dataset,
    ]
    run_and_stream(cmd, cwd=repo_root)


def run_train_one_seed(code_dir, dataset, seed, base_args, topks):
    hp = DATASET_HPARAMS[dataset]
    cmd = [
        sys.executable,
        '-u',
        'main.py',
        f'--dataset={dataset}',
        '--model=pcsrec',
        f"--alpha={hp['alpha']}",
        f"--beta={hp['beta']}",
        f"--gamma={hp['gamma']}",
        f"--tau={hp['tau']}",
        f"--bpr_batch={base_args.bpr_batch}",
        f"--recdim={base_args.recdim}",
        f"--layer={base_args.layer}",
        f"--lr={base_args.lr}",
        f"--decay={base_args.decay}",
        f"--testbatch={base_args.testbatch}",
        f"--epochs={base_args.epochs}",
        f"--topks={topks}",
        f"--tensorboard={base_args.tensorboard}",
        f"--eval_interval={base_args.eval_interval}",
        f"--early_stop_patience={base_args.early_stop_patience}",
        f"--early_stop_min_delta={base_args.early_stop_min_delta}",
        f'--seed={seed}',
    ]

    print(f"\n===== Train: {dataset}, seed={seed} =====")
    lines = run_and_stream(cmd, cwd=code_dir)

    seen_test_best = False
    metric_line = None
    for line in lines:
        if '[TEST-BEST]' in line:
            seen_test_best = True
            continue
        if seen_test_best and "{'precision':" in line and "'recall':" in line and "'ndcg':" in line:
            metric_line = line

    if metric_line is None:
        raise RuntimeError(f'No [TEST-BEST] metric found for dataset={dataset}, seed={seed}')

    metrics = parse_metric_line(metric_line)
    if metrics is None:
        raise RuntimeError(f'Cannot parse [TEST-BEST] line: {metric_line}')

    return metrics


def summarize_dataset(seed_metrics, topk_index):
    recall_vals = [m['recall'][topk_index] for m in seed_metrics]
    ndcg_vals = [m['ndcg'][topk_index] for m in seed_metrics]

    return {
        'recall20_mean': statistics.mean(recall_vals),
        'recall20_std': statistics.pstdev(recall_vals) if len(recall_vals) > 1 else 0.0,
        'ndcg20_mean': statistics.mean(ndcg_vals),
        'ndcg20_std': statistics.pstdev(ndcg_vals) if len(ndcg_vals) > 1 else 0.0,
    }


def save_reports(report_dir, run_report, table_rows):
    os.makedirs(report_dir, exist_ok=True)

    json_path = os.path.join(report_dir, 'benchmark_summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(run_report, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(report_dir, 'benchmark_summary.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
        writer.writeheader()
        writer.writerows(table_rows)

    md_path = os.path.join(report_dir, 'benchmark_summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('| Dataset | Ours NDCG@20(mean±std) | Paper NDCG@20 | NDCG Gap | Ours Recall@20(mean±std) | Paper Recall@20 | Recall Gap |\n')
        f.write('|---|---:|---:|---:|---:|---:|---:|\n')
        for row in table_rows:
            f.write(
                f"| {row['dataset']} | {row['ours_ndcg20']} | {row['paper_ndcg20']:.5f} | {row['ndcg_gap_pct']} | "
                f"{row['ours_recall20']} | {row['paper_recall20']:.5f} | {row['recall_gap_pct']} |\n"
            )

    return json_path, csv_path, md_path


def parse_args():
    parser = argparse.ArgumentParser(description='Run PCSRec on 3 paper datasets and compare with paper results')
    parser.add_argument('--datasets', type=str, default='amazon_video,amazon_instrument,amazon_crafts')
    parser.add_argument('--seeds', type=str, default='2020,2021')
    parser.add_argument('--preprocess', type=int, default=1, help='1: run preprocess before training')

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--bpr_batch', type=int, default=4096)
    parser.add_argument('--recdim', type=int, default=64)
    parser.add_argument('--layer', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--decay', type=float, default=5e-4)
    parser.add_argument('--testbatch', type=int, default=256)
    parser.add_argument('--topks', type=str, default='[10,20]')
    parser.add_argument('--tensorboard', type=int, default=0)
    parser.add_argument('--eval_interval', type=int, default=5)
    parser.add_argument('--early_stop_patience', type=int, default=6)
    parser.add_argument('--early_stop_min_delta', type=float, default=1e-4)

    parser.add_argument('--output_dir', type=str, default='reports/paper_benchmark')
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    topks = ast.literal_eval(args.topks)
    if 20 not in topks:
        raise ValueError('--topks must include 20 for paper comparison')
    topk_index = topks.index(20)

    for d in datasets:
        if d not in DATASET_HPARAMS:
            raise ValueError(f'Unsupported dataset: {d}')

    code_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(code_dir)

    all_results = {}
    table_rows = []

    for dataset in datasets:
        if args.preprocess == 1:
            run_preprocess(repo_root, dataset)

        seed_metrics = []
        for seed in seeds:
            metrics = run_train_one_seed(code_dir, dataset, seed, args, args.topks)
            seed_metrics.append(metrics)

        summary = summarize_dataset(seed_metrics, topk_index)
        paper = PAPER_RESULTS.get(dataset)

        if paper is not None:
            ndcg_gap_pct = ((summary['ndcg20_mean'] - paper['ndcg20']) / paper['ndcg20']) * 100.0
            recall_gap_pct = ((summary['recall20_mean'] - paper['recall20']) / paper['recall20']) * 100.0
        else:
            ndcg_gap_pct = float('nan')
            recall_gap_pct = float('nan')

        all_results[dataset] = {
            'seeds': seeds,
            'seed_metrics': seed_metrics,
            'summary': summary,
            'paper': paper,
            'gap_percent': {
                'ndcg20': ndcg_gap_pct,
                'recall20': recall_gap_pct,
            },
            'hparams': DATASET_HPARAMS[dataset],
        }

        table_rows.append({
            'dataset': dataset,
            'ours_ndcg20': f"{summary['ndcg20_mean']:.5f}±{summary['ndcg20_std']:.5f}",
            'paper_ndcg20': paper['ndcg20'] if paper else 0.0,
            'ndcg_gap_pct': f"{ndcg_gap_pct:+.2f}%",
            'ours_recall20': f"{summary['recall20_mean']:.5f}±{summary['recall20_std']:.5f}",
            'paper_recall20': paper['recall20'] if paper else 0.0,
            'recall_gap_pct': f"{recall_gap_pct:+.2f}%",
        })

    run_report = {
        'time': datetime.now().isoformat(),
        'datasets': datasets,
        'seeds': seeds,
        'topks': topks,
        'train_args': {
            'epochs': args.epochs,
            'bpr_batch': args.bpr_batch,
            'recdim': args.recdim,
            'layer': args.layer,
            'lr': args.lr,
            'decay': args.decay,
            'testbatch': args.testbatch,
            'tensorboard': args.tensorboard,
            'eval_interval': args.eval_interval,
            'early_stop_patience': args.early_stop_patience,
            'early_stop_min_delta': args.early_stop_min_delta,
        },
        'results': all_results,
    }

    out_dir = os.path.join(code_dir, args.output_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
    json_path, csv_path, md_path = save_reports(out_dir, run_report, table_rows)

    print('\n===== Final Comparison (Ours vs Paper) =====')
    for row in table_rows:
        print(
            f"{row['dataset']}: NDCG@20 {row['ours_ndcg20']} (paper {row['paper_ndcg20']:.5f}, {row['ndcg_gap_pct']}), "
            f"Recall@20 {row['ours_recall20']} (paper {row['paper_recall20']:.5f}, {row['recall_gap_pct']})"
        )
    print(f'\nSaved reports:\n- {json_path}\n- {csv_path}\n- {md_path}')


if __name__ == '__main__':
    main()
