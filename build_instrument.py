import argparse
import pandas as pd
import gzip
import json
import os
import urllib.request
import numpy as np
from collections import defaultdict
from time import time
from tqdm import tqdm


DATASET_META = {
    'amazon_instrument': {
        'url': 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Musical_Instruments_5.json.gz',
        'file': 'Musical_Instruments_5.json.gz',
        'name': 'Amazon-Instrument',
    },
    'amazon_video': {
        'url': 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Amazon_Instant_Video_5.json.gz',
        'file': 'Amazon_Instant_Video_5.json.gz',
        'name': 'Amazon-Video',
    },
    'amazon_crafts': {
        'url': 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Arts_Crafts_and_Sewing_5.json.gz',
        'file': 'Arts_Crafts_and_Sewing_5.json.gz',
        'name': 'Amazon-Crafts',
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description='Build signed datasets for PCSRec from SNAP Amazon 5-core reviews')
    parser.add_argument('--dataset', type=str, default='amazon_instrument',
                        choices=list(DATASET_META.keys()),
                        help='Target dataset name in data/<dataset>')
    parser.add_argument('--skip_download', type=int, default=0,
                        help='Set 1 to skip downloading and use existing raw file')
    return parser.parse_args()


def _format_seconds(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f'{h}h {m}m {s}s'
    if m > 0:
        return f'{m}m {s}s'
    return f'{s}s'


def _download_with_progress(url, file_path):
    pbar = None

    def reporthook(block_num, block_size, total_size):
        nonlocal pbar
        if pbar is None:
            total = total_size if total_size > 0 else None
            pbar = tqdm(total=total, unit='B', unit_scale=True, desc='Downloading')
        downloaded = block_num * block_size
        if pbar.total is not None:
            pbar.update(downloaded - pbar.n)
        else:
            pbar.update(block_size)

    try:
        urllib.request.urlretrieve(url, file_path, reporthook=reporthook)
    finally:
        if pbar is not None:
            pbar.close()


def _load_gzip_json_with_progress(file_path):
    print('Counting raw lines for ETA...')
    t0 = time()
    total_lines = 0
    with gzip.open(file_path, 'rb') as f:
        for _ in f:
            total_lines += 1
    print(f'Line counting done: {total_lines} lines ({_format_seconds(time() - t0)})')

    data = []
    with gzip.open(file_path, 'rb') as f:
        for line in tqdm(f, total=total_lines, desc='Parsing JSON lines', unit='line'):
            data.append(json.loads(line))
    return data


def split_by_user(data_df):
    data_df = data_df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
    train_list, valid_list, test_list = [], [], []
    user_groups = data_df.groupby('user_id')
    n_users = data_df['user_id'].nunique()
    for _, group in tqdm(user_groups, total=n_users, desc='Splitting by user', unit='user'):
        n = len(group)
        if n >= 3:
            train_n = max(1, int(n * 0.7))
            valid_n = max(1, int(n * 0.1))
            if train_n + valid_n >= n:
                valid_n = 1
                train_n = n - valid_n - 1
            train_list.append(group.iloc[:train_n])
            valid_list.append(group.iloc[train_n:train_n + valid_n])
            test_list.append(group.iloc[train_n + valid_n:])
        elif n == 2:
            train_list.append(group.iloc[:1])
            test_list.append(group.iloc[1:])
        else:
            train_list.append(group)
    train_df = pd.concat(train_list) if train_list else pd.DataFrame(columns=data_df.columns)
    valid_df = pd.concat(valid_list) if valid_list else pd.DataFrame(columns=data_df.columns)
    test_df = pd.concat(test_list) if test_list else pd.DataFrame(columns=data_df.columns)
    return train_df, valid_df, test_df


def write_grouped_txt(data, path):
    if data.empty:
        open(path, 'w').close()
        return
    grouped = data.groupby('user_id')['item_id'].apply(list)
    with open(path, 'w', encoding='utf-8') as f:
        for u, items in grouped.items():
            uniq_items = sorted(set(items))
            f.write(f"{u} {' '.join(map(str, uniq_items))}\n")


def write_signed_triplet(train_pos_df, train_neg_df, path):
    with open(path, 'w', encoding='utf-8') as f:
        for row in train_pos_df[['user_id', 'item_id']].drop_duplicates().itertuples(index=False):
            f.write(f"{int(row.user_id)} {int(row.item_id)} 1\n")
        for row in train_neg_df[['user_id', 'item_id']].drop_duplicates().itertuples(index=False):
            f.write(f"{int(row.user_id)} {int(row.item_id)} -1\n")


def write_mapping(path, mapping):
    with open(path, 'w', encoding='utf-8') as f:
        for idx, raw_id in enumerate(mapping):
            f.write(f"{idx}\t{raw_id}\n")


def build_signed_summary(train_pos_df, train_neg_df):
    pos_map = defaultdict(set)
    neg_map = defaultdict(set)
    for row in train_pos_df[['user_id', 'item_id']].itertuples(index=False):
        pos_map[int(row.user_id)].add(int(row.item_id))
    for row in train_neg_df[['user_id', 'item_id']].itertuples(index=False):
        neg_map[int(row.user_id)].add(int(row.item_id))
    return sum(1 for u in pos_map if len(pos_map[u]) > 0 and len(neg_map[u]) > 0)


def main():
    total_t0 = time()
    args = parse_args()
    meta = DATASET_META[args.dataset]

    root = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(root, 'raw_data', 'amazon')
    os.makedirs(raw_dir, exist_ok=True)
    file_path = os.path.join(raw_dir, meta['file'])
    out_dir = os.path.join(root, 'data', args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(file_path):
        if args.skip_download == 1:
            raise FileNotFoundError(f'Raw file missing: {file_path}')
        print(f"Downloading {meta['name']} dataset from SNAP...")
        stage_t0 = time()
        _download_with_progress(meta['url'], file_path)
        print('Download success!')
        print(f'Download time: {_format_seconds(time() - stage_t0)}')
    else:
        print(f'File already exists at {file_path}, skipping download.')

    print('Processing data (this may take a moment)...')
    stage_t0 = time()
    data = _load_gzip_json_with_progress(file_path)
    print(f'Load+parse time: {_format_seconds(time() - stage_t0)}')

    stage_t0 = time()
    df = pd.DataFrame(data)[['reviewerID', 'asin', 'overall', 'unixReviewTime']]
    df.columns = ['user', 'item', 'rating', 'timestamp']
    df['user_id'], user_index = pd.factorize(df['user'])
    df['item_id'], item_index = pd.factorize(df['item'])
    print(f'Build DataFrame+ID map time: {_format_seconds(time() - stage_t0)}')

    print(f"Stats -> Users: {len(user_index)}, Items: {len(item_index)}, Total Interactions: {len(df)}")
    df['sign'] = np.where(df['rating'] >= 4.0, 1, -1)
    df_pos = df[df['sign'] == 1].copy()
    df_neg = df[df['sign'] == -1].copy()
    print(f"Positive Edges: {len(df_pos)}, Negative Edges: {len(df_neg)}")

    print('Splitting positive edges...')
    stage_t0 = time()
    train_pos, valid_pos, test_pos = split_by_user(df_pos)
    print('Splitting negative edges...')
    train_neg, valid_neg, test_neg = split_by_user(df_neg)
    print(f'Splitting time: {_format_seconds(time() - stage_t0)}')

    print(f'Writing files to {out_dir}...')
    stage_t0 = time()
    write_grouped_txt(train_pos, os.path.join(out_dir, 'train.txt'))
    write_grouped_txt(valid_pos, os.path.join(out_dir, 'valid.txt'))
    write_grouped_txt(test_pos, os.path.join(out_dir, 'test.txt'))
    write_signed_triplet(train_pos, train_neg, os.path.join(out_dir, 'train_signed.txt'))
    write_mapping(os.path.join(out_dir, 'user_list.txt'), user_index)
    write_mapping(os.path.join(out_dir, 'item_list.txt'), item_index)
    print(f'Write files time: {_format_seconds(time() - stage_t0)}')

    users_with_pos_and_neg = build_signed_summary(train_pos, train_neg)
    print('All Done! Preprocessing complete.')
    print('Files generated: train.txt, valid.txt, test.txt, train_signed.txt, user_list.txt, item_list.txt')
    print(f'Users with both positive and negative train edges: {users_with_pos_and_neg}')
    print(f'Train/Valid/Test positive edges: {len(train_pos)}/{len(valid_pos)}/{len(test_pos)}')
    print(f'Train/Valid/Test negative edges: {len(train_neg)}/{len(valid_neg)}/{len(test_neg)}')
    print(f'Total time: {_format_seconds(time() - total_t0)}')


if __name__ == '__main__':
    main()