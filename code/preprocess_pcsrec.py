import argparse
import os
import random
import numpy as np
import pandas as pd
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess raw ratings for PCSRec')
    parser.add_argument('--input', type=str, required=True, help='Raw file path (.csv/.tsv/.txt)')
    parser.add_argument('--output_dir', type=str, required=True, help='Output dataset directory')
    parser.add_argument('--sep', type=str, default=',', help='Separator for input file')
    parser.add_argument('--user_col', type=str, default='user', help='User column name')
    parser.add_argument('--item_col', type=str, default='item', help='Item column name')
    parser.add_argument('--rating_col', type=str, default='rating', help='Rating column name')
    parser.add_argument('--time_col', type=str, default='', help='Optional timestamp column name')
    parser.add_argument('--pos_thres', type=float, default=4.0, help='Positive threshold')
    parser.add_argument('--neg_thres', type=float, default=2.0, help='Negative threshold')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='Per-user test ratio for positive edges')
    parser.add_argument('--seed', type=int, default=2020)
    return parser.parse_args()


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def remap_ids(df, user_col, item_col):
    unique_users = df[user_col].unique().tolist()
    unique_items = df[item_col].unique().tolist()
    user_map = {u: idx for idx, u in enumerate(unique_users)}
    item_map = {i: idx for idx, i in enumerate(unique_items)}
    df['uid'] = df[user_col].map(user_map).astype(int)
    df['iid'] = df[item_col].map(item_map).astype(int)
    return df, user_map, item_map


def build_signed_edges(df, rating_col, pos_thres, neg_thres):
    pos_df = df[df[rating_col] >= pos_thres][['uid', 'iid']].copy()
    neg_df = df[df[rating_col] <= neg_thres][['uid', 'iid']].copy()
    return pos_df, neg_df


def split_train_test_pos(pos_df, test_ratio, seed, time_series=None):
    random.seed(seed)
    train_user_items = defaultdict(list)
    test_user_items = defaultdict(list)

    if time_series is not None:
        pos_df = pos_df.sort_values(by=['uid', time_series])
        grouped = pos_df.groupby('uid')
        for uid, group in grouped:
            items = group['iid'].tolist()
            if len(items) == 1:
                train_user_items[uid].append(items[0])
                continue
            test_n = max(1, int(len(items) * test_ratio))
            train_items = items[:-test_n]
            test_items = items[-test_n:]
            train_user_items[uid].extend(train_items)
            test_user_items[uid].extend(test_items)
    else:
        grouped = pos_df.groupby('uid')
        for uid, group in grouped:
            items = group['iid'].tolist()
            if len(items) == 1:
                train_user_items[uid].append(items[0])
                continue
            random.shuffle(items)
            test_n = max(1, int(len(items) * test_ratio))
            test_items = items[:test_n]
            train_items = items[test_n:]
            if len(train_items) == 0:
                train_items = test_items[:1]
                test_items = test_items[1:]
            train_user_items[uid].extend(train_items)
            test_user_items[uid].extend(test_items)

    return train_user_items, test_user_items


def write_grouped_txt(path, grouped):
    with open(path, 'w', encoding='utf-8') as f:
        for uid in sorted(grouped.keys()):
            items = sorted(set(grouped[uid]))
            if len(items) == 0:
                continue
            f.write(str(uid) + ' ' + ' '.join(map(str, items)) + '\n')


def write_signed_triplet(path, train_pos, train_neg):
    with open(path, 'w', encoding='utf-8') as f:
        for uid, items in train_pos.items():
            for iid in set(items):
                f.write(f'{uid} {iid} 1\n')
        for uid, items in train_neg.items():
            for iid in set(items):
                f.write(f'{uid} {iid} -1\n')


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    df = pd.read_csv(args.input, sep=args.sep)
    required_cols = [args.user_col, args.item_col, args.rating_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f'Missing column: {col}')

    df, user_map, item_map = remap_ids(df, args.user_col, args.item_col)
    pos_df, neg_df = build_signed_edges(df, args.rating_col, args.pos_thres, args.neg_thres)

    time_col = args.time_col if args.time_col and args.time_col in df.columns else None
    if time_col is not None:
        pos_with_time = df[df[args.rating_col] >= args.pos_thres][['uid', 'iid', time_col]].copy()
        train_pos, test_pos = split_train_test_pos(pos_with_time, args.test_ratio, args.seed, time_series=time_col)
    else:
        train_pos, test_pos = split_train_test_pos(pos_df, args.test_ratio, args.seed)

    train_neg = defaultdict(list)
    for _, row in neg_df.iterrows():
        train_neg[int(row['uid'])].append(int(row['iid']))

    write_grouped_txt(os.path.join(args.output_dir, 'train.txt'), train_pos)
    write_grouped_txt(os.path.join(args.output_dir, 'test.txt'), test_pos)
    write_signed_triplet(os.path.join(args.output_dir, 'train_signed.txt'), train_pos, train_neg)

    with open(os.path.join(args.output_dir, 'user_list.txt'), 'w', encoding='utf-8') as f:
        for raw_user, uid in user_map.items():
            f.write(f'{uid}\t{raw_user}\n')

    with open(os.path.join(args.output_dir, 'item_list.txt'), 'w', encoding='utf-8') as f:
        for raw_item, iid in item_map.items():
            f.write(f'{iid}\t{raw_item}\n')

    print('Preprocess done')
    print(f'Users: {len(user_map)}, Items: {len(item_map)}')
    print(f'Positive interactions: {len(pos_df)}, Negative interactions: {len(neg_df)}')


if __name__ == '__main__':
    main()
