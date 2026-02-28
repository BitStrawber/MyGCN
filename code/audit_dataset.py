import argparse
import os
from collections import defaultdict


def parse_grouped_interactions(file_path):
    user_items = defaultdict(set)
    interactions = set()
    if not os.path.exists(file_path):
        return user_items, interactions
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            user = int(parts[0])
            items = [int(x) for x in parts[1:]]
            for item in items:
                user_items[user].add(item)
                interactions.add((user, item))
    return user_items, interactions


def parse_signed_triplets(file_path):
    pos = set()
    neg = set()
    if not os.path.exists(file_path):
        return pos, neg
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            user, item, sign = int(parts[0]), int(parts[1]), int(parts[2])
            if sign > 0:
                pos.add((user, item))
            elif sign < 0:
                neg.add((user, item))
    return pos, neg


def interaction_stats(interactions):
    if not interactions:
        return 0, 0, 0
    users = {u for u, _ in interactions}
    items = {i for _, i in interactions}
    return len(users), len(items), len(interactions)


def user_level_overlap(source_user_items, target_user_items):
    leaked = 0
    total = 0
    for user, items in target_user_items.items():
        total += len(items)
        if user in source_user_items:
            leaked += len(items & source_user_items[user])
    ratio = 0.0 if total == 0 else leaked / total
    return leaked, total, ratio


def set_overlap(name_a, set_a, name_b, set_b):
    inter = len(set_a & set_b)
    ratio_a = 0.0 if len(set_a) == 0 else inter / len(set_a)
    ratio_b = 0.0 if len(set_b) == 0 else inter / len(set_b)
    print(f"[Overlap] {name_a} ∩ {name_b}: {inter} ({ratio_a:.6%} of {name_a}, {ratio_b:.6%} of {name_b})")


def main():
    parser = argparse.ArgumentParser(description='Audit dataset consistency and leakage risks for MyGCN/PCSRec')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset folder name under ../data, e.g., yelp2018, amazon-book, amazon_instrument')
    parser.add_argument('--data_root', type=str, default='../data',
                        help='Root data folder')
    parser.add_argument('--paper_users', type=int, default=-1,
                        help='Optional expected #users from paper table')
    parser.add_argument('--paper_items', type=int, default=-1,
                        help='Optional expected #items from paper table')
    args = parser.parse_args()

    dataset_dir = os.path.join(args.data_root, args.dataset)
    train_path = os.path.join(dataset_dir, 'train.txt')
    valid_path = os.path.join(dataset_dir, 'valid.txt')
    test_path = os.path.join(dataset_dir, 'test.txt')
    signed_path = os.path.join(dataset_dir, 'train_signed.txt')

    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f'Dataset folder not found: {dataset_dir}')

    train_u2i, train_set = parse_grouped_interactions(train_path)
    valid_u2i, valid_set = parse_grouped_interactions(valid_path)
    test_u2i, test_set = parse_grouped_interactions(test_path)
    signed_pos, signed_neg = parse_signed_triplets(signed_path)

    print(f'=== Dataset Audit: {args.dataset} ===')
    print(f'Dataset dir: {os.path.abspath(dataset_dir)}')
    print(f'Files present: train={os.path.exists(train_path)}, valid={os.path.exists(valid_path)}, test={os.path.exists(test_path)}, train_signed={os.path.exists(signed_path)}')

    train_users, train_items, train_inter = interaction_stats(train_set)
    valid_users, valid_items, valid_inter = interaction_stats(valid_set)
    test_users, test_items, test_inter = interaction_stats(test_set)

    union_set = train_set | valid_set | test_set
    all_users, all_items, all_inter = interaction_stats(union_set)

    print('--- Basic Stats ---')
    print(f'Train: users={train_users}, items={train_items}, interactions={train_inter}')
    print(f'Valid: users={valid_users}, items={valid_items}, interactions={valid_inter}')
    print(f'Test : users={test_users}, items={test_items}, interactions={test_inter}')
    print(f'All  : users={all_users}, items={all_items}, interactions={all_inter}')

    if args.paper_users > 0:
        diff_users = all_users - args.paper_users
        print(f'[Compare Paper] users={all_users}, paper={args.paper_users}, diff={diff_users}')
    if args.paper_items > 0:
        diff_items = all_items - args.paper_items
        print(f'[Compare Paper] items={all_items}, paper={args.paper_items}, diff={diff_items}')

    print('--- Leakage / Overlap Checks ---')
    set_overlap('train', train_set, 'valid', valid_set)
    set_overlap('train', train_set, 'test', test_set)
    set_overlap('valid', valid_set, 'test', test_set)

    leaked_v, total_v, ratio_v = user_level_overlap(train_u2i, valid_u2i)
    leaked_t, total_t, ratio_t = user_level_overlap(train_u2i, test_u2i)
    print(f'[User-Level Leak] valid items already in same-user train: {leaked_v}/{total_v} ({ratio_v:.6%})')
    print(f'[User-Level Leak] test  items already in same-user train: {leaked_t}/{total_t} ({ratio_t:.6%})')

    if os.path.exists(signed_path):
        print('--- Signed Consistency Checks ---')
        print(f'train_signed positive edges: {len(signed_pos)}')
        print(f'train_signed negative edges: {len(signed_neg)}')
        overlap_pos = len(train_set & signed_pos)
        only_train = len(train_set - signed_pos)
        only_signed_pos = len(signed_pos - train_set)
        print(f'pos overlap(train.txt ∩ train_signed +1): {overlap_pos}')
        print(f'pos only in train.txt: {only_train}')
        print(f'pos only in train_signed(+1): {only_signed_pos}')

    print('--- Interpretation Hints ---')
    print('* Full-ranking evaluation requires scoring against all items and masking train positives.')
    print('* If train∩test > 0 or same-user train/test leak ratio > 0, metrics can be inflated.')
    print('* If paper stats differ a lot, preprocessing protocol is likely inconsistent.')


if __name__ == '__main__':
    main()
