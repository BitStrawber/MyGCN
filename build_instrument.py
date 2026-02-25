import pandas as pd
import gzip
import json
import os
import urllib.request
import numpy as np
from collections import defaultdict

# ================= 配置区 =================
url = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Musical_Instruments_5.json.gz"
ROOT = os.path.dirname(os.path.abspath(__file__))
raw_dir = os.path.join(ROOT, "raw_data", "amazon")
os.makedirs(raw_dir, exist_ok=True)
file_path = os.path.join(raw_dir, "Musical_Instruments_5.json.gz")
out_dir = os.path.join(ROOT, "data", "amazon_instrument")
os.makedirs(out_dir, exist_ok=True)

# ================= 下载逻辑 =================
if not os.path.exists(file_path):
    print(f"Downloading Amazon-Instrument dataset from SNAP...")
    try:
        urllib.request.urlretrieve(url, file_path)
        print("Download success!")
    except Exception as e:
        print(f"Error downloading: {e}")
        exit()
else:
    print(f"File already exists at {file_path}, skipping download.")

# ================= 处理逻辑 =================
print("Processing data (this may take a moment)...")

data = []
with gzip.open(file_path, 'rb') as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)[['reviewerID', 'asin', 'overall', 'unixReviewTime']]
df.columns = ['user', 'item', 'rating', 'timestamp']

# 优化4: 使用 pd.factorize 高效进行从0开始的连续ID映射
df['user_id'], user_index = pd.factorize(df['user'])
df['item_id'], item_index = pd.factorize(df['item'])

print(f"Stats -> Users: {len(user_index)}, Items: {len(item_index)}, Total Interactions: {len(df)}")

# 论文口径：rating >= 4 为正反馈，rating < 4 为负反馈
df['sign'] = np.where(df['rating'] >= 4.0, 1, -1)
df_pos = df[df['sign'] == 1].copy()
df_neg = df[df['sign'] == -1].copy()

print(f"Positive Edges: {len(df_pos)}, Negative Edges: {len(df_neg)}")


# ================= 优化1 & 3: 按用户分组进行 7:1:2 划分 =================
def split_by_user(data_df):
    """按时间切分 7:1:2（train/valid/test）。"""
    data_df = data_df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

    train_list, valid_list, test_list = [], [], []

    for _, group in data_df.groupby('user_id'):
        n = len(group)
        if n >= 3:
            train_n = max(1, int(n * 0.7))
            valid_n = max(1, int(n * 0.1))
            if train_n + valid_n >= n:
                valid_n = 1
                train_n = n - valid_n - 1
            test_n = n - train_n - valid_n
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


print("Splitting positive edges...")
train_pos, valid_pos, test_pos = split_by_user(df_pos)
print("Splitting negative edges...")
train_neg, valid_neg, test_neg = split_by_user(df_neg)


# ================= 文件写入 =================
def write_grouped_txt(data, path):
    if data.empty:
        open(path, 'w').close()
        return
    grouped = data.groupby('user_id')['item_id'].apply(list)
    with open(path, 'w', encoding='utf-8') as f:
        for u, items in grouped.items():
            uniq_items = sorted(set(items))
            item_str = " ".join(map(str, uniq_items))
            f.write(f"{u} {item_str}\n")


def write_signed_triplet(train_pos_df, train_neg_df, path):
    with open(path, 'w', encoding='utf-8') as f:
        if not train_pos_df.empty:
            for row in train_pos_df[['user_id', 'item_id']].drop_duplicates().itertuples(index=False):
                f.write(f"{int(row.user_id)} {int(row.item_id)} 1\n")
        if not train_neg_df.empty:
            for row in train_neg_df[['user_id', 'item_id']].drop_duplicates().itertuples(index=False):
                f.write(f"{int(row.user_id)} {int(row.item_id)} -1\n")


def write_mapping(path, mapping):
    with open(path, 'w', encoding='utf-8') as f:
        for idx, raw_id in enumerate(mapping):
            f.write(f"{idx}\t{raw_id}\n")


def build_signed_summary(train_pos_df, train_neg_df):
    pos_map = defaultdict(list)
    neg_map = defaultdict(list)
    if not train_pos_df.empty:
        for row in train_pos_df[['user_id', 'item_id']].itertuples(index=False):
            pos_map[int(row.user_id)].append(int(row.item_id))
    if not train_neg_df.empty:
        for row in train_neg_df[['user_id', 'item_id']].itertuples(index=False):
            neg_map[int(row.user_id)].append(int(row.item_id))
    users_with_pos_and_neg = sum(1 for u in pos_map if len(set(pos_map[u])) > 0 and len(set(neg_map[u])) > 0)
    return users_with_pos_and_neg


print(f"Writing files to {out_dir}...")
# LightGCN / PCSRec 读取的标准文件
write_grouped_txt(train_pos, os.path.join(out_dir, "train.txt"))
write_grouped_txt(valid_pos, os.path.join(out_dir, "valid.txt"))
write_grouped_txt(test_pos, os.path.join(out_dir, "test.txt"))
write_signed_triplet(train_pos, train_neg, os.path.join(out_dir, "train_signed.txt"))

# 辅助映射文件
write_mapping(os.path.join(out_dir, "user_list.txt"), user_index)
write_mapping(os.path.join(out_dir, "item_list.txt"), item_index)

users_with_pos_and_neg = build_signed_summary(train_pos, train_neg)
print("All Done! Preprocessing complete.")
print(f"Files generated: train.txt, valid.txt, test.txt, train_signed.txt, user_list.txt, item_list.txt")
print(f"Users with both positive and negative train edges: {users_with_pos_and_neg}")
print(f"Train/Valid/Test positive edges: {len(train_pos)}/{len(valid_pos)}/{len(test_pos)}")
print(f"Train/Valid/Test negative edges: {len(train_neg)}/{len(valid_neg)}/{len(test_neg)}")