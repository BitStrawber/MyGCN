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

df = pd.DataFrame(data)[['reviewerID', 'asin', 'overall']]
df.columns = ['user', 'item', 'rating']

# 优化4: 使用 pd.factorize 高效进行从0开始的连续ID映射
df['user_id'], user_index = pd.factorize(df['user'])
df['item_id'], item_index = pd.factorize(df['item'])

print(f"Stats -> Users: {len(user_index)}, Items: {len(item_index)}, Total Interactions: {len(df)}")

# 区分正负反馈: >=4 为正，<=2 为负，(2,4) 视为中性并丢弃
df['sign'] = np.where(df['rating'] >= 4.0, 1, np.where(df['rating'] <= 2.0, -1, 0))
df = df[df['sign'] != 0].copy()
df_pos = df[df['sign'] == 1].copy()
df_neg = df[df['sign'] == -1].copy()

print(f"Positive Edges: {len(df_pos)}, Negative Edges: {len(df_neg)}")


# ================= 优化1 & 3: 按用户分组进行 7:1:2 划分 =================
def split_by_user(data_df):
    """按用户切分为 train/test（8:2），保证每个用户至少有一个 train 正样本。"""
    data_df = data_df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_list, test_list = [], []

    for _, group in data_df.groupby('user_id'):
        n = len(group)
        if n >= 2:
            test_n = max(1, int(n * 0.2))
            train_n = max(1, n - test_n)
            train_list.append(group.iloc[:train_n])
            test_list.append(group.iloc[train_n:])
        else:
            train_list.append(group)

    train_df = pd.concat(train_list) if train_list else pd.DataFrame(columns=data_df.columns)
    test_df = pd.concat(test_list) if test_list else pd.DataFrame(columns=data_df.columns)
    return train_df, test_df


print("Splitting positive edges...")
train_pos, test_pos = split_by_user(df_pos)
train_neg = df_neg.copy()


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
write_grouped_txt(test_pos, os.path.join(out_dir, "test.txt"))
write_signed_triplet(train_pos, train_neg, os.path.join(out_dir, "train_signed.txt"))

# 辅助映射文件
write_mapping(os.path.join(out_dir, "user_list.txt"), user_index)
write_mapping(os.path.join(out_dir, "item_list.txt"), item_index)

users_with_pos_and_neg = build_signed_summary(train_pos, train_neg)
print("All Done! Preprocessing complete.")
print(f"Files generated: train.txt, test.txt, train_signed.txt, user_list.txt, item_list.txt")
print(f"Users with both positive and negative train edges: {users_with_pos_and_neg}")