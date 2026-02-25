import pandas as pd
import gzip
import json
import os
import urllib.request
import numpy as np

# ================= 配置区 =================
url = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Musical_Instruments_5.json.gz"
raw_dir = "../raw_data/amazon"
os.makedirs(raw_dir, exist_ok=True)
file_path = os.path.join(raw_dir, "Musical_Instruments_5.json.gz")
out_dir = "../data/amazon_instrument"
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

# 区分正负反馈 (论文标准: >=4为正, <4为负)
df['sign'] = np.where(df['rating'] >= 4.0, 1, -1)
df_pos = df[df['sign'] == 1].copy()
df_neg = df[df['sign'] == -1].copy()

print(f"Positive Edges: {len(df_pos)}, Negative Edges: {len(df_neg)}")


# ================= 优化1 & 3: 按用户分组进行 7:1:2 划分 =================
def split_by_user(data_df):
    """按用户将数据划分为 7:1:2，确保尽量避免孤立节点"""
    # 打乱数据
    data_df = data_df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_list, valid_list, test_list = [], [], []

    for _, group in data_df.groupby('user_id'):
        n = len(group)
        if n >= 3:
            train_end = int(n * 0.7)
            valid_end = int(n * 0.8)
            # 保证训练集至少有1条
            train_end = max(1, train_end)
            valid_end = max(train_end + 1, valid_end)

            train_list.append(group.iloc[:train_end])
            valid_list.append(group.iloc[train_end:valid_end])
            test_list.append(group.iloc[valid_end:])
        else:
            # 如果某个用户正/负交互极少，全放入训练集防止报错
            train_list.append(group)

    return pd.concat(train_list), pd.concat(valid_list) if valid_list else pd.DataFrame(), pd.concat(
        test_list) if test_list else pd.DataFrame()


print("Splitting positive edges...")
train_pos, valid_pos, test_pos = split_by_user(df_pos)
print("Splitting negative edges...")
train_neg, valid_neg, test_neg = split_by_user(df_neg)


# ================= 文件写入 =================
def write_txt(data, path):
    if data.empty:
        open(path, 'w').close()
        return
    grouped = data.groupby('user_id')['item_id'].apply(list)
    with open(path, 'w') as f:
        for u, items in grouped.items():
            item_str = " ".join(map(str, items))
            f.write(f"{u} {item_str}\n")


print(f"Writing files to {out_dir}...")
# 保存正向图数据
write_txt(train_pos, f"{out_dir}/train_pos.txt")
write_txt(valid_pos, f"{out_dir}/valid_pos.txt")
write_txt(test_pos, f"{out_dir}/test_pos.txt")

# 保存负向图数据 (优化2: 必须保存负反馈的测试数据以供后续分析)
write_txt(train_neg, f"{out_dir}/train_neg.txt")
write_txt(valid_neg, f"{out_dir}/valid_neg.txt")
write_txt(test_neg, f"{out_dir}/test_neg.txt")

print("All Done! Preprocessing complete.")