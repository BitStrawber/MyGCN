import pandas as pd
import os

# 1. 读取你下载的原始 Yelp 评分数据
# (假设你已经从 Kaggle 下载了 yelp_academic_dataset_review.json)
df = pd.read_json("raw_data/yelp/yelp_academic_dataset_review.json", lines=True, chunksize=100000)
df = pd.concat([chunk[['user_id', 'business_id', 'stars']] for chunk in df])
df.columns = ['user', 'item', 'rating']


# 2. K-Core 过滤 (为了匹配论文中 3000 user / 7000 item 的规模，K 大约设为 30)
def filter_k_core(data, k=30):
    while True:
        user_counts = data['user'].value_counts()
        item_counts = data['item'].value_counts()
        valid_users = user_counts[user_counts >= k].index
        valid_items = item_counts[item_counts >= k].index

        data_new = data[data['user'].isin(valid_users) & data['item'].isin(valid_items)]
        if len(data_new) == len(data):
            break
        data = data_new.copy()
    return data


df = filter_k_core(df, k=30)

# 3. 重新映射 ID (0 到 N-1)
user_map = {u: i for i, u in enumerate(df['user'].unique())}
item_map = {i: j for j, i in enumerate(df['item'].unique())}
df['user_id'] = df['user'].map(user_map)
df['item_id'] = df['item'].map(item_map)
print(f"Users: {len(user_map)}, Items: {len(item_map)}, Edges: {len(df)}")

# 4. 区分正负反馈 (论文标准：>= 4星为正，其余为负)
df_pos = df[df['rating'] >= 4.0]
df_neg = df[df['rating'] < 4.0]

# 5. 划分数据集 7:1:2 (仅对正反馈划分)
shuffled_pos = df_pos.sample(frac=1, random_state=42).reset_index(drop=True)
n = len(shuffled_pos)
train_pos = shuffled_pos.iloc[:int(n * 0.7)]
valid_pos = shuffled_pos.iloc[int(n * 0.7):int(n * 0.8)]
test_pos = shuffled_pos.iloc[int(n * 0.8):]

# 写入 TXT
out_dir = "../data/yelp_pcsrec"
os.makedirs(out_dir, exist_ok=True)


def write_txt(data, path):
    grouped = data.groupby('user_id')['item_id'].apply(list)
    with open(path, 'w') as f:
        for u, items in grouped.items():
            f.write(f"{u} " + " ".join(map(str, items)) + "\n")


write_txt(train_pos, f"{out_dir}/train_pos.txt")
write_txt(df_neg, f"{out_dir}/train_neg.txt")
write_txt(valid_pos, f"{out_dir}/valid.txt")
write_txt(test_pos, f"{out_dir}/test.txt")
print("Yelp PCSRec dataset generated!")