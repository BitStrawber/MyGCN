import pandas as pd
import gzip
import json
import os
import urllib.request

# 1. 自动下载
url = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/Musical_Instruments_5.json.gz"
file_path = "Musical_Instruments_5.json.gz"
if not os.path.exists(file_path):
    print("Downloading Amazon-Instrument dataset (fast)...")
    urllib.request.urlretrieve(url, file_path)

print("Processing data...")
# 2. 读取压缩包
data =[]
with gzip.open(file_path, 'rb') as f:
    for line in f:
        data.append(json.loads(line))
df = pd.DataFrame(data)[['reviewerID', 'asin', 'overall']]
df.columns = ['user', 'item', 'rating']

# 3. 重新映射 ID (0 到 N-1)
user_map = {u: i for i, u in enumerate(df['user'].unique())}
item_map = {i: j for j, i in enumerate(df['item'].unique())}
df['user_id'] = df['user'].map(user_map)
df['item_id'] = df['item'].map(item_map)
print(f"Users: {len(user_map)}, Items: {len(item_map)}, Edges: {len(df)}")

# 4. 区分正负反馈 (论文标准: >=4为正, <4为负)
df_pos = df[df['rating'] >= 4.0]
df_neg = df[df['rating'] < 4.0]

# 5. 划分数据集 7:1:2 (仅对正反馈进行划分)
shuffled_pos = df_pos.sample(frac=1, random_state=42).reset_index(drop=True)
n = len(shuffled_pos)
train_pos = shuffled_pos.iloc[:int(n*0.7)]
valid_pos = shuffled_pos.iloc[int(n*0.7):int(n*0.8)]
test_pos = shuffled_pos.iloc[int(n*0.8):]

# 6. 保存为 LightGCN 格式
out_dir = "data/amazon_instrument"
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
print(f"Success! Data saved to {out_dir}")