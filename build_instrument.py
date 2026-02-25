import pandas as pd
import gzip
import json
import os
import urllib.request

# ================= 配置区 =================
# 1. 修改下载链接为 Stanford SNAP 源 (2014版，稳定可用)
url = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Musical_Instruments_5.json.gz"

# 2. 设置文件保存路径 (建议放在 raw_data 目录下)
raw_dir = "raw_data/amazon"
os.makedirs(raw_dir, exist_ok=True) # 自动创建目录
file_path = os.path.join(raw_dir, "Musical_Instruments_5.json.gz")

# ================= 下载逻辑 =================
if not os.path.exists(file_path):
    print(f"Downloading Amazon-Instrument dataset from SNAP...")
    print(f"Source: {url}")
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
try:
    with gzip.open(file_path, 'rb') as f:
        for line in f:
            data.append(json.loads(line))
except OSError:
    print("Error: The file seems corrupted. Please delete it and run the script again.")
    exit()

df = pd.DataFrame(data)[['reviewerID', 'asin', 'overall']]
df.columns = ['user', 'item', 'rating']

# 3. 重新映射 ID (0 到 N-1)
user_map = {u: i for i, u in enumerate(df['user'].unique())}
item_map = {i: j for j, i in enumerate(df['item'].unique())}

df['user_id'] = df['user'].map(user_map)
df['item_id'] = df['item'].map(item_map)

print(f"Stats -> Users: {len(user_map)}, Items: {len(item_map)}, Interactions: {len(df)}")

# 4. 区分正负反馈 (论文标准: >=4为正, <4为负)
df_pos = df[df['rating'] >= 4.0].copy()
df_neg = df[df['rating'] < 4.0].copy()

print(f"Positive Edges: {len(df_pos)}, Negative Edges: {len(df_neg)}")

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
    # 聚合操作
    grouped = data.groupby('user_id')['item_id'].apply(list)
    with open(path, 'w') as f:
        for u, items in grouped.items():
            # 将 item list 转为字符串，中间用空格隔开
            item_str = " ".join(map(str, items))
            f.write(f"{u} {item_str}\n")

print(f"Writing files to {out_dir}...")
write_txt(train_pos, f"{out_dir}/train_pos.txt")
write_txt(df_neg, f"{out_dir}/train_neg.txt")
write_txt(valid_pos, f"{out_dir}/valid.txt")
write_txt(test_pos, f"{out_dir}/test.txt")

print("All Done! Ready to run main.py.")