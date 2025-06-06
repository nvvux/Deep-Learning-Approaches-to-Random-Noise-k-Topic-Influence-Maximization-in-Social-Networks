""""
import numpy as np
import pandas as pd
import os
import ast
# Đọc ma trận A đã lưu
A = np.load("processed_data/A_graph1.npy")
print("A.shape:", A.shape)  # (n, n, k)

# Đọc graph1.csv
df = pd.read_csv("graphs_data/graph1.csv")
print("Các cột csv:", df.columns.tolist())

# Chuẩn hóa tên cột weight (weight1, weight2,...)
weight_cols = [col for col in df.columns if col.lower().startswith('weight')]
weight_cols = sorted(weight_cols, key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
k = len(weight_cols)
print("Các cột weight:", weight_cols)

# Kiểm tra từng cạnh (edge)
not_match = []
for idx, row in df.iterrows():
    src = int(row['source'])
    tgt = int(row['target'])
    for t, w_col in enumerate(weight_cols):
        v_csv = row[w_col]
        v_A = A[tgt, src, t]
        if not np.isclose(v_A, v_csv, atol=1e-6):
            not_match.append((src, tgt, t, v_csv, v_A))

print(f"\nSố cạnh KHÔNG khớp: {len(not_match)}/{len(df)*k}")

# Nếu có sai lệch, in một số ví dụ
if not_match:
    print("\nMột vài cạnh không khớp:")
    for i in range(min(10, len(not_match))):
        src, tgt, t, v_csv, v_A = not_match[i]
        print(f"  Edge {src}->{tgt}, topic {t}: csv={v_csv}, A={v_A}")

else:
    print("✅ Ma trận A hoàn toàn khớp với file graph1.csv cho mọi topic!")

# Có thể kiểm tra ngẫu nhiên vài cạnh đúng:
print("\nKiểm tra 5 cạnh đầu tiên:")
for i, row in df.head(5).iterrows():
    src, tgt = int(row['source']), int(row['target'])
    for t, w_col in enumerate(weight_cols):
        print(f"  Edge {src}->{tgt}, topic {t}: csv={row[w_col]}, A={A[tgt,src,t]}")

"""
"""""
import ast


X = np.load("processed_data/X_graph1.npy")
df = pd.read_csv("graphs_data/graph1_labels.csv")
seed_col = 'seed'

# Số node và số topic
n = X.shape[1] // 2  # k = 2 (thay đổi nếu k != 2)
k = 2

def encode_seed(seed_str, n, k):
    seed_sets = ast.literal_eval(seed_str)
    arr = np.zeros((k, n), dtype=int)
    for topic_idx, topic_seeds in enumerate(seed_sets):
        for node in topic_seeds:
            arr[topic_idx, node] = 1
    return arr.flatten()

print(f"Số dòng X: {X.shape[0]}, Số dòng seed: {df.shape[0]}")
print("So sánh 3 dòng đầu:")
for i in range(3):
    print("-" * 60)
    print(f"Dòng {i}:")
    print("Seed gốc:", df[seed_col][i])
    print("Encode lại:", encode_seed(df[seed_col][i], n, k))
    print("Trong X:   ", X[i])
    is_equal = np.array_equal(X[i], encode_seed(df[seed_col][i], n, k))
    print("==> Giống nhau?" , is_equal)


# Đọc y từ file .npy
y = np.load('processed_data/y_graph1.npy')

# Đọc label gốc từ file csv
df = pd.read_csv('graphs_data/graph1_labels.csv')

# Xác định cột chứa label, ví dụ cột tên là INF (có thể là INF, INF_0, INF_1 ...)
inf_cols = [col for col in df.columns if col.strip().startswith('INF')]
y_true = df[inf_cols].values

# So sánh
all_equal = np.allclose(y, y_true)
print("==> Label y và INF trong file csv giống nhau?", all_equal)

# Nếu muốn kiểm tra từng dòng:
for i in range(len(y)):
    if not np.allclose(y[i], y_true[i]):
        print(f"Sai tại dòng {i}: y_npy={y[i]}, y_csv={y_true[i]}")
print("5 dòng đầu y (npy):\n", y[:5])
print("5 dòng đầu INF trong csv:\n", y_true[:5])
"""
"""
import numpy as np

A = np.load('processed_data/A_graph1.npy')
X = np.load('processed_data/X_graph1.npy')
y = np.load('processed_data/y_graph1.npy')

print("A_graph1.npy shape:", A.shape)
print("X_graph1.npy shape:", X.shape)
print("y_graph1.npy shape:", y.shape)
print("A_graph1.npy dtype:", A.dtype)
print("X_graph1.npy dtype:", X.dtype)
print("y_graph1.npy dtype:", y.dtype)

# Show vài phần tử đầu để kiểm tra nội dung
print("A[0:2,0:2,:]:", A[0:2,0:2,:])
print("X[0:5]:", X[0:5])
print("y:", y)
"""
import numpy as np

A = np.load('processed_data/A_graph1.npy')  # shape [N,N,k]
for t in range(A.shape[2]):
    row_sums = A[:,:,t].sum(axis=1)
    print(f"Topic {t}: min={row_sums.min()}, max={row_sums.max()}")
