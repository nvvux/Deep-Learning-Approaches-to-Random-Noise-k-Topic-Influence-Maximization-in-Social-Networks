# ======= load_data.py =======

import numpy as np
import pandas as pd
import ast
import os
import pickle

def encode_seed_multitopic(seed_str, n, k):
    seed_sets = ast.literal_eval(seed_str)
    arr = np.zeros((k, n), dtype=int)
    for topic_idx, topic_seeds in enumerate(seed_sets):
        for node in topic_seeds:
            arr[topic_idx, node] = 1
    return arr.flatten()

def load_graph_multitopic(graph_csv, label_csv):
    edge_df = pd.read_csv(graph_csv)
    weight_cols = [col for col in edge_df.columns if col.lower().startswith('weight')]
    weight_cols = sorted(weight_cols, key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
    n = max(edge_df['source'].max(), edge_df['target'].max()) + 1
    k = len(weight_cols)
    A = np.zeros((n, n, k))
    for _, row in edge_df.iterrows():
        src = int(row['source'])
        tgt = int(row['target'])
        for i, w_col in enumerate(weight_cols):
            A[tgt, src, i] = row[w_col]
    df = pd.read_csv(label_csv)
    if 'time' in df.columns:
        df = df.drop(columns=['time'])
    inf_cols = [col for col in df.columns if col.strip().upper().startswith('INF')]
    seed_col = 'seed'
    X = np.stack([encode_seed_multitopic(s, n, k) for s in df[seed_col]])
    y = df[inf_cols].values
    return A, X, y, inf_cols

def process_all_graphs(data_dir, save_dir, num_graphs=30):
    os.makedirs(save_dir, exist_ok=True)
    label_cols = None
    for idx in range(1, num_graphs+1):
        graph_csv = os.path.join(data_dir, f"graph{idx}.csv")
        label_csv = os.path.join(data_dir, f"graph{idx}_labels.csv")
        if not (os.path.exists(graph_csv) and os.path.exists(label_csv)):
            print(f"[WARN] Không tìm thấy {graph_csv} hoặc {label_csv}. Bỏ qua.")
            continue
        print(f"\n=== Loading: graph{idx} ===")
        A, X, y, inf_cols = load_graph_multitopic(graph_csv, label_csv)
        print(f"  A.shape = {A.shape}, X.shape = {X.shape}, y.shape = {y.shape}, label_cols = {inf_cols}")
        np.save(os.path.join(save_dir, f"A_graph{idx}.npy"), A)
        np.save(os.path.join(save_dir, f"X_graph{idx}.npy"), X)
        np.save(os.path.join(save_dir, f"y_graph{idx}.npy"), y)
        if label_cols is None:
            label_cols = inf_cols
    with open(os.path.join(save_dir, "label_cols.pkl"), "wb") as f:
        pickle.dump(label_cols, f)
    print(f"\n===> Đã lưu toàn bộ dữ liệu vào: {save_dir}")

def load_all_processed_graphs(save_dir, num_graphs=30):
    """
    Load toàn bộ file .npy đã lưu từ process_all_graphs.
    Pad tất cả các graph về cùng kích thước maxN.
    """
    A_list = []
    X_all, y_all, graph_idx = [], [], []
    maxN = 0
    k = None
    shapes = []
    # 1. Đầu tiên, xác định maxN, k
    for idx in range(1, num_graphs+1):
        A_path = os.path.join(save_dir, f"A_graph{idx}.npy")
        if not os.path.exists(A_path):
            continue
        A = np.load(A_path)
        shapes.append(A.shape)
        if A.shape[0] > maxN:
            maxN = A.shape[0]
        if k is None:
            k = A.shape[2]
    # 2. Load và pad từng graph
    for gidx, idx in enumerate(range(1, num_graphs+1)):
        A_path = os.path.join(save_dir, f"A_graph{idx}.npy")
        X_path = os.path.join(save_dir, f"X_graph{idx}.npy")
        y_path = os.path.join(save_dir, f"y_graph{idx}.npy")
        if not (os.path.exists(A_path) and os.path.exists(X_path) and os.path.exists(y_path)):
            continue
        A = np.load(A_path)
        X = np.load(X_path)
        y = np.load(y_path)
        n = A.shape[0]
        # Pad A về (maxN, maxN, k)
        Apad = np.zeros((maxN, maxN, k), dtype=A.dtype)
        Apad[:n, :n, :] = A
        # Pad X về (num_sample, maxN*k)
        Xpad = np.zeros((X.shape[0], maxN*k), dtype=X.dtype)
        for topic in range(k):
            Xpad[:, topic*maxN:topic*maxN+n] = X[:, topic*n:(topic+1)*n]
        # Append
        A_list.append(Apad)
        X_all.append(Xpad)
        y_all.append(y)
        graph_idx.extend([gidx]*X.shape[0])
    X_all = np.vstack(X_all)
    y_all = np.vstack(y_all)
    graph_idx = np.array(graph_idx)
    # Load tên label
    with open(os.path.join(save_dir, "label_cols.pkl"), "rb") as f:
        label_cols = pickle.load(f)
    print(f"Đã load {len(A_list)} graphs, tổng {X_all.shape[0]} sample, maxN={maxN}, k={k}")
    return A_list, X_all, y_all, graph_idx, label_cols

if __name__ == "__main__":
    # Chạy 1 lần để xử lý toàn bộ file .csv sang .npy
    data_dir = "./graphs_data"       # Thư mục chứa graph*.csv, graph*_labels.csv
    save_dir = "./processed_data"    # Thư mục lưu file .npy, .pkl
    process_all_graphs(data_dir, save_dir, num_graphs=30)
