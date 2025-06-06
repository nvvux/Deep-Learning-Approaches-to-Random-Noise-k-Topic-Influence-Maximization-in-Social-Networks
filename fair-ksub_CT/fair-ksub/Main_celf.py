import torch
import pandas as pd
import numpy as np
import scipy.sparse as sp
import time
from train_glie import GNN_MultiTopic_Skip, normalize, sparse_mx_to_torch_sparse_tensor
from IC_basic import IC
from celf_gile import celf_gile

def create_adj_list_from_df(df, N, device):
    """Tạo danh sách adjacency matrix đã chuẩn hóa cho từng topic (GPU)"""
    adj_list = []
    for weight_col in ['weight1', 'weight2']:
        rows = df['source'].values
        cols = df['target'].values
        weights = df[weight_col].values
        A = sp.coo_matrix((weights, (rows, cols)), shape=(N, N))
        A = normalize(A)
        A_torch = sparse_mx_to_torch_sparse_tensor(A, device)
        adj_list.append(A_torch)
    return adj_list

def check_device():
    print("\n==== KIỂM TRA GPU ====")
    print("Torch version:", torch.__version__)
    cuda_ok = torch.cuda.is_available()
    print("Torch CUDA available:", cuda_ok)
    if cuda_ok:
        print("CUDA device:", torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print("❌ KHÔNG tìm thấy GPU CUDA! Chỉ chạy trên CPU (sẽ rất chậm).")
        device = torch.device("cpu")
    print("=======================\n")
    return device

def main():
    device = check_device()
    print(f"⚙️ Đang sử dụng device: {device}")

    # ==== Load mô hình đã huấn luyện chỉ 1 lần ====
    model = GNN_MultiTopic_Skip(
        k=2, feat_d=50, h1=64, h2=32, h3=16, dropout=0.5
    ).to(device)
    model.load_state_dict(torch.load("best_multitopic_model.pt", map_location=device))
    model.eval()

    graph_files = [
        "graphs_data/graph1.csv",
        "graphs_data/graph2.csv",
        "facebook_multi_topic.csv",
    ]
    mae_list, pred_spreads, true_spreads = [], [], []
    gnn_times, ic_times = [], []

    for graph_path in graph_files:
        print(f"\n==== Đang chạy trên {graph_path} ====")
        df_graph = pd.read_csv(graph_path)
        assert {'source', 'target', 'weight1', 'weight2'}.issubset(df_graph.columns), "CSV thiếu cột bắt buộc"
        N = max(df_graph['source'].max(), df_graph['target'].max()) + 1

        # ==== Tạo danh sách adjacency matrices (GPU) ====
        adj_list = create_adj_list_from_df(df_graph, N, device)

        # ==== CELF-GILE: chọn seed + dự đoán spread (GNN) + thực nghiệm IC ====
        with torch.no_grad():
            t1 = time.time()
            S_topic, pred_spread = celf_gile(
                model, adj_list, 50, 2, device, seed_size=4, verbose=False
            )
            t2 = time.time()
            t3 = time.time()
            true_spread = IC(
                df_graph, S_topic, sigmas=[0.1, 0.1], mc=1000
            )
            t4 = time.time()
        gnn_time = t2 - t1
        ic_time = t4 - t3

        print("🎯 Seed Topic 0:", S_topic[0])
        print("🎯 Seed Topic 1:", S_topic[1])
        print("🔮 GILE Prediction:", pred_spread)
        print("🧪 IC Spread:", true_spread)
        print(f"⏱️ GNN Time: {gnn_time:.4f} s | ⏱️ IC Time: {ic_time:.4f} s")
        mae = abs(pred_spread - true_spread)
        print(f"📉 MAE (GILE vs IC): {mae:.4f}")

        mae_list.append(mae)
        pred_spreads.append(pred_spread)
        true_spreads.append(true_spread)
        gnn_times.append(gnn_time)
        ic_times.append(ic_time)

    # ==== Tổng hợp kết quả ====
    mae_arr = np.array(mae_list)
    mae_norm = (mae_arr - mae_arr.min()) / (mae_arr.max() - mae_arr.min() + 1e-8)
    print("\n===== Tổng hợp Kết quả CHUẨN NHƯ PAPER =====")
    for i, graph_path in enumerate(graph_files):
        mae = mae_list[i]
        spread = true_spreads[i]
        mae_norm2 = mae / spread if spread != 0 else 0
        print(
            f"{graph_path}: MAE = {mae:.4f} | MAE/Spread = {mae_norm2:.4f} | "
            f"GILE = {pred_spreads[i]:.2f} | IC = {spread:.2f} | "
            f"GNN Time = {gnn_times[i]:.4f}s | IC Time = {ic_times[i]:.4f}s"
        )

if __name__ == '__main__':
    main()
