import torch
import pandas as pd
import numpy as np
import scipy.sparse as sp
from train_glie import GNN_MultiTopic_Skip,normalize,sparse_mx_to_torch_sparse_tensor
from IC_basic import IC
from celf_gile import celf_gile


def create_adj_list_from_df(df, N):
    adj_list = []
    for weight_col in ['weight1', 'weight2']:
        rows = df['source'].values
        cols = df['target'].values
        weights = df[weight_col].values
        A = sp.coo_matrix((weights, (rows, cols)), shape=(N, N))
        A = normalize(A)
        A_torch = sparse_mx_to_torch_sparse_tensor(A).to("cuda")
        adj_list.append(A_torch)
    return adj_list


def main():
    # ==== Load graph ====
    df_graph = pd.read_csv("processed_data/A_graph1.npy")

    # ==== Load model đã huấn luyện ====
    model = GNN_MultiTopic_Skip(k=2, feat_d=50, h1=128, h2=64, h3=32, dropout=0.5).to("cuda")
    model.load_state_dict(torch.load("best_multitopic_model.pt"))
    model.eval()

    # ==== Tạo adjacency matrices ====
    N = max(df_graph['source'].max(), df_graph['target'].max()) + 1
    adj_list = create_adj_list_from_df(df_graph, N)

    # ==== Gọi hàm CELF-GILE ====
    S, pred_spread, true_spread, runtime = celf_gile(
        model=model,
        G=df_graph,
        adj_list=adj_list,
        feat_d=50,
        k=2,
        device=torch.device("cuda"),
        seed_size=100,
        eval_func=IC
    )

    print("✅ Top-100 Seed Nodes:", S)
    print("🔮 Dự đoán GILE:", pred_spread)
    print("🧪 Lan truyền IC thật:", true_spread)
    print("⏱️ Thời gian chạy (s):", runtime)


if __name__ == '__main__':
    main()