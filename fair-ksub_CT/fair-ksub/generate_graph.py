import networkx as nx
import pandas as pd
import numpy as np
import argparse


def generate_er_graph_to_csv(n_nodes, p, n_topics, output_file, seed=None):
    # Bước 1: Tạo đồ thị ER
    G = nx.erdos_renyi_graph(n=n_nodes, p=p, seed=seed, directed=True)

    # Bước 2: Sinh trọng số ban đầu theo quy chuẩn trong [0.1, 0.9] (chưa chuẩn hóa tổng)
    edge_list = []
    for u, v in G.edges():
        edge = {'source': u, 'target': v}
        for i in range(1, n_topics + 1):
            edge[f'weight{i}'] = round(np.random.uniform(0.1, 0.9), 3)
        edge_list.append(edge)

    df = pd.DataFrame(edge_list)

    # 👉 Ghi file CSV ban đầu với trọng số ngẫu nhiên
    df.to_csv(output_file, index=False)
    print(f"📄 Đã tạo file ban đầu: {output_file} với trọng số chưa chuẩn hóa.")

    # Bước 3: Chuẩn hóa theo công thức:
    # p^{(i)}(w, v) = p'^{(i)}(w, v) / ∑_{x∈N(v)} p'^{(i)}(x, v)
    for i in range(1, n_topics + 1):
        weight_col = f'weight{i}'
        df[weight_col] = df.groupby('target')[weight_col].transform(lambda x: x / x.sum())

    # 👉 Ghi đè lại file CSV với dữ liệu đã chuẩn hóa
    df.to_csv(output_file, index=False)
    print(f"✅ Đã CHUẨN HÓA xác suất theo đích v (theo công thức) và ghi đè lại vào '{output_file}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sinh đồ thị Erdős–Rényi ngẫu nhiên và xuất ra CSV.")
    parser.add_argument("--n_nodes", type=int, default=20, help="Số lượng node (mặc định: 20)")
    parser.add_argument("--p", type=float, default=0.2, help="Xác suất tạo cạnh (mặc định: 0.2)")
    parser.add_argument("--n_topics", type=int, default=2, help="Số chủ đề (mặc định: 2)")
    parser.add_argument("--output_file", type=str, default="graph250.csv",
                        help="Tên file đầu ra (mặc định: graph.csv)")
    parser.add_argument("--seed", type=int, default=None, help="Seed ngẫu nhiên (tùy chọn)")

    args = parser.parse_args()

    generate_er_graph_to_csv(
        n_nodes=args.n_nodes,
        p=args.p,
        n_topics=args.n_topics,
        output_file=args.output_file,
        seed=args.seed
    )
