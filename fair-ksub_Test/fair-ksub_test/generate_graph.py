# generate_graph.py

import networkx as nx
import pandas as pd
import numpy as np
import argparse

def generate_er_graph_to_csv(n_nodes, p, n_topics, output_file, seed=None):
    """
    Sinh đồ thị Erdős–Rényi và ghi ra file CSV.

    Args:
        n_nodes (int): Số node.
        p (float): Xác suất kết nối cạnh.
        n_topics (int): Số chủ đề => số cột weight.
        output_file (str): Tên file CSV.
        seed (int or None): Seed ngẫu nhiên (nếu cần tái lập).
    """

    G = nx.erdos_renyi_graph(n=n_nodes, p=p, seed=seed, directed=True)

    edge_list = []
    for u, v in G.edges():
        edge = {'source': u, 'target': v}
        for i in range(1, n_topics + 1):
            edge[f'weight{i}'] = round(np.random.uniform(0.1, 0.9), 3)
        edge_list.append(edge)

    df = pd.DataFrame(edge_list)
    df.to_csv(output_file, index=False)
    print(f"✅ Đã sinh đồ thị ER với {n_nodes} node, {G.number_of_edges()} cạnh, lưu vào '{output_file}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sinh đồ thị Erdős–Rényi ngẫu nhiên và xuất ra CSV.")
    parser.add_argument("--n_nodes", type=int, default=20, help="Số lượng node (mặc định: 20)")
    parser.add_argument("--p", type=float, default=0.2, help="Xác suất tạo cạnh (mặc định: 0.2)")
    parser.add_argument("--n_topics", type=int, default=3, help="Số chủ đề (mặc định: 3)")
    parser.add_argument("--output_file", type=str, default="graph.csv", help="Tên file đầu ra (mặc định: graph.csv)")
    parser.add_argument("--seed", type=int, default=None, help="Seed ngẫu nhiên (tùy chọn)")

    args = parser.parse_args()

    generate_er_graph_to_csv(
        n_nodes=args.n_nodes,
        p=args.p,
        n_topics=args.n_topics,
        output_file=args.output_file,
        seed=args.seed
    )
