import os
import random
import pandas as pd
import time
from generate_graph import generate_er_graph_to_csv

# == Copy lại 2 hàm này vào file mới (hoặc import từ file bạn đã lưu sẵn) ==
from greedy_lazy import lazy_greedy
from IC_ss import IC

def format_seed_tuple(x):
    return str([tuple(s) for s in x])

def random_seed_sets(all_nodes, n_nodes, K):
    selected = random.sample(all_nodes, n_nodes)
    x = [[] for _ in range(K)]
    for i, node in enumerate(selected):
        x[i % K].append(node)
    return x

def generate_40_labels(
    graph_file, k, output_csv="labels.csv", mc=1000, sigmas=None, graph_name="graph"
):
    # Đọc graph, chuẩn bị thông tin
    G = pd.read_csv(graph_file)
    weight_cols = [col for col in G.columns if col.startswith("weight")]
    K = len(weight_cols)
    if sigmas is None:
        sigmas = [0.05] * K
    all_nodes = list(set(G['source']).union(set(G['target'])))

    rows = []

    # 10 dòng Greedy từng bước
    x, _, df_greedy = lazy_greedy(graph_file, k, sigmas=sigmas, mc=mc, graph_name=graph_name, print_table=False)
    for i in range(min(10, len(df_greedy))):
        row = df_greedy.iloc[i]
        rows.append({
            "graph": row["graph"],
            "seed": row["seed"],
            "INF": row["INF"],
            "time": row["time"]
        })

    # 30 dòng Random độc lập từng mức (mỗi mức số node từ 1-10, mỗi mức 3 dòng)
    for n_nodes in range(1, 11):
        for _ in range(3):
            seed_sets = random_seed_sets(all_nodes, n_nodes, K)
            start = time.time()
            spread = IC(G, seed_sets, sigmas=sigmas, mc=mc)
            elapsed = time.time() - start
            rows.append({
                "graph": graph_name,
                "seed": format_seed_tuple(seed_sets),
                "INF": round(spread, 2),
                "time": round(elapsed, 6)
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved {len(df)} labels to {output_csv}")

# == Hàm batch sinh graph và label tổng hợp ==
def batch_generate_all_graphs_and_labels():
    output_dir = "graphs_data"
    os.makedirs(output_dir, exist_ok=True)
    all_rows = []

    for i in range(1, 2):  # Đổi range(1, N+1) nếu muốn nhiều graph
        n_nodes = random.randint(200, 250)
        graph_name = f"graph{i}"
        graph_file = os.path.join(output_dir, f"{graph_name}.csv")
        label_file = os.path.join(output_dir, f"{graph_name}_labels.csv")

        # Sinh đồ thị Erdos-Renyi
        generate_er_graph_to_csv(
            n_nodes=n_nodes,
            p=0.02,
            n_topics=2,
            output_file=graph_file,
            seed=i
        )

        # Sinh labels 40 dòng chuẩn (10 greedy + 30 random)
        generate_40_labels(
            graph_file=graph_file,
            k=2,
            output_csv=label_file,
            mc=1000,        # hoặc 100 cho test nhanh
            sigmas=[0.05, 0.05],
            graph_name=graph_name
        )

        # Đọc kết quả và gộp lại
        df = pd.read_csv(label_file)
        all_rows.append(df)

    # Gộp tất cả label lại một file tổng
    final_df = pd.concat(all_rows, ignore_index=True)
    final_csv = os.path.join(output_dir, "all_train_labels.csv")
    final_df.to_csv(final_csv, index=False)
    print(f"\n📦 Đã lưu file label tổng: {final_csv}")

if __name__ == "__main__":
    batch_generate_all_graphs_and_labels()
