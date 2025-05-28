# generate_labels.py

import pandas as pd
import random
import time
from tqdm import tqdm

# Import hàm Lazy Greedy MP
from lazy_mp import greedy_lazy_mp
from ic_parallel_noisy import IC  # module IC đã hỗ trợ song song & noise

def random_seed_sets(nodes, k, K):
    selected = random.sample(list(nodes), k)
    x = [[] for _ in range(K)]
    for i, node in enumerate(selected):
        x[i % K].append(node)
    return x


def format_seed_set(x):
    """Chuyển List[List[int]] -> chuỗi như '[{0,1},{3},{7,8}]'"""
    return "[" + ",".join("{" + ",".join(map(str, s)) + "}" for s in x) + "]"


def generate_training_labels_csv(
    filename,
    k,
    output_csv="train_labels.csv",
    lazy_n=10,        # số mẫu Lazy Greedy (CELF)
    random_n=30,
    mc=1000,
    sigma=0.15,
    n_jobs=None,
    graph_name="g0"
):
    # Đọc graph từ file CSV
    G = pd.read_csv(filename)
    weight_cols = [col for col in G.columns if col.startswith("weight")]
    K = len(weight_cols)
    all_nodes = set(G['source']).union(set(G['target']))

    rows = []

    # ——— Sinh mẫu bằng Lazy Greedy MP (CELF + multiprocessing) ———
    print(f"\n🎯 Sinh {lazy_n} mẫu bằng Lazy Greedy (CELF + MP):")
    for _ in tqdm(range(lazy_n), desc="LazyGreedyMP"):
        start = time.time()
        # Gọi greedy_lazy_mp ngay trên DataFrame G
        seeds, spread = greedy_lazy_mp(G, k, mc=mc, n_workers=n_jobs)
        elapsed = time.time() - start

        rows.append({
            "graph": graph_name,
            "seed": format_seed_set(seeds),
            "INF": round(spread, 2),
            "time": round(elapsed, 6)
        })

    # ——— Sinh mẫu ngẫu nhiên để bổ trợ ———
    print(f"\n🎲 Sinh {random_n} mẫu bằng Random:")
    for _ in tqdm(range(random_n), desc="Random"):
        seeds = random_seed_sets(all_nodes, k, K)
        start = time.time()
        # Tính spread với module IC (phiên bản noisy & parallel)
        spread = IC(G, seeds, mc=mc, n_jobs=n_jobs, sigma=sigma)
        elapsed = time.time() - start

        rows.append({
            "graph": graph_name,
            "seed": format_seed_set(seeds),
            "INF": round(spread, 2),
            "time": round(elapsed, 6)
        })

    # Ghi kết quả ra CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Đã lưu {len(rows)} dòng vào '{output_csv}' đúng định dạng yêu cầu.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Test generate_training_labels_csv với Lazy Greedy MP"
    )
    parser.add_argument(
        "-f", "--filename", required=True, help="File CSV graph (source,target,weight*)"
    )
    parser.add_argument(
        "-k", "--k", type=int, required=True, help="Số seed tổng"
    )
    parser.add_argument(
        "-o", "--output", default="test_labels.csv", help="File CSV đầu ra"
    )
    parser.add_argument(
        "--lazy_n", type=int, default=1, help="Số mẫu CELF để test"
    )
    parser.add_argument(
        "--random_n", type=int, default=1, help="Số mẫu Random để test"
    )
    parser.add_argument(
        "--mc", type=int, default=100, help="Số MC cho IC"
    )
    parser.add_argument(
        "--sigma", type=float, default=0.05, help="Sigma cho noise"
    )
    parser.add_argument(
        "--n_jobs", type=int, default=None, help="Số process cho IC"
    )
    parser.add_argument(
        "--graph_name", default="test_graph", help="Tên graph"
    )
    args = parser.parse_args()

    generate_training_labels_csv(
        filename=args.filename,
        k=args.k,
        output_csv=args.output,
        lazy_n=args.lazy_n,
        random_n=args.random_n,
        mc=args.mc,
        sigma=args.sigma,
        n_jobs=args.n_jobs,
        graph_name=args.graph_name
    )
