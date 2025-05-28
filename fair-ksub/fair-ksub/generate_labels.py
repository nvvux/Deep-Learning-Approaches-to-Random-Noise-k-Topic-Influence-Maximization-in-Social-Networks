# generate_labels.py

import pandas as pd
import random
import time
from tqdm import tqdm
from lazy_greedy import greedy_celf
from ic_parallel_noisy import IC  # phải có bản đã chỉnh cho chạy song song & noise

def random_seed_sets(nodes, k, K):
    selected = random.sample(list(nodes), k)
    x = [[] for _ in range(K)]
    for i, node in enumerate(selected):
        x[i % K].append(node)
    return x

def format_seed_set(x):
    """Chuyển List[List[int]] -> chuỗi như '[{0,1},{3},{7,8}]' """
    return "[" + ",".join("{" + ",".join(map(str, s)) + "}" for s in x) + "]"

def generate_training_labels_csv(
    filename,
    k,
    output_csv="train_labels.csv",
    greedy_n=10,
    random_n=30,
    mc=1000,
    sigma=0.05,
    n_jobs=None,
    graph_name="g0"
):
    G = pd.read_csv(filename)
    weight_cols = [col for col in G.columns if col.startswith("weight")]
    K = len(weight_cols)
    all_nodes = set(G['source']).union(set(G['target']))

    rows = []

    print(f"\n🎯 Sinh {greedy_n} mẫu bằng Greedy:")
    for _ in tqdm(range(greedy_n), desc="Greedy"):
        x, _ = greedy_celf(filename, k)
        start = time.time()
        spread = IC(G, x, mc=mc, n_jobs=n_jobs, sigma=sigma)
        elapsed = time.time() - start
        rows.append({
            "graph": graph_name,
            "seed": format_seed_set(x),
            "INF": round(spread, 2),
            "time": round(elapsed, 6)
        })

    print(f"\n🎲 Sinh {random_n} mẫu bằng Random:")
    for _ in tqdm(range(random_n), desc="Random"):
        x = random_seed_sets(all_nodes, k, K)
        start = time.time()
        spread = IC(G, x, mc=mc, n_jobs=n_jobs, sigma=sigma)
        elapsed = time.time() - start
        rows.append({
            "graph": graph_name,
            "seed": format_seed_set(x),
            "INF": round(spread, 2),
            "time": round(elapsed, 6)
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Đã lưu {len(rows)} dòng vào '{output_csv}' đúng định dạng yêu cầu.")
