import heapq
import random
import time
from multiprocessing import Pool, cpu_count
from collections import defaultdict
import pandas as pd
from diffuse import IC  # Hoặc từ ic_parallel_noisy import IC nếu bạn dùng phiên bản mới

def build_adj_list_per_topic(df):
    weight_cols = [c for c in df.columns if c.startswith("weight")]
    adjs = [defaultdict(list) for _ in weight_cols]
    for _, row in df.iterrows():
        u = int(row["source"])
        v = int(row["target"])
        for t, col in enumerate(weight_cols):
            adjs[t][u].append((v, row[col]))
    return adjs

def simulate_ic_topic(adj_t, seeds, mc=1000):
    total = 0.0
    for _ in range(mc):
        activated = set(seeds)
        frontier = list(seeds)
        while frontier:
            new_frontier = []
            for u in frontier:
                for v, p in adj_t.get(u, []):
                    if v not in activated and random.random() <= p:
                        activated.add(v)
                        new_frontier.append(v)
            frontier = new_frontier
        total += len(activated)
    return total / mc

def compute_initial_gain(args):
    t, u, adj_t, mc = args
    gain = simulate_ic_topic(adj_t, [u], mc)
    return (t, u, gain)

def greedy_lazy_mp(df, k, mc=1000, n_workers=None):
    random.seed()  # <-- seed khác mỗi lần theo hệ thống
    if n_workers is None:
        n_workers = cpu_count()

    adjs = build_adj_list_per_topic(df)
    K = len(adjs)
    seeds = [[] for _ in range(K)]
    selected_global = set()

    tasks = [(t, u, adjs[t], mc) for t in range(K) for u in adjs[t].keys()]
    random.shuffle(tasks)  # <-- shuffle task để pool mỗi lần khởi tạo khác nhau
    with Pool(n_workers) as pool:
        init_results = pool.map(compute_initial_gain, tasks)

    heap = []
    for t, u, gain in init_results:
        jitter = random.uniform(0, 1e-3)  # <-- thêm jitter nhỏ để phá vỡ tie
        heapq.heappush(heap, (-(gain + jitter), u, t, 0))

    for i in range(1, k + 1):
        while True:
            neg_gain, u, t, last_iter = heapq.heappop(heap)
            if u in selected_global:
                continue
            gain = -neg_gain
            if last_iter < i:
                new_gain = simulate_ic_topic(adjs[t], seeds[t] + [u], mc)
                jitter = random.uniform(0, 1e-3)
                heapq.heappush(heap, (-(new_gain + jitter), u, t, i))
            else:
                seeds[t].append(u)
                selected_global.add(u)
                break

    total_spread = IC(df, seeds, mc)
    return seeds, total_spread

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Lazy Greedy + multiprocessing for Multi-Topic IC"
    )
    parser.add_argument("--input", type=str, default="graph.csv", help="CSV source,target,weight*")
    parser.add_argument("--k", type=int, default=5, help="Số seed cần chọn")
    parser.add_argument("--mc", type=int, default=1000, help="MC simulations per estimate")
    parser.add_argument("--workers", type=int, default=None, help="Số process for parallel")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    start = time.time()
    seeds, spread = greedy_lazy_mp(df, args.k, mc=args.mc, n_workers=args.workers)
    elapsed = time.time() - start

    print("🕒 Kết quả lazy greedy mp (CELF + MP):")
    for idx, s in enumerate(seeds, 1):
        print(f"  Chủ đề {idx}: {s}")
    print(f"  ➤ Spread (union) = {spread:.2f}")
    print(f"  ⏱ Time = {elapsed:.3f}s")
