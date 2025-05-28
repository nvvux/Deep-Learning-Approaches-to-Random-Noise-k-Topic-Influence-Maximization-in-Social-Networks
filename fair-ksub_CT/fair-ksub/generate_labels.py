import pandas as pd
import random
import time
from tqdm import tqdm
from greedy_lazy import greedy   # Đã sửa greedy trả seed đúng format
from IC_ss import IC      # IC tuần tự/noise

def random_seed_sets(nodes, k, K):
    selected = random.sample(list(nodes), k)
    x = [[] for _ in range(K)]
    for i, node in enumerate(selected):
        x[i % K].append(node)
    return x

def format_seed_tuple(x):
    """Chuyển List[List[int]] -> dạng [(6,), (15,)] (list of tuple)"""
    return str([tuple(s) for s in x])

def generate_training_labels_csv(
    filename,
    k,
    output_csv="train_labels.csv",
    greedy_n=10,
    random_n=30,
    mc=100,
    sigmas=None,
    graph_name="g0"
):
    G = pd.read_csv(filename)
    weight_cols = [col for col in G.columns if col.startswith("weight")]
    K = len(weight_cols)
    all_nodes = set(G['source']).union(set(G['target']))

    if sigmas is None:
        sigmas = [0.05] * K  # Default mỗi topic một sigma = 0.05

    rows = []

    print(f"\n🎯 Sinh {greedy_n} mẫu bằng Greedy:")
    for _ in tqdm(range(greedy_n), desc="Greedy"):
        x, _, _ = greedy(filename, k, sigmas=sigmas, mc=mc)
        start = time.time()
        spread = IC(G, x, sigmas=sigmas, mc=mc)
        elapsed = time.time() - start
        rows.append({
            "graph": graph_name,
            "seed": format_seed_tuple(x),
            "INF": round(spread, 2),
            "time": round(elapsed, 6)
        })

    print(f"\n🎲 Sinh {random_n} mẫu bằng Random:")
    for _ in tqdm(range(random_n), desc="Random"):
        x = random_seed_sets(all_nodes, k, K)
        start = time.time()
        spread = IC(G, x, sigmas=sigmas, mc=mc)
        elapsed = time.time() - start
        rows.append({
            "graph": graph_name,
            "seed": format_seed_tuple(x),
            "INF": round(spread, 2),
            "time": round(elapsed, 6)
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Đã lưu {len(rows)} dòng vào '{output_csv}' đúng định dạng yêu cầu.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sinh nhãn cho bài toán tối đa hóa ảnh hưởng (greedy/random)")
    parser.add_argument("--input", type=str, default="graph250.csv")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--greedy_n", type=int, default=10)
    parser.add_argument("--random_n", type=int, default=30)
    parser.add_argument("--mc", type=int, default=1000)
    parser.add_argument("--sigmas", type=float, nargs='*', default=None, help="List sigma cho mỗi topic (VD: --sigmas 0.05 0.1)")
    parser.add_argument("--graph_name", type=str, default="g0")
    parser.add_argument("--output_csv", type=str, default="train_labels.csv")
    args = parser.parse_args()

    # Tự động nhận số topic từ file input nếu chưa truyền sigmas
    tmp = pd.read_csv(args.input)
    K = len([c for c in tmp.columns if c.startswith("weight")])
    if args.sigmas is None:
        sigmas = [0.05] * K
    else:
        sigmas = args.sigmas

    generate_training_labels_csv(
        args.input,
        k=args.k,
        output_csv=args.output_csv,
        greedy_n=args.greedy_n,
        random_n=args.random_n,
        mc=args.mc,
        sigmas=sigmas,
        graph_name=args.graph_name
    )
