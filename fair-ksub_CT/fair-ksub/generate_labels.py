""""
import pandas as pd
import random
import time
from tqdm import tqdm
from greedy_lazy import lazy_greedy   # Đã sửa greedy trả seed đúng format
from IC_ss import IC      # IC tuần tự/noise

def random_seed_sets(nodes, k, K):
    selected = random.sample(list(nodes), k)
    x = [[] for _ in range(K)]
    for i, node in enumerate(selected):
        x[i % K].append(node)
    return x

def format_seed_tuple(x):
    """"Chuyển List[List[int]] -> dạng [(6,), (15,)] (list of tuple)""""
    return str([tuple(s) for s in x])

def generate_training_labels_csv(
    filename,
    k,
    output_csv="train_labels.csv",
    greedy_n=1,
    random_n=3,
    mc=1000,
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
        x, _, _ = lazy_greedy(filename, k, sigmas=sigmas, mc=mc)
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
"""
"""
import pandas as pd
import random
import time
from tqdm import tqdm
from greedy_lazy import lazy_greedy    # Hoặc sửa import theo đúng tên file greedy_lazy bạn dùng
from IC_ss import IC                  # IC tuần tự/noise

def random_seed_sets(nodes, k, K):
    """"Sinh seed sets ngẫu nhiên, chia đều vào các chủ đề, không overlap.""""
    selected = random.sample(list(nodes), k)
    x = [[] for _ in range(K)]
    for i, node in enumerate(selected):
        x[i % K].append(node)
    return x

def format_seed_tuple(x):
    """"Chuyển List[List[int]] -> [(n1,), (n2, n3), ...] (list of tuple)""""
    return str([tuple(s) for s in x])

def generate_training_labels_csv(
    filename,
    k,
    output_csv="train_labels.csv",
    greedy_n=1,
    random_n=3,
    mc=1000,
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

    print(f"\n🎯 Sinh {greedy_n} mẫu bằng Greedy-Lazy (CELF):")
    for _ in tqdm(range(greedy_n), desc="Greedy-Lazy"):
        x, _, _ = lazy_greedy(filename, k, sigmas=sigmas, mc=mc)
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
    print(f"\n✅ Đã lưu {len(rows)} dòng vào '{output_csv}' đúng định dạng.")

# Example CLI usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sinh nhãn cho tối đa hóa ảnh hưởng (greedy-lazy/random)")
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
    """
import pandas as pd
import random
import time
from tqdm import tqdm
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

def generate_greedy_steps(filename, k, sigmas, mc, graph_name, max_steps=10):
    x, _, df_greedy = lazy_greedy(filename, k, sigmas=sigmas, mc=mc, graph_name=graph_name, print_table=False)
    rows = []
    for i in range(min(max_steps, len(df_greedy))):
        row = df_greedy.iloc[i]
        rows.append({
            "graph": row["graph"],
            "seed": row["seed"],
            "INF": row["INF"],
            "time": row["time"]
        })
    return pd.DataFrame(rows)

def generate_random_independent_seeds(
    G,
    K,
    mc=1000,
    sigmas=None,
    graph_name="graph250",
    n_repeat=3,
    max_nodes=10
):
    all_nodes = list(set(G['source']).union(set(G['target'])))
    if sigmas is None:
        sigmas = [0.05] * K

    rows = []
    for n_nodes in range(1, max_nodes + 1):
        for _ in range(n_repeat):
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
    return pd.DataFrame(rows)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ghép 10 dòng Greedy (từng bước) + 30 dòng Random (độc lập) thành 40 dòng tổng")
    parser.add_argument("--input", type=str, default="graph250.csv")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--mc", type=int, default=1000)
    parser.add_argument("--sigmas", type=float, nargs='*', default=None)
    parser.add_argument("--graph_name", type=str, default="graph250")
    parser.add_argument("--output_csv", type=str, default="train_labels_40.csv")
    parser.add_argument("--max_nodes", type=int, default=10)
    parser.add_argument("--n_repeat", type=int, default=3)
    args = parser.parse_args()

    tmp = pd.read_csv(args.input)
    weight_cols = [col for col in tmp.columns if col.startswith("weight")]
    K = len(weight_cols)
    sigmas = [0.05] * K if args.sigmas is None else args.sigmas

    # 1. Greedy từng bước (10 dòng)
    print("=== Sinh 10 dòng Greedy ===")
    df_greedy = generate_greedy_steps(
        args.input, args.k, sigmas, args.mc, args.graph_name, max_steps=args.max_nodes
    )

    # 2. Random hoàn toàn độc lập (3 lần x 10 mức số node)
    print("=== Sinh 30 dòng Random độc lập ===")
    G = pd.read_csv(args.input)
    df_random = generate_random_independent_seeds(
        G, K, mc=args.mc, sigmas=sigmas, graph_name=args.graph_name, n_repeat=args.n_repeat, max_nodes=args.max_nodes
    )

    # 3. Ghép lại thành 40 dòng
    df_all = pd.concat([df_greedy, df_random], ignore_index=True)
    df_all.to_csv(args.output_csv, index=False)
    print(f"\n✅ Đã lưu {len(df_all)} dòng vào {args.output_csv}")

