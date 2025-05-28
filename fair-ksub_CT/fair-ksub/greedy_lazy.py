import pandas as pd
import time
from tqdm import tqdm
import numpy as np
# IC tuần tự ở đây, hoặc import từ file khác với tên IC_serial
def IC_serial(G, seed_sets, sigmas, mc=100):
    weight_cols = [c for c in G.columns if c.startswith("weight")]
    assert len(weight_cols) == len(seed_sets) == len(sigmas)
    spread = []
    for _ in range(mc):
        activated_global = set()
        for topic_idx, (S_topic, w_col) in enumerate(zip(seed_sets, weight_cols)):
            sigma = sigmas[topic_idx]
            active = list(S_topic)
            activated = set(S_topic)
            while active:
                temp = G[G['source'].isin(active)]
                targets = temp['target'].to_numpy()
                probs = temp[w_col].to_numpy()
                gau_noise = np.clip(np.random.normal(0, sigma, len(probs)), -0.1, 0.1)
                probs_noisy = np.clip(probs + gau_noise, 0, 1)
                coins = np.random.rand(len(targets))
                new_nodes = targets[coins < probs_noisy]
                new_nodes = [v for v in new_nodes if v not in activated]
                activated.update(new_nodes)
                active = new_nodes
            activated_global.update(activated)
        spread.append(len(activated_global))
    return float(np.mean(spread))

def greedy(filename, k, sigmas=None, mc=100, graph_name="graph250", print_table=True):
    G = pd.read_csv(filename)
    weight_cols = [col for col in G.columns if col.startswith("weight")]
    K = len(weight_cols)
    nodes = set(G['source']).union(set(G['target']))
    selected_nodes = set()

    if sigmas is None:
        sigmas = [0.0] * K

    x = [[] for _ in range(K)]
    current_spread = 0
    result_table = []

    for i in range(k):
        best_gain = -1
        best_node = None
        best_topic = None
        best_spread = None
        t0 = time.time()

        candidates = nodes - selected_nodes

        for node in tqdm(candidates, desc=f"Chọn node thứ {i+1}"):
            for topic in range(K):
                x_temp = [s[:] for s in x]
                x_temp[topic].append(node)
                tmp_spread = IC_serial(G, x_temp, sigmas=sigmas, mc=mc)
                gain = tmp_spread - current_spread
                if gain > best_gain:
                    best_gain = gain
                    best_node = node
                    best_topic = topic
                    best_spread = tmp_spread
        if best_node is None:
            break

        x[best_topic].append(best_node)
        selected_nodes.add(best_node)
        current_spread = best_spread
        elapsed = time.time() - t0

        result_table.append({
            "graph": graph_name,
            "seed": str([tuple(s) for s in x]),
            "INF": round(current_spread, 3),
            "time": round(elapsed, 2)
        })

    df_result = pd.DataFrame(result_table)
    if print_table:
        print(df_result)
    return x, current_spread, df_result

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tính spread cho các seed cố định hoặc chạy Greedy tối ưu")
    parser.add_argument("--input", type=str, default="graph250.csv",
                        help="Đường dẫn tới file CSV (source,target,weight1…weightK)")
    parser.add_argument("--k", type=int, default=2, help="Số node seed muốn chọn")
    parser.add_argument("--sigmas", type=float, nargs='*', default=None, help="List độ lệch chuẩn nhiễu cho mỗi topic (VD: --sigmas 0.05 0.1)")
    parser.add_argument("--mc", type=int, default=100, help="Số lần Monte Carlo mỗi lần đánh giá spread")
    args = parser.parse_args()

    seeds, spread, df_result = greedy(
        args.input, k=args.k, sigmas=args.sigmas, mc=args.mc
    )
    print("\nBảng kết quả từng vòng greedy:")
    print(df_result)
    df_result.to_csv("greedy_lazy_result.csv", index=False)
