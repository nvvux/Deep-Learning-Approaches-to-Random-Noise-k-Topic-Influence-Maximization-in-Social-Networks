# IC basic and Greedy lazy
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import heapq
# from  IC_ss import IC
from IC_basic import IC

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

def lazy_greedy(filename, k, sigmas=None, mc=100, B=11, graph_name="graph250", print_table=True):
    G = pd.read_csv(filename)
    weight_cols = [col for col in G.columns if col.startswith("weight")]
    K = len(weight_cols)
    nodes = set(G['source']).union(set(G['target']))
    selected_nodes = set()

    if sigmas is None:
        sigmas = [0.0] * K

    x = [[] for _ in range(K)]  # seed sets for each topic
    current_spread = 0
    result_table = []

    # Init CELF queue: [( -gain, update_iter, node, topic, last_gain )]
    heap = []
    update_iter = 0
    for topic in range(K):
        for node in nodes:
            if node not in selected_nodes:
                temp_x = [s[:] for s in x]
                temp_x[topic].append(node)
                gain = IC(G, temp_x, sigmas, mc) - current_spread
                heapq.heappush(heap, (-gain, update_iter, node, topic, gain))

    total_elapsed = 0  # Thời gian tích lũy

    for it in range(B):
        t0 = time.time()
        found = False
        while heap:
            minus_gain, last_update, node, topic, last_gain = heapq.heappop(heap)
            if node in selected_nodes:
                continue
            if last_update == update_iter:
                x[topic].append(node)
                selected_nodes.add(node)
                current_spread += -minus_gain
                elapsed = round(time.time() - t0, 2)
                total_elapsed += elapsed
                # Dùng str([tuple(s) for s in x]) để seed giống mẫu bạn gửi
                seeds_str = str([tuple(s) for s in x])
                result_table.append({
                    "graph": graph_name,
                    "seed": seeds_str,
                    "INF": round(current_spread, 3),
                    "time": total_elapsed
                })
                print(f'{graph_name},"{seeds_str}",{round(current_spread, 3)},{total_elapsed}')
                found = True
                update_iter += 1
                break
            else:
                temp_x = [s[:] for s in x]
                temp_x[topic].append(node)
                new_gain = IC(G, temp_x, sigmas, mc) - current_spread
                heapq.heappush(heap, (-new_gain, update_iter, node, topic, new_gain))
        if not found:
            print("Không còn node ứng viên hợp lệ, dừng.")
            break

    df_result = pd.DataFrame(result_table)
    if print_table:
        print(df_result)
    return x, current_spread, df_result

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Lazy Greedy (CELF) cho Influence Maximization nhiều chủ đề")
    parser.add_argument("--input", type=str, default="graph250.csv",
                        help="Đường dẫn tới file CSV (source,target,weight1…weightK)")
    parser.add_argument("--k", type=int, default=2, help="Số topic")
    parser.add_argument("--sigmas", type=float, nargs='*', default=None, help="List độ lệch chuẩn nhiễu cho mỗi topic (VD: --sigmas 0.05 0.1)")
    parser.add_argument("--mc", type=int, default=1000, help="Số lần Monte Carlo mỗi lần đánh giá spread")
    parser.add_argument("--B", type=int, default=11, help="Tổng số node seed tối đa chọn")
    args = parser.parse_args()

    seeds, spread, df_result = lazy_greedy(
        args.input, k=args.k, sigmas=args.sigmas, mc=args.mc, B=args.B
    )

    print("\nBảng kết quả từng vòng CELF:")
    print(df_result)
    df_result.to_csv("greedy_lazyr2_result.csv", index=False)

"""
import pandas as pd
import numpy as np
import time
import heapq
from tqdm import tqdm

from IC_ss import IC  # Đổi lại import phù hợp với tên file IC song song của bạn

def lazy_greedy(
    filename,
    k,
    sigmas=None,
    mc=100,
    B=11,
    graph_name="graph250",
    print_table=True,
    n_jobs=None,
    random_state=None
):
    G = pd.read_csv(filename)
    weight_cols = [col for col in G.columns if col.startswith("weight")]
    K = len(weight_cols)
    nodes = set(G['source']).union(set(G['target']))
    selected_nodes = set()

    if sigmas is None:
        sigmas = [0.0] * K

    x = [[] for _ in range(K)]  # seed sets for each topic
    current_spread = 0
    result_table = []

    # Khởi tạo heap CELF: (-gain, update_iter, node, topic, last_gain)
    heap = []
    update_iter = 0
    print("Khởi tạo heap CELF...")
    for topic in range(K):
        for node in tqdm(nodes, desc=f"Init CELF - topic {topic}"):
            temp_x = [s[:] for s in x]
            temp_x[topic].append(node)
            gain = IC(G, temp_x, sigmas, mc=mc, n_jobs=n_jobs, random_state=random_state) - current_spread
            heapq.heappush(heap, (-gain, update_iter, node, topic, gain))

    total_elapsed = 0  # Tổng thời gian tích lũy

    for it in range(B):
        t0 = time.time()
        found = False
        while heap:
            minus_gain, last_update, node, topic, last_gain = heapq.heappop(heap)
            if node in selected_nodes:
                continue
            if last_update == update_iter:
                # Node này đã được tính với seed set mới nhất, chọn nó!
                x[topic].append(node)
                selected_nodes.add(node)
                current_spread += -minus_gain
                elapsed = round(time.time() - t0, 2)
                total_elapsed += elapsed
                seeds_str = "[" + ", ".join(str(tuple(seed)) for seed in x) + "]"
                result_table.append({
                    "graph": graph_name,
                    "seed": seeds_str,
                    "INF": round(current_spread, 3),
                    "time": total_elapsed
                })
                print(f'{graph_name},"{seeds_str}",{round(current_spread, 3)},{total_elapsed}')
                found = True
                update_iter += 1
                break
            else:
                # Tính lại gain cho node này với seed set mới nhất (IC song song)
                temp_x = [s[:] for s in x]
                temp_x[topic].append(node)
                new_gain = IC(G, temp_x, sigmas, mc=mc, n_jobs=n_jobs, random_state=random_state) - current_spread
                heapq.heappush(heap, (-new_gain, update_iter, node, topic, new_gain))
        if not found:
            print("Không còn node ứng viên phù hợp hoặc heap rỗng, dừng lại.")
            break

    df_result = pd.DataFrame(result_table)
    if print_table:
        print(df_result)
    return x, current_spread, df_result

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Lazy Greedy (CELF) + IC song song cho Influence Maximization đa chủ đề")
    parser.add_argument("--input", type=str, default="graph250.csv",
                        help="Đường dẫn tới file CSV (source,target,weight1…weightK)")
    parser.add_argument("--k", type=int, default=2, help="Số topic")
    parser.add_argument("--sigmas", type=float, nargs='*', default=None, help="List độ lệch chuẩn nhiễu cho mỗi topic (VD: --sigmas 0.05 0.1)")
    parser.add_argument("--mc", type=int, default=100, help="Số lần Monte Carlo mỗi lần đánh giá spread")
    parser.add_argument("--B", type=int, default=11, help="Tổng số node seed tối đa chọn")
    parser.add_argument("--n_jobs", type=int, default=None, help="Số core CPU")
    parser.add_argument("--random_state", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    seeds, spread, df_result = lazy_greedy(
        args.input,
        k=args.k,
        sigmas=args.sigmas,
        mc=args.mc,
        B=args.B,
        graph_name=args.input,
        print_table=True,
        n_jobs=args.n_jobs,
        random_state=args.random_state
    )

    print("\nBảng kết quả từng vòng CELF:")
    print(df_result)
    df_result.to_csv("greedy_lazy_ss_result.csv", index=False)
"""