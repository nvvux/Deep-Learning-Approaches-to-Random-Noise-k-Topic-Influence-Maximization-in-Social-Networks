"""
import pandas as pd
import time
from tqdm import tqdm
import numpy as np
#from IC_ss import IC
from IC_basic import IC
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

    x = [[] for _ in range(K)] # seed sets cho từng topic
    current_spread = 0
    result_table = []
    B = 11
    total_elapsed = 0  # Thời gian tích lũy

    for i in range(B):
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
                tmp_spread = IC(G, x_temp, sigmas=sigmas, mc=mc)
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
        elapsed = round(time.time() - t0, 2)
        total_elapsed += elapsed  # Cộng dồn

        # Biểu diễn từng seed set của từng topic thành tuple cho dễ đọc
        seeds_str = "[" + ", ".join(str(tuple(seed)) for seed in x) + "]"

        result_table.append({
            "graph": graph_name,
            "seed": seeds_str,
            "INF": round(current_spread, 3),
            "time": total_elapsed
        })

        print(f'{graph_name},"{seeds_str}",{round(current_spread,3)},{total_elapsed}')

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

"""
import pandas as pd
import numpy as np
import time
import heapq
from tqdm import tqdm
from multiprocessing import Pool
from IC_ss import IC

def lazy_greedy(
    filename,
    k,
    sigmas=None,
    mc=1000,
    B=11,
    graph_name="graphs_data/graph1",
    print_table=True,
    n_jobs=None,
    random_state=None
):
    G = pd.read_csv(filename)
    weight_cols = [col for col in G.columns if col.startswith("weight")]
    K = len(weight_cols)
    assert K == k, "Số topic (k) phải đúng số cột weight trong dữ liệu!"
    nodes = set(G['source']).union(set(G['target']))
    selected_nodes = set()

    if sigmas is None:
        sigmas = [0.0] * K

    x = [[] for _ in range(K)]  # seed sets cho từng chủ đề
    current_spread = 0
    result_table = []
    n_jobs = n_jobs or 4  # Default số core nếu không truyền

    with Pool(processes=n_jobs) as pool:
        # Khởi tạo heap CELF
        heap = []
        update_iter = 0
        print("==> Khởi tạo heap CELF...")
        for topic in range(K):
            for node in tqdm(nodes, desc=f"Init CELF - topic {topic}"):
                temp_x = [s[:] for s in x]
                temp_x[topic].append(node)
                gain = IC(
                    G, temp_x, sigmas, mc=mc,
                    pool=pool, random_state=random_state
                ) - current_spread
                heapq.heappush(heap, (-gain, update_iter, node, topic, gain))

        total_elapsed = 0

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
                    seeds_str = str([tuple(s) for s in x])
                    result_table.append({
                        "graph": graph_name,
                        "seed": seeds_str,
                        "INF": round(current_spread, 3),
                        "time": total_elapsed
                    })
                    print(f'{graph_name},"{seeds_str}",{round(current_spread,3)},{total_elapsed}')
                    found = True
                    update_iter += 1
                    break
                else:
                    temp_x = [s[:] for s in x]
                    temp_x[topic].append(node)
                    new_gain = IC(
                        G, temp_x, sigmas, mc=mc,
                        pool=pool, random_state=random_state
                    ) - current_spread
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

    parser = argparse.ArgumentParser(description="Lazy Greedy (CELF) + IC song song có nhiễu, duy trì pool cho IM đa chủ đề")
    parser.add_argument("--input", type=str, default="graphs_data/graph1.csv",
                        help="Đường dẫn tới file CSV (source,target,weight1…weightK)")
    parser.add_argument("--k", type=int, default=2, help="Số topic (cột weight)")
    parser.add_argument("--sigmas", type=float, nargs='*', default=None, help="List sigma cho mỗi topic (VD: --sigmas 0.05 0.1)")
    parser.add_argument("--mc", type=int, default=1000, help="Số lần Monte Carlo mỗi lần đánh giá spread")
    parser.add_argument("--B", type=int, default=11, help="Tổng số node seed tối đa chọn")
    parser.add_argument("--n_jobs", type=int, default=4, help="Số core CPU")
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
    df_result.to_csv("greedy_lazy_noisy_ss_result.csv", index=False)
