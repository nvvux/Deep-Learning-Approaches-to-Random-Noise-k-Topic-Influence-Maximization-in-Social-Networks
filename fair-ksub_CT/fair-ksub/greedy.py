import pandas as pd
import time
from tqdm import tqdm
from IC_basic import IC  # Đảm bảo IC nhận đủ tham số như IC_serial

def greedy(filename, k, sigmas=None, mc=1000, graph_name="graph250", print_table=True):
    G = pd.read_csv(filename)
    weight_cols = [col for col in G.columns if col.startswith("weight")]
    K = len(weight_cols)
    nodes = set(G['source']).union(set(G['target']))
    selected_nodes = set()

    if sigmas is None:
        sigmas = [0.0] * K  # Không nhiễu nếu không truyền

    x = [[] for _ in range(K)]
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
        total_elapsed += elapsed

        # Lưu vào bảng kết quả, seed dạng tuple cho giống mẫu
        result_table.append({
            "graph": graph_name,
            "seed": str([tuple(s) for s in x]),
            "INF": round(current_spread, 3),
            "time": total_elapsed
        })

    df_result = pd.DataFrame(result_table)
    if print_table:
        print(df_result)
    return x, current_spread, df_result

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tính spread cho các seed cố định hoặc chạy Greedy tối ưu")
    parser.add_argument("--input", type=str,
                        default="graph250.csv",
                        help="Đường dẫn tới file CSV (source,target,weight1…weightK)")
    parser.add_argument("--k", type=int, default=2, help="Số node seed muốn chọn")
    parser.add_argument("--sigmas", type=float, nargs='*', default=None, help="List độ lệch chuẩn nhiễu cho mỗi topic (VD: --sigmas 0.05 0.1)")
    parser.add_argument("--mc", type=int, default=1000, help="Số lần Monte Carlo mỗi lần đánh giá spread")
    args = parser.parse_args()

    seeds, spread, df_result = greedy(
        args.input, k=args.k, sigmas=args.sigmas, mc=args.mc
    )
    print("\nBảng kết quả từng vòng greedy:")
    print(df_result)
    # Nếu muốn lưu file csv:
    df_result.to_csv("greedy_result.csv", index=False)
