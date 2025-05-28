# greedy.py

import pandas as pd
from tqdm import tqdm
from diffuse import IC

def greedy(filename, k):
    G = pd.read_csv(filename)
    weight_cols = [col for col in G.columns if col.startswith("weight")]
    K = len(weight_cols)
    nodes = set(G['source']).union(set(G['target']))
    selected_nodes = set()

    x = [[] for _ in range(K)]
    current_spread = 0

    for i in range(k):
        best_gain = -1
        best_node = None
        best_topic = None
        best_spread = None

        candidates = nodes - selected_nodes

        for node in tqdm(candidates, desc=f"Chọn node thứ {i+1}"):
            for topic in range(K):
                x_temp = [s[:] for s in x]
                x_temp[topic].append(node)
                tmp_spread = IC(G, x_temp)
                gain = tmp_spread - current_spread
                if gain > best_gain:
                    best_gain = gain
                    best_node = node
                    best_topic = topic
                    best_spread = tmp_spread
        if(best_node is None):
            break
        x[best_topic].append(best_node)
        selected_nodes.add(best_node)
        current_spread = best_spread
    return x, current_spread
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tính spread cho các seed cố định")
    parser.add_argument("--input", type=str, default="graph.csv",
                        help="Đường dẫn tới file CSV (source,target,weight1…weightK)")
    args = parser.parse_args()

    # Load graph
    G = pd.read_csv(args.input)

    # Đặt seed sets theo yêu cầu
    #    Chủ đề 1: [17, 2]
    #    Chủ đề 2: [ 8,13]
    #    Chủ đề 3: [    4]
    seeds = [
        [ 2],
        [8, 13],
        [4]
    ]

    # Tính spread
    spread = IC(G, seeds)
    print(f"Spread cho seeds {seeds} là {spread:.2f}")
