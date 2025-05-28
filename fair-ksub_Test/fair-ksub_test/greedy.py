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
