import time
import pandas as pd
import numpy as np


def IC(G, x, mc=1000):
    weight_cols = [col for col in G.columns if col.startswith("weight")]
    assert len(weight_cols) == len(x), "Number of topics (weight columns) must be euqual to number of seed sets."
    spread = []
    for i in range(mc):
        active_nodes=[]
        for topic_idx, (S, weight_col) in enumerate(zip(x, weight_cols)):          
            new_active, A = S[:], S[:]
            while new_active:
                temp = G.loc[G['source'].isin(new_active)]
                targets = temp['target'].tolist()
                ic = temp[weight_col].tolist()
                #thêm nhiễu gausei    

                #chuẩn hóa sau khi thêm nhiễu


                coins = np.random.uniform(0, 1, len(targets))
                choice = [ic[c] > coins[c] for c in range(len(coins))]
                new_ones = np.extract(choice, targets)
                new_active = list(set(new_ones) - set(A))
                A += new_active
            active_nodes.append(set(A))
        total_active = set().union(*active_nodes)
        spread.append(len(total_active))
    return (np.mean(spread))