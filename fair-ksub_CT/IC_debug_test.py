import pandas as pd
import numpy as np

def IC(G, x, sigmas=None, mc=100):
    weight_cols = [col for col in G.columns if col.startswith("weight")]
    assert len(weight_cols) == len(x)
    if sigmas is None:
        sigmas = [0.0] * len(weight_cols)
    spread = []
    for _ in range(mc):
        active_nodes_per_topic = []
        for topic_idx, (S, weight_col) in enumerate(zip(x, weight_cols)):
            sigma = sigmas[topic_idx]
            A = set(S)
            new_active = set(S)
            while new_active:
                temp = G[G['source'].isin(new_active)]
                targets = temp['target'].to_numpy()
                ic = temp[weight_col].to_numpy()
                gau_noise = np.clip(np.random.normal(0, sigma, len(ic)), -0.1, 0.1)
                ic_noisy = np.clip(ic + gau_noise, 0, 1)
                coins = np.random.uniform(0, 1, len(targets))
                choice = ic_noisy > coins
                new_ones = set(targets[choice])
                new_active = new_ones - A
                A |= new_active
            active_nodes_per_topic.append(A)
        total_active = set().union(*active_nodes_per_topic)
        spread.append(len(total_active))
    return np.mean(spread)

def main():
    G = pd.read_csv("fair-ksub/facebook_multi_topic.csv")
    seeds_topic1 = [0, 3437, 107, 414, 3980]
    seeds_topic2 = [1912, 698, 1684, 348, 686]
    x = [seeds_topic1, seeds_topic2]
    sigmas = [0.1, 0.1]
    result = IC(G, x, sigmas=sigmas, mc=100)
    print(f"Ảnh hưởng union cho 2 topic: {result:.2f}")

if __name__ == "__main__":
    main()
