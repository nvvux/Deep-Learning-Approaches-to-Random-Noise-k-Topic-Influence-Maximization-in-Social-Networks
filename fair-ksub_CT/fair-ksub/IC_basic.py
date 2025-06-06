import time
import pandas as pd
import numpy as np


# Giả sử có một list sigmas (mỗi topic một giá trị), truyền thêm vào hàm IC
def IC(G, x, sigmas=None, mc=1000):
    weight_cols = [col for col in G.columns if col.startswith("weight")]
    assert len(weight_cols) == len(x), "Number of topics (weight columns) must be euqual to number of seed sets."
    if sigmas is None:
        sigmas = [0.0] * len(weight_cols)  # không nhiễu nếu không truyền
    spread = []
    for i in range(mc):
        active_nodes = []
        for topic_idx, (S, weight_col) in enumerate(zip(x, weight_cols)):
            sigma = sigmas[topic_idx]
            new_active, A = S[:], S[:]
            while new_active:
                temp = G.loc[G['source'].isin(new_active)]
                targets = temp['target'].tolist()
                ic = temp[weight_col].to_numpy()
                # Thêm nhiễu Gaussian [-0.1, 0.1] cho từng cạnh của topic hiện tại
                gau_noise = np.clip(np.random.normal(0, sigma, len(ic)), -0.1, 0.1)
                ic_noisy = np.clip(ic + gau_noise, 0, 1)
                coins = np.random.uniform(0, 1, len(targets))
                choice = ic_noisy > coins  # vector so sánh
                new_ones = np.array(targets)[choice]
                new_active = list(set(new_ones) - set(A))
                A += new_active
            active_nodes.append(set(A))
        total_active = set().union(*active_nodes)
        spread.append(len(total_active))
    return np.mean(spread)

def main():
    # Đọc file graph với 2 topic (giả sử file có weight1, weight2)
    G = pd.read_csv("fb_clone.csv")

    # Seed chọn tay hoặc random, không giao nhICau
    #(183, 86, 181, 178, 8), (7, 16, 124, 48, 68) inf 124
    # (86, 20, 136, 29, 21), (77, 127, 22, 124, 56)inf 87
    seeds_topic1 = [1137, 245]
    seeds_topic2 = [0, 1]
    x = [seeds_topic1, seeds_topic2]

    sigmas = [0.1, 0.1]  # hoặc tuỳ ý
    result = IC(G, x, sigmas=sigmas, mc=1000)
    print(f"Ảnh hưởng union cho 2 topic (file): {result:.2f}")

if __name__ == "__main__":
    main()
