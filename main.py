# load_data.py

import networkx as nx
# main.py

import random
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Set, Tuple

def load_generated_data(file_path):
    """
    Đọc file 'output.txt' do generate_data.py sinh ra:
    - Dòng 1: n m
    - Dòng i>1: u v p1 p2 … p_k
    Trả về:
      n, m, k,
      edges: list of (u,v),
      probs: dict {(u,v): [p1,…,p_k]},
      G: networkx.DiGraph với edge attribute 'p' = [p1,…,p_k]
    """
    with open(file_path, 'r') as f:
        header = f.readline().strip().split()
        n, m = map(int, header)
        lines = f.readlines()

    edges = []
    probs = {}
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        u, v = map(int, parts[:2])
        p_list = list(map(float, parts[2:]))
        edges.append((u, v))
        probs[(u, v)] = p_list

    k = len(p_list) if edges else 0

    # Xây graph có hướng (vì dùng in-adj lúc sinh)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for (u, v), p in probs.items():
        G.add_edge(u, v, p=p)

    return n, m, k, edges, probs, G






# --- 1. Multi-Topic IC với Gaussian noise clamp in [-0.1,0.1] ---
class MultiTopicIC:
    def __init__(self,
                 G: nx.DiGraph,
                 k: int,
                 base_prob: Dict[int, Dict[Tuple[int,int], float]],
                 sigma: float = 0.05):
        self.G = G
        self.k = k
        self.base_prob = base_prob
        self.sigma = sigma

    def _noisy_prob(self, t: int, u: int, v: int) -> float:
        p0 = self.base_prob[t].get((u, v), 0.0)
        noise = random.gauss(0, self.sigma)
        # clamp noise vào [-0.1,0.1]
        noise = max(min(noise, 0.1), -0.1)
        p = p0 + noise
        # clamp p vào [0,1]
        return float(min(max(p, 0.0), 1.0))

    def simulate(self, seeds: List[Set[int]]) -> List[Set[int]]:
        activated = [set(S) for S in seeds]
        frontier  = [set(S) for S in seeds]
        global_active = {v: -1 for v in self.G.nodes()}
        for t, S in enumerate(seeds):
            for v in S:
                global_active[v] = t

        while any(frontier):
            new_f = [set() for _ in range(self.k)]
            for t in range(self.k):
                for u in frontier[t]:
                    for v in self.G.successors(u):
                        if global_active[v] == -1 and random.random() <= self._noisy_prob(t, u, v):
                            global_active[v] = t
                            new_f[t].add(v)
                            activated[t].add(v)
            frontier = new_f
        return activated

    def expected_spread(self, seeds: List[Set[int]], runs: int = 100) -> List[float]:
        cum = [0.0]*self.k
        for _ in range(runs):
            act = self.simulate(seeds)
            for t in range(self.k):
                cum[t] += len(act[t])
        return [x/runs for x in cum]


# --- 2. MLP cho dự đoán spread ---
class InfluenceMLP(nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dims: List[int] = [64, 32, 16]):
        super().__init__()
        layers = []
        prev = input_dim
        # 3 hidden layers: 64, 32, 16
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        # output layer
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

# --- 3. Feature extraction (4-dim) ---
def extract_feature(
    seed_set: Set[int],
    degree_centrality: Dict[int, float],
    base_prob_topic: Dict[Tuple[int,int], float],
    G: nx.DiGraph,
    topic: int
) -> torch.Tensor:
    # 1) |S|
    f1 = float(len(seed_set))
    # 2) sum degree_centrality
    f2 = sum(degree_centrality[v] for v in seed_set)
    # 3) sum base_prob cho tất cả (u->w), u in S
    sprob = 0.0
    for u in seed_set:
        for w in G.successors(u):
            sprob += base_prob_topic.get((u, w), 0.0)
    f3 = sprob
    # 4) topic id
    f4 = float(topic)
    return torch.tensor([f1, f2, f3, f4], dtype=torch.float32)


# --- 4. Greedy search giống trước ---
def search_optimal_seeds(
    model: InfluenceMLP,
    G: nx.DiGraph,
    degree_centrality: Dict[int,float],
    base_prob: Dict[int, Dict[Tuple[int,int],float]],
    k: int,
    budget: List[int]
) -> List[Set[int]]:
    chosen    = [set() for _ in range(k)]
    available = set(G.nodes())
    for t in range(k):
        for _ in range(budget[t]):
            best_v, best_gain = None, -1e9
            base_feat = extract_feature(chosen[t], degree_centrality, base_prob[t], G, t).unsqueeze(0)
            base_pred = model(base_feat).item()
            for v in available - chosen[t]:
                feat = extract_feature(chosen[t]|{v}, degree_centrality, base_prob[t], G, t).unsqueeze(0)
                gain = model(feat).item() - base_pred
                if gain > best_gain:
                    best_gain, best_v = gain, v
            if best_v is None:
                break
            chosen[t].add(best_v)
            available.remove(best_v)
    return chosen


# --- 5. Main flow ---
if __name__ == "__main__":


    # 5.1 Load dữ liệu bạn đã sinh ra
    generated_file = "output.txt"
    n, m, k, edges, loaded_probs, G = load_generated_data(generated_file)

    # 5.2 Chuẩn base_prob: Dict[topic]→dict[(u,v)]→p0
    base_prob: Dict[int, Dict[Tuple[int,int],float]] = {t:{} for t in range(k)}
    for (u,v), p_list in loaded_probs.items():
        for t in range(k):
            base_prob[t][(u,v)] = p_list[t]

    # 5.3 Khởi IC và centrality
    ic    = MultiTopicIC(G, k, base_prob, sigma=0.05)
    deg_c = nx.degree_centrality(G)

    # 5.4 Tạo samples, targets
    budget  = [5]*k
    samples = []
    targets = []
    for t in range(k):
        for _ in range(300):
            S = set(random.sample(list(G.nodes()), budget[t]))
            # chỉ tính spread cho topic t
            spreads = ic.expected_spread([S if i==t else set() for i in range(k)], runs=30)
            feat = extract_feature(S, deg_c, base_prob[t], G, t)
            samples.append(feat)
            targets.append(spreads[t])

    X = torch.stack(samples)           # [N,4]
    y = torch.tensor(targets)          # [N]
    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    # 5.5 Train MLP
    model = InfluenceMLP(input_dim=4, hidden_dims=[64, 32, 16])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn   = nn.MSELoss()
    for epoch in range(25):
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()*xb.size(0)
        print(f"Epoch {epoch+1:02d}  Loss = {total_loss/len(ds):.4f}")

    # 5.6 Tìm và đánh giá seed set tối ưu
    optimal = search_optimal_seeds(model, G, deg_c, base_prob, k, budget)
    print("Optimal seeds per topic:", optimal)
    final_spread = ic.expected_spread(optimal, runs=100)
    print("Final expected spread:", final_spread)
