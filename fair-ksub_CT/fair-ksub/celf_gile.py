import numpy as np
import torch
import heapq

def gnn_spread_pred(model, adj_list, S, feat_d, k, device):
    """
    Dự đoán spread với GNN multitopic skip.
    - adj_list: list [k] adjacency [N,N] (torch tensor, đã normalize, trên device)
    - S: list các seed node cho mỗi topic [[node,...],[node,...]]
    """
    N = adj_list[0].shape[0]
    feat_list = []
    for t in range(k):
        x = torch.zeros(N, feat_d, device=device)
        if len(S[t]) > 0:
            x[S[t], t] = 1  # ĐÚNG cột topic t
        feat_list.append(x)
    idx = torch.zeros(N, dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        spread = model(adj_list, feat_list, idx)
    return spread.item()

def celf_gile(model, adj_list, feat_d, k, device, seed_size=6, verbose=False):
    """
    CELF-GILE chuẩn cho multitopic GNN: mỗi node chỉ thuộc 1 topic seed set, greedy lazy, không trùng lặp node.
    """
    N = adj_list[0].shape[0]
    S = [[] for _ in range(k)]
    chosen = set()  # tập hợp node đã chọn
    cur_spread = gnn_spread_pred(model, adj_list, S, feat_d, k, device)
    Q = []
    # Khởi tạo: tất cả node, tất cả topic
    for t in range(k):
        for u in range(N):
            if u in chosen:
                continue
            S_try = [list(seeds) for seeds in S]
            S_try[t].append(u)
            gain = gnn_spread_pred(model, adj_list, S_try, feat_d, k, device) - cur_spread
            heapq.heappush(Q, (-gain, t, u, 0))
    selected = 0
    while selected < seed_size:
        minus_gain, t, u, last_upd = heapq.heappop(Q)
        if u in chosen:
            continue
        if last_upd < selected:
            S_try = [list(seeds) for seeds in S]
            S_try[t].append(u)
            gain = gnn_spread_pred(model, adj_list, S_try, feat_d, k, device) - cur_spread
            heapq.heappush(Q, (-gain, t, u, selected))
            continue
        # Chọn node u cho topic t
        S[t].append(u)
        chosen.add(u)
        cur_spread += -minus_gain
        selected += 1
        if verbose:
            print(f"Thêm node {u} cho topic {t}, spread = {cur_spread:.3f} (total seed: {selected})")
    return S, cur_spread
