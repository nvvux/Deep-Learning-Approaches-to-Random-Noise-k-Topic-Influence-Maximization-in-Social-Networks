import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

# Biến toàn cục sẽ được khởi tạo 1 lần trong mỗi worker
_G = None
_weight_cols = []
_sigmas = []

def _init_pool(G: pd.DataFrame, weight_cols, sigmas):
    global _G, _weight_cols, _sigmas
    _G = G
    _weight_cols = weight_cols
    _sigmas = sigmas

def _simulate_once(args):
    """args = (seed_sets, rng_seed)"""
    seed_sets, rng_seed = args
    rng = np.random.RandomState(rng_seed)

    activated_global = set()

    for topic_idx, (S_topic, w_col) in enumerate(zip(seed_sets, _weight_cols)):
        sigma = _sigmas[topic_idx]
        active = list(S_topic)
        activated = set(S_topic)

        while active:
            temp = _G[_G['source'].isin(active)]
            targets = temp['target'].to_numpy()
            probs   = temp[w_col].to_numpy()

            # Sinh nhiễu Gaussian riêng cho từng topic
            gau_noise = np.clip(rng.normal(0, sigma, len(probs)), -0.1, 0.1)
            probs_noisy = np.clip(probs + gau_noise, 0, 1)

            coins = rng.rand(len(targets))
            new_nodes = targets[coins < probs_noisy]

            # Loại trùng
            new_nodes = [v for v in new_nodes if v not in activated]
            activated.update(new_nodes)
            active = new_nodes

        activated_global.update(activated)

    return len(activated_global)

def IC(
        G: pd.DataFrame,
        seed_sets: list[list[int]],
        sigmas: list[float],
        mc: int = 1000,
        n_jobs: int | None = None,
        random_state: int | None = None
):
    """
    G         : DataFrame (source, target, weight0, weight1, …)
    seed_sets : list[K] list[int]
    sigmas    : list[K] float – mỗi topic một giá trị sigma (nhiễu Gaussian)
    mc        : số lần Monte-Carlo
    n_jobs    : số CPU (mặc định = tất cả)
    """
    weight_cols = [c for c in G.columns if c.startswith("weight")]
    assert len(weight_cols) == len(seed_sets) == len(sigmas), \
        "Số cột weight, số seed set và số sigma phải khớp với số topic."

    n_jobs = n_jobs or cpu_count()
    rng = np.random.RandomState(random_state)
    MAX_SEED = np.iinfo(np.int32).max  # 2_147_483_647
    seeds_for_runs = rng.randint(0, MAX_SEED, size=mc)

    arg_iter = [(seed_sets, s) for s in seeds_for_runs]

    with Pool(
        processes=n_jobs,
        initializer=_init_pool,
        initargs=(G, weight_cols, sigmas)
    ) as pool:
        results = pool.map(_simulate_once, arg_iter, chunksize=max(1, mc // (10 * n_jobs)))

    return float(np.mean(results))
