# ic_parallel_noisy.py
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

_G = None
_weight_cols = []

def _init_pool(G: pd.DataFrame, weight_cols):
    global _G, _weight_cols
    _G = G
    _weight_cols = weight_cols


def _simulate_once(args):
    seed_sets, rng_seed, sigma = args
    rng = np.random.RandomState(rng_seed)

    activated_global = set()

    for topic_idx, (S_topic, w_col) in enumerate(zip(seed_sets, _weight_cols)):
        active = list(S_topic)
        activated = set(S_topic)

        while active:
            temp = _G[_G['source'].isin(active)]

            targets = temp['target'].to_numpy()
            probs = temp[w_col].to_numpy()

            # ✅ Thêm nhiễu Gaussian giới hạn [-0.1, 0.1]
            gau = np.clip(rng.normal(0, sigma, len(probs)), -0.1, 0.1)
            probs_noisy = np.clip(probs + gau, 0, 1)

            # Tung đồng xu
            coins = rng.rand(len(targets))
            new_nodes = targets[coins < probs_noisy]

            new_nodes = [v for v in new_nodes if v not in activated]
            activated.update(new_nodes)
            active = new_nodes

        activated_global.update(activated)

    return len(activated_global)


def IC(
        G: pd.DataFrame,
        seed_sets: list[list[int]],
        mc: int = 1000,
        n_jobs: int | None = None,
        sigma: float = 0.05,
        random_state: int | None = None
):
    weight_cols = [c for c in G.columns if c.startswith("weight")]
    assert len(weight_cols) == len(seed_sets), \
        "Số cột weight phải khớp số seed set (chủ đề)."

    n_jobs = n_jobs or cpu_count()
    rng = np.random.RandomState(random_state)
    max_int32 = np.iinfo(np.int32).max  # 2**31 - 1
    seeds_for_runs = rng.randint(0, max_int32, size=mc)

    # ✅ Truyền sigma vào từng lần gọi worker
    arg_iter = [(seed_sets, s, sigma) for s in seeds_for_runs]

    with Pool(
        processes=n_jobs,
        initializer=_init_pool,
        initargs=(G, weight_cols)
    ) as pool:
        results = pool.map(_simulate_once, arg_iter, chunksize=max(1, mc // (10 * n_jobs)))

    return float(np.mean(results))
