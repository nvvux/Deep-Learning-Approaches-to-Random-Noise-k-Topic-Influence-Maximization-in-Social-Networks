import os
import pandas as pd
from generate_graph import generate_er_graph_to_csv
from lazy_mp import greedy_lazy_mp
import random
output_dir = "many_graphs"
os.makedirs(output_dir, exist_ok=True)
all_rows = []

n_graphs = 5
n_repeat = 5

for i in range(1, n_graphs + 1):
    n_nodes = random.randint(200, 250)
    graph_file = os.path.join(output_dir, f"graph{i}.csv")
    # Sinh đồ thị mới mỗi lần
    generate_er_graph_to_csv(n_nodes=n_nodes, p=0.02, n_topics=3, output_file=graph_file, seed=i)
    df = pd.read_csv(graph_file)
    for rep in range(1, n_repeat + 1):
        seeds, spread = greedy_lazy_mp(df, k=5, mc=1000)
        all_rows.append({
            "graph_id": i,
            "repeat": rep,
            "seed1": seeds[0],
            "seed2": seeds[1],
            "seed3": seeds[2],
            "spread": spread,
        })

final_df = pd.DataFrame(all_rows)
final_df.to_csv(os.path.join(output_dir, "train_labels.csv"), index=False)
