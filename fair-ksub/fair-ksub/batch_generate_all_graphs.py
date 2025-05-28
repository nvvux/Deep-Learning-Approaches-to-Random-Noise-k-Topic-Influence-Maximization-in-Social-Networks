import os
import random
import pandas as pd
from generate_graph import generate_er_graph_to_csv
from generate_labels_glm import generate_training_labels_csv

def batch_generate_all_graphs_and_labels():
    output_dir = "graphs_data"
    os.makedirs(output_dir, exist_ok=True)
    all_rows = []

    for i in range(1, 4):
        # Generate Erdos-Renyi graphs with random node sizes
        n_nodes = random.randint(200, 250)
        graph_name = f"graph{i}"
        graph_file = os.path.join(output_dir, f"{graph_name}.csv")
        label_file = os.path.join(output_dir, f"{graph_name}_labels.csv")

        # Create graph and save to CSV
        generate_er_graph_to_csv(
            n_nodes=n_nodes,
            p=0.02,
            n_topics=2,
            output_file=graph_file,
            seed=i
        )

        # Generate labels using Lazy Greedy and Random
        generate_training_labels_csv(
            filename=graph_file,
            k=2,
            output_csv=label_file,
            #greedy_n =10,
            lazy_n=10,       # Use Lazy Greedy MP (CELF)
            random_n=30,
            mc=1000,
            sigma=0.05,
            n_jobs=None,     # Replace incorrect parameter
            graph_name=graph_name
        )

        # Read and aggregate results
        df = pd.read_csv(label_file)
        all_rows.append(df)

    # Combine all labels and save
    final_df = pd.concat(all_rows, ignore_index=True)
    final_csv = os.path.join(output_dir, "all_train_labels.csv")
    final_df.to_csv(final_csv, index=False)
    print(f"\n📦 Process complete. Combined label file saved: {final_csv}")

if __name__ == "__main__":
    batch_generate_all_graphs_and_labels()