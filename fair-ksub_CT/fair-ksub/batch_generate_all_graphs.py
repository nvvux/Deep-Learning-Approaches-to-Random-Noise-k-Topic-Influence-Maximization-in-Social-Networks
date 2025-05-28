import os
import random
import pandas as pd
from generate_graph import generate_er_graph_to_csv
from generate_labels import generate_training_labels_csv

def batch_generate_all_graphs_and_labels():
    output_dir = "graphs_data"
    os.makedirs(output_dir, exist_ok=True)
    all_rows = []

    for i in range(1, 2):
        # Sinh ngẫu nhiên số node cho mỗi graph
        n_nodes = random.randint(200, 250)
        graph_name = f"graph{i}"
        graph_file = os.path.join(output_dir, f"{graph_name}.csv")
        label_file = os.path.join(output_dir, f"{graph_name}_labels.csv")

        # Sinh đồ thị Erdos-Renyi
        generate_er_graph_to_csv(
            n_nodes=n_nodes,
            p=0.02,
            n_topics=2,      # Số topic cố định là 2, hoặc cho vào biến nếu cần
            output_file=graph_file,
            seed=i
        )

        # Xác định số topic để sinh đúng sigmas
        K = 2  # hoặc tự động lấy từ file csv sau khi sinh graph
        sigmas = [0.05] * K

        # Sinh labels (greedy/lazy/random)
        generate_training_labels_csv(
            filename=graph_file,
            k=2,
            output_csv=label_file,
            greedy_n=10,      # Số mẫu greedy
            random_n=30,      # Số mẫu random
            mc=1000,           # Để 100 khi test, tăng lên khi nghiệm cuối
            sigmas=sigmas,
            graph_name=graph_name
        )

        # Đọc kết quả và gộp lại
        df = pd.read_csv(label_file)
        all_rows.append(df)

    # Gộp tất cả label lại một file tổng
    final_df = pd.concat(all_rows, ignore_index=True)
    final_csv = os.path.join(output_dir, "all_train_labels.csv")
    final_df.to_csv(final_csv, index=False)
    print(f"\n📦 Đã lưu file label tổng: {final_csv}")

if __name__ == "__main__":
    batch_generate_all_graphs_and_labels()
