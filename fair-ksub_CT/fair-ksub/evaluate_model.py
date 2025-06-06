import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Tránh lỗi OpenMP khi dùng matplotlib + torch

import torch
import numpy as np
import matplotlib.pyplot as plt
from load_data import load_all_processed_graphs
from train_glie import GNN_MultiTopic_Skip, prepare_batch  # <- dùng đúng model gốc đã huấn luyện

def evaluate_trained_model(model_path="best_multitopic_model.pt", num_graphs=30, k=2, feat_d=50):
    print("🔍 Đang tải mô hình và dữ liệu...")
    A_list, X_all, y_all, graph_idx, _ = load_all_processed_graphs("processed_data", num_graphs)
    print(f"✅ Đã load {num_graphs} graphs, tổng {len(graph_idx)} sample, k={k}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Khởi tạo đúng kiến trúc huấn luyện ban đầu
    model = GNN_MultiTopic_Skip(k, feat_d, h1 =64, h2=32, h3=16, dropout=0.5).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Chuẩn bị dữ liệu toàn bộ để đánh giá
    all_ids = np.arange(len(graph_idx))
    adj_list, feat_list, idx_tensor, y_tensor = prepare_batch(
        A_list, X_all, y_all, graph_idx, all_ids, k, feat_d, device
    )

    with torch.no_grad():
        pred = model(adj_list, feat_list, idx_tensor).cpu().numpy()
        y_true = y_tensor.view(-1, 1).cpu().numpy()

        # Tính toán chỉ số đánh giá
        mse = np.mean((pred - y_true) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(pred - y_true))

    print(f"\n📊 Kết quả đánh giá:")
    print(f"➡ MAE  = {mae:.4f}")
    print(f"➡ RMSE = {rmse:.4f}")
    print(f"➡ MSE  = {mse:.4f}")

    # === Vẽ biểu đồ so sánh dự đoán vs nhãn thật ===
    plt.figure(figsize=(10, 5))
    plt.plot(pred, label='Dự đoán', marker='o', linestyle='--', linewidth=1)
    plt.plot(y_true, label='Giá trị thật', marker='x', linestyle='-', linewidth=1)
    plt.xlabel('Chỉ số mẫu')
    plt.ylabel('Độ lan truyền ảnh hưởng')
    plt.title('So sánh Dự đoán và Thực tế')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    evaluate_trained_model(model_path="best_multitopic_model.pt", feat_d=50)
