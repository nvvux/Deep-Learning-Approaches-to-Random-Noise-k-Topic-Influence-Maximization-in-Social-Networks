import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.sparse as sp
from load_data import load_all_processed_graphs
from tqdm import tqdm

# ==== GNN MultiTopic Skip với 3 tầng ẩn, Eq. (19)-(21), skip H0 || H1 || H2 (thêm H3 nếu muốn) ====
class GNN_MultiTopic_Skip(nn.Module):
    def __init__(self, k, feat_d, h1, h2, h3, dropout):
        super().__init__()
        self.k = k
        self.feat_d = feat_d
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc1 = nn.ModuleList([nn.Linear(2 * feat_d, h1) for _ in range(k)])
        self.bn1 = nn.ModuleList([nn.BatchNorm1d(h1) for _ in range(k)])
        self.fc2 = nn.ModuleList([nn.Linear(2 * h1, h2) for _ in range(k)])
        self.bn2 = nn.ModuleList([nn.BatchNorm1d(h2) for _ in range(k)])
        self.fc3 = nn.ModuleList([nn.Linear(2 * h2, h3) for _ in range(k)])
        self.bn3 = nn.ModuleList([nn.BatchNorm1d(h3) for _ in range(k)])
        # Có thể chọn: out_layer có H3 hoặc không
        self.out_layer = nn.ModuleList([
            nn.Linear(feat_d + h1 + h2 + h3, 1) for _ in range(k)
        ])
    def forward(self, adj_list, feat_list, idx):
        topic_outputs = []
        for i in range(self.k):
            H0 = feat_list[i]
            H0_neigh = torch.mm(adj_list[i], H0)
            H1_input = torch.cat([H0, H0_neigh], dim=1)
            H1 = self.relu(self.bn1[i](self.fc1[i](H1_input)))
            H1 = self.dropout(H1)
            H1_neigh = torch.mm(adj_list[i], H1)
            H2_input = torch.cat([H1, H1_neigh], dim=1)
            H2 = self.relu(self.bn2[i](self.fc2[i](H2_input)))
            H2 = self.dropout(H2)
            H2_neigh = torch.mm(adj_list[i], H2)
            H3_input = torch.cat([H2, H2_neigh], dim=1)
            H3 = self.relu(self.bn3[i](self.fc3[i](H3_input)))
            H3 = self.dropout(H3)
            # === Skip connection (H0 || H1 || H2 || H3) ===
            H_all = torch.cat([H0, H1, H2, H3], dim=1)
            H_out = self.out_layer[i](H_all)   # [total_node, 1]
            # Tổng hóa từng graph trong batch bằng idx (số graph = batch_size)
            idx_exp = idx.unsqueeze(1).expand(-1, H_out.shape[1])
            num_graphs = int(torch.max(idx).item()) + 1
            H_agg = torch.zeros(num_graphs, H_out.shape[1], device=H_out.device)
            H_agg = H_agg.scatter_add_(0, idx_exp, H_out)
            topic_outputs.append(H_agg)   # [batch_size, 1]
        # Ghép topic lại, tổng hóa các branch
        H_concat = torch.cat(topic_outputs, dim=1)    # [batch_size, k]
        return torch.sum(H_concat, dim=1, keepdim=True)   # [batch_size, 1]

def normalize(mx):
    if not sp.issparse(mx):
        mx = sp.csr_matrix(mx)
    rowsum = np.array(mx.sum(1)).flatten()
    r_inv = np.zeros_like(rowsum)
    r_inv[rowsum != 0] = 1.0 / rowsum[rowsum != 0]
    r_mat_inv = sp.diags(r_inv)
    mx_norm = r_mat_inv.dot(mx)
    return mx_norm.tocsr()

def sparse_mx_to_torch_dense_tensor(sparse_mx, device):
    return torch.FloatTensor(sparse_mx.toarray()).to(device)

def prepare_batch(A_list, X_all, y_all, graph_idx, sample_ids, k, feat_d, device):
    """
    Tạo batch gồm nhiều graph. Với mỗi topic t:
      - ghép block_diag adjacency (của từng graph trong batch) thành 1 ma trận lớn.
      - ghép feature từng graph thành batch cho topic đó.
    """
    A_batch = [[] for _ in range(k)]
    X_batch = [[] for _ in range(k)]
    y_batch = []
    idx_batch = []
    node_offset = 0
    for b, i in enumerate(sample_ids):
        gi = graph_idx[i]
        N = A_list[gi].shape[0]
        for t in range(k):
            A_norm = normalize(A_list[gi][:, :, t])
            A_batch[t].append(A_norm)
            x = np.zeros((N, feat_d), dtype=np.float32)
            x_topic = X_all[i]
            if x_topic.ndim == 1:
                x_topic = x_topic.reshape((N, k))
            x[:, t] = x_topic[:, t]
            X_batch[t].append(x)
        y_batch.append(y_all[i])
        idx_batch.extend([b] * N)
        node_offset += N
    # Chuẩn: dùng vstack, chuyển thẳng sang tensor
    adj_list = [sparse_mx_to_torch_dense_tensor(sp.block_diag(A_batch[i]), device) for i in range(k)]
    feat_list = [torch.tensor(np.vstack(X_batch[i]), dtype=torch.float32, device=device) for i in range(k)]
    idx_tensor = torch.LongTensor(idx_batch).to(device)
    y_tensor = torch.FloatTensor(np.array(y_batch)).view(-1, 1).to(device)
    return adj_list, feat_list, idx_tensor, y_tensor

def mae(pred, target):
    return torch.mean(torch.abs(pred - target))

def minmax_scale(arr):
    arr = np.array(arr)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

def train_model(
    A_list, X_all, y_all, graph_idx,
    k=2, feat_d=50, h1=128, h2=64, h3=32, dropout=0.5,
    epochs=100, trials=3, batch_size=32
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_ids = np.arange(len(graph_idx))
    np.random.shuffle(all_ids)
    train_ids = all_ids[:int(0.8 * len(all_ids))]
    val_ids = all_ids[int(0.8 * len(all_ids)):]
    best_val = float('inf')
    best_state = None
    train_losses, val_losses = [], []
    train_maes, val_maes = [], []
    for trial in range(trials):
        print(f"\n====== TRIAL {trial + 1}/{trials} ======")
        model = GNN_MultiTopic_Skip(k, feat_d, h1, h2, h3, dropout).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        patience = 20
        wait = 0
        for epoch in range(epochs):
            model.train()
            np.random.shuffle(train_ids)
            for b in range(0, len(train_ids), batch_size):
                batch_ids = train_ids[b:b + batch_size]
                adj_list, feat_list, idx_tensor, y_tensor = prepare_batch(
                    A_list, X_all, y_all, graph_idx, batch_ids, k, feat_d, device)
                optimizer.zero_grad()
                output = model(adj_list, feat_list, idx_tensor)  # [batch_size, 1]
                loss = F.mse_loss(output, y_tensor)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                # Train MAE
                train_loss_list, train_mae_list = [], []
                for b in range(0, len(train_ids), batch_size):
                    batch_ids = train_ids[b:b + batch_size]
                    adj_list, feat_list, idx_tensor, y_tensor = prepare_batch(
                        A_list, X_all, y_all, graph_idx, batch_ids, k, feat_d, device)
                    train_output = model(adj_list, feat_list, idx_tensor)
                    train_loss_list.append(F.mse_loss(train_output, y_tensor).item())
                    train_mae_list.append(mae(train_output, y_tensor).item())
                train_loss = np.mean(train_loss_list)
                train_mae = np.mean(train_mae_list)
                # Valid MAE
                val_loss_list, val_mae_list = [], []
                for b in range(0, len(val_ids), batch_size):
                    batch_ids = val_ids[b:b + batch_size]
                    adj_list, feat_list, idx_tensor, y_tensor = prepare_batch(
                        A_list, X_all, y_all, graph_idx, batch_ids, k, feat_d, device)
                    val_output = model(adj_list, feat_list, idx_tensor)
                    val_loss_list.append(F.mse_loss(val_output, y_tensor).item())
                    val_mae_list.append(mae(val_output, y_tensor).item())
                val_loss = np.mean(val_loss_list)
                val_mae = np.mean(val_mae_list)
            train_losses.append(train_loss)
            train_maes.append(train_mae)
            val_losses.append(val_loss)
            val_maes.append(val_mae)
            train_losses_norm = minmax_scale(train_losses)
            val_losses_norm = minmax_scale(val_losses)
            train_maes_norm = minmax_scale(train_maes)
            val_maes_norm = minmax_scale(val_maes)
            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_loss:.4f} (Norm: {train_losses_norm[-1]:.4f}) | "
                f"Train MAE: {train_mae:.4f} (Norm: {train_maes_norm[-1]:.4f}) | "
                f"Val Loss: {val_loss:.4f} (Norm: {val_losses_norm[-1]:.4f}) | "
                f"Val MAE: {val_mae:.4f} (Norm: {val_maes_norm[-1]:.4f})"
            )
            scheduler.step(val_loss)
            if val_loss < best_val:
                best_val = val_loss
                best_state = model.state_dict()
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at epoch {epoch:03d}")
                    break
    torch.save(best_state, "best_multitopic_model.pt")
    print("\n✅ Best model saved with Val Loss = {:.4f}".format(best_val))

def train_model_from_processed():
    A_list, X_all, y_all, graph_idx, _ = load_all_processed_graphs("processed_data", num_graphs=30)
    print(f"Đã load {len(A_list)} graphs, tổng {len(graph_idx)} sample, maxN={max(A.shape[0] for A in A_list)}, k={A_list[0].shape[2]}")
    train_model(A_list, X_all, y_all, graph_idx, k=2, feat_d=50, h1=64, h2=32, h3=16, dropout=0.5, epochs=100, trials=3, batch_size=8)

if __name__ == '__main__':
    train_model_from_processed()
