import argparse
import os
import datetime as dt
import json
import math
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ==========================================
# 1. Model Architecture (GST-GNN & DynEC)
# ==========================================

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT Layer implementation (Single Head for clarity, can be extended to Multi-Head).
    Paper Ref: Eq. (350) - (359)
    """
    def __init__(self, in_features: int, out_features: int, dropout: float, alpha: float, concat: bool = True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h: torch.Tensor, adj: torch.Tensor):
        # h: (N, in_features)
        # adj: (N, N) - dense or sparse
        Wh = torch.mm(h, self.W)  # (N, out_features)
        
        # Attention Mechanism: e_ij = a^T [Wh_i || Wh_j]
        # Efficient broadcasting implementation
        a_input = self._prepare_attentional_mechanism_input(Wh) # (N, N, 2*out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2)) # (N, N)

        # Masked Attention (using adjacency matrix)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        h_prime = torch.matmul(attention, Wh) # (N, out_features)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]
        # Repeat Wh to create pairs (Wh_i, Wh_j)
        # (N, 1, out) -> (N, N, out)
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0).view(N, N, self.out_features)
        # (1, N, out) -> (N, N, out)
        Wh_repeated_alternating = Wh.repeat(N, 1).view(N, N, self.out_features)
        # (N, N, 2*out)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)
        return all_combinations_matrix


class GCNLayer(nn.Module):
    """
    Simple GCN Layer for ablation (w/o Gating/Attention).
    """
    def __init__(self, in_features: int, out_features: int, dropout: float):
        super(GCNLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
    def forward(self, h: torch.Tensor, adj: torch.Tensor):
        Wh = torch.mm(h, self.W)
        # Simple aggregation: A * Wh
        # Assuming adj is already normalized
        h_prime = torch.mm(adj, Wh)
        return F.elu(h_prime)

class GST_GNN_Encoder(nn.Module):
    """
    Gated Spatio-Temporal Graph Neural Network (GST-GNN).
    Paper Ref: Section 3.4
    Structure: GAT -> GRU
    """
    def __init__(self, nfeat: int, nhid: int, nout: int, dropout: float, alpha: float, use_gating: bool = True):
        super(GST_GNN_Encoder, self).__init__()
        self.use_gating = use_gating
        
        if self.use_gating:
            self.layer1 = GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
            self.layer2 = GraphAttentionLayer(nhid, nout, dropout=dropout, alpha=alpha, concat=True)
        else:
            self.layer1 = GCNLayer(nfeat, nhid, dropout=dropout)
            self.layer2 = GCNLayer(nhid, nout, dropout=dropout)
        
        # GRU for temporal update
        self.gru_cell = nn.GRUCell(nout, nout) 
        self.nout = nout

    def forward(self, x: torch.Tensor, adj: torch.Tensor, h_prev: Optional[torch.Tensor] = None):
        # Spatial Aggregation
        x_1 = self.layer1(x, adj)
        h_spat = self.layer2(x_1, adj)
        
        # Temporal Update (GRU)
        if h_prev is None:
            h_prev = torch.zeros_like(h_spat)
        
        h_curr = self.gru_cell(h_spat, h_prev)
        return h_curr

class DynEC(nn.Module):
    """
    DynEC Full Model.
    Paper Ref: Algorithm 2
    """
    def __init__(self, nfeat: int, nhid: int, nout: int, n_clusters: int, dropout: float = 0.2, alpha: float = 0.2, use_gating: bool = True):
        super(DynEC, self).__init__()
        self.encoder = GST_GNN_Encoder(nfeat, nhid, nout, dropout, alpha, use_gating=use_gating)
        
        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, nout))
        nn.init.xavier_uniform_(self.cluster_layer.data)
        self.v = 1.0

    def forward(self, x: torch.Tensor, adj: torch.Tensor, h_prev: Optional[torch.Tensor] = None):
        z = self.encoder(x, adj, h_prev)
        dist = torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2)
        q = 1.0 / (1.0 + dist / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return z, q


# ==========================================
# 2. Loss Functions & Helpers
# ==========================================

def target_distribution(q):
    """
    Compute target distribution P from soft assignment Q.
    Paper Ref: Eq. (395)
    p_ij = q_ij^2 / f_j / sum(q_ik^2 / f_k)
    """
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def kl_divergence(p, q):
    """
    KL Divergence Loss.
    """
    return torch.sum(p * torch.log(p / (q + 1e-12)))

# ==========================================
# 3. Data Processing & Graph Construction
# ==========================================

def construct_multiview_graph(X: np.ndarray, knn: int = 10, shift_band: int = 2, use_dtw: bool = True, use_mi: bool = True) -> torch.Tensor:
    """
    Construct Multi-View Graph (Geometric + Temporal Alignment + Dependency).
    Paper Ref: Section 3.3
    """
    # X: (N, D) - Normalized load profiles
    N = X.shape[0]
    X_tensor = torch.FloatTensor(X)
    
    # 1. Geometric View (Euclidean)
    # -----------------------------
    # Pairwise distance
    dist = torch.cdist(X_tensor, X_tensor, p=2) # (N, N)
    # kNN Graph
    # Get indices of k nearest neighbors
    _, indices = torch.topk(dist, k=knn, largest=False)
    A_geo = torch.zeros(N, N)
    rows = torch.arange(N).view(-1, 1).repeat(1, knn)
    A_geo[rows, indices] = 1.0
    
    views = [A_geo]
    
    if use_dtw:
        # 2. Temporal Alignment View (Approximate cDTW)
        # ---------------------------------------------
        sim_max = torch.zeros(N, N) - 1.0 # Init with -1
        
        # Standardize for correlation
        X_mean = X_tensor.mean(dim=1, keepdim=True)
        X_std = X_tensor.std(dim=1, keepdim=True) + 1e-8
        X_norm = (X_tensor - X_mean) / X_std
        
        for s in range(-shift_band, shift_band + 1):
            if s == 0:
                X_shift = X_norm
            else:
                X_shift = torch.roll(X_norm, shifts=s, dims=1)
                
            sim = torch.mm(X_norm, X_shift.t()) / X_norm.shape[1] # Correlation
            sim_max = torch.max(sim_max, sim)
            
        # kNN on Similarity
        _, indices = torch.topk(sim_max, k=knn, largest=True)
        A_dtw = torch.zeros(N, N)
        A_dtw[rows, indices] = 1.0
        views.append(A_dtw)
    
    if use_mi:
        # 3. Dependency View (Statistical)
        # --------------------------------
        X_mean = X_tensor.mean(dim=1, keepdim=True)
        X_std = X_tensor.std(dim=1, keepdim=True) + 1e-8
        X_norm = (X_tensor - X_mean) / X_std
        
        corr = torch.abs(torch.mm(X_norm, X_norm.t()) / X_norm.shape[1])
        _, indices = torch.topk(corr, k=knn, largest=True)
        A_mi = torch.zeros(N, N)
        A_mi[rows, indices] = 1.0
        views.append(A_mi)
    
    # Fusion
    A_fused = torch.zeros(N, N)
    for v in views:
        A_fused += v
    
    A_fused = A_fused / len(views)
    
    # Symmetrization and Normalization
    A_fused = (A_fused + A_fused.t()) / 2.0
    A_fused[A_fused > 0] = 1.0 
    
    # Add self-loop
    A_fused = A_fused + torch.eye(N)
    
    # Normalize: D^{-1/2} A D^{-1/2}
    D = A_fused.sum(1)
    D_inv_sqrt = torch.pow(D, -0.5)
    D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.
    D_mat_inv_sqrt = torch.diag(D_inv_sqrt)
    A_norm = torch.mm(torch.mm(D_mat_inv_sqrt, A_fused), D_mat_inv_sqrt)
    
    return A_norm

# ==========================================
# 4. Training & Evaluation Pipeline
# ==========================================

def train_dynec(data_root, city_name, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Data
    # Reuse existing CSV reading logic (simplified)
    profiles_path = os.path.join(data_root, city_name, "profiles_daily.csv")
    if not os.path.exists(profiles_path):
        print(f"Data not found: {profiles_path}")
        return

    df = pd.read_csv(profiles_path)
    df["date"] = pd.to_datetime(df["date"])
    dates = sorted(df["date"].unique())
    
    # Limit Days
    if args.max_days > 0:
        dates = dates[:args.max_days]
        df = df[df["date"].isin(dates)]
        
    # Get Consistent User Set (Intersection of all days)
    # Or just simplify: take users present in ALL selected days
    user_counts = df["user_id"].value_counts()
    valid_users = user_counts[user_counts == len(dates)].index
    
    if args.limit_users > 0 and len(valid_users) > args.limit_users:
        valid_users = valid_users[:args.limit_users]
        
    print(f"Selected {len(valid_users)} users for {len(dates)} days.")
    df = df[df["user_id"].isin(valid_users)]
    
    # Sort users once to ensure consistent order
    valid_users = sorted(valid_users)
    user_map = {u: i for i, u in enumerate(valid_users)}
    
    # Prepare Model
    # Input dim = 24 (hours), Hidden = 64/32, Output = 32
    model = DynEC(nfeat=24, nhid=64, nout=32, n_clusters=args.k, dropout=0.2, use_gating=not args.no_gating).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Initialize Centroids (KMeans on Embeddings)
    # We need to run the encoder once to get initial embeddings for KMeans
    print("Initializing centroids...")
    
    # Prepare data for t=0
    d0 = dates[0]
    daily_df0 = df[df["date"] == d0].sort_values("user_id")
    X0 = daily_df0[[f"h{i}" for i in range(24)]].values
    X0 = (X0 - X0.mean(1, keepdims=True)) / (X0.std(1, keepdims=True) + 1e-8)
    
    # Build initial graph
    adj0 = construct_multiview_graph(X0, knn=args.knn, shift_band=args.shift_band, use_dtw=not args.no_dtw, use_mi=not args.no_mi).to(device)
    X0_tensor = torch.FloatTensor(X0).to(device)
    
    # Get initial embeddings
    model.eval()
    with torch.no_grad():
        # h_prev is None initially
        z0 = model.encoder(X0_tensor, adj0, None)
        
    # Run KMeans on embeddings
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=args.k, n_init=20)
    y_pred = kmeans.fit_predict(z0.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float).to(device)
    
    # Temporal Storage
    h_prev = None # (N, nout)
    q_prev = None # (N, K)
    
    results = []
    
    print(f"Starting training loop for {len(dates)} steps...")
    
    for t, date in enumerate(dates):
        # Prepare Data for Step t
        # (Assuming user set is constant for simplicity, or handle mask)
        daily_df = df[df["date"] == date]
        # Sort by user_id to ensure alignment
        daily_df = daily_df.sort_values("user_id")
        user_ids = daily_df["user_id"].values
        X = daily_df[[f"h{i}" for i in range(24)]].values
        
        # Normalize (Standard Score per user)
        X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-8)
        
        # Build Graph
        adj = construct_multiview_graph(X, knn=args.knn, shift_band=args.shift_band, use_dtw=not args.no_dtw, use_mi=not args.no_mi).to(device)
        X_tensor = torch.FloatTensor(X).to(device)
        
        # Training Step (Fine-tuning Clustering)
        model.train()
        for epoch in range(args.epochs):
            optimizer.zero_grad()
            
            # Forward
            z, q = model(X_tensor, adj, h_prev)
            
            # Calculate Target Distribution
            p = target_distribution(q).detach()
            
            # Loss
            loss_clus = F.kl_div(q.log(), p, reduction='batchmean')
            
            loss_temp = 0
            if q_prev is not None and args.lambda_temp > 0:
                # Consistency Loss: KL(Q_{t-1} || Q_t)
                # Note: users must align. Assuming same users for now.
                # If users change, need alignment logic (Hungarian).
                loss_temp = F.kl_div(q.log(), q_prev.detach(), reduction='batchmean')
                
            loss = loss_clus + args.lambda_temp * loss_temp
            
            loss.backward()
            optimizer.step()
            
        # Inference & State Update
        model.eval()
        with torch.no_grad():
            # Before inference, h_prev is from t-1
            # But for inference at t, we need to pass h_prev (t-1)
            z, q = model(X_tensor, adj, h_prev)
            
            ts_score = 0.0
            if q_prev is not None:
                # Calculate TS (Temporal Smoothness): KL(Q_t || Q_{t-1})
                # q is current (t), q_prev is previous (t-1)
                ts_score = F.kl_div(q.log(), q_prev, reduction='batchmean').item()
            
            # Update temporal state
            h_prev = z.detach() # GRU state for next step
            q_prev = q.detach() # Consistency target for next step
            
            # Metrics
            pred = q.argmax(1).cpu().numpy()
            
            # Get True Labels (if available)
            if "true_cluster" in daily_df.columns:
                y_true = daily_df["true_cluster"].values
                from sklearn.metrics import adjusted_rand_score
                ari = adjusted_rand_score(y_true, pred)
            else:
                ari = None
                
            # Internal Metrics
            if not args.skip_internal_metrics:
                from sklearn.metrics import silhouette_score, davies_bouldin_score
                sil = silhouette_score(X, pred) if len(np.unique(pred)) > 1 else -1
                dbi = davies_bouldin_score(X, pred) if len(np.unique(pred)) > 1 else -1
            else:
                sil = None
                dbi = None
                
            # Calculate CSR (Cluster Switching Rate)
            csr = 0.0
            if t > 0:
                # Compare current pred with previous pred (need to store prev pred)
                # Assuming user order is consistent (which it is)
                prev_pred = results[-1]["labels"]
                switching_users = np.sum(pred != prev_pred)
                csr = switching_users / len(pred)
                
            print(f"Date: {date.date()} | ARI: {ari:.4f} | Sil: {sil if sil else 'N/A'} | DBI: {dbi:.4f} | CSR: {csr:.4f} | TS: {ts_score:.4f}")
            
            results.append({
                "date": str(date.date()),
                "ari": ari,
                "silhouette": sil,
                "dbi": dbi,
                "csr": csr,
                "ts": ts_score,
                "labels": pred # Store labels for CSR calculation
            })
            
    # Calculate Average Metrics
    avg_ari = np.mean([r["ari"] for r in results if r["ari"] is not None])
    avg_sil = np.mean([r["silhouette"] for r in results if r["silhouette"] is not None])
    avg_dbi = np.mean([r["dbi"] for r in results if r["dbi"] is not None])
    # Average CSR (exclude first day)
    if len(results) > 1:
        avg_csr = np.mean([r["csr"] for r in results[1:]])
        avg_ts = np.mean([r["ts"] for r in results[1:]])
    else:
        avg_csr = 0.0
        avg_ts = 0.0
        
    print(f"\nFinal Results for {city_name}:")
    print(f"Avg ARI: {avg_ari:.4f}")
    print(f"Avg SC:  {avg_sil:.4f}")
    print(f"Avg DBI: {avg_dbi:.4f}")
    print(f"Avg CSR: {avg_csr:.4f}")
    print(f"Avg TS:  {avg_ts:.4f}")
            
    # Save Results
    # Convert numpy arrays in labels to list for JSON
    for r in results:
        if "labels" in r and isinstance(r["labels"], np.ndarray):
            r["labels"] = r["labels"].tolist()

    # Determine suffix
    suffix = ""
    if args.no_dtw: suffix += "_no_dtw"
    if args.no_mi: suffix += "_no_mi"
    if args.no_gating: suffix += "_no_gating"
    if args.lambda_temp == 0: suffix += "_no_temp"
    
    out_path = os.path.join(args.out_dir, f"results_dynec_impl_{city_name}{suffix}_seed{args.current_seed}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")
    
    return {
        "ari": avg_ari,
        "sc": avg_sil,
        "dbi": avg_dbi,
        "csr": avg_csr,
        "ts": avg_ts
    }


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    # Default paths relative to script location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_root = os.path.join(base_dir, "Sichuan2024Dataset")
    default_out_dir = os.path.join(base_dir, "Sichuan2024_Experiments", "results")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=default_data_root)
    parser.add_argument("--city", default="City-A")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--knn", type=int, default=10)
    parser.add_argument("--shift-band", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=10) # Epochs per time step
    parser.add_argument("--lambda-temp", type=float, default=0.1)
    parser.add_argument("--out-dir", default=default_out_dir)
    parser.add_argument("--skip-internal-metrics", action="store_true")
    parser.add_argument("--limit-users", type=int, default=500, help="Limit number of users for quick test")
    parser.add_argument("--max-days", type=int, default=2, help="Limit number of days for quick test")
    parser.add_argument("--seeds", type=str, default="42", help="Comma-separated list of seeds (e.g. 1,2,3,4,5)")
    
    # Ablation Flags
    parser.add_argument("--no-dtw", action="store_true", help="Disable cDTW view")
    parser.add_argument("--no-mi", action="store_true", help="Disable MI view")
    parser.add_argument("--no-gating", action="store_true", help="Disable Gating (replace GAT with GCN)")
    
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    seed_list = [int(s) for s in args.seeds.split(",")]
    all_metrics = []
    
    print(f"Running experiments for {len(seed_list)} seeds: {seed_list}")
    
    for seed in seed_list:
        print(f"\n=== Running Seed {seed} ===")
        set_seed(seed)
        args.current_seed = seed
        metrics = train_dynec(args.data_root, args.city, args)
        if metrics:
            all_metrics.append(metrics)
            
    # Calculate Statistics
    if len(all_metrics) > 0:
        print("\n=== Final Statistics ===")
        keys = ["ari", "sc", "dbi", "csr", "ts"]
        stats = {}
        for k in keys:
            vals = [m[k] for m in all_metrics if m[k] is not None]
            if vals:
                mean_val = np.mean(vals)
                std_val = np.std(vals)
                stats[k] = {"mean": mean_val, "std": std_val}
                print(f"{k.upper()}: {mean_val:.4f} ± {std_val:.4f}")
            else:
                print(f"{k.upper()}: N/A")
        
        # Save Summary
        summary_path = os.path.join(args.out_dir, f"summary_{args.city}_seeds.json")
        with open(summary_path, "w") as f:
            json.dump({"seeds": seed_list, "metrics": all_metrics, "stats": stats}, f, indent=2)
        print(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    main()
