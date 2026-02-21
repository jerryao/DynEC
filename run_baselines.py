import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score, davies_bouldin_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
from baselines.evolvegcn import EvolveGCN
from baselines.time2graph import Time2Graph

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Utility Functions ---
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_data(city, base_path, limit_users=None, max_days=None):
    print(f"Loading data for {city}...")
    events_path = os.path.join(base_path, city, 'events.csv')
    profiles_path = os.path.join(base_path, city, 'profiles_daily.csv')
    users_path = os.path.join(base_path, city, 'users.csv')

    # Load users to get consistent user set
    users_df = pd.read_csv(users_path)
    all_user_ids = users_df['user_id'].unique()
    
    if limit_users:
        all_user_ids = all_user_ids[:limit_users]
        
    # Load daily profiles
    profiles_df = pd.read_csv(profiles_path)
    profiles_df['date'] = pd.to_datetime(profiles_df['date'])
    dates = sorted(profiles_df['date'].unique())
    
    if max_days:
        dates = dates[:max_days]
    
    # Filter users who are present in ALL selected days
    valid_users = set(all_user_ids)
    for d in dates:
        day_users = set(profiles_df[profiles_df['date'] == d]['user_id'].unique())
        valid_users = valid_users.intersection(day_users)
    
    final_users = sorted(list(valid_users))
    print(f"Selected {len(final_users)} users for {len(dates)} days.")
    
    # Create tensor: (T, N, F)
    data_list = []
    labels_seq = [] # Placeholder if we had ground truth labels per day
    
    # For evaluation, we might need ground truth. Here we assume profiles have 'label' or similar if available,
    # but the provided files don't guarantee labels. We will use `events` to generate pseudo-labels or just ignore.
    # Actually, we will return raw features.
    
    # We need to construct feature matrix. Assuming profiles have 24 hourly values + other features
    # Let's verify columns. We'll stick to 24 hourly load values if available, or just all numeric columns.
    # Based on previous knowledge, profiles has 'h0'...'h23'.
    feature_cols = [f'h{i}' for i in range(24)]
    
    # Pre-filter dataframe for speed
    profiles_df = profiles_df[profiles_df['date'].isin(dates)]
    profiles_df = profiles_df[profiles_df['user_id'].isin(final_users)]
    
    # Check for labels
    has_labels = 'true_cluster' in profiles_df.columns

    # Pivot to ensure order: (Date, User, Features)
    # We iterate dates to build the tensor
    X_seq = []
    Y_seq = []
    
    for d in dates:
        day_df = profiles_df[profiles_df['date'] == d].set_index('user_id')
        day_df = day_df.reindex(final_users) # Ensure consistent order
        X_t = day_df[feature_cols].values
        X_seq.append(X_t)
        
        if has_labels:
            Y_t = day_df['true_cluster'].values
            Y_seq.append(Y_t)
        
    X_seq = np.array(X_seq) # (T, N, F)
    if has_labels:
        Y_seq = np.array(Y_seq) # (T, N)
    else:
        Y_seq = None
    
    # Normalize features
    N, T, F_dim = len(final_users), len(dates), len(feature_cols)
    X_seq_reshaped = X_seq.reshape(-1, F_dim)
    scaler = StandardScaler()
    X_seq_norm = scaler.fit_transform(X_seq_reshaped).reshape(T, N, F_dim)
    
    return X_seq_norm, Y_seq, final_users, dates

# --- Baseline Models ---

class BaselineModel:
    def __init__(self, n_clusters, device='cuda'):
        self.n_clusters = n_clusters
        self.device = device
        
    def fit_predict(self, X_seq, seed=None):
        raise NotImplementedError

# 1. K-Means (Static)
class KMeansBaseline(BaselineModel):
    def fit_predict(self, X_seq, seed=None):
        print(f"Running K-Means (Static per day) with seed {seed}...")
        T, N, F_dim = X_seq.shape
        results = []
        
        for t in range(T):
            X_t = X_seq[t]
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=seed, n_init=10)
            labels = kmeans.fit_predict(X_t)
            sil = silhouette_score(X_t, labels) if len(np.unique(labels)) > 1 else 0
            results.append({
                'step': t,
                'labels': labels,
                'silhouette': sil
            })
        return results

# 2. Spectral Clustering (Static)
class SpectralBaseline(BaselineModel):
    def fit_predict(self, X_seq, seed=None):
        print(f"Running Spectral Clustering (Static per day) with seed {seed}...")
        T, N, F_dim = X_seq.shape
        results = []
        
        for t in range(T):
            X_t = X_seq[t]
            # Construct affinity matrix (RBF kernel or KNN)
            # Using nearest_neighbors for scalability
            spectral = SpectralClustering(n_clusters=self.n_clusters, 
                                          affinity='nearest_neighbors',
                                          random_state=seed,
                                          n_neighbors=10,
                                          n_jobs=-1)
            labels = spectral.fit_predict(X_t)
            sil = silhouette_score(X_t, labels) if len(np.unique(labels)) > 1 else 0
            results.append({
                'step': t,
                'labels': labels,
                'silhouette': sil
            })
        return results


# 5. Time2Graph (Official-aligned)
class Time2GraphBaseline(BaselineModel):
    def fit_predict(self, X_seq, seed=None):
        print(f"Running Time2Graph (Official-aligned) with seed {seed}...")
        # Use the dedicated implementation in baselines/time2graph.py
        # We need to ensure X_seq is compatible (T, N, F)
        
        # Initialize the model
        # segment_length=6 (4 hours) or similar based on data freq
        # If data is daily 24h, segment_length=6 means 4 segments per day.
        # F_dim is 24 usually. 24/6 = 4 segments.
        # Passing seed if supported, otherwise rely on global set_seed
        t2g = Time2Graph(n_shapelets=20, segment_length=4, n_clusters=self.n_clusters)
        
        # Fit and Predict
        results = t2g.fit_predict(X_seq)
        return results

# 6. EvolveGCN (Official-aligned)
# Uses LSTM to evolve GCN weights (EvolveGCN-O)
class EvolveGCNBaseline(BaselineModel):
    def fit_predict(self, X_seq, seed=None):
        print(f"Running EvolveGCN (Official-aligned) with seed {seed}...")
        T, N, F_dim = X_seq.shape
        results = []
        
        # Build Adjacency (KNN) for each step
        adjs = []
        print("Constructing adjacency matrices...")
        for t in range(T):
            # Use kneighbors_graph
            A = kneighbors_graph(X_seq[t], 10, mode='connectivity', include_self=True)
            A = torch.FloatTensor(A.toarray()).to(self.device)
            
            # Normalize A
            D = torch.diag(torch.sum(A, dim=1))
            D_inv_sqrt = torch.pow(D, -0.5)
            D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.
            A_norm = torch.mm(torch.mm(D_inv_sqrt, A), D_inv_sqrt)
            adjs.append(A_norm)
            
        # Initialize EvolveGCN model
        # Input dim: F_dim
        # Hidden dim: 64
        # Output dim: n_clusters (embedding size for clustering)
        model = EvolveGCN(in_feat=F_dim, hidden_feat=64, out_feat=self.n_clusters, n_layers=2).to(self.device)
        
        # Lower learning rate and add gradient clipping
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        print("Training EvolveGCN with Truncated BPTT...")
        X_seq_tensor = torch.FloatTensor(X_seq).to(self.device)
        
        chunk_size = 30 # Train in chunks of 30 steps
        
        for epoch in range(50):
            model.train()
            
            # Initialize hidden states for the sequence
            hx_list = None 
            
            total_loss = 0
            
            # Iterate over chunks
            for start_t in range(0, T, chunk_size):
                end_t = min(start_t + chunk_size, T)
                
                # Detach hidden states from previous chunk to truncate BPTT
                if hx_list is not None:
                    hx_list = [(h.detach(), c.detach()) for (h, c) in hx_list]
                
                optimizer.zero_grad()
                
                # Get chunk data
                X_chunk = X_seq_tensor[start_t:end_t]
                adj_chunk = adjs[start_t:end_t]
                
                # Forward pass for the chunk
                outputs, hx_list = model(X_chunk, adj_chunk, hx_list)
                
                chunk_loss = 0
                for i, t in enumerate(range(start_t, end_t)):
                    z = outputs[i]
                    adj = adjs[t]
                    
                    # Reconstruction Loss
                    recon = torch.mm(z, z.t())
                    loss = F.mse_loss(recon, adj)
                    chunk_loss += loss
                
                if torch.isnan(chunk_loss):
                    print(f"Warning: NaN loss detected at epoch {epoch}, chunk {start_t}-{end_t}")
                    optimizer.zero_grad() # Clear gradients
                    continue

                # Backward pass for the chunk
                chunk_loss.backward()
                
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                
                total_loss += chunk_loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
            
        # Predict
        model.eval()
        with torch.no_grad():
            # For prediction, we can run full sequence or chunked (doesn't matter for gradients)
            # But we need state continuity.
            outputs, _ = model(X_seq_tensor, adjs)
            
            for t in range(T):
                z = outputs[t]
                
                # Cluster the embeddings
                kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
                labels = kmeans.fit_predict(z.cpu().numpy())
                
                sil = silhouette_score(X_seq[t], labels) if len(np.unique(labels)) > 1 else 0
                results.append({
                    'step': t,
                    'labels': labels,
                    'silhouette': sil
                })
                
        return results

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='City-A')
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--limit-users', type=int, default=1000)
    parser.add_argument('--max-days', type=int, default=7)
    parser.add_argument('--out-dir', type=str, default='.')
    parser.add_argument('--suffix', type=str, default='')
    # Default paths relative to script location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_root = os.path.join(base_dir, "Sichuan2024Dataset")
    parser.add_argument('--data-root', type=str, default=default_data_root)
    parser.add_argument('--seeds', type=str, default='42', help='Comma-separated list of seeds (e.g. 1,2,3,4,5)')
    args = parser.parse_args()
    
    base_path = args.data_root
    X_seq, Y_seq, users, dates = load_data(args.city, base_path, args.limit_users, args.max_days)
    os.makedirs(args.out_dir, exist_ok=True)
    
    seed_list = [int(s) for s in args.seeds.split(',')]
    
    baselines = {
        'KMeans': KMeansBaseline(args.k),
        'Spectral': SpectralBaseline(args.k),
        'Time2Graph': Time2GraphBaseline(args.k),
        'EvolveGCN': EvolveGCNBaseline(args.k)
    }
    
    # Structure to hold stats across seeds
    final_stats = {name: {'ari': [], 'sc': [], 'dbi': [], 'csr': []} for name in baselines}
    
    import json

    for seed in seed_list:
        print(f"\n=== Running Seed {seed} ===")
        set_seed(seed)
        
        for name, model in baselines.items():
            print(f"\n--- Running {name} (Seed {seed}) ---")
            
            # Construct output path for this seed
            out_path = os.path.join(args.out_dir, f'results_baseline_{name}_{args.city}{args.suffix}_seed{seed}.json')
            
            # Check existing results
            if os.path.exists(out_path):
                try:
                    with open(out_path, 'r') as f:
                        existing_data = json.load(f)
                    if len(existing_data) == len(dates):
                        print(f"Skipping {name} - Results already exist for seed {seed}.")
                        
                        # Load stats from existing file
                        aris = [r['ari'] for r in existing_data if 'ari' in r]
                        scs = [r['silhouette'] for r in existing_data]
                        dbis = [r['dbi'] for r in existing_data if 'dbi' in r and r['dbi'] is not None]
                        csrs = [r['csr'] for r in existing_data] # CSR is usually 0 for first day, but average over T-1
                        
                        if aris: final_stats[name]['ari'].append(np.mean(aris))
                        if scs: final_stats[name]['sc'].append(np.mean(scs))
                        if dbis: final_stats[name]['dbi'].append(np.mean(dbis))
                        # CSR average usually excludes the first day (which is 0)
                        if len(csrs) > 1: final_stats[name]['csr'].append(np.mean(csrs[1:]))
                        else: final_stats[name]['csr'].append(0)
                        
                        continue
                except Exception as e:
                    print(f"Error checking existing results: {e}")
            
            try:
                results = model.fit_predict(X_seq, seed=seed)
                
                # Calculate Metrics (ARI, DBI, CSR)
                aris = []
                dbis = []
                csrs = []
                scs = []
                
                prev_labels = None
                
                for i, r in enumerate(results):
                    t = r['step']
                    labels = r['labels']
                    current_X = X_seq[t]
                    
                    # ARI
                    if Y_seq is not None and t < len(Y_seq):
                        ari = adjusted_rand_score(Y_seq[t], labels)
                        r['ari'] = ari
                        aris.append(ari)
                    
                    # DBI
                    if len(np.unique(labels)) > 1:
                        dbi = davies_bouldin_score(current_X, labels)
                        r['dbi'] = dbi
                        dbis.append(dbi)
                    else:
                        r['dbi'] = None
                    
                    # SC
                    sc = r['silhouette']
                    scs.append(sc)
                    
                    # CSR
                    csr = 0.0
                    if prev_labels is not None:
                        # Assuming user order is consistent
                        switching = np.sum(labels != prev_labels)
                        csr = switching / len(labels)
                    r['csr'] = csr
                    if i > 0: # Skip first day for CSR average
                        csrs.append(csr)
                    
                    prev_labels = labels

                avg_ari = np.mean(aris) if aris else 0
                avg_dbi = np.mean(dbis) if dbis else 0
                avg_sc = np.mean(scs) if scs else 0
                avg_csr = np.mean(csrs) if csrs else 0
                
                # Store seed stats
                final_stats[name]['ari'].append(avg_ari)
                final_stats[name]['sc'].append(avg_sc)
                final_stats[name]['dbi'].append(avg_dbi)
                final_stats[name]['csr'].append(avg_csr)
                
                print(f"{name} (Seed {seed}) Avg SC: {avg_sc:.4f}, ARI: {avg_ari:.4f}, DBI: {avg_dbi:.4f}, CSR: {avg_csr:.4f}")
                
                # Save results (Simplified)
                # Convert numpy types to python for json
                serializable_results = []
                for r in results:
                    res_dict = {
                        'step': r['step'],
                        'silhouette': float(r['silhouette']),
                        'csr': float(r['csr'])
                        # 'labels': r['labels'].tolist() # Skip saving labels to keep file small for now
                    }
                    if 'ari' in r:
                        res_dict['ari'] = float(r['ari'])
                    if 'dbi' in r and r['dbi'] is not None:
                        res_dict['dbi'] = float(r['dbi'])
                    
                    serializable_results.append(res_dict)
                
                with open(out_path, 'w') as f:
                    json.dump(serializable_results, f, indent=2)
                    
            except Exception as e:
                print(f"Error running {name}: {e}")
                import traceback
                traceback.print_exc()

    # Calculate and Print Final Statistics
    print("\n=== Final Statistics (Mean ± Std) ===")
    summary_stats = {}
    
    for name in baselines:
        print(f"\nModel: {name}")
        summary_stats[name] = {}
        for metric in ['ari', 'sc', 'dbi', 'csr']:
            vals = final_stats[name][metric]
            if vals:
                mean_val = np.mean(vals)
                std_val = np.std(vals)
                summary_stats[name][metric] = {'mean': mean_val, 'std': std_val}
                print(f"  {metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")
            else:
                print(f"  {metric.upper()}: N/A")
                
    # Save Summary
    summary_path = os.path.join(args.out_dir, f"summary_baselines_{args.city}_seeds.json")
    with open(summary_path, "w") as f:
        json.dump({"seeds": seed_list, "stats": summary_stats}, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

if __name__ == '__main__':
    main()
