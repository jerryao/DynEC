import json
import numpy as np
import os

files = [
    ("DynEC (Full)", "results_dynec_impl_City-A.json"),
    ("w/o cDTW View", "results_dynec_impl_City-A_no_dtw.json"),
    ("w/o MI View", "results_dynec_impl_City-A_no_mi.json"),
    ("w/o Gating (GAT+GRU)", "results_dynec_impl_City-A_no_gating.json"),
    ("w/o Temporal Loss (lambda=0)", "results_dynec_impl_City-A_no_temp.json")
]

base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Sichuan2024_Experiments", "results", "ablation")

print(f"{'Variant':<30} {'ARI':<10} {'SC':<10} {'DBI':<10} {'CSR':<10}")
print("-" * 75)

for name, filename in files:
    path = os.path.join(base_path, filename)
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
            # Calculate mean of metrics
            ari_vals = [d['ari'] for d in data if d.get('ari') is not None]
            sc_vals = [d['silhouette'] for d in data if d.get('silhouette') is not None]
            dbi_vals = [d['dbi'] for d in data if d.get('dbi') is not None]
            # CSR excludes first day usually, as it is 0
            csr_vals = [d['csr'] for d in data[1:]] if len(data) > 1 else [0.0]
            
            avg_ari = np.mean(ari_vals) if ari_vals else 0.0
            avg_sc = np.mean(sc_vals) if sc_vals else 0.0
            avg_dbi = np.mean(dbi_vals) if dbi_vals else 0.0
            avg_csr = np.mean(csr_vals) if csr_vals else 0.0
            
            print(f"{name:<30} {avg_ari:.4f}     {avg_sc:.4f}     {avg_dbi:.4f}     {avg_csr:.4f}")
    else:
        print(f"{name:<30} FILE NOT FOUND")
