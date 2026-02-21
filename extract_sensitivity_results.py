import json
import numpy as np
import os

lambdas = [0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
base_path = r"Sichuan2024_Experiments\results\sensitivity"

print(f"{'Lambda':<10} {'ARI':<10} {'SC':<10} {'CSR':<10} {'DBI':<10}")
print("-" * 50)

for lam in lambdas:
    dir_name = f"lambda_{lam}"
    suffix = "_no_temp" if lam == 0 else ""
    filename = f"results_dynec_impl_City-A{suffix}.json"
    path = os.path.join(base_path, dir_name, filename)
    
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
            
        # Calculate averages
        # Note: usually we skip the first day for CSR as it's 0 by definition or not meaningful
        # But let's see how run_dynec_impl calculates it. It seems to include all days in the json.
        # Let's align with the standard metric calculation: mean over all available valid values.
        
        ari_vals = [d['ari'] for d in data if d.get('ari') is not None]
        sc_vals = [d['silhouette'] for d in data if d.get('silhouette') is not None]
        csr_vals = [d['csr'] for d in data if d.get('csr') is not None]
        dbi_vals = [d['dbi'] for d in data if d.get('dbi') is not None]

        # For CSR, often the first value is 0.0 because there is no previous day. 
        # Ideally we should exclude the first day for CSR.
        if len(csr_vals) > 1:
            avg_csr = np.mean(csr_vals[1:])
        else:
            avg_csr = np.mean(csr_vals) if csr_vals else 0.0
            
        avg_ari = np.mean(ari_vals) if ari_vals else 0.0
        avg_sc = np.mean(sc_vals) if sc_vals else 0.0
        avg_dbi = np.mean(dbi_vals) if dbi_vals else 0.0
        
        print(f"{lam:<10} {avg_ari:.4f}     {avg_sc:.4f}     {avg_csr:.4f}     {avg_dbi:.4f}")
    else:
        print(f"{lam:<10} File not found: {path}")
