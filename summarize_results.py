
import json
import os
import glob
import numpy as np

def load_baseline_results():
    # Look in current directory
    pattern = "results_baseline_*.json"
    files = glob.glob(pattern)
    print(f"Found {len(files)} baseline files.")
    
    summary = {}
    
    for f in files:
        # Filename format: results_baseline_{method}_{city}.json
        basename = os.path.basename(f)
        parts = basename.replace("results_baseline_", "").replace(".json", "").split("_")
        
        if len(parts) >= 2:
            method = parts[0]
            city = parts[1] # "City-A"
            
            try:
                with open(f, 'r') as fp:
                    data = json.load(fp)
            except:
                print(f"Error loading {f}")
                continue
            
            count = len(data)
            aris = [d.get('ari', 0) for d in data if d.get('ari') is not None]
            avg_ari = np.mean(aris) if aris else 0
            
            sils = [d.get('silhouette', 0) for d in data if d.get('silhouette') is not None]
            avg_sil = np.mean(sils) if sils else 0

            dbis = [d.get('dbi', 0) for d in data if d.get('dbi') is not None]
            avg_dbi = np.mean(dbis) if dbis else 0

            # CSR - exclude first step if needed, or just mean
            csrs = [d.get('csr', 0) for d in data if d.get('csr') is not None]
            avg_csr = np.mean(csrs) if csrs else 0
            
            if city not in summary:
                summary[city] = {}
            summary[city][method] = {
                'ari': avg_ari, 
                'sil': avg_sil, 
                'dbi': avg_dbi, 
                'csr': avg_csr,
                'count': count
            }
            
    return summary

def load_dynec_results():
    # Look in Sichuan2024_Experiments/results/
    pattern = "Sichuan2024_Experiments/results/results_dynec_impl_*.json"
    files = glob.glob(pattern)
    print(f"Found {len(files)} DynEC result files.")
    
    dynec_summary = {}
    
    for f in files:
        basename = os.path.basename(f)
        # format: results_dynec_impl_City-A.json
        if "results_dynec_impl_" not in basename: continue
        
        city = basename.replace("results_dynec_impl_", "").replace(".json", "")
        
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
        except:
            print(f"Error loading {f}")
            continue
            
        count = len(data)
        aris = [d.get('ari', 0) for d in data]
        avg_ari = np.mean(aris) if aris else 0
        
        sils = [d.get('silhouette', 0) for d in data]
        avg_sil = np.mean(sils) if sils else 0
        
        dbis = [d.get('dbi', 0) for d in data]
        avg_dbi = np.mean(dbis) if dbis else 0
        
        csrs = [d.get('csr', 0) for d in data]
        avg_csr = np.mean(csrs) if csrs else 0
        
        dynec_summary[city] = {
            'ari': avg_ari, 
            'sil': avg_sil, 
            'dbi': avg_dbi, 
            'csr': avg_csr,
            'count': count
        }
        
    return dynec_summary

def main():
    baselines = load_baseline_results()
    dynec = load_dynec_results()
    
    # Merge DynEC into baselines
    for city, res in dynec.items():
        if city not in baselines:
            baselines[city] = {}
        baselines[city]['DynEC'] = res
        
    # Print Table
    cities = sorted(list(baselines.keys()))
    if not cities:
        print("No results found.")
        return

    # Collect all methods
    methods = set()
    for city in cities:
        methods.update(baselines[city].keys())
    methods = sorted(list(methods))
    
    # 1. ARI Table
    print("\n" + "="*80)
    print("Table 1: Adjusted Rand Index (ARI) Comparison (Higher is Better)")
    print("="*80)
    header = f"{'Method':<20} | " + " | ".join([f"{c:<15}" for c in cities])
    print(header)
    print("-" * len(header))
    
    for m in methods:
        row = f"{m:<20} | "
        for c in cities:
            if m in baselines[c]:
                val = baselines[c][m].get('ari', 0)
                row += f"{val:.4f}          | " # fixed width
            else:
                row += f"{'N/A':<15} | "
        print(row.strip(" |"))

    # 2. Silhouette Table
    print("\n" + "="*80)
    print("Table 2: Silhouette Score (Higher is Better)")
    print("="*80)
    print(header)
    print("-" * len(header))
    
    for m in methods:
        row = f"{m:<20} | "
        for c in cities:
            if m in baselines[c]:
                val = baselines[c][m].get('sil', 0)
                row += f"{val:.4f}          | "
            else:
                row += f"{'N/A':<15} | "
        print(row.strip(" |"))

    # 3. DBI Table
    print("\n" + "="*80)
    print("Table 3: Davies-Bouldin Index (DBI) Comparison (Lower is Better)")
    print("="*80)
    print(header)
    print("-" * len(header))
    
    for m in methods:
        row = f"{m:<20} | "
        for c in cities:
            if m in baselines[c]:
                val = baselines[c][m].get('dbi', 0)
                row += f"{val:.4f}          | "
            else:
                row += f"{'N/A':<15} | "
        print(row.strip(" |"))

    # 4. CSR Table
    print("\n" + "="*80)
    print("Table 4: Cluster Stability Ratio (CSR) Comparison (Lower is Better)")
    print("="*80)
    print(header)
    print("-" * len(header))
    
    for m in methods:
        row = f"{m:<20} | "
        for c in cities:
            if m in baselines[c]:
                val = baselines[c][m].get('csr', 0)
                row += f"{val:.4f}          | "
            else:
                row += f"{'N/A':<15} | "
        print(row.strip(" |"))

    # --- Table 3: DynEC Comprehensive Metrics ---
    print("\n" + "="*90)
    print("Table 3: DynEC Comprehensive Metrics (Full Year Analysis)")
    print("="*90)
    print(f"{'City':<10} | {'Days':<6} | {'ARI':<10} | {'Silhouette':<10} | {'DBI (Lower Better)':<18} | {'CSR (Lower Better)':<18}")
    print("-" * 90)
    
    for c in cities:
        res = dynec.get(c, {})
        if res:
            print(f"{c:<10} | {res.get('count',0):<6} | {res.get('ari',0):<10.4f} | {res.get('sil',0):<10.4f} | {res.get('dbi',0):<18.4f} | {res.get('csr',0):<18.4f}")

if __name__ == "__main__":
    main()
