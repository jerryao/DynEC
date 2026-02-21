
import json
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def load_results(city):
    results = {}
    
    # Load DynEC
    dynec_files = glob.glob(f"Sichuan2024_Experiments/results/results_dynec_impl_{city}.json")
    if dynec_files:
        with open(dynec_files[0], 'r') as f:
            data = json.load(f)
            # Normalize step/date
            # DynEC has "date" and index is implicit step
            df = pd.DataFrame(data)
            df['step'] = range(len(df))
            df['method'] = 'DynEC'
            results['DynEC'] = df
            
    # Load Baselines
    baseline_files = glob.glob(f"results_baseline_*_{city}.json")
    for f in baseline_files:
        basename = os.path.basename(f)
        method = basename.replace("results_baseline_", "").replace(f"_{city}.json", "")
        
        with open(f, 'r') as fp:
            data = json.load(fp)
            df = pd.DataFrame(data)
            df['method'] = method
            results[method] = df
            
    return results

def plot_city(city, results):
    if not results:
        print(f"No results for {city}")
        return
        
    metrics = ['ari', 'silhouette', 'dbi', 'csr']
    titles = ['ARI (Higher Better)', 'Silhouette (Higher Better)', 'DBI (Lower Better)', 'CSR (Lower Better)']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Comparison on {city}", fontsize=16)
    
    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]
        
        for method, df in results.items():
            if metric in df.columns:
                # Smooth lines for better visualization if needed, but raw is fine
                # Use step as x-axis
                sns.lineplot(data=df, x='step', y=metric, label=method, ax=ax, alpha=0.7)
        
        ax.set_title(titles[i])
        ax.set_xlabel('Time Step (Days)')
        ax.set_ylabel(metric.upper())
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(f"comparison_{city}.png", dpi=300)
    print(f"Saved comparison_{city}.png")
    plt.close()

def main():
    cities = ['City-A', 'City-B', 'City-C']
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    for city in cities:
        print(f"Processing {city}...")
        results = load_results(city)
        plot_city(city, results)

if __name__ == "__main__":
    main()
