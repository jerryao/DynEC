import pandas as pd
import os
import shutil

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DATA_DIR = os.path.join(BASE_DIR, "Sichuan2024Dataset")
DST_DATA_DIR = os.path.join(BASE_DIR, "Sichuan2024Dataset_Monthly")

CITIES = ["City-A", "City-B", "City-C"]

def prepare_monthly_data():
    if os.path.exists(DST_DATA_DIR):
        shutil.rmtree(DST_DATA_DIR)
    os.makedirs(DST_DATA_DIR)

    for city in CITIES:
        print(f"Processing {city}...")
        city_src = os.path.join(SRC_DATA_DIR, city)
        city_dst = os.path.join(DST_DATA_DIR, city)
        os.makedirs(city_dst)

        # 1. Load profiles_daily.csv
        profiles_path = os.path.join(city_src, "profiles_daily.csv")
        df = pd.read_csv(profiles_path)
        
        # 2. Convert date to datetime
        df["date"] = pd.to_datetime(df["date"])
        
        # 3. Create 'month' column (e.g., 2024-01-01)
        # Use start of month as the "date"
        df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
        
        # 4. Aggregate by (user_id, month)
        # Features h0-h23: mean
        # true_cluster: mode (most frequent)
        
        feature_cols = [f"h{i}" for i in range(24)]
        
        # Define aggregation dictionary
        agg_dict = {col: "mean" for col in feature_cols}
        if "true_cluster" in df.columns:
            agg_dict["true_cluster"] = lambda x: x.mode().iloc[0] if not x.mode().empty else -1
            
        print(f"  Aggregating daily to monthly...")
        monthly_df = df.groupby(["user_id", "month"]).agg(agg_dict).reset_index()
        
        # Rename 'month' back to 'date' for compatibility with existing scripts
        monthly_df = monthly_df.rename(columns={"month": "date"})
        
        # Sort by date, user_id
        monthly_df = monthly_df.sort_values(["date", "user_id"])
        
        # Save to destination
        dst_path = os.path.join(city_dst, "profiles_daily.csv") # Keep name as profiles_daily.csv
        monthly_df.to_csv(dst_path, index=False)
        print(f"  Saved monthly data to {dst_path} with {len(monthly_df)} rows.")
        
        # 5. Copy users.csv and events.csv (if exists)
        for f in ["users.csv", "events.csv"]:
            src_f = os.path.join(city_src, f)
            if os.path.exists(src_f):
                shutil.copy2(src_f, os.path.join(city_dst, f))
                print(f"  Copied {f}")

    # Create meta.json if needed
    meta_src = os.path.join(SRC_DATA_DIR, "meta.json")
    if os.path.exists(meta_src):
        shutil.copy2(meta_src, os.path.join(DST_DATA_DIR, "meta.json"))

if __name__ == "__main__":
    prepare_monthly_data()
