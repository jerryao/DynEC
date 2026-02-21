import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

class Time2Graph:
    """
    Time2Graph Implementation (Simplified for Unsupervised/Clustering).
    
    Core Idea:
    1. Shapelet Discovery: Extract representative sub-sequences (shapelets) from time series.
    2. Shapelet Transformation: Convert time series into sequences of shapelet IDs.
    3. Graph Construction: Build a transition graph (shapelet-to-shapelet) for each user.
    4. Embedding: Use the transition probabilities (or graph embeddings) as user features.
    """
    def __init__(self, n_shapelets=50, segment_length=6, n_clusters=8):
        self.n_shapelets = n_shapelets
        self.segment_length = segment_length
        self.n_clusters = n_clusters # Final clustering target
        self.shapelets = None
        self.kmeans_shapelets = None

    def fit(self, X_seq):
        """
        Learn shapelets from the data.
        X_seq: (T, N, F) - Time series data.
        """
        T, N, F = X_seq.shape
        n_segments = F // self.segment_length
        
        # 1. Collect all segments
        all_segments = []
        # Subsample to avoid memory explosion if T*N is large
        # For 366 days * 1000 users, that's huge. 
        # We should sample a subset of days and users to learn shapelets.
        
        # Increase sample size for better coverage on year-long data
        sample_days_count = min(T, 60) # 2 months equivalent
        sample_users_count = min(N, 200) # 20% users
        
        sample_days = np.random.choice(range(T), sample_days_count, replace=False)
        sample_users = np.random.choice(range(N), sample_users_count, replace=False)
        
        for t in sample_days:
            for i in sample_users:
                ts = X_seq[t, i]
                for s in range(n_segments):
                    segment = ts[s*self.segment_length : (s+1)*self.segment_length]
                    if len(segment) == self.segment_length:
                        all_segments.append(segment)
        
        all_segments = np.array(all_segments)
        
        # 2. Learn Shapelets via K-Means
        print(f"Learning {self.n_shapelets} shapelets from {len(all_segments)} segments...")
        self.kmeans_shapelets = KMeans(n_clusters=self.n_shapelets, random_state=42, n_init=10)
        self.kmeans_shapelets.fit(all_segments)
        self.shapelets = self.kmeans_shapelets.cluster_centers_
        
    def transform(self, X_seq):
        """
        Convert time series to graph embeddings.
        Returns: embeddings (T, N, Feature_Dim)
        """
        T, N, F = X_seq.shape
        n_segments = F // self.segment_length
        
        embeddings = []
        
        print("Transforming time series to graph embeddings...")
        for t in range(T):
            day_embeddings = []
            # Batch processing for efficiency?
            # Process all users for day t
            
            # Extract segments for all users at day t
            # Shape: (N * n_segments, segment_length)
            day_segments = []
            for i in range(N):
                ts = X_seq[t, i]
                for s in range(n_segments):
                    day_segments.append(ts[s*self.segment_length : (s+1)*self.segment_length])
            
            day_segments = np.array(day_segments)
            
            # Predict shapelet IDs
            # (N * n_segments,)
            shapelet_ids = self.kmeans_shapelets.predict(day_segments)
            
            # Reshape back to (N, n_segments)
            user_shapelet_seqs = shapelet_ids.reshape(N, n_segments)
            
            # Build Transition Matrix for each user
            # Feature dim = n_shapelets * n_shapelets (Flattened Transition Matrix)
            # If n_shapelets=50, dim=2500. A bit large but manageable.
            # Alternatively, we can use just the counts (Bag of Patterns) + specific transitions.
            # Let's use flattened transition matrix.
            
            for i in range(N):
                seq = user_shapelet_seqs[i]
                trans_mat = np.zeros((self.n_shapelets, self.n_shapelets))
                
                for k in range(len(seq)-1):
                    src = seq[k]
                    dst = seq[k+1]
                    trans_mat[src, dst] += 1
                
                # Normalize (Row-stochastic)
                row_sums = trans_mat.sum(axis=1, keepdims=True) + 1e-8
                trans_mat = trans_mat / row_sums
                
                day_embeddings.append(trans_mat.flatten())
                
            embeddings.append(day_embeddings)
            
        return np.array(embeddings) # (T, N, n_shapelets^2)

    def fit_predict(self, X_seq):
        """
        Full pipeline: Fit shapelets, Transform, Cluster.
        """
        if self.kmeans_shapelets is None:
            self.fit(X_seq)
            
        embeddings = self.transform(X_seq)
        
        # Cluster per day
        T, N, _ = embeddings.shape
        results = []
        
        from sklearn.metrics import silhouette_score
        
        print("Clustering Time2Graph embeddings...")
        for t in range(T):
            X_t = embeddings[t]
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
            labels = kmeans.fit_predict(X_t)
            
            # Calculate silhouette on original data or embeddings?
            # Usually on the embeddings used for clustering.
            sil = silhouette_score(X_t, labels) if len(np.unique(labels)) > 1 else 0
            
            results.append({
                'step': t,
                'labels': labels,
                'silhouette': sil
            })
            
        return results
