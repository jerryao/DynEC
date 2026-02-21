import argparse
import datetime as dt
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StepResult:
    method: str
    city: str
    period: str
    seed: int
    n_users: int
    n_users_overlap: Optional[int]
    k: int
    ari: float
    silhouette: Optional[float]
    dbi: Optional[float]
    csr: Optional[float]
    ts: Optional[float]


def _stable_ints(arr: Sequence[int]) -> np.ndarray:
    return np.asarray([int(x) for x in arr], dtype=np.int64)


def adjusted_rand_index(labels_true: Sequence[int], labels_pred: Sequence[int]) -> float:
    lt = _stable_ints(labels_true)
    lp = _stable_ints(labels_pred)
    if lt.shape[0] != lp.shape[0] or lt.shape[0] == 0:
        return float("nan")

    _, lt_ids = np.unique(lt, return_inverse=True)
    _, lp_ids = np.unique(lp, return_inverse=True)
    n_true = int(lt_ids.max()) + 1
    n_pred = int(lp_ids.max()) + 1

    contingency = np.zeros((n_true, n_pred), dtype=np.int64)
    for i in range(lt_ids.shape[0]):
        contingency[int(lt_ids[i]), int(lp_ids[i])] += 1

    def comb2(x: np.ndarray) -> np.ndarray:
        return x * (x - 1) // 2

    sum_comb_c = int(comb2(contingency).sum())
    sum_comb_true = int(comb2(contingency.sum(axis=1)).sum())
    sum_comb_pred = int(comb2(contingency.sum(axis=0)).sum())
    comb_n = int(lt_ids.shape[0] * (lt_ids.shape[0] - 1) // 2)
    if comb_n == 0:
        return 1.0

    expected = (sum_comb_true * sum_comb_pred) / comb_n
    max_index = 0.5 * (sum_comb_true + sum_comb_pred)
    denom = max_index - expected
    if denom <= 0:
        return 0.0
    return float((sum_comb_c - expected) / denom)


def pairwise_sqeuclidean(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    G = X @ X.T
    sq = np.maximum(np.diag(G)[:, None] - 2.0 * G + np.diag(G)[None, :], 0.0)
    return sq


def silhouette_score_euclidean(X: np.ndarray, labels: Sequence[int]) -> Optional[float]:
    labels = _stable_ints(labels)
    n = X.shape[0]
    if n < 2:
        return None

    uniq = np.unique(labels)
    if uniq.shape[0] <= 1 or uniq.shape[0] >= n:
        return None

    sq = pairwise_sqeuclidean(X)
    D = np.sqrt(sq)

    sil = np.zeros(n, dtype=np.float64)
    for i in range(n):
        li = labels[i]
        in_mask = labels == li
        in_mask[i] = False
        if np.any(in_mask):
            a = float(D[i, in_mask].mean())
        else:
            sil[i] = 0.0
            continue

        b = float("inf")
        for l in uniq:
            if int(l) == int(li):
                continue
            mask = labels == l
            if not np.any(mask):
                continue
            b = min(b, float(D[i, mask].mean()))

        if not np.isfinite(b):
            sil[i] = 0.0
            continue
        sil[i] = (b - a) / max(a, b)

    return float(np.mean(sil))


def davies_bouldin_index(X: np.ndarray, labels: Sequence[int]) -> Optional[float]:
    labels = _stable_ints(labels)
    uniq = np.unique(labels)
    if uniq.shape[0] <= 1:
        return None

    X = np.asarray(X, dtype=np.float64)
    centroids: Dict[int, np.ndarray] = {}
    scatters: Dict[int, float] = {}
    for l in uniq:
        mask = labels == l
        if not np.any(mask):
            continue
        Xi = X[mask]
        ci = Xi.mean(axis=0)
        centroids[int(l)] = ci
        scatters[int(l)] = float(np.mean(np.linalg.norm(Xi - ci[None, :], axis=1)))

    keys = list(centroids.keys())
    if len(keys) <= 1:
        return None

    R = []
    for i in keys:
        ci = centroids[i]
        si = scatters[i]
        worst = -float("inf")
        for j in keys:
            if i == j:
                continue
            cj = centroids[j]
            sj = scatters[j]
            denom = float(np.linalg.norm(ci - cj))
            if denom <= 1e-12:
                val = float("inf")
            else:
                val = (si + sj) / denom
            if val > worst:
                worst = val
        R.append(worst)

    if not R:
        return None
    return float(np.mean(R))


def kmeans_pp_init(rng: np.random.Generator, X: np.ndarray, k: int) -> np.ndarray:
    n = X.shape[0]
    centroids = np.empty((k, X.shape[1]), dtype=np.float64)
    idx0 = int(rng.integers(0, n))
    centroids[0] = X[idx0]
    closest_sq = np.sum((X - centroids[0][None, :]) ** 2, axis=1)

    for c in range(1, k):
        probs = closest_sq / max(float(closest_sq.sum()), 1e-12)
        idx = int(rng.choice(n, p=probs))
        centroids[c] = X[idx]
        d_sq = np.sum((X - centroids[c][None, :]) ** 2, axis=1)
        closest_sq = np.minimum(closest_sq, d_sq)
    return centroids


def assign_labels(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    d2 = (
        np.sum(X * X, axis=1)[:, None]
        - 2.0 * (X @ centroids.T)
        + np.sum(centroids * centroids, axis=1)[None, :]
    )
    return np.argmin(d2, axis=1).astype(np.int64)


def run_kmeans(
    X: np.ndarray, k: int, seed: int, max_iter: int = 100, tol: float = 1e-5
) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    centroids = kmeans_pp_init(rng, X, k)

    for _ in range(max_iter):
        labels = assign_labels(X, centroids)
        new_centroids = np.empty_like(centroids)
        for j in range(k):
            mask = labels == j
            if not np.any(mask):
                new_centroids[j] = X[int(rng.integers(0, n))]
            else:
                new_centroids[j] = X[mask].mean(axis=0)

        shift = float(np.linalg.norm(new_centroids - centroids))
        centroids = new_centroids
        if shift <= tol:
            break

    labels = assign_labels(X, centroids)
    return labels, centroids


def mode_int(series: pd.Series) -> int:
    vc = series.value_counts()
    if len(vc) == 0:
        return 0
    return int(vc.index[0])


def parse_seeds(seeds: Optional[str], fallback_seed: int) -> List[int]:
    if seeds is None or str(seeds).strip() == "":
        return [int(fallback_seed)]
    out: List[int] = []
    for part in str(seeds).split(","):
        s = part.strip()
        if not s:
            continue
        out.append(int(s))
    if not out:
        out = [int(fallback_seed)]
    return out


def sample_indices(n: int, sample_size: int, seed: int) -> np.ndarray:
    n = int(n)
    sample_size = int(sample_size)
    if sample_size <= 0 or sample_size >= n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(n, size=sample_size, replace=False).astype(np.int64))



def read_city_monthly_matrix(city_dir: str) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    profiles_path = os.path.join(city_dir, "profiles_daily.csv")
    df = pd.read_csv(profiles_path)
    hcols = [f"h{i}" for i in range(24)]
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df["period"] = df["date"].dt.to_period("M").astype(str)

    out: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for period, g in df.groupby("period", sort=True):
        agg = g.groupby("user_id", sort=True)[hcols].mean()
        y = g.groupby("user_id", sort=True)["true_cluster"].apply(mode_int)
        user_ids = agg.index.to_numpy(dtype=np.int64)
        X = agg.to_numpy(dtype=np.float64)
        y_true = y.to_numpy(dtype=np.int64)
        out[str(period)] = (user_ids, X, y_true)
    return out


def read_city_daily_matrix(city_dir: str) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    profiles_path = os.path.join(city_dir, "profiles_daily.csv")
    df = pd.read_csv(profiles_path)
    hcols = [f"h{i}" for i in range(24)]
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df["period"] = df["date"].dt.strftime("%Y-%m-%d")

    out: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for period, g in df.groupby("period", sort=True):
        agg = g.groupby("user_id", sort=True)[hcols].mean()
        y = g.groupby("user_id", sort=True)["true_cluster"].apply(mode_int)
        user_ids = agg.index.to_numpy(dtype=np.int64)
        X = agg.to_numpy(dtype=np.float64)
        y_true = y.to_numpy(dtype=np.int64)
        out[str(period)] = (user_ids, X, y_true)
    return out


def cluster_switching_rate(prev: np.ndarray, curr: np.ndarray) -> float:
    if prev.shape[0] != curr.shape[0] or prev.shape[0] == 0:
        return float("nan")
    return float(np.mean(prev != curr))


def hungarian_min_cost(cost: np.ndarray) -> np.ndarray:
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
        raise ValueError("cost must be a square matrix")

    n = int(cost.shape[0])
    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(n + 1, dtype=np.float64)
    p = np.zeros(n + 1, dtype=np.int64)
    way = np.zeros(n + 1, dtype=np.int64)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, float("inf"), dtype=np.float64)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = int(p[j0])
            delta = float("inf")
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = float(minv[j])
                    j1 = j
            for j in range(0, n + 1):
                if used[j]:
                    u[int(p[j])] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    col_for_row = np.empty(n, dtype=np.int64)
    for j in range(1, n + 1):
        i = int(p[j])
        col_for_row[i - 1] = j - 1
    return col_for_row


def align_labels_q_centroids_by_centroids(
    prev_centroids: np.ndarray,
    curr_centroids: np.ndarray,
    curr_labels: np.ndarray,
    curr_q: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    prev_centroids = np.asarray(prev_centroids, dtype=np.float64)
    curr_centroids = np.asarray(curr_centroids, dtype=np.float64)
    curr_labels = np.asarray(curr_labels, dtype=np.int64)
    curr_q = np.asarray(curr_q, dtype=np.float64)
    if prev_centroids.shape != curr_centroids.shape:
        raise ValueError("prev_centroids and curr_centroids must have the same shape")
    if curr_q.shape[1] != prev_centroids.shape[0]:
        raise ValueError("curr_q must have shape (n_users, k)")

    cost = np.linalg.norm(prev_centroids[:, None, :] - curr_centroids[None, :, :], axis=2)
    col_for_row = hungarian_min_cost(cost)

    inv = np.empty(col_for_row.shape[0], dtype=np.int64)
    for prev_idx, curr_idx in enumerate(col_for_row.tolist()):
        inv[int(curr_idx)] = int(prev_idx)
    aligned_labels = inv[curr_labels]

    aligned_centroids = np.empty_like(curr_centroids)
    for prev_idx, curr_idx in enumerate(col_for_row.tolist()):
        aligned_centroids[int(prev_idx)] = curr_centroids[int(curr_idx)]

    aligned_q = np.empty_like(curr_q)
    for prev_idx, curr_idx in enumerate(col_for_row.tolist()):
        aligned_q[:, int(prev_idx)] = curr_q[:, int(curr_idx)]

    return aligned_labels, aligned_q, aligned_centroids


def k_neighbors_from_distance_matrix(d2: np.ndarray, k: int) -> np.ndarray:
    d2 = np.asarray(d2, dtype=np.float64)
    n = int(d2.shape[0])
    d2 = d2.copy()
    np.fill_diagonal(d2, float("inf"))
    idx = np.argpartition(d2, kth=min(k, n - 1), axis=1)[:, :k]
    A = np.zeros((n, n), dtype=np.float64)
    rows = np.arange(n)[:, None]
    A[rows, idx] = 1.0
    return A


def k_neighbors_from_similarity_matrix(sim: np.ndarray, k: int) -> np.ndarray:
    sim = np.asarray(sim, dtype=np.float64)
    n = int(sim.shape[0])
    sim = sim.copy()
    np.fill_diagonal(sim, -float("inf"))
    idx = np.argpartition(-sim, kth=min(k, n - 1), axis=1)[:, :k]
    A = np.zeros((n, n), dtype=np.float64)
    rows = np.arange(n)[:, None]
    A[rows, idx] = 1.0
    return A


def row_standardize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return (X - mu) / (sd + 1e-12)


def normalize_adjacency(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    A = (A + A.T) / 2.0
    A[A > 0] = 1.0
    A = A + np.eye(A.shape[0], dtype=np.float64)
    deg = A.sum(axis=1)
    return A / (deg[:, None] + 1e-12)


def build_fused_graph(
    X: np.ndarray,
    knn: int,
    shift_band: int,
    w_geo: float = 1.0,
    w_align: float = 1.0,
    w_corr: float = 1.0,
) -> np.ndarray:
    Xs = row_standardize(X)
    d2 = pairwise_sqeuclidean(Xs)
    A_geo = k_neighbors_from_distance_matrix(d2, knn)

    sim_max = None
    for s in range(-shift_band, shift_band + 1):
        X_shift = np.roll(Xs, shift=s, axis=1)
        sim = Xs @ X_shift.T
        sim_max = sim if sim_max is None else np.maximum(sim_max, sim)
    A_align = k_neighbors_from_similarity_matrix(sim_max, knn)

    corr = (Xs @ Xs.T) / max(float(Xs.shape[1] - 1), 1.0)
    A_corr = k_neighbors_from_similarity_matrix(np.abs(corr), knn)

    A = w_geo * A_geo + w_align * A_align + w_corr * A_corr
    return normalize_adjacency(A)


def fit_pca(X: np.ndarray, dim: int) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    cov = (Xc.T @ Xc) / max(float(Xc.shape[0] - 1), 1.0)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    r = min(int(dim), int(eigvecs.shape[1]))
    comps = eigvecs[:, order[:r]].T.copy()
    return mean, comps


def transform_pca(X: np.ndarray, mean: np.ndarray, comps: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    comps = np.asarray(comps, dtype=np.float64)
    Xc = X - mean
    return Xc @ comps.T


def compute_embedding(X: np.ndarray, A_norm: np.ndarray, smooth_steps: int) -> np.ndarray:
    H = np.asarray(X, dtype=np.float64)
    for _ in range(int(smooth_steps)):
        H = A_norm @ H
    return H


def soft_assignment_student_t(Z: np.ndarray, centroids: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    Z = np.asarray(Z, dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64)
    dist = np.sum(Z * Z, axis=1)[:, None] - 2.0 * (Z @ centroids.T) + np.sum(centroids * centroids, axis=1)[None, :]
    q = 1.0 / (1.0 + dist / alpha)
    q = q ** ((alpha + 1.0) / 2.0)
    q = q / (q.sum(axis=1, keepdims=True) + 1e-12)
    return q


def target_distribution(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    w = (q * q) / (q.sum(axis=0, keepdims=True) + 1e-12)
    p = w / (w.sum(axis=1, keepdims=True) + 1e-12)
    return p


def apply_temporal_consistency(p: np.ndarray, q_prev: np.ndarray, lam: float) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    q_prev = np.asarray(q_prev, dtype=np.float64)
    lam = float(lam)
    if lam <= 0:
        q = p
    else:
        a = 1.0 / (1.0 + lam)
        b = lam / (1.0 + lam)
        q = (p**a) * (q_prev**b)
    q = q / (q.sum(axis=1, keepdims=True) + 1e-12)
    return q


def ts_from_assignments(prev_q: np.ndarray, curr_q: np.ndarray) -> float:
    prev_q = np.asarray(prev_q, dtype=np.float64)
    curr_q = np.asarray(curr_q, dtype=np.float64)
    if prev_q.shape != curr_q.shape or prev_q.shape[0] == 0:
        return float("nan")
    p = np.clip(prev_q, 1e-12, 1.0)
    q = np.clip(curr_q, 1e-12, 1.0)
    p = p / (p.sum(axis=1, keepdims=True) + 1e-12)
    q = q / (q.sum(axis=1, keepdims=True) + 1e-12)
    return float(np.mean(np.sum(p * (np.log(p) - np.log(q)), axis=1)))


def one_hot_smooth(labels: np.ndarray, k: int, eps: float = 1e-3) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    n = int(labels.shape[0])
    q = np.full((n, int(k)), eps / float(k), dtype=np.float64)
    q[np.arange(n), labels] += 1.0 - eps
    q = q / (q.sum(axis=1, keepdims=True) + 1e-12)
    return q


def run_dynec(
    Z: np.ndarray,
    k: int,
    seed: int,
    lam: float,
    prev_centroids: Optional[np.ndarray],
    prev_q: Optional[np.ndarray],
    max_iter: int = 50,
    tol: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Z = np.asarray(Z, dtype=np.float64)
    n = Z.shape[0]
    rng = np.random.default_rng(seed)

    if prev_centroids is None:
        centroids = kmeans_pp_init(rng, Z, k)
    else:
        centroids = np.asarray(prev_centroids, dtype=np.float64).copy()

    if prev_q is None:
        q_prev = np.full((n, k), 1.0 / float(k), dtype=np.float64)
    else:
        q_prev = np.asarray(prev_q, dtype=np.float64)

    for _ in range(int(max_iter)):
        q_raw = soft_assignment_student_t(Z, centroids, alpha=1.0)
        p = target_distribution(q_raw)
        q = apply_temporal_consistency(p, q_prev, lam)

        denom = q.sum(axis=0) + 1e-12
        new_centroids = (q.T @ Z) / denom[:, None]

        shift = float(np.linalg.norm(new_centroids - centroids))
        centroids = new_centroids
        if shift <= tol:
            break

    labels = np.argmax(q, axis=1).astype(np.int64)
    return labels, q, centroids


def list_city_dirs(data_root: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for name in sorted(os.listdir(data_root)):
        d = os.path.join(data_root, name)
        if not os.path.isdir(d):
            continue
        if os.path.exists(os.path.join(d, "profiles_daily.csv")):
            out.append((name, d))
    return out


def evaluate_city_with_method(
    method: str,
    city_name: str,
    city_dir: str,
    time_step: str,
    k: int,
    seed: int,
    max_iter: int,
    embed_dim: int,
    knn: int,
    shift_band: int,
    smooth_steps: int,
    lam: float,
    compute_internal_metrics: bool,
    internal_metrics_sample: int,
) -> List[StepResult]:
    if time_step == "day":
        periods = read_city_daily_matrix(city_dir)
    elif time_step == "month":
        periods = read_city_monthly_matrix(city_dir)
    else:
        raise ValueError(f"unknown time_step: {time_step}")
    results: List[StepResult] = []
    prev_pred: Optional[np.ndarray] = None
    prev_users: Optional[np.ndarray] = None
    prev_centroids: Optional[np.ndarray] = None
    prev_q: Optional[np.ndarray] = None

    pca_mean = None
    pca_comps = None
    if method == "dynec":
        X_all = np.vstack([row_standardize(periods[p][1]) for p in sorted(periods.keys())])
        pca_mean, pca_comps = fit_pca(X_all, embed_dim)

    for i, period in enumerate(sorted(periods.keys())):
        users, X, y_true = periods[period]
        X_base = row_standardize(X)
        if method == "kmeans":
            pred, centroids = run_kmeans(X_base, k=k, seed=seed + 17 * i, max_iter=max_iter)
            q = one_hot_smooth(pred, k=k, eps=1e-3)
        elif method == "dynec":
            A = build_fused_graph(X_base, knn=knn, shift_band=shift_band)
            H = compute_embedding(X_base, A, smooth_steps=smooth_steps)
            Z = transform_pca(H, pca_mean, pca_comps)
            pred, q, centroids = run_dynec(
                Z,
                k=k,
                seed=seed + 17 * i,
                lam=lam,
                prev_centroids=prev_centroids,
                prev_q=prev_q,
                max_iter=max_iter,
            )
        else:
            raise ValueError(f"unknown method: {method}")

        ari = adjusted_rand_index(y_true, pred)
        sil = None
        dbi = None
        if compute_internal_metrics:
            metric_seed = int(seed + 17 * i + 99991)
            if method == "kmeans":
                idx = sample_indices(X_base.shape[0], internal_metrics_sample, metric_seed)
                sil = silhouette_score_euclidean(X_base[idx], pred[idx])
                dbi = davies_bouldin_index(X_base[idx], pred[idx])
            else:
                idx = sample_indices(Z.shape[0], internal_metrics_sample, metric_seed)
                sil = silhouette_score_euclidean(Z[idx], pred[idx])
                dbi = davies_bouldin_index(Z[idx], pred[idx])

        csr = None
        ts = None
        n_users_overlap = None
        if prev_pred is not None and prev_users is not None and prev_centroids is not None and prev_q is not None:
            pred_aligned, q_aligned, centroids_aligned = align_labels_q_centroids_by_centroids(
                prev_centroids, centroids, pred, q
            )

            common, prev_idx, curr_idx = np.intersect1d(prev_users, users, assume_unique=False, return_indices=True)
            n_users_overlap = int(common.shape[0])
            if n_users_overlap >= 2:
                csr = cluster_switching_rate(prev_pred[prev_idx], pred_aligned[curr_idx])
                ts = ts_from_assignments(prev_q[prev_idx], q_aligned[curr_idx])

            pred = pred_aligned
            q = q_aligned
            centroids = centroids_aligned

        results.append(
            StepResult(
                method=method,
                city=city_name,
                period=period,
                seed=int(seed),
                n_users=int(users.shape[0]),
                n_users_overlap=None if n_users_overlap is None else int(n_users_overlap),
                k=int(k),
                ari=float(ari),
                silhouette=None if sil is None else float(sil),
                dbi=None if dbi is None else float(dbi),
                csr=None if csr is None else float(csr),
                ts=None if ts is None else float(ts),
            )
        )
        prev_pred = pred
        prev_users = users
        prev_centroids = centroids
        prev_q = q

    return results


def write_outputs(out_dir: str, rows: List[StepResult]) -> Dict[str, object]:
    os.makedirs(out_dir, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(out_dir, f"results_{ts}.csv")
    json_path = os.path.join(out_dir, f"summary_{ts}.json")

    df = pd.DataFrame([r.__dict__ for r in rows])
    df.to_csv(csv_path, index=False, encoding="utf-8")

    summary: Dict[str, object] = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "csv": csv_path,
        "results": {},
        "by_period": {},
        "by_seed": {},
    }

    for (method, city), g in df.groupby(["method", "city"]):
        def _stat(col: str) -> Dict[str, Optional[float]]:
            s = pd.to_numeric(g[col], errors="coerce").dropna()
            if len(s) == 0:
                return {"mean": None, "std": None, "n": 0}
            return {"mean": float(s.mean()), "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0, "n": int(len(s))}

        if method not in summary["results"]:
            summary["results"][method] = {}
        summary["results"][method][city] = {
            "ari": _stat("ari"),
            "silhouette": _stat("silhouette"),
            "dbi": _stat("dbi"),
            "csr": _stat("csr"),
            "ts": _stat("ts"),
        }

    for (method, city, period), g in df.groupby(["method", "city", "period"]):
        def _stat_period(col: str) -> Dict[str, Optional[float]]:
            s = pd.to_numeric(g[col], errors="coerce").dropna()
            if len(s) == 0:
                return {"mean": None, "std": None, "n": 0}
            return {"mean": float(s.mean()), "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0, "n": int(len(s))}

        if method not in summary["by_period"]:
            summary["by_period"][method] = {}
        if city not in summary["by_period"][method]:
            summary["by_period"][method][city] = {}
        summary["by_period"][method][city][period] = {
            "ari": _stat_period("ari"),
            "silhouette": _stat_period("silhouette"),
            "dbi": _stat_period("dbi"),
            "csr": _stat_period("csr"),
            "ts": _stat_period("ts"),
        }

    for (method, city, seed), g in df.groupby(["method", "city", "seed"]):
        def _stat_seed(col: str) -> Dict[str, Optional[float]]:
            s = pd.to_numeric(g[col], errors="coerce").dropna()
            if len(s) == 0:
                return {"mean": None, "std": None, "n": 0}
            return {"mean": float(s.mean()), "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0, "n": int(len(s))}

        if method not in summary["by_seed"]:
            summary["by_seed"][method] = {}
        if city not in summary["by_seed"][method]:
            summary["by_seed"][method][city] = {}
        summary["by_seed"][method][city][str(int(seed))] = {
            "ari": _stat_seed("ari"),
            "silhouette": _stat_seed("silhouette"),
            "dbi": _stat_seed("dbi"),
            "csr": _stat_seed("csr"),
            "ts": _stat_seed("ts"),
        }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    summary["json"] = json_path
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Sichuan2024Dataset")))
    p.add_argument("--out-dir", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "results")))
    p.add_argument("--method", choices=["kmeans", "dynec", "both"], default="both")
    p.add_argument("--time-step", choices=["day", "month"], default="day")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260217)
    p.add_argument("--seeds", default=None)
    p.add_argument("--max-iter", type=int, default=50)
    p.add_argument("--embed-dim", type=int, default=16)
    p.add_argument("--knn", type=int, default=12)
    p.add_argument("--shift-band", type=int, default=2)
    p.add_argument("--smooth-steps", type=int, default=2)
    p.add_argument("--lambda-temp", type=float, default=0.1)
    p.add_argument("--city", default=None, help="Specific city to process (e.g., City-A)")
    p.add_argument("--skip-internal-metrics", action="store_true")
    p.add_argument("--internal-metrics-sample", type=int, default=0)
    args = p.parse_args()

    data_root = os.path.abspath(args.data_root)
    rows: List[StepResult] = []

    methods = ["kmeans", "dynec"] if args.method == "both" else [args.method]
    seeds = parse_seeds(args.seeds, args.seed)
    compute_internal_metrics = not bool(args.skip_internal_metrics)
    internal_metrics_sample = int(args.internal_metrics_sample)

    for seed in seeds:
        for method in methods:
            for city_name, city_dir in list_city_dirs(data_root):
                if args.city and city_name != args.city:
                    continue
                print(f"Processing {city_name} with method {method} (seed={seed})")
                rows.extend(
                    evaluate_city_with_method(
                        method=method,
                        city_name=city_name,
                        city_dir=city_dir,
                        time_step=args.time_step,
                        k=args.k,
                        seed=int(seed),
                        max_iter=args.max_iter,
                        embed_dim=args.embed_dim,
                        knn=args.knn,
                        shift_band=args.shift_band,
                        smooth_steps=args.smooth_steps,
                        lam=args.lambda_temp,
                        compute_internal_metrics=compute_internal_metrics,
                        internal_metrics_sample=internal_metrics_sample,
                    )
                )

    summary = write_outputs(args.out_dir, rows)
    summary["config"] = {
        "data_root": data_root,
        "out_dir": os.path.abspath(args.out_dir),
        "method": args.method,
        "time_step": args.time_step,
        "k": int(args.k),
        "seeds": [int(x) for x in seeds],
        "max_iter": int(args.max_iter),
        "embed_dim": int(args.embed_dim),
        "knn": int(args.knn),
        "shift_band": int(args.shift_band),
        "smooth_steps": int(args.smooth_steps),
        "lambda_temp": float(args.lambda_temp),
        "compute_internal_metrics": bool(compute_internal_metrics),
        "internal_metrics_sample": int(internal_metrics_sample),
    }
    with open(summary["json"], "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
