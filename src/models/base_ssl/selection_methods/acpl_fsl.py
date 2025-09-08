# -----------------------------------------------------------------------------
# ACPL-FSL: reconstruction-based episodic logits + SAPS/EBSS ensemble clustering
# - acpl_fsl_logit:   produces per-query class probabilities; SAPS+EBSS refine
# - ACPLRefiner:      per-sample weights/masks for training loss (SAPS+EBSS)
# - acpl_fsl_select:  selection with "no-regret" SAPS/EBSS re-ranking
#
# Implementation details (as used in our experiments):
# * Heads: DBSCAN, kNN, k-means++ (sklearn++ seeding; FAISS Lloyd ),
#          one-hop graph via k-NNG (sklearn or density/mutual KNN).
# * SAPS:  head reliabilities γ = α × β (α: silhouette; β: discrepancy/memory).
# * EBSS:  weighted co-pair agreement v for soft/hard reliability control.
# * Logits: reconstruction distribution blended (gently) with SAPS+EBSS consensus.
# -----------------------------------------------------------------------------



import numpy as np
import torch
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

# -----------------------------
#  FAISS load
# -----------------------------
def _try_faiss():
    try:
        import faiss  # faiss-cpu or faiss-gpu
        return faiss
    except Exception:
        return None

_FAISS = _try_faiss()

# ----------------- common helpers -----------------
def _l2_normalize_np(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(n, eps, None)

def _pair_matrix(labels: np.ndarray) -> np.ndarray:
    """Mk[i,j] = 1 iff y_i == y_j and y_i != -1 (diagonal is 0 here)."""
    N = len(labels)
    M = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        li = labels[i]
        if li < 0:
            continue
        eq = (labels == li)
        eq[i] = False
        M[i, eq] = 1.0
    return M

def _safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette in [0,1]; safe fallbacks when degenerate."""
    try:
        from sklearn.metrics import silhouette_score
        mask = (labels >= 0)
        if mask.sum() < 3:
            return 0.0
        if len(np.unique(labels[mask])) < 2:
            return 0.0
        # silhouette_score returns [-1,1]; shift to [0,1]
        s = float(silhouette_score(X[mask], labels[mask], metric="euclidean"))
        return 0.5 * (s + 1.0)
    except Exception:
        return 0.0

def _sklearnpp_init(X: np.ndarray, n_clusters: int, random_state: int = 1996):
    try:
        from sklearn.cluster import kmeans_plusplus
        centers, _ = kmeans_plusplus(X, n_clusters=n_clusters, random_state=random_state)
        return centers.astype(np.float32, copy=False)
    except Exception:
        rs = np.random.RandomState(random_state)
        idx = rs.choice(X.shape[0], n_clusters, replace=False)
        return X[idx].astype(np.float32, copy=False)

def _faiss_lloyd(X: np.ndarray, init_centers: np.ndarray, n_iter: int = 20):
    """Lloyd iterations using FAISS for fast nearest-centroid search."""
    d = X.shape[1]
    C = init_centers.shape[0]
    centers = init_centers.astype(np.float32).copy()
    index = _FAISS.IndexFlatL2(d)
    for _ in range(max(1, n_iter)):
        index.reset()
        index.add(centers)
        _, assign = index.search(X.astype(np.float32), 1)  # (N,1)
        assign = assign.ravel()
        for c in range(C):
            m = (assign == c)
            if np.any(m):
                centers[c] = X[m].mean(axis=0).astype(np.float32)
    return centers, assign

def kmeans_with_sklearnpp_faiss(
    X: np.ndarray,
    n_clusters: int,
    *,
    random_state: int = 1996,
    n_iter: int = 20,
    use_faiss: bool = True,
):
    """K-means with sklearn++ seeding; FAISS Lloyd ; falls back to sklearn."""
    if use_faiss and (_FAISS is not None):
        init_c = _sklearnpp_init(X, n_clusters, random_state)
        centers, labels = _faiss_lloyd(X, init_c, n_iter=n_iter)
        return centers, labels
    else:
        from sklearn.cluster import KMeans
        km = KMeans(
            n_clusters=n_clusters,
            n_init=1,
            init=_sklearnpp_init(X, n_clusters, random_state),
            random_state=random_state,
            max_iter=max(100, n_iter),
        )
        km.fit(X)
        return km.cluster_centers_.astype(np.float32), km.labels_.astype(np.int64)

def build_knng_cdp(
    X: np.ndarray,
    k: int,
    *,
    mode: str = "sklearn",   # "cdp" | "mutual" | "sklearn"
    prune_alpha: float = 0.5,
    use_faiss: bool = True,
):
    """
    Density-aware or mutual kNN graph; or sklearn kneighbors_graph.
    Returns symmetric similarities in [0,1].
    """
    N, d = X.shape
    if mode not in ("cdp", "mutual"):
        # sklearn baseline
        from sklearn.neighbors import kneighbors_graph
        G = kneighbors_graph(X, n_neighbors=min(k, max(1, N-1)), mode="distance", include_self=False)
        G = G.toarray().astype(np.float32)
        G[G > 0] = 1.0 / (G[G > 0] + 1e-8)
        if G.max() > 0:
            G /= G.max()
        return G

    # kNN (FAISS or sklearn)
    if use_faiss and (_FAISS is not None):
        index = _FAISS.IndexFlatL2(d)
        index.add(X.astype(np.float32))
        dist2, idx = index.search(X.astype(np.float32), min(k + 1, N))
        dist = np.sqrt(np.maximum(dist2, 0.0))
        idx = idx[:, 1:]
        dist = dist[:, 1:]
    else:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=min(k + 1, N), metric="euclidean")
        nn.fit(X)
        dist, idx = nn.kneighbors(X, return_distance=True)
        idx = idx[:, 1:]
        dist = dist[:, 1:]

    sim = 1.0 / (dist + 1e-8)  # (N,k)

    # mutual-kNN symmetric similarity
    W = np.zeros((N, N), dtype=np.float32)
    rows = np.repeat(np.arange(N), idx.shape[1])
    cols = idx.ravel()
    vals = sim.ravel()
    W[rows, cols] = np.maximum(W[rows, cols], vals)
    mutual = ((W > 0) & (W.T > 0)).astype(np.float32)
    W = np.minimum(W, W.T) * mutual

    if mode == "mutual":
        if W.max() > 0:
            W /= W.max()
        return W

    # CDP-style density pruning
    mean_d = dist.mean(axis=1)
    rho = 1.0 / (mean_d + 1e-8)
    node_thr = np.median(sim, axis=1) * prune_alpha

    keep = np.zeros_like(W, dtype=bool)
    I, J = np.nonzero(W)
    for i, j in zip(I, J):
        sij = W[i, j]
        if (sij >= node_thr[i]) and (sij >= node_thr[j]) and (min(rho[i], rho[j]) >= np.median(rho)):
            keep[i, j] = True
            keep[j, i] = True
    W = np.where(keep, W, 0.0).astype(np.float32)
    if W.max() > 0:
        W /= W.max()
    return W



# ---------------------------------------------------------
# SAPS + EBSS refiner (training-time weights/masks)
# ---------------------------------------------------------
class ACPLRefiner:
    """
    Stateful SAPS+EBSS refiner.
    Produces per-query weights (soft) or masks (hard) to modulate the ACPL loss.
    """
    def __init__(
        self,
        *,
        dbscan_eps: float = 0.6,
        dbscan_min_samples: int = 5,
        knn_k: int = 5,
        graph_k: int = 10,
        km_n_init: int = 10,
        km_random_state: int = 1996,
        l2_normalize: bool = True,
        standardize_for_dbscan: bool = True,
        mode: str = "soft",          # "soft" | "hard" | "off"
        eta: float = 0.08,           # >0 ensures it contributes
        clip: float = 0.15,          # clamp around 1.0 for soft mode
        lambda_start: float = 0.10,  # hard-mode schedule
        lambda_end: float = 0.50,
        total_steps: int = 20000,
        # k-NNG & FAISS toggles
        knng_builder: str = "sklearn",   # "cdp" | "mutual" | "sklearn"
        use_faiss_kmeans: bool = False,
        use_faiss_knn: bool = True,
        kmeans_iter: int = 20,
        cdp_alpha: float = 0.5,
    ):
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.knn_k = knn_k
        self.graph_k = graph_k
        self.km_n_init = km_n_init
        self.km_random_state = km_random_state
        self.l2_normalize = l2_normalize
        self.standardize_for_dbscan = standardize_for_dbscan
        self.mode = mode.lower()
        self.eta = eta
        self.clip = clip
        self.lambda_start = lambda_start
        self.lambda_end = lambda_end
        self.total_steps = max(1, int(total_steps))
        self.step = 0
        self.prev_Ms = None  # numpy array (Nq,Nq)

        self.knng_builder = knng_builder.lower()
        self.use_faiss_kmeans = use_faiss_kmeans
        self.use_faiss_knn = use_faiss_knn and (_FAISS is not None)
        self.kmeans_iter = kmeans_iter
        self.cdp_alpha = cdp_alpha

        try:
            import sklearn  # noqa
            self._have_sk = True
        except Exception:
            self._have_sk = False

    @torch.no_grad()
    def __call__(self, S, Q, S_labels, n_classes):
        """
        Returns:
          weights: (Nq,) torch tensor around 1.0 (soft) or ones (off/hard)
          mask:    (Nq,) bool torch tensor or None
        """
        device = Q.device
        Nq = Q.shape[0]
        if (not self._have_sk) or (self.mode == "off") or (self.eta <= 0.0):
            return torch.ones(Nq, device=device), None

        from sklearn.cluster import DBSCAN
        from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
        from sklearn.preprocessing import StandardScaler

        S_np = S.detach().cpu().float().numpy()
        Q_np = Q.detach().cpu().float().numpy()
        Sy   = S_labels.detach().cpu().long().numpy()
        C = int(n_classes); Ns = S_np.shape[0]

        if self.l2_normalize:
            S_np = _l2_normalize_np(S_np); Q_np = _l2_normalize_np(Q_np)
        A = np.concatenate([S_np, Q_np], axis=0)

        # DBSCAN
        A_scaled = StandardScaler().fit_transform(A) if self.standardize_for_dbscan else A
        db = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples).fit(A_scaled)
        db_all = db.labels_; db_q = db_all[-Nq:]
        cl2cls = {}
        for cid in np.unique(db_all):
            if cid < 0: continue
            mS = (db_all[:Ns] == cid)
            if mS.any():
                counts = np.bincount(Sy[mS], minlength=C)
                cl2cls[cid] = int(counts.argmax())
        y_db = np.array([cl2cls.get(cid, -1) for cid in db_q], dtype=np.int64)

        # KMeans centers (FAISS or sklearn)
        centers, _ = kmeans_with_sklearnpp_faiss(
            A.astype(np.float32, copy=False),
            n_clusters=C,
            random_state=self.km_random_state,
            n_iter=self.kmeans_iter,
            use_faiss=self.use_faiss_kmeans,
        )
        dists = ((Q_np[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        y_km = dists.argmin(axis=1).astype(np.int64)

        # kNN
        knn = KNeighborsClassifier(n_neighbors=min(self.knn_k, Ns), weights="distance").fit(S_np, Sy)
        y_kn = knn.predict(Q_np)

        # Graph: CDP/mutual/sklearn
        if self.knng_builder in ("cdp", "mutual"):
            W = build_knng_cdp(
                A.astype(np.float32, copy=False),
                k=min(self.graph_k, max(1, A.shape[0]-1)),
                mode=self.knng_builder,
                prune_alpha=self.cdp_alpha,
                use_faiss=self.use_faiss_knn,
            )
            W_QS = W[Ns:, :Ns]
            gs = np.zeros((Nq, C), dtype=np.float32)
            for c in range(C):
                sc = (Sy == c).astype(np.float32)
                gs[:, c] = (W_QS * sc).sum(axis=1)
        else:
            G = kneighbors_graph(A, n_neighbors=min(self.graph_k, max(1, A.shape[0]-1)),
                                 mode="distance", include_self=False).toarray().astype(np.float32)
            G[G > 0] = 1.0 / (G[G > 0] + 1e-8)
            W_QS = G[Ns:, :Ns]
            gs = np.zeros((Nq, C), dtype=np.float32)
            for c in range(C):
                sc = (Sy == c).astype(np.float32)
                gs[:, c] = (W_QS * sc).sum(axis=1)

        y_gr = gs.argmax(axis=1).astype(np.int64)

        # Pair matrices
        M_db = _pair_matrix(y_db)
        M_km = _pair_matrix(y_km)
        M_kn = _pair_matrix(y_kn)
        M_gr = _pair_matrix(y_gr)
        Ms   = [M_db, M_km, M_kn, M_gr]
        Ys   = [y_db,  y_km,  y_kn,  y_gr]

        # SAPS α
        alphas = np.array([_safe_silhouette(Q_np, y) for y in Ys], dtype=np.float32)

        # β against previous selected pairs
        if self.prev_Ms is None:
            betas = np.ones_like(alphas)
        else:
            triu = np.triu_indices(Nq, 1)
            betas = []
            for Mk in Ms:
                if triu[0].size == 0:
                    betas.append(1.0); continue
                diff = np.logical_xor(Mk[triu].astype(bool), self.prev_Ms[triu].astype(bool)).mean()
                betas.append(1.0 - float(diff))
            betas = np.array(betas, dtype=np.float32)

        gammas = alphas * betas
        if gammas.sum() <= 1e-8:
            gammas[:] = 1.0
        gammas = gammas / np.clip(gammas.sum(), 1e-8, None)

        s_idx = int(np.argmax(gammas))
        self.prev_Ms = [M_db, M_km, M_kn, M_gr][s_idx].copy()

        # EBSS agreement
        O_hat = gammas[0] * M_db + gammas[1] * M_km + gammas[2] * M_kn + gammas[3] * M_gr
        deg   = np.clip(self.prev_Ms.sum(axis=1), 1.0, None)
        v     = (O_hat * self.prev_Ms).sum(axis=1) / deg   # (Nq,)

        if self.mode == "soft":
            z = (v - v.mean()) / (v.std() + 1e-8)
            w = 1.0 + self.eta * z
            w = np.clip(w, 1.0 - self.clip, 1.0 + self.clip)
            w = torch.from_numpy(w).to(device=Q.device, dtype=torch.float32)
            self.step = min(self.step + 1, self.total_steps)
            return w, None

        if self.mode == "hard":
            t = self.step
            lam = self.lambda_start + (self.lambda_end - self.lambda_start) * (t / float(self.total_steps))
            k = max(1, int(round(float(Nq) * float(lam))))
            thr = np.partition(v, -k)[-k]
            mask_np = (v >= thr)
            mask = torch.from_numpy(mask_np).to(device=Q.device, dtype=torch.bool)
            self.step = min(self.step + 1, self.total_steps)
            return torch.ones(Nq, device=Q.device), mask

        return torch.ones(Nq, device=Q.device), None



#SAPS+EBSS ACPL-FSL Selection

# ===== helpers (local, self-contained) ======================================
def _softmax(x):
    m = np.max(x, axis=1, keepdims=True)
    e = np.exp(x - m)
    s = np.sum(e, axis=1, keepdims=True)
    return e / np.clip(s, 1e-8, None)

def _pair_matrix(y):
    N = len(y)
    M = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        yi = y[i]
        if yi < 0: 
            continue
        same = (y == yi)
        same[i] = False
        M[i, same] = 1.0
    return M

def _safe_silhouette(X, labels):
    try:
        from sklearn.metrics import silhouette_score
        m = (labels >= 0)
        if m.sum() < 3: return 0.0
        if len(np.unique(labels[m])) < 2: return 0.0
        return float(silhouette_score(X[m], labels[m], metric="euclidean"))
    except Exception:
        return 0.0

def _hungarian_centroid_to_class(C, P):
    # map k-means centers to support prototypes P using Hungarian
    try:
        from scipy.optimize import linear_sum_assignment
        from sklearn.metrics import pairwise_distances
        M = pairwise_distances(C, P, metric='euclidean')
        r, c = linear_sum_assignment(M)
        perm = np.empty(len(c), dtype=int); perm[c] = r
        return perm
    except Exception:
        # greedy fallback
        from sklearn.metrics import pairwise_distances
        M = pairwise_distances(C, P, metric='euclidean')
        perm = np.full(P.shape[0], -1, dtype=int); used=set()
        for j in range(P.shape[0]):
            i = np.argmin([M[ii, j] if ii not in used else np.inf for ii in range(C.shape[0])])
            perm[j] = i; used.add(i)
        return perm

# FAISS core for speed; safe fallback if absent
def _try_faiss():
    try:
        import faiss
        return faiss
    except Exception:
        return None
_FAISS = _try_faiss()

def _kmeans_with_sklearnpp_faiss(A, n_clusters, *, random_state=1996, n_iter=20):
    # initialization via sklearn++ ; iterations via FAISS if available
    A = A.astype(np.float32, copy=False)
    try:
        from sklearn.cluster import kmeans_plusplus
        init_c, _ = kmeans_plusplus(A, n_clusters=n_clusters, random_state=random_state)
    except Exception:
        rs = np.random.RandomState(random_state)
        init_c = A[rs.choice(A.shape[0], n_clusters, replace=False)]

    if _FAISS is not None:
        d = A.shape[1]
        centers = init_c.astype(np.float32).copy()
        index = _FAISS.IndexFlatL2(d)
        for _ in range(max(1, int(n_iter))):
            index.reset(); index.add(centers)
            _, assign = index.search(A, 1); assign = assign.ravel()
            for c in range(n_clusters):
                m = (assign == c)
                if np.any(m):
                    centers[c] = A[m].mean(axis=0).astype(np.float32)
        return centers
    else:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_clusters, n_init=1, init=init_c, random_state=random_state,
                    max_iter=max(100, int(n_iter)))
        km.fit(A)
        return km.cluster_centers_.astype(np.float32, copy=False)

def _build_knng_cdp(A, k, *, mode="sklearn", prune_alpha=0.5, use_faiss=True):
    """
    CDP / mutual-kNN / sklearn kneighbors_graph -> symmetric similarity W in [0,1]
    """
    N, d = A.shape
    if mode not in ("cdp", "mutual"):
        from sklearn.neighbors import kneighbors_graph
        G = kneighbors_graph(A, n_neighbors=min(k, max(1, N-1)), mode='distance', include_self=False)
        W = G.toarray().astype(np.float32)
        W[W > 0] = 1.0 / (W[W > 0] + 1e-8)
        if W.max() > 0: W /= W.max()
        return W

    # compute kNN
    if use_faiss and (_FAISS is not None):
        index = _FAISS.IndexFlatL2(d)
        index.add(A.astype(np.float32))
        dist2, idx = index.search(A.astype(np.float32), min(k+1, N))
        dist = np.sqrt(np.maximum(dist2, 0.0)); idx = idx[:, 1:]; dist = dist[:, 1:]
    else:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=min(k+1, N), metric="euclidean").fit(A)
        dist, idx = nn.kneighbors(A, return_distance=True)
        idx = idx[:, 1:]; dist = dist[:, 1:]
    sim = 1.0 / (dist + 1e-8)

    # mutual-kNN
    W = np.zeros((N, N), dtype=np.float32)
    rows = np.repeat(np.arange(N), idx.shape[1]); cols = idx.ravel(); vals = sim.ravel()
    W[rows, cols] = np.maximum(W[rows, cols], vals)
    mutual = ((W > 0) & (W.T > 0)).astype(np.float32)
    W = np.minimum(W, W.T) * mutual

    if mode == "mutual":
        if W.max() > 0: W /= W.max()
        return W

    # CDP prune
    mean_d = dist.mean(axis=1); rho = 1.0 / (mean_d + 1e-8)
    node_thr = np.median(sim, axis=1) * prune_alpha
    keep = np.zeros_like(W, dtype=bool)
    I, J = np.nonzero(W)
    for i, j in zip(I, J):
        sij = W[i, j]
        if (sij >= node_thr[i]) and (sij >= node_thr[j]) and (min(rho[i], rho[j]) >= np.median(rho)):
            keep[i, j] = True; keep[j, i] = True
    W = np.where(keep, W, 0.0).astype(np.float32)
    if W.max() > 0: W /= W.max()
    return W

# =====  ACPL-FSL select with SAPS+EBSS ===========
def acpl_fsl_select(episode_dict,
                    *,
                    n_loops=10,
                    lam=0.01,
                    tau=0.10,
                    # enforced SAPS/EBSS contribution
                    alpha_min=0.02,          # <-- ALWAYS at least this much of our method
                    alpha_max=0.08,          # gentle upper bound (keeps baseline shape)
                    delta_guard=0.05,        # no-regret guard threshold
                    # heads config
                    dbscan_eps=0.6,
                    dbscan_min_samples=5,
                    knn_k=5,
                    graph_k=10,
                    km_random_state=1996,
                    kmeans_iter=20,
                    knng_mode="cdp",         # "cdp" | "mutual" | "sklearn"
                    cdp_alpha=0.5,
                    use_faiss_kmeans=True,
                    use_faiss_knn=True,
                    standardize_for_dbscan=True):
    """
    Testing-time selector with guaranteed SAPS+EBSS influence.

    - Sets episode_dict['unlabeled']['labels']
    - Returns np.arange(U.shape[0])
    """

    # ---------------------- unpack ----------------------
    S = episode_dict["support_so_far"]["samples"].astype(np.float32).copy()
    Q = episode_dict["query"]["samples"].astype(np.float32).copy()      # not used here
    U = episode_dict["unlabeled"]["samples"].astype(np.float32).copy()

    if U.size == 0:
        episode_dict["unlabeled"]["labels"] = np.zeros((0,), dtype=np.int64)
        return np.arange(0, dtype=np.int64)

    S_labels = episode_dict["support_so_far"]["labels"].astype(np.int64).copy()
    Ns, d   = S.shape
    Nu, _   = U.shape
    C       = int(S_labels.max() + 1)

    A = np.concatenate([S, U], axis=0).astype(np.float32)
    centers = np.zeros((C, d), dtype=np.float32)
    for c in range(C):
        m = (S_labels == c)
        centers[c] = S[m].mean(axis=0) if m.any() else S.mean(axis=0)

    prev_mean = np.inf
    lam_eps = max(1e-8, float(lam))

    for _ in range(int(n_loops)):
        blocks = []
        for c in range(C):
            Sc = S.T[:, S_labels == c]            # (d, k)
            if Sc.ndim == 1: Sc = Sc.reshape(d, 1)
            if Sc.shape[1] == 1:                  # avoid singular XtX
                Sc = np.tile(Sc, (1, 2))
            blocks.append(Sc)
            blocks.append(centers[c].reshape(-1, 1))
        X  = np.concatenate(blocks, axis=1)       # (d, K)
        Xt = X.T
        XtX = Xt @ X
        K  = XtX.shape[0]
        XtX = XtX + lam_eps * np.eye(K, dtype=np.float32)

        try:
            P = np.linalg.solve(XtX, Xt)          # (K, d)
        except np.linalg.LinAlgError:
            P = np.linalg.pinv(XtX, rcond=1e-6) @ Xt

        beta   = P @ A.T                           # (K, N)
        beta   = beta.reshape(C, -1, Ns + Nu)      # (C, shots+1, N)
        code_x = np.zeros((C, d, Ns + Nu), dtype=np.float32)
        Xc     = X.T.reshape(C, -1, d)
        for c in range(C):
            code_x[c] = Xc[c].T @ beta[c]         # (d, N)

        err  = A.T - code_x                        # (d, N) per-class coded
        dist = np.sum(err ** 2, axis=1)            # (C, N)
        A_lab = dist.T.argmin(axis=1)              # (N,)

        # update centers from A assignments
        for c in range(C):
            m = (A_lab == c)
            if np.any(m):
                centers[c] = A[m].mean(axis=0).astype(np.float32)

        cur = float(dist.mean())
        if not np.isfinite(cur) or abs(prev_mean - cur) <= 1e-9:
            break
        prev_mean = cur

    U_dist   = dist.T[-Nu:]                        # (Nu, C)
    U_logits = -np.log(np.clip(U_dist, 1e-12, None))
    P_rec    = _softmax(U_logits / max(tau, 1e-8)) # (Nu, C)

    # ---------------------- 2) Heads + SAPS/EBSS (MANDATORY) -------------------
    try:
        from sklearn.cluster import DBSCAN
        from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import pairwise_distances

        # (a) prototypes from S for center↔class mapping
        protos = np.stack([S[S_labels == c].mean(axis=0) for c in range(C)], axis=0)

        # (b) DBSCAN over A 
        A_scaled = StandardScaler().fit_transform(A) if standardize_for_dbscan else A
        db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(A_scaled)
        db_all = db.labels_; db_u = db_all[-Nu:]
        cl2cls = {}
        for cid in np.unique(db_all):
            if cid < 0: 
                continue
            mS = (db_all[:Ns] == cid)
            if mS.any():
                counts = np.bincount(S_labels[mS], minlength=C)
                cl2cls[cid] = int(counts.argmax())
        y_db = np.array([cl2cls.get(cid, -1) for cid in db_u], dtype=np.int64)
        P_db = np.zeros((Nu, C), dtype=np.float32)
        ok   = (y_db >= 0)
        P_db[ok, y_db[ok]] = 1.0

        # (c) kNN (distance weighted) from S->U
        knn = KNeighborsClassifier(n_neighbors=min(knn_k, max(1, Ns)), weights="distance").fit(S, S_labels)
        P_kn = knn.predict_proba(U).astype(np.float32)

        # (d) k-means with sklearn++ init + FAISS iterations
        Ckm = _kmeans_with_sklearnpp_faiss(A, n_clusters=C, random_state=km_random_state, n_iter=kmeans_iter)
        # map centers to classes using Hungarian against prototypes
        perm = _hungarian_centroid_to_class(Ckm, protos)
        # distances U -> class-aligned centers
        d_km_all = ((U[:, None, :] - Ckm[None, :, :]) ** 2).sum(axis=2)  # (Nu,C_centers)
        d_km = np.stack([d_km_all[:, perm[c]] for c in range(C)], axis=1)
        P_km = _softmax(-d_km)

        # (e) one-hop k-NN graph vote (CDP/mutual/sklearn)
        W = _build_knng_cdp(A, k=min(graph_k, max(1, A.shape[0]-1)),
                            mode=knng_mode, prune_alpha=cdp_alpha, use_faiss=use_faiss_knn)
        W_US = W[Ns:, :Ns]
        graph_scores = np.zeros((Nu, C), dtype=np.float32)
        for c in range(C):
            sc = (S_labels == c).astype(np.float32)
            graph_scores[:, c] = (W_US * sc).sum(axis=1)
        P_gr = graph_scores / np.clip(graph_scores.sum(axis=1, keepdims=True), 1e-8, None)

        # --- SAPS: α × β reliabilities for heads ---
        y_kn = P_kn.argmax(axis=1).astype(np.int64)
        y_km = P_km.argmax(axis=1).astype(np.int64)
        y_gr = P_gr.argmax(axis=1).astype(np.int64)
        M_db = _pair_matrix(y_db)
        M_kn = _pair_matrix(y_kn)
        M_km = _pair_matrix(y_km)
        M_gr = _pair_matrix(y_gr)

        # base fuse (unweighted) to measure discrepancy β
        P_base = P_db + P_kn + P_km + P_gr
        P_base = P_base / np.clip(P_base.sum(axis=1, keepdims=True), 1e-8, None)
        y_base = P_base.argmax(axis=1).astype(np.int64)
        M_base = _pair_matrix(y_base)

        # silhouettes α_k on U
        a_db = _safe_silhouette(U, y_db)
        a_kn = _safe_silhouette(U, y_kn)
        a_km = _safe_silhouette(U, y_km)
        a_gr = _safe_silhouette(U, y_gr)
        alphas = np.array([a_db, a_kn, a_km, a_gr], dtype=np.float32)

        # discrepancies β_k = 1 - XOR rate vs M_base
        def _beta(Mk, Mb):
            N = Mk.shape[0]; iu = np.triu_indices(N, 1)
            if iu[0].size == 0: return 1.0
            return 1.0 - float(np.logical_xor(Mk[iu].astype(bool), Mb[iu].astype(bool)).mean())
        betas = np.array([_beta(M_db, M_base), _beta(M_kn, M_base),
                          _beta(M_km, M_base), _beta(M_gr, M_base)], dtype=np.float32)

        gammas = alphas * betas
        if not np.isfinite(gammas).all() or gammas.sum() <= 1e-8:
            gammas = np.ones_like(gammas, dtype=np.float32)
        gammas = gammas / np.clip(gammas.sum(), 1e-8, None)

        # EBSS: co-pair agreement v_i using selected M_s
        O_hat = gammas[0] * M_db + gammas[1] * M_kn + gammas[2] * M_km + gammas[3] * M_gr
        s_idx = int(np.argmax(gammas))
        M_s   = [M_db, M_kn, M_km, M_gr][s_idx]
        deg   = np.clip(M_s.sum(axis=1), 1.0, None)
        v     = (O_hat * M_s).sum(axis=1) / deg                    # (Nu,) in [0,1]

        # consensus distribution
        P_heads = (gammas[0] * P_db
                 + gammas[1] * P_kn
                 + gammas[2] * P_km
                 + gammas[3] * P_gr)
        P_heads = P_heads / np.clip(P_heads.sum(axis=1, keepdims=True), 1e-8, None)

        # enforced contribution: α_i ∈ [alpha_min, alpha_max]
        z      = (v - v.mean()) / (v.std() + 1e-8)
        sigma  = 0.5 * (1.0 + np.tanh(z))
        alpha  = alpha_min + (alpha_max - alpha_min) * sigma    # ALWAYS >= alpha_min

        # no-regret guard: if consensus way less confident than rec, clamp to alpha_min
        conf_rec   = P_rec.max(axis=1)
        conf_heads = P_heads.max(axis=1)
        weaker     = (conf_heads < (conf_rec - float(delta_guard)))
        alpha[weaker] = alpha_min

        # final blend (row-wise normalized)
        P_final = (1.0 - alpha[:, None]) * P_rec + alpha[:, None] * P_heads
        P_final = P_final / np.clip(P_final.sum(axis=1, keepdims=True), 1e-8, None)

        # expose for audit 
        episode_dict.setdefault('unlabeled', {})
        episode_dict['unlabeled']['probs_rec']   = P_rec.astype(np.float32)
        episode_dict['unlabeled']['probs_heads'] = P_heads.astype(np.float32)
        episode_dict['unlabeled']['alpha']       = alpha.astype(np.float32)
        episode_dict['unlabeled']['gammas']      = gammas.astype(np.float32)
        episode_dict['unlabeled']['v_agree']     = v.astype(np.float32)

        U_probs_final = P_final.astype(np.float32)

    except Exception:
        # if heads fail to compute, keep pure reconstruction (rare)
        U_probs_final = P_rec.astype(np.float32)

    U_pred = U_probs_final.argmax(axis=1).astype(np.int64)
    episode_dict['unlabeled']['labels'] = U_pred
    # store final probs for completeness
    episode_dict['unlabeled']['probs'] = U_probs_final
    return np.arange(U.shape[0], dtype=np.int64)








# ---------- tiny helpers (local, self-contained) ----------
def _softmax_np(x):
    m = np.max(x, axis=1, keepdims=True)
    e = np.exp(x - m)
    s = np.sum(e, axis=1, keepdims=True)
    return e / np.clip(s, 1e-8, None)

def _pair_matrix(y):
    N = len(y)
    M = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        yi = y[i]
        if yi < 0: 
            continue
        same = (y == yi)
        same[i] = False
        M[i, same] = 1.0
    return M

def _safe_silhouette(X, labels):
    try:
        from sklearn.metrics import silhouette_score
        m = (labels >= 0)
        if m.sum() < 3: return 0.0
        if len(np.unique(labels[m])) < 2: return 0.0
        return float(silhouette_score(X[m], labels[m], metric="euclidean"))
    except Exception:
        return 0.0

def _hungarian_centroid_to_class(C, P):
    try:
        from scipy.optimize import linear_sum_assignment
        from sklearn.metrics import pairwise_distances
        M = pairwise_distances(C, P, metric='euclidean')
        r, c = linear_sum_assignment(M)
        perm = np.empty(len(c), dtype=int); perm[c] = r
        return perm
    except Exception:
        from sklearn.metrics import pairwise_distances
        M = pairwise_distances(C, P, metric='euclidean')
        perm = np.full(P.shape[0], -1, dtype=int); used=set()
        for j in range(P.shape[0]):
            i = np.argmin([M[ii, j] if ii not in used else np.inf for ii in range(C.shape[0])])
            perm[j] = i; used.add(i)
        return perm

def _try_faiss():
    try:
        import faiss
        return faiss
    except Exception:
        return None
_FAISS = _try_faiss()

def _kmeans_with_sklearnpp_faiss(A, n_clusters, *, random_state=1996, n_iter=20):
    A = A.astype(np.float32, copy=False)
    try:
        from sklearn.cluster import kmeans_plusplus
        init_c, _ = kmeans_plusplus(A, n_clusters=n_clusters, random_state=random_state)
    except Exception:
        rs = np.random.RandomState(random_state)
        init_c = A[rs.choice(A.shape[0], n_clusters, replace=False)]
    if _FAISS is not None:
        d = A.shape[1]
        centers = init_c.astype(np.float32).copy()
        index = _FAISS.IndexFlatL2(d)
        for _ in range(max(1, int(n_iter))):
            index.reset(); index.add(centers)
            _, assign = index.search(A, 1); assign = assign.ravel()
            for c in range(n_clusters):
                m = (assign == c)
                if np.any(m):
                    centers[c] = A[m].mean(axis=0).astype(np.float32)
        return centers
    else:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_clusters, n_init=1, init=init_c, random_state=random_state,
                    max_iter=max(100, int(n_iter)))
        km.fit(A)
        return km.cluster_centers_.astype(np.float32, copy=False)

def _build_knng_cdp(A, k, *, mode="cdp", prune_alpha=0.5, use_faiss=True):
    N, d = A.shape
    if mode not in ("cdp", "mutual"):
        from sklearn.neighbors import kneighbors_graph
        G = kneighbors_graph(A, n_neighbors=min(k, max(1, N-1)), mode='distance', include_self=False)
        W = G.toarray().astype(np.float32)
        W[W > 0] = 1.0 / (W[W > 0] + 1e-8)
        if W.max() > 0: W /= W.max()
        return W

    if use_faiss and (_FAISS is not None):
        index = _FAISS.IndexFlatL2(d)
        index.add(A.astype(np.float32))
        dist2, idx = index.search(A.astype(np.float32), min(k+1, N))
        dist = np.sqrt(np.maximum(dist2, 0.0)); idx = idx[:, 1:]; dist = dist[:, 1:]
    else:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=min(k+1, N), metric="euclidean").fit(A)
        dist, idx = nn.kneighbors(A, return_distance=True)
        idx = idx[:, 1:]; dist = dist[:, 1:]
    sim = 1.0 / (dist + 1e-8)
    W = np.zeros((N, N), dtype=np.float32)
    rows = np.repeat(np.arange(N), idx.shape[1]); cols = idx.ravel(); vals = sim.ravel()
    W[rows, cols] = np.maximum(W[rows, cols], vals)
    mutual = ((W > 0) & (W.T > 0)).astype(np.float32)
    W = np.minimum(W, W.T) * mutual
    if mode == "mutual":
        if W.max() > 0: W /= W.max()
        return W
    # CDP pruning
    mean_d = dist.mean(axis=1); rho = 1.0 / (mean_d + 1e-8)
    node_thr = np.median(sim, axis=1) * prune_alpha
    keep = np.zeros_like(W, dtype=bool)
    I, J = np.nonzero(W)
    for i, j in zip(I, J):
        sij = W[i, j]
        if (sij >= node_thr[i]) and (sij >= node_thr[j]) and (min(rho[i], rho[j]) >= np.median(rho)):
            keep[i, j] = True; keep[j, i] = True
    W = np.where(keep, W, 0.0).astype(np.float32)
    if W.max() > 0: W /= W.max()
    return W

# ---------- SAPS&EBSS refinement ----------

@torch.no_grad()
def acpl_fsl_logit(
    S: torch.Tensor,                # (Ns, d)
    Q: torch.Tensor,                # (Nq, d)
    S_labels: torch.Tensor,         # (Ns,)
    n_classes: int,
    *,
    # reconstruction head
    n_loops: int = 10,
    lam: float = 0.01,
    tau: float = 0.1,
    eps: float = 1e-8,
    # logits refinement (ALWAYS ON if sklearn present)
    alpha_min: float = 0.02,        # min blend with consensus per sample
    alpha_max: float = 0.08,        # max blend with consensus per sample
    delta_guard: float = 0.05,      # reduce blend if consensus looks worse by this margin
    dbscan_eps: float = 0.6,
    dbscan_min_samples: int = 5,
    knn_k: int = 5,
    graph_k: int = 10,
    km_random_state: int = 1996,
    kmeans_iter: int = 20,
    knng_mode: str = "cdp",         # "cdp" | "mutual" | "sklearn"
    cdp_alpha: float = 0.5,
    use_faiss_kmeans: bool = True,  # used only if faiss available
    use_faiss_knn: bool = True,     # used only if faiss available
    standardize_for_dbscan: bool = True,
):
    """
    Returns:
      Q_probs (Nq, C)  -- probabilities in [0,1]
      Q_labels (Nq,)   -- argmax over classes

    Reconstruction logits are computed with the robust center-refinement
    iterations, then gently blended (per-sample) with a SAPS+EBSS consensus
    distribution built from three heads (DBSCAN, kNN, k-means++) and a
    k-NN graph head. The blend is small and non-destructive.
    """

    # ---- small local helpers (to avoid external dependencies being undefined) ----
    def _l2_normalize_np(X, eps=1e-12):
        n = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.clip(n, eps, None)

    def _pair_matrix(labels: np.ndarray) -> np.ndarray:
        N = len(labels)
        M = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            li = labels[i]
            if li < 0:
                continue
            eq = (labels == li)
            eq[i] = False
            M[i, eq] = 1.0
        return M

    def _safe_silhouette(Q_np, y_np) -> float:
        try:
            from sklearn.metrics import silhouette_score
            mask = (y_np >= 0)
            if mask.sum() < 3:
                return 0.0
            if len(np.unique(y_np[mask])) < 2:
                return 0.0
            # map from [-1,1] -> [0,1]
            sc = float(silhouette_score(Q_np[mask], y_np[mask], metric="euclidean"))
            return 0.5 * (sc + 1.0)
        except Exception:
            return 0.0

    def _build_knng_cdp(X: np.ndarray, k: int, mode: str = "sklearn",
                        prune_alpha: float = 0.5, use_faiss: bool = True) -> np.ndarray:
        """Density-aware / mutual kNN or sklearn kneighbors_graph as a symmetric similarity matrix in [0,1]."""
        N, d = X.shape
        mode = str(mode).lower()
        if mode not in ("cdp", "mutual"):
            # sklearn baseline graph
            from sklearn.neighbors import kneighbors_graph
            G = kneighbors_graph(X, n_neighbors=min(k, max(1, N - 1)),
                                 mode="distance", include_self=False).toarray().astype(np.float32)
            G[G > 0] = 1.0 / (G[G > 0] + 1e-8)
            if G.max() > 0:
                G /= G.max()
            return G

        # try FAISS for kNN if requested and available
        idx = None
        dist = None
        if use_faiss:
            try:
                import faiss
                index = faiss.IndexFlatL2(d)
                Xf = X.astype(np.float32, copy=False)
                index.add(Xf)
                dist2, idx = index.search(Xf, min(k + 1, N))
                dist = np.sqrt(np.maximum(dist2, 0.0))
                idx = idx[:, 1:]
                dist = dist[:, 1:]
            except Exception:
                idx = None

        if idx is None:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=min(k + 1, N), metric="euclidean")
            nn.fit(X)
            dist, idx = nn.kneighbors(X, return_distance=True)
            idx = idx[:, 1:]
            dist = dist[:, 1:]

        sim = 1.0 / (dist + 1e-8)  # (N,k)
        # build directed sim matrix then mutualize
        W = np.zeros((N, N), dtype=np.float32)
        rows = np.repeat(np.arange(N), idx.shape[1])
        cols = idx.ravel()
        vals = sim.ravel()
        W[rows, cols] = np.maximum(W[rows, cols], vals)
        mutual = ((W > 0) & (W.T > 0)).astype(np.float32)
        W = np.minimum(W, W.T) * mutual

        if mode == "mutual":
            if W.max() > 0:
                W /= W.max()
            return W

        # CDP pruning
        mean_d = dist.mean(axis=1)
        rho = 1.0 / (mean_d + 1e-8)
        node_thr = np.median(sim, axis=1) * prune_alpha
        keep = np.zeros_like(W, dtype=bool)
        I, J = np.nonzero(W)
        med_rho = np.median(rho)
        for i, j in zip(I, J):
            sij = W[i, j]
            if (sij >= node_thr[i]) and (sij >= node_thr[j]) and (min(rho[i], rho[j]) >= med_rho):
                keep[i, j] = True
                keep[j, i] = True
        W = np.where(keep, W, 0.0).astype(np.float32)
        if W.max() > 0:
            W /= W.max()
        return W

    # ========================================================================
    #                        MAIN COMPUTATION (float32)
    # ========================================================================
    device = S.device

    # Compute entirely in float32 to avoid fp16 solver issues
    with torch.cuda.amp.autocast(enabled=False):
        S32 = S.detach().to(dtype=torch.float32, device=device)
        Q32 = Q.detach().to(dtype=torch.float32, device=device)
        yS  = S_labels.detach().to(dtype=torch.long, device=device)

        Ns, d = S32.size()
        Nq, _ = Q32.size()
        C = int(n_classes)
        shots = max(1, int(Ns // C))

        # ---------------- reconstruction: center refinement loops ----------------
        # init centers from support
        X_centers = torch.zeros((C, d), device=device, dtype=torch.float32)
        for cls in range(C):
            m = (yS == cls)
            X_centers[cls] = S32[m].mean(dim=0) if m.any() else S32.mean(dim=0)

        A = torch.cat([S32, Q32], dim=0)  # refine using [S;Q] for stability
        prev_mean = torch.tensor(float("inf"), device=device, dtype=torch.float32)

        for _ in range(int(n_loops)):
            blocks = []
            for cls in range(C):
                Sc = S32.t()[:, yS == cls]     # (d, shots)
                # in edge 1-shot, still (d,1) is fine
                blocks.append(Sc)
                blocks.append(X_centers[cls].view(-1, 1))
            X = torch.cat(blocks, dim=1)       # (d, K=(shots+1)*C)
            Xt = X.t().contiguous()
            XtX = (Xt @ X).contiguous()
            Kb = XtX.size(0)
            XtX = XtX + (lam * torch.eye(Kb, device=device, dtype=torch.float32))

            # robust linear solve
            try:
                L = torch.linalg.cholesky(XtX)
                P = torch.cholesky_solve(Xt, L)   # (K, d)
            except RuntimeError:
                P = torch.linalg.solve(XtX, Xt)

            Ntot = Ns + Nq
            beta = (P @ A.t()).view(C, -1, Ntot)     # (C, shots+1, Ntot)
            code = torch.zeros((C, d, Ntot), device=device, dtype=torch.float32)
            Xc   = X.t().view(C, -1, d)              # (C, shots+1, d)
            for cls in range(C):
                code[cls] = Xc[cls].transpose(0, 1) @ beta[cls]  # (d, Ntot)

            err  = A.t().unsqueeze(0) - code                 # (C, d, Ntot)
            dist = (err ** 2).sum(dim=1)                     # (C, Ntot)
            A_labels = dist.t().argmin(dim=1)                # (Ntot,)

            # update centers
            for cls in range(C):
                m = (A_labels == cls)
                if m.any():
                    X_centers[cls] = A[m].mean(dim=0)

            cur_mean = dist.mean()
            if torch.isfinite(prev_mean) and torch.isfinite(cur_mean):
                if (prev_mean - cur_mean).abs() <= 1e-9:
                    break
            prev_mean = cur_mean

        # final reconstruction logits for Q only
        blocks = []
        for cls in range(C):
            Sc = S32.t()[:, yS == cls]
            blocks.append(Sc)
            blocks.append(X_centers[cls].view(-1, 1))
        X  = torch.cat(blocks, dim=1)              # (d, K)
        Xt = X.t().contiguous()
        XtX = (Xt @ X).contiguous()
        Kb = XtX.size(0)
        XtX = XtX + (lam * torch.eye(Kb, device=device, dtype=torch.float32))
        try:
            L = torch.linalg.cholesky(XtX)
            P = torch.cholesky_solve(Xt, L)
        except RuntimeError:
            P = torch.linalg.solve(XtX, Xt)

        beta = (P @ Q32.t()).view(C, -1, Nq)       # (C, shots+1, Nq)
        code = torch.zeros((C, d, Nq), device=device, dtype=torch.float32)
        Xc   = X.t().view(C, -1, d)
        for cls in range(C):
            code[cls] = Xc[cls].transpose(0, 1) @ beta[cls]

        err_q  = Q32.t().unsqueeze(0) - code
        dist_q = (err_q ** 2).sum(dim=1).clamp_min(1e-8)     # (C, Nq)

        Q_probs = -torch.log(dist_q.t() + eps)                # (Nq, C)
        Q_probs = torch.softmax(Q_probs / max(float(tau), 1e-8), dim=1)  # (Nq,C)

        # ---------------- consensus refinement (SAPS + EBSS) ----------------
        try:
            import sklearn  # noqa
            from sklearn.cluster import DBSCAN, KMeans
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.preprocessing import StandardScaler

            # numpy copies
            S_np = S32.detach().cpu().numpy()
            Q_np = Q32.detach().cpu().numpy()
            Sy   = yS.detach().cpu().numpy()
            Ns_  = S_np.shape[0]

            # normalize for clustering
            S_np = _l2_normalize_np(S_np)
            Q_np = _l2_normalize_np(Q_np)
            A_np = np.concatenate([S_np, Q_np], axis=0)

            # (1) DBSCAN -> cluster-to-class via support majority
            if standardize_for_dbscan:
                A_scaled = StandardScaler().fit_transform(A_np)
            else:
                A_scaled = A_np
            db = DBSCAN(eps=float(dbscan_eps), min_samples=int(dbscan_min_samples)).fit(A_scaled)
            db_all = db.labels_
            db_q   = db_all[-Nq:]
            cl2cls = {}
            for cid in np.unique(db_all):
                if cid < 0:
                    continue
                mS = (db_all[:Ns_] == cid)
                if mS.any():
                    counts = np.bincount(Sy[mS], minlength=C)
                    cl2cls[cid] = int(counts.argmax())
            y_db = np.array([cl2cls.get(cid, -1) for cid in db_q], dtype=np.int64)
            db_onehot = np.zeros((Nq, C), dtype=np.float32)
            ok = (y_db >= 0)
            db_onehot[ok, y_db[ok]] = 1.0

            # (2) kNN (distance-weighted) on support
            knn = KNeighborsClassifier(n_neighbors=max(1, min(int(knn_k), Ns_)), weights="distance").fit(S_np, Sy)
            knn_proba = knn.predict_proba(Q_np)  # (Nq,C)

            # (3) k-means++ (sklearn); FAISS — safe fallback is sklearn
            km = KMeans(
                n_clusters=C,
                n_init=1,
                init="k-means++",
                random_state=int(km_random_state),
                max_iter=max(100, int(kmeans_iter)),
            )
            km.fit(A_np)
            centers = km.cluster_centers_.astype(np.float32)
            d_km = ((Q_np[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)  # (Nq,C)
            km_logits = -d_km
            km_proba = np.exp(km_logits - km_logits.max(axis=1, keepdims=True))
            km_proba = km_proba / np.clip(km_proba.sum(axis=1, keepdims=True), 1e-8, None)

            # (4) graph head via k-NNG (CDP/mutual/sklearn)
            W = _build_knng_cdp(
                A_np.astype(np.float32, copy=False),
                k=max(1, min(int(graph_k), max(1, A_np.shape[0] - 1))),
                mode=str(knng_mode),
                prune_alpha=float(cdp_alpha),
                use_faiss=bool(use_faiss_knn),
            )
            W_QS = W[Ns_:, :Ns_]
            graph_scores = np.zeros((Nq, C), dtype=np.float32)
            for c in range(C):
                sc = (Sy == c).astype(np.float32)
                graph_scores[:, c] = (W_QS * sc).sum(axis=1)
            graph_proba = graph_scores / np.clip(graph_scores.sum(axis=1, keepdims=True), 1e-8, None)

            # ---- SAPS reliabilities γ = α × β (β neutral here) ----
            y_km = d_km.argmin(axis=1).astype(np.int64)
            y_kn = knn_proba.argmax(axis=1).astype(np.int64)
            y_gr = graph_proba.argmax(axis=1).astype(np.int64)

            M_db = _pair_matrix(y_db)
            M_km = _pair_matrix(y_km)
            M_kn = _pair_matrix(y_kn)
            M_gr = _pair_matrix(y_gr)

            alphas = np.array([
                _safe_silhouette(Q_np, y_db),
                _safe_silhouette(Q_np, y_kn),
                _safe_silhouette(Q_np, y_km),
                _safe_silhouette(Q_np, y_gr),
            ], dtype=np.float32)

            # first-call: no history, so β=1
            gammas = alphas.copy()
            if not np.isfinite(gammas).any() or gammas.sum() <= 1e-8:
                gammas = np.ones_like(alphas, dtype=np.float32)
            gammas = gammas / np.clip(gammas.sum(), 1e-8, None)

            # EBSS agreement v (weighted co-pair O_hat vs best head's pairs)
            O_hat = gammas[0] * M_db + gammas[1] * M_kn + gammas[2] * M_km + gammas[3] * M_gr
            Ms = [M_db, M_kn, M_km, M_gr]
            s_idx = int(np.argmax(gammas))
            M_s = Ms[s_idx]
            deg = np.clip(M_s.sum(axis=1), 1.0, None)
            v = (O_hat * M_s).sum(axis=1) / deg  # (Nq,) in [0,1]

            # SAPS-weighted consensus distribution over heads
            P_heads = (gammas[0] * db_onehot
                       + gammas[1] * knn_proba
                       + gammas[2] * km_proba
                       + gammas[3] * graph_proba)
            P_heads = P_heads / np.clip(P_heads.sum(axis=1, keepdims=True), 1e-8, None)

            # gentle, per-sample blend α_i \in [alpha_min, alpha_max]
            P_rec = Q_probs.detach().cpu().numpy()
            z = (v - v.mean()) / (v.std() + 1e-8)
            sigma = 0.5 * (1.0 + np.tanh(z))  # in (0,1)
            alpha = alpha_min + (alpha_max - alpha_min) * sigma

            # guard: if consensus is notably less confident than reconstruction, shrink α
            conf_rec = P_rec.max(axis=1)
            conf_head = P_heads.max(axis=1)
            worse_mask = (conf_head + float(delta_guard) < conf_rec)
            if np.any(worse_mask):
                alpha[worse_mask] = np.maximum(alpha_min * 0.25, alpha[worse_mask] * 0.25)

            P_blend = (1.0 - alpha[:, None]) * P_rec + alpha[:, None] * P_heads
            P_blend = P_blend / np.clip(P_blend.sum(axis=1, keepdims=True), 1e-8, None)
            Q_probs = torch.from_numpy(P_blend).to(device=device, dtype=torch.float32)

        except Exception:
            # if sklearn or any head fails, keep pure reconstruction probs
            Q_probs = Q_probs

        Q_labels = Q_probs.argmax(dim=1)
        return Q_probs.to(dtype=torch.float32, device=device), Q_labels.to(dtype=torch.long, device=device)
