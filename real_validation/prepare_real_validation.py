from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# FROZEN CONFIGURATION
# ============================================================

DATA_DIR = Path("data/real_acn")
RESULTS_DIR = Path("results_real")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ENERGY_FILE = DATA_DIR / "caltech_hourly_energy.npz"
ADJ_FILE = DATA_DIR / "caltech_pod_adjacency.npy"
MAPPING_FILE = DATA_DIR / "caltech_station_mapping.csv"

LOOKBACK_HOURS = 168
HORIZON_HOURS = 24
STEP_HOURS = 24

# There are exactly 670 complete days.
#
# Freeze whole-day chronological split:
# 469 train + 100 validation + 101 test = 670 days.
TRAIN_DAYS = 469
VAL_DAYS = 100
TEST_DAYS = 101

# Fixed negative controls.
N_PERMUTATIONS = 100
PERMUTATION_SEED = 777

# IMPORTANT:
# Keep this value frozen.
#
# This is only a lightweight ridge diagnostic.
# Before final paper reporting, confirm that this value matches
# the RIDGE regularization used in your frozen Washington RSPS code.
RIDGE_ALPHA = 1.0


# ============================================================
# LOAD DATA
# ============================================================

npz = np.load(ENERGY_FILE, allow_pickle=True)

energy = np.asarray(
    npz["energy"],
    dtype=np.float64,
)

timestamps = pd.to_datetime(
    npz["timestamps"],
    utc=True,
)

stations = np.asarray(
    npz["stations"],
).astype(str)

A_real = np.load(ADJ_FILE).astype(np.float64)

mapping = pd.read_csv(MAPPING_FILE)
mapping = mapping.sort_values("node_id").reset_index(drop=True)


print("=" * 78)
print("REAL ACN VALIDATION PREPARATION")
print("=" * 78)

print("Hourly matrix shape:", energy.shape)
print("Adjacency shape:", A_real.shape)

assert energy.shape == (16080, 37)
assert A_real.shape == (37, 37)
assert len(stations) == 37


# ============================================================
# VERIFY COMPLETE DAILY STRUCTURE
# ============================================================

N_HOURS = energy.shape[0]

assert N_HOURS % 24 == 0

TOTAL_DAYS = N_HOURS // 24

assert TOTAL_DAYS == 670
assert TRAIN_DAYS + VAL_DAYS + TEST_DAYS == TOTAL_DAYS


train_end_hour = TRAIN_DAYS * 24
val_end_hour = (TRAIN_DAYS + VAL_DAYS) * 24


print()
print("Chronological split")
print("-------------------")

print(
    "TRAIN:",
    timestamps[0],
    "to",
    timestamps[train_end_hour - 1],
)

print(
    "VALIDATION:",
    timestamps[train_end_hour],
    "to",
    timestamps[val_end_hour - 1],
)

print(
    "TEST:",
    timestamps[val_end_hour],
    "to",
    timestamps[-1],
)

print()
print("Train days:", TRAIN_DAYS)
print("Validation days:", VAL_DAYS)
print("Test days:", TEST_DAYS)


# ============================================================
# FORECAST-WINDOW COUNTS
#
# Training targets stay inside train.
#
# Validation/test windows may use preceding historical context,
# but their targets remain entirely inside their own split.
# ============================================================

train_windows = TRAIN_DAYS - LOOKBACK_HOURS // 24

val_windows = VAL_DAYS
test_windows = TEST_DAYS


print()
print("Forecast windows")
print("----------------")
print("Training windows:", train_windows)
print("Validation windows:", val_windows)
print("Test windows:", test_windows)


# ============================================================
# DAILY ENERGY
# ============================================================

daily = energy.reshape(
    TOTAL_DAYS,
    24,
    energy.shape[1],
).sum(axis=1)

# Shape:
# days x nodes

print()
print("Daily matrix:", daily.shape)


# ============================================================
# TRAIN-ONLY NORMALIZATION
# ============================================================

train_daily = daily[:TRAIN_DAYS]

mu = train_daily.mean(axis=0)
sigma = train_daily.std(axis=0)

# Prevent divide-by-zero.
sigma[sigma < 1e-8] = 1.0

z = (daily - mu) / sigma


# ============================================================
# WEEKLY DIFFERENCING
#
# r_k = z_k - z_(k-7)
# ============================================================

r = z[7:] - z[:-7]

# r index 0 corresponds to original day 7.
#
# Original day k maps to r index k - 7.


# ============================================================
# GRAPH UTILITIES
# ============================================================

def row_normalize(A):
    A = np.asarray(A, dtype=np.float64)

    degree = A.sum(axis=1)

    W = np.zeros_like(A)

    good = degree > 0

    W[good] = (
        A[good]
        / degree[good, None]
    )

    return W


def permute_adjacency(A, permutation):
    """
    Node-label permutation:
        A_perm = P A P^T

    This preserves graph topology and degree distribution
    while changing which station occupies each graph position.
    """

    permutation = np.asarray(permutation)

    return A[
        np.ix_(
            permutation,
            permutation,
        )
    ]


# ============================================================
# RIDGE REGRESSION
# ============================================================

def ridge_fit(X, y, alpha):
    """
    Ridge with unregularized intercept.
    """

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    X_aug = np.column_stack(
        [
            np.ones(len(X)),
            X,
        ]
    )

    I = np.eye(X_aug.shape[1])
    I[0, 0] = 0.0

    beta = np.linalg.solve(
        X_aug.T @ X_aug + alpha * I,
        X_aug.T @ y,
    )

    return beta


def ridge_predict(X, beta):

    X = np.asarray(X, dtype=np.float64)

    X_aug = np.column_stack(
        [
            np.ones(len(X)),
            X,
        ]
    )

    return X_aug @ beta


# ============================================================
# RSPS
# ============================================================

def calculate_rsps(
    residuals,
    adjacency,
    node_indices,
):
    """
    Matches the paper's conceptual diagnostic:

    self model:
        r_i,k <- r_i,k-1

    spatial model:
        r_i,k <- [r_i,k-1, neighbor residual at k-1]

    Training uses training days only.
    Evaluation uses validation days only.
    """

    node_indices = np.asarray(
        node_indices,
        dtype=int,
    )

    W = row_normalize(adjacency)

    # residuals correspond to original days 7...669.
    #
    # To predict original day k:
    # current residual index = k - 7
    # lag residual index     = k - 8

    # Original target days:
    #
    # training:
    # day 8 ... TRAIN_DAYS-1
    #
    # validation:
    # day TRAIN_DAYS ... TRAIN_DAYS+VAL_DAYS-1

    train_target_days = np.arange(
        8,
        TRAIN_DAYS,
    )

    val_target_days = np.arange(
        TRAIN_DAYS,
        TRAIN_DAYS + VAL_DAYS,
    )


    def make_samples(target_days):

        self_x = []
        neigh_x = []
        target_y = []

        for day in target_days:

            curr_idx = day - 7
            lag_idx = day - 8

            r_prev = residuals[lag_idx]
            r_now = residuals[curr_idx]

            neighbor_prev = W @ r_prev

            for i in node_indices:

                self_x.append(
                    [r_prev[i]]
                )

                neigh_x.append(
                    [
                        r_prev[i],
                        neighbor_prev[i],
                    ]
                )

                target_y.append(
                    r_now[i]
                )

        return (
            np.asarray(self_x),
            np.asarray(neigh_x),
            np.asarray(target_y),
        )


    X_self_train, X_sp_train, y_train = make_samples(
        train_target_days
    )

    X_self_val, X_sp_val, y_val = make_samples(
        val_target_days
    )


    b_self = ridge_fit(
        X_self_train,
        y_train,
        RIDGE_ALPHA,
    )

    b_sp = ridge_fit(
        X_sp_train,
        y_train,
        RIDGE_ALPHA,
    )


    pred_self = ridge_predict(
        X_self_val,
        b_self,
    )

    pred_sp = ridge_predict(
        X_sp_val,
        b_sp,
    )


    mse_self = np.mean(
        (y_val - pred_self) ** 2
    )

    mse_sp = np.mean(
        (y_val - pred_sp) ** 2
    )


    rsps = (
        mse_self - mse_sp
    ) / (
        mse_self + 1e-12
    )


    neighbor_coef = float(
        b_sp[-1]
    )


    return {
        "rsps": float(rsps),
        "mse_self": float(mse_self),
        "mse_spatial": float(mse_sp),
        "neighbor_coef": neighbor_coef,
        "n_train_samples": int(len(y_train)),
        "n_val_samples": int(len(y_val)),
    }


# ============================================================
# IDENTIFY REAL POD NODES
# ============================================================

degrees = A_real.sum(axis=1)

pod_nodes = np.where(
    degrees > 0
)[0]

isolated_nodes = np.where(
    degrees == 0
)[0]


assert len(pod_nodes) == 16
assert len(isolated_nodes) == 21


print()
print("Pod-connected nodes:", len(pod_nodes))
print("Isolated nodes:", len(isolated_nodes))


# ============================================================
# REAL GRAPH RSPS -- ALL 37
# ============================================================

real_all = calculate_rsps(
    r,
    A_real,
    np.arange(37),
)


# ============================================================
# REAL GRAPH RSPS -- POD 16
#
# Extract induced graph.
# ============================================================

A_pod = A_real[
    np.ix_(
        pod_nodes,
        pod_nodes,
    )
]

assert A_pod.shape == (16, 16)

pod_degrees = A_pod.sum(axis=1)

assert np.all(
    pod_degrees == 7
)


# For pod-only analysis, residual matrix is also restricted.
r_pod = r[:, pod_nodes]


real_pod = calculate_rsps(
    r_pod,
    A_pod,
    np.arange(16),
)


# ============================================================
# PERMUTATION CONTROLS
# ============================================================

rng = np.random.default_rng(
    PERMUTATION_SEED
)

perm_rows_all = []
perm_rows_pod = []

saved_permutations = []


for p in range(N_PERMUTATIONS):

    # --------------------------------------------------------
    # ALL-37 control
    # --------------------------------------------------------

    perm_all = rng.permutation(37)

    A_perm_all = permute_adjacency(
        A_real,
        perm_all,
    )

    result_all = calculate_rsps(
        r,
        A_perm_all,
        np.arange(37),
    )

    perm_rows_all.append({
        "permutation_id": p,
        **result_all,
    })


    # --------------------------------------------------------
    # POD-16 matched control
    #
    # This retains:
    #   16 nodes
    #   degree 7 for every node
    #   two 8-node cliques
    #
    # Only clique membership alignment is changed.
    # --------------------------------------------------------

    perm_pod = rng.permutation(16)

    A_perm_pod = permute_adjacency(
        A_pod,
        perm_pod,
    )

    result_pod = calculate_rsps(
        r_pod,
        A_perm_pod,
        np.arange(16),
    )

    perm_rows_pod.append({
        "permutation_id": p,
        **result_pod,
    })


    # Save first 20 permutations for later GNN controls.
    if p < 20:

        saved_permutations.append({
            "permutation_id": p,
            "all37": perm_all.tolist(),
            "pod16": perm_pod.tolist(),
        })


# ============================================================
# RESULTS
# ============================================================

perm_all_df = pd.DataFrame(
    perm_rows_all
)

perm_pod_df = pd.DataFrame(
    perm_rows_pod
)


def summarize(
    name,
    real_result,
    perm_df,
):

    perm_mean = perm_df["rsps"].mean()
    perm_std = perm_df["rsps"].std(ddof=1)

    advantage = (
        real_result["rsps"]
        - perm_mean
    )

    percentile = (
        100.0
        * np.mean(
            real_result["rsps"]
            > perm_df["rsps"]
        )
    )

    print()
    print("=" * 78)
    print(name)
    print("=" * 78)

    print(
        "REAL RSPS:",
        f"{real_result['rsps']:.6f}",
    )

    print(
        "Permutation mean:",
        f"{perm_mean:.6f}",
    )

    print(
        "Permutation std:",
        f"{perm_std:.6f}",
    )

    print(
        "RSPS alignment advantage:",
        f"{advantage:.6f}",
    )

    print(
        "Real graph beats:",
        f"{percentile:.1f}% of permutations",
    )

    print(
        "Self validation MSE:",
        f"{real_result['mse_self']:.6f}",
    )

    print(
        "Spatial validation MSE:",
        f"{real_result['mse_spatial']:.6f}",
    )

    print(
        "Neighbor coefficient:",
        f"{real_result['neighbor_coef']:.6f}",
    )

    return {
        "analysis": name,
        "real_rsps": real_result["rsps"],
        "perm_rsps_mean": perm_mean,
        "perm_rsps_std": perm_std,
        "rsps_alignment_advantage": advantage,
        "percent_permutations_beaten": percentile,
        "mse_self": real_result["mse_self"],
        "mse_spatial": real_result["mse_spatial"],
        "neighbor_coef": real_result["neighbor_coef"],
    }


summary_all = summarize(
    "ALL 37 STATIONS",
    real_all,
    perm_all_df,
)

summary_pod = summarize(
    "POD-CONNECTED 16 STATIONS",
    real_pod,
    perm_pod_df,
)


# ============================================================
# SAVE EVERYTHING
# ============================================================

perm_all_df.to_csv(
    RESULTS_DIR / "real_rsps_permutations_all37.csv",
    index=False,
)

perm_pod_df.to_csv(
    RESULTS_DIR / "real_rsps_permutations_pod16.csv",
    index=False,
)

summary_df = pd.DataFrame(
    [
        summary_all,
        summary_pod,
    ]
)

summary_df.to_csv(
    RESULTS_DIR / "real_rsps_summary.csv",
    index=False,
)


with open(
    RESULTS_DIR / "real_graph_permutations.json",
    "w",
) as f:

    json.dump(
        saved_permutations,
        f,
        indent=2,
    )


split_metadata = {
    "total_days": TOTAL_DAYS,
    "train_days": TRAIN_DAYS,
    "validation_days": VAL_DAYS,
    "test_days": TEST_DAYS,
    "train_hours": TRAIN_DAYS * 24,
    "validation_hours": VAL_DAYS * 24,
    "test_hours": TEST_DAYS * 24,
    "lookback_hours": LOOKBACK_HOURS,
    "forecast_horizon_hours": HORIZON_HOURS,
    "forecast_stride_hours": STEP_HOURS,
    "training_forecast_windows": train_windows,
    "validation_forecast_windows": val_windows,
    "test_forecast_windows": test_windows,
    "n_nodes_all": 37,
    "n_nodes_pod": 16,
    "n_rsps_permutations": N_PERMUTATIONS,
    "n_saved_gnn_permutations": 20,
    "permutation_seed": PERMUTATION_SEED,
    "ridge_alpha": RIDGE_ALPHA,
}


with open(
    RESULTS_DIR / "real_validation_split.json",
    "w",
) as f:

    json.dump(
        split_metadata,
        f,
        indent=2,
    )


print()
print("=" * 78)
print("SAVED")
print("=" * 78)

print(
    RESULTS_DIR
    / "real_rsps_summary.csv"
)

print(
    RESULTS_DIR
    / "real_rsps_permutations_all37.csv"
)

print(
    RESULTS_DIR
    / "real_rsps_permutations_pod16.csv"
)

print(
    RESULTS_DIR
    / "real_graph_permutations.json"
)

print(
    RESULTS_DIR
    / "real_validation_split.json"
)
