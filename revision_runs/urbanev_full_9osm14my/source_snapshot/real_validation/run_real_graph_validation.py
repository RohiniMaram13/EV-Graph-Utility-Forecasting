"""
Real-world ACN validation for the graph-utility paper.

PRIMARY REAL-DATA ANALYSIS
--------------------------
Observed Caltech charging data:
    16,080 hourly observations
    37 selected EVSEs
    primary neural analysis = 16 pod-connected EVSEs

Forecasting:
    previous 168 h -> next 24 h

Chronological split:
    469 days training
    100 days validation
    101 days testing

Models:
    1. GRU baseline
    2. GRU-FT matched temporal control
    3. REAL-POD-GNN
    4. 20 topology-preserving PERM-GNN controls

Training seeds:
    42, 52, 62

The Washington synthetic benchmark is not modified.
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


from experiments_v2.config import (
    HIDDEN_DIM,
    HORIZON,
    DROPOUT,
    LR,
    PATIENCE,
)

from experiments_v2.models import (
    GRUForecast,
)

from experiments_v2.train_eval import (
    train_model,
    set_seed,
)

from experiments_v2.metrics import metrics

# Reuse the exact frozen matched-control implementation.
from experiments_v2.run_full_graph_control import (
    train_gnn_branch,
    evaluate,
    cpu_state,
    make_graph_dict,
)

from experiments_v2.device_utils import (
    resolve_device,
    device_report,
)


# ============================================================
# FROZEN REAL-DATA CONFIGURATION
# ============================================================

DATA_DIR = Path("data/real_acn")
RESULTS_DIR = Path("results_real")

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


ENERGY_FILE = DATA_DIR / "caltech_hourly_energy.npz"

ADJ_FILE = DATA_DIR / "caltech_pod_adjacency.npy"

MAPPING_FILE = DATA_DIR / "caltech_station_mapping.csv"

PERM_FILE = RESULTS_DIR / "real_graph_permutations.json"


LOOKBACK = 168
FORECAST_HORIZON = 24
STEP = 24


TRAIN_DAYS = 469
VAL_DAYS = 100
TEST_DAYS = 101


TRAIN_SEEDS = [
    42,
    52,
    62,
]


# Frozen matched-training protocol.
GRU_EPOCHS = 100
GRAPH_EPOCHS = 30
FINETUNE_EPOCHS = 70

GRAPH_LR = 1e-3
FINETUNE_LR = 5e-4

# Use same validation patience as frozen experiment.
EARLY_STOPPING_PATIENCE = PATIENCE
GRAPH_PATIENCE = PATIENCE


# ============================================================
# DATASET CONSTRUCTION
# ============================================================

def load_real_pod_dataset():

    npz = np.load(
        ENERGY_FILE,
        allow_pickle=True,
    )

    energy = np.asarray(
        npz["energy"],
        dtype=np.float32,
    )

    timestamps = pd.to_datetime(
        npz["timestamps"],
        utc=True,
    )

    station_ids = np.asarray(
        npz["stations"]
    ).astype(str)

    A_all = np.load(
        ADJ_FILE
    ).astype(np.float32)


    assert energy.shape == (16080, 37)
    assert A_all.shape == (37, 37)


    # --------------------------------------------------------
    # Select the 16 nodes that have real pod edges.
    # --------------------------------------------------------

    degrees = A_all.sum(axis=1)

    pod_nodes = np.where(
        degrees > 0
    )[0]

    assert len(pod_nodes) == 16


    A_pod = A_all[
        np.ix_(
            pod_nodes,
            pod_nodes,
        )
    ]

    assert A_pod.shape == (16, 16)

    assert np.all(
        A_pod.sum(axis=1) == 7
    )


    energy = energy[
        :,
        pod_nodes,
    ]

    station_ids = station_ids[
        pod_nodes
    ]


    # --------------------------------------------------------
    # Training-only normalization.
    # --------------------------------------------------------

    train_end_hour = (
        TRAIN_DAYS * 24
    )

    train_values = energy[
        :train_end_hour
    ]


    mean = train_values.mean(
        axis=0,
        keepdims=False,
    )

    std = train_values.std(
        axis=0,
        keepdims=False,
    )

    std[
        std < 1e-8
    ] = 1.0


    normalized = (
        energy - mean[None, :]
    ) / std[None, :]


    # scaler is shaped [N, 1] so it broadcasts
    # against [windows, nodes, horizon].
    scaler = {
        "mean": mean[:, None],
        "std": std[:, None],
    }


    # --------------------------------------------------------
    # Window creation.
    #
    # target_day is the first day being predicted.
    #
    # The previous 168 hours form X.
    # The following 24 hours form Y.
    # --------------------------------------------------------

    def build_windows(target_days):

        X = []
        Y = []

        for day in target_days:

            target_start = (
                day * 24
            )

            input_start = (
                target_start
                - LOOKBACK
            )

            target_end = (
                target_start
                + FORECAST_HORIZON
            )

            x = normalized[
                input_start:target_start
            ]

            y = normalized[
                target_start:target_end
            ]


            # convert:
            #
            # [time, nodes]
            #
            # to
            #
            # [nodes, time]

            X.append(
                x.T
            )

            Y.append(
                y.T
            )


        X = torch.tensor(
            np.stack(X),
            dtype=torch.float32,
        )

        Y = torch.tensor(
            np.stack(Y),
            dtype=torch.float32,
        )

        return X, Y


    # --------------------------------------------------------
    # Chronological target periods.
    # --------------------------------------------------------

    train_target_days = range(
        LOOKBACK // 24,
        TRAIN_DAYS,
    )

    validation_target_days = range(
        TRAIN_DAYS,
        TRAIN_DAYS + VAL_DAYS,
    )

    test_target_days = range(
        TRAIN_DAYS + VAL_DAYS,
        TRAIN_DAYS + VAL_DAYS + TEST_DAYS,
    )


    train_set = build_windows(
        train_target_days
    )

    val_set = build_windows(
        validation_target_days
    )

    test_set = build_windows(
        test_target_days
    )


    assert len(train_set[0]) == 462
    assert len(val_set[0]) == 100
    assert len(test_set[0]) == 101


    print()
    print("=" * 76)
    print("REAL POD-16 DATASET")
    print("=" * 76)

    print(
        "Hourly observations:",
        len(energy),
    )

    print(
        "Stations:",
        len(station_ids),
    )

    print(
        "Training windows:",
        len(train_set[0]),
    )

    print(
        "Validation windows:",
        len(val_set[0]),
    )

    print(
        "Test windows:",
        len(test_set[0]),
    )

    print()

    print(
        "Train:",
        timestamps[0],
        "to",
        timestamps[
            TRAIN_DAYS * 24 - 1
        ],
    )

    print(
        "Validation:",
        timestamps[
            TRAIN_DAYS * 24
        ],
        "to",
        timestamps[
            (TRAIN_DAYS + VAL_DAYS)
            * 24 - 1
        ],
    )

    print(
        "Test:",
        timestamps[
            (TRAIN_DAYS + VAL_DAYS)
            * 24
        ],
        "to",
        timestamps[-1],
    )


    return {
        "energy": energy,
        "station_ids": station_ids,
        "pod_nodes": pod_nodes,
        "adjacency": A_pod,
        "train": train_set,
        "validation": val_set,
        "test": test_set,
        "scaler": scaler,
    }


# ============================================================
# ADJACENCY -> PYTORCH GEOMETRIC GRAPH
# ============================================================

def adjacency_to_graph(A):

    rows, cols = np.where(
        A > 0
    )

    edge_index = torch.tensor(
        np.stack(
            [
                rows,
                cols,
            ]
        ),
        dtype=torch.long,
    )

    return make_graph_dict(
        edge_index
    )


# ============================================================
# PERMUTED GRAPH
# ============================================================

def permute_adjacency(
    A,
    permutation,
):

    permutation = np.asarray(
        permutation,
        dtype=int,
    )

    return A[
        np.ix_(
            permutation,
            permutation,
        )
    ]


# ============================================================
# TRAIN SHARED TEMPORAL BASELINE
# ============================================================

def train_temporal_controls(
    train_set,
    val_set,
    test_set,
    scaler,
    train_seed,
    runtime,
):

    # --------------------------------------------------------
    # A: shared GRU baseline
    # --------------------------------------------------------

    print()
    print(
        "  A shared GRU baseline",
        flush=True,
    )

    set_seed(
        train_seed
    )

    gru = GRUForecast(
        HIDDEN_DIM,
        HORIZON,
    )

    gru, gru_best, gru_epochs = train_model(
        gru,
        train_set,
        val_set,
        None,
        None,
        seed=train_seed,
        verbose=runtime["verbose"],
        device=runtime["device"],
        batch_size=runtime["batch_size"],
        max_epochs=runtime["gru_epochs"],
        patience=runtime["patience"],
        amp=runtime["amp"],
        lr=LR,
    )


    baseline_metrics = evaluate(
        gru,
        test_set,
        scaler,
        None,
        runtime,
    )


    shared_state = cpu_state(
        gru
    )


    # --------------------------------------------------------
    # B: matched GRU fine-tuning control
    # --------------------------------------------------------

    print(
        "  B matched GRU fine-tuning control",
        flush=True,
    )


    set_seed(
        train_seed + 1000
    )


    gru_ft = GRUForecast(
        HIDDEN_DIM,
        HORIZON,
    )

    gru_ft.load_state_dict(
        shared_state
    )


    gru_ft, gru_ft_best, gru_ft_epochs = train_model(
        gru_ft,
        train_set,
        val_set,
        None,
        None,
        seed=train_seed + 1000,
        verbose=runtime["verbose"],
        device=runtime["device"],
        batch_size=runtime["batch_size"],
        max_epochs=runtime["finetune_epochs"],
        patience=runtime["patience"],
        amp=runtime["amp"],
        lr=runtime["finetune_lr"],
        include_initial=True,
    )


    gru_ft_metrics = evaluate(
        gru_ft,
        test_set,
        scaler,
        None,
        runtime,
    )


    return {
        "shared_state": shared_state,

        "baseline": baseline_metrics,

        "baseline_best_val_mse": gru_best,

        "baseline_epochs": gru_epochs,

        "gru_ft": gru_ft_metrics,

        "gru_ft_best_val_mse": gru_ft_best,

        "gru_ft_epochs": gru_ft_epochs,
    }


# ============================================================
# RUN ONE TRAINING SEED
# ============================================================

def run_seed(
    dataset,
    permutations,
    train_seed,
    runtime,
    n_permutations,
):

    tr = dataset["train"]
    va = dataset["validation"]
    te = dataset["test"]

    scaler = dataset["scaler"]

    A_real = dataset["adjacency"]


    real_graph = adjacency_to_graph(
        A_real
    )


    print()
    print("=" * 76)
    print(
        f"TRAINING SEED {train_seed}"
    )
    print("=" * 76)


    t0 = time.perf_counter()


    temporal = train_temporal_controls(
        tr,
        va,
        te,
        scaler,
        train_seed,
        runtime,
    )


    shared_state = temporal[
        "shared_state"
    ]


    # --------------------------------------------------------
    # REAL POD GRAPH
    # --------------------------------------------------------

    print()
    print(
        "  C REAL-POD-GNN",
        flush=True,
    )


    real_out = train_gnn_branch(
        shared_state,
        tr,
        va,
        te,
        scaler,
        real_graph,
        train_seed,
        runtime,
        "REAL-POD",
    )


    # --------------------------------------------------------
    # PERMUTED GRAPH CONTROLS
    # --------------------------------------------------------

    perm_rows = []


    for p_idx in range(
        n_permutations
    ):

        saved = permutations[
            p_idx
        ]

        permutation = np.asarray(
            saved["pod16"],
            dtype=int,
        )


        A_perm = permute_adjacency(
            A_real,
            permutation,
        )


        # Exact topology checks.
        assert np.all(
            A_perm.sum(axis=1) == 7
        )

        assert int(
            A_perm.sum() / 2
        ) == 56


        perm_graph = adjacency_to_graph(
            A_perm
        )


        print()
        print(
            f"  PERM-GNN "
            f"{p_idx + 1}/{n_permutations}",
            flush=True,
        )


        perm_out = train_gnn_branch(
            shared_state,
            tr,
            va,
            te,
            scaler,
            perm_graph,
            train_seed,
            runtime,
            f"PERM-{p_idx:02d}",
        )


        eps = 1e-12


        real_vs_perm = (
            100.0
            * (
                perm_out["mae"]
                - real_out["mae"]
            )
            / (
                perm_out["mae"]
                + eps
            )
        )


        perm_rows.append(
            {
                "train_seed": train_seed,

                "permutation_id": p_idx,

                "gru_baseline_mae":
                    temporal["baseline"]["mae"],

                "gru_ft_mae":
                    temporal["gru_ft"]["mae"],

                "real_gnn_mae":
                    real_out["mae"],

                "perm_gnn_mae":
                    perm_out["mae"],

                "real_vs_perm_gain_pct":
                    real_vs_perm,

                "perm_final_gate":
                    perm_out["final_gate"],

                "perm_graph_stage_gate":
                    perm_out["graph_stage_gate"],

                "perm_graph_stage_epochs":
                    perm_out["graph_stage_epochs"],

                "perm_finetune_epochs":
                    perm_out["finetune_epochs"],
            }
        )


        # Save continuously.
        existing_path = (
            RESULTS_DIR
            / "real_gnn_permutation_runs.csv"
        )

        current = pd.DataFrame(
            perm_rows
        )

        # This per-seed temporary file protects progress.
        current.to_csv(
            RESULTS_DIR
            / (
                f"real_gnn_permutations_"
                f"seed{train_seed}.csv"
            ),
            index=False,
        )


    # --------------------------------------------------------
    # SEED SUMMARY
    # --------------------------------------------------------

    perm_df = pd.DataFrame(
        perm_rows
    )


    perm_mae_mean = (
        perm_df["perm_gnn_mae"].mean()
    )

    perm_mae_std = (
        perm_df["perm_gnn_mae"].std(
            ddof=1
        )
    )


    eps = 1e-12


    real_gain_over_gru_ft = (
        100.0
        * (
            temporal["gru_ft"]["mae"]
            - real_out["mae"]
        )
        / (
            temporal["gru_ft"]["mae"]
            + eps
        )
    )


    real_vs_mean_perm = (
        100.0
        * (
            perm_mae_mean
            - real_out["mae"]
        )
        / (
            perm_mae_mean
            + eps
        )
    )


    pct_permutations_beaten = (
        100.0
        * np.mean(
            real_out["mae"]
            < perm_df["perm_gnn_mae"]
        )
    )


    summary = {
        "train_seed":
            train_seed,

        "gru_baseline_mae":
            temporal["baseline"]["mae"],

        "gru_baseline_rmse":
            temporal["baseline"]["rmse"],

        "gru_baseline_r2":
            temporal["baseline"]["r2"],

        "gru_ft_mae":
            temporal["gru_ft"]["mae"],

        "gru_ft_rmse":
            temporal["gru_ft"]["rmse"],

        "gru_ft_r2":
            temporal["gru_ft"]["r2"],

        "real_gnn_mae":
            real_out["mae"],

        "real_gnn_rmse":
            real_out["rmse"],

        "real_gnn_r2":
            real_out["r2"],

        "real_gain_over_gru_ft_pct":
            real_gain_over_gru_ft,

        "perm_gnn_mae_mean":
            perm_mae_mean,

        "perm_gnn_mae_std":
            perm_mae_std,

        "real_vs_mean_perm_gain_pct":
            real_vs_mean_perm,

        "percent_permutations_beaten":
            pct_permutations_beaten,

        "real_initial_gate":
            real_out["initial_gate"],

        "real_graph_stage_gate":
            real_out["graph_stage_gate"],

        "real_final_gate":
            real_out["final_gate"],

        "real_graph_stage_epochs":
            real_out["graph_stage_epochs"],

        "real_finetune_epochs":
            real_out["finetune_epochs"],

        "runtime_minutes":
            (
                time.perf_counter() - t0
            ) / 60.0,
    }


    print()
    print("=" * 76)

    print(
        f"SEED {train_seed} RESULT"
    )

    print("=" * 76)

    print(
        "GRU-FT MAE:",
        f"{summary['gru_ft_mae']:.6f}",
    )

    print(
        "REAL-GNN MAE:",
        f"{summary['real_gnn_mae']:.6f}",
    )

    print(
        "Mean PERM-GNN MAE:",
        f"{summary['perm_gnn_mae_mean']:.6f}",
    )

    print(
        "REAL gain over GRU-FT:",
        f"{summary['real_gain_over_gru_ft_pct']:+.2f}%",
    )

    print(
        "REAL vs mean PERM:",
        f"{summary['real_vs_mean_perm_gain_pct']:+.2f}%",
    )

    print(
        "REAL beats:",
        f"{summary['percent_permutations_beaten']:.1f}% "
        "of permuted graphs",
    )

    print(
        "Final graph gate:",
        f"{summary['real_final_gate']:.4f}",
    )


    return (
        summary,
        perm_df,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--device",
        default="auto",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--n-permutations",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
    )


    args = parser.parse_args()


    if not (
        1
        <= args.n_permutations
        <= 20
    ):

        raise ValueError(
            "--n-permutations must be between 1 and 20."
        )


    device = resolve_device(
        args.device
    )


    runtime = {
        "device": device,

        "batch_size":
            args.batch_size,

        "amp":
            not args.no_amp,

        "verbose":
            args.verbose,

        "gru_epochs":
            GRU_EPOCHS,

        "graph_epochs":
            GRAPH_EPOCHS,

        "finetune_epochs":
            FINETUNE_EPOCHS,

        "patience":
            EARLY_STOPPING_PATIENCE,

        "graph_patience":
            GRAPH_PATIENCE,

        "graph_lr":
            GRAPH_LR,

        "finetune_lr":
            FINETUNE_LR,
    }


    print(
        device_report(
            device
        )
    )

    print()

    print(
        "Runtime:"
    )

    print(
        "batch_size=",
        runtime["batch_size"],
    )

    print(
        "GRU epochs=",
        runtime["gru_epochs"],
    )

    print(
        "graph-only epochs=",
        runtime["graph_epochs"],
    )

    print(
        "fine-tune epochs=",
        runtime["finetune_epochs"],
    )

    print(
        "patience=",
        runtime["patience"],
    )

    print(
        "graph patience=",
        runtime["graph_patience"],
    )

    print(
        "AMP=",
        runtime["amp"],
    )


    # --------------------------------------------------------
    # Load data and frozen permutations.
    # --------------------------------------------------------

    dataset = load_real_pod_dataset()


    with open(
        PERM_FILE,
        "r",
    ) as f:

        permutations = json.load(
            f
        )


    assert len(
        permutations
    ) >= args.n_permutations


    # --------------------------------------------------------
    # Save exact Pod-16 station IDs.
    # --------------------------------------------------------

    pd.DataFrame(
        {
            "pod_node_index":
                np.arange(16),

            "original_node_id":
                dataset["pod_nodes"],

            "stationID":
                dataset["station_ids"],
        }
    ).to_csv(
        RESULTS_DIR
        / "real_pod16_station_list.csv",
        index=False,
    )


    # --------------------------------------------------------
    # Run all three training seeds.
    # --------------------------------------------------------

    summaries = []

    all_perm_rows = []


    for train_seed in TRAIN_SEEDS:

        summary, perm_df = run_seed(
            dataset,
            permutations,
            train_seed,
            runtime,
            args.n_permutations,
        )

        summaries.append(
            summary
        )

        all_perm_rows.append(
            perm_df
        )


        # Save progressive summary.
        pd.DataFrame(
            summaries
        ).to_csv(
            RESULTS_DIR
            / "real_gnn_seed_summary.csv",
            index=False,
        )


    # --------------------------------------------------------
    # Final combined outputs.
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        summaries
    )

    perm_df = pd.concat(
        all_perm_rows,
        ignore_index=True,
    )


    perm_df.to_csv(
        RESULTS_DIR
        / "real_gnn_permutation_runs.csv",
        index=False,
    )


    # --------------------------------------------------------
    # Aggregate final result across training seeds.
    # --------------------------------------------------------

    final = {
        "n_nodes": 16,

        "n_real_edges": 56,

        "n_train_seeds":
            len(TRAIN_SEEDS),

        "n_permutations_per_seed":
            args.n_permutations,

        "gru_ft_mae_mean":
            summary_df[
                "gru_ft_mae"
            ].mean(),

        "gru_ft_mae_std":
            summary_df[
                "gru_ft_mae"
            ].std(ddof=1),

        "real_gnn_mae_mean":
            summary_df[
                "real_gnn_mae"
            ].mean(),

        "real_gnn_mae_std":
            summary_df[
                "real_gnn_mae"
            ].std(ddof=1),

        "real_gain_over_gru_ft_mean_pct":
            summary_df[
                "real_gain_over_gru_ft_pct"
            ].mean(),

        "real_gain_over_gru_ft_std_pct":
            summary_df[
                "real_gain_over_gru_ft_pct"
            ].std(ddof=1),

        "real_vs_mean_perm_gain_mean_pct":
            summary_df[
                "real_vs_mean_perm_gain_pct"
            ].mean(),

        "real_vs_mean_perm_gain_std_pct":
            summary_df[
                "real_vs_mean_perm_gain_pct"
            ].std(ddof=1),

        "mean_percent_permutations_beaten":
            summary_df[
                "percent_permutations_beaten"
            ].mean(),

        "real_final_gate_mean":
            summary_df[
                "real_final_gate"
            ].mean(),

        "real_final_gate_std":
            summary_df[
                "real_final_gate"
            ].std(ddof=1),
    }


    pd.DataFrame(
        [final]
    ).to_csv(
        RESULTS_DIR
        / "real_gnn_final_summary.csv",
        index=False,
    )


    print()
    print("=" * 76)
    print("FINAL REAL-DATA VALIDATION")
    print("=" * 76)

    print(
        "GRU-FT MAE:",
        f"{final['gru_ft_mae_mean']:.6f}"
        f" +/- "
        f"{final['gru_ft_mae_std']:.6f}",
    )

    print(
        "REAL-POD-GNN MAE:",
        f"{final['real_gnn_mae_mean']:.6f}"
        f" +/- "
        f"{final['real_gnn_mae_std']:.6f}",
    )

    print(
        "REAL gain over GRU-FT:",
        f"{final['real_gain_over_gru_ft_mean_pct']:+.2f}"
        f" +/- "
        f"{final['real_gain_over_gru_ft_std_pct']:.2f}%",
    )

    print(
        "REAL vs mean PERM:",
        f"{final['real_vs_mean_perm_gain_mean_pct']:+.2f}"
        f" +/- "
        f"{final['real_vs_mean_perm_gain_std_pct']:.2f}%",
    )

    print(
        "Mean permutations beaten:",
        f"{final['mean_percent_permutations_beaten']:.1f}%",
    )

    print(
        "Final graph gate:",
        f"{final['real_final_gate_mean']:.4f}"
        f" +/- "
        f"{final['real_final_gate_std']:.4f}",
    )

    print()
    print("Saved:")

    print(
        RESULTS_DIR
        / "real_gnn_seed_summary.csv"
    )

    print(
        RESULTS_DIR
        / "real_gnn_permutation_runs.csv"
    )

    print(
        RESULTS_DIR
        / "real_gnn_final_summary.csv"
    )

    print(
        RESULTS_DIR
        / "real_pod16_station_list.csv"
    )


if __name__ == "__main__":
    main()
