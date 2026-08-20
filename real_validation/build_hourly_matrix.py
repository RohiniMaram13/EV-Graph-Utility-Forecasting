import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION -- FROZEN BEFORE MODEL TRAINING
# ============================================================

JSON_DIR = Path("data/real_acn/timeseries_sessions")

STATIONS_FILE = Path(
    "data/real_acn/caltech_selected_stations.csv"
)

OUT_DIR = Path("data/real_acn")
OUT_DIR.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2018-05-01 00:00:00", tz="UTC")
END = pd.Timestamp("2020-03-01 00:00:00", tz="UTC")


# ============================================================
# LOAD FROZEN STATION LIST
# ============================================================

stations_df = pd.read_csv(STATIONS_FILE)

stations_df = stations_df.sort_values("node_id")

stations = stations_df["stationID"].astype(str).tolist()

station_to_idx = {
    station: i
    for i, station in enumerate(stations)
}

N = len(stations)


# ============================================================
# CREATE COMPLETE HOURLY TIMELINE
# ============================================================

hours = pd.date_range(
    start=START,
    end=END,
    freq="h",
    inclusive="left",
)

T = len(hours)

hour_to_idx = {
    timestamp: i
    for i, timestamp in enumerate(hours)
}


# Matrix is:
#
# rows    = hours
# columns = stations
#
# values = delivered kWh assigned to that hour

energy = np.zeros(
    (T, N),
    dtype=np.float32,
)


print("=" * 72)
print("BUILDING ACN HOURLY ENERGY MATRIX")
print("=" * 72)

print("Stations:", N)
print("Hours:", T)
print("Start:", START)
print("End:", END)
print()


# ============================================================
# PROCESS EACH SESSION
# ============================================================

files = list(JSON_DIR.glob("*.json"))

print("JSON files:", len(files))
print()

processed = 0
usable = 0
missing_timeseries = 0
bad_timeseries = 0
zero_current = 0
outside_period = 0

input_energy_usable = 0.0
allocated_energy = 0.0


for file_number, path in enumerate(files, start=1):

    try:

        with open(path, "r", encoding="utf-8") as f:
            session = json.load(f)

        station = str(session.get("stationID"))

        if station not in station_to_idx:
            continue

        kwh = session.get("kWhDelivered")

        if kwh is None:
            continue

        try:
            kwh = float(kwh)
        except (TypeError, ValueError):
            continue

        if not np.isfinite(kwh) or kwh < 0:
            continue

        charging = session.get("chargingCurrent")

        if charging is None:
            missing_timeseries += 1
            continue

        timestamps = charging.get("timestamps")
        currents = charging.get("current")

        if timestamps is None or currents is None:
            bad_timeseries += 1
            continue

        if len(timestamps) != len(currents):
            bad_timeseries += 1
            continue

        if len(timestamps) < 2:
            bad_timeseries += 1
            continue


        # ----------------------------------------------------
        # Parse measurements
        # ----------------------------------------------------

        ts = pd.to_datetime(
            timestamps,
            utc=True,
            errors="coerce",
        )

        current = pd.to_numeric(
            pd.Series(currents),
            errors="coerce",
        ).to_numpy(dtype=float)


        valid = (
            ~pd.isna(ts)
            & np.isfinite(current)
        )

        ts = ts[valid]
        current = current[valid]

        if len(ts) < 2:
            bad_timeseries += 1
            continue


        # Negative current is not meaningful charging load.
        current = np.maximum(current, 0.0)


        # ----------------------------------------------------
        # Sort measurements by timestamp
        # ----------------------------------------------------

        order = np.argsort(ts.asi8)

        ts = ts[order]
        current = current[order]


        # ----------------------------------------------------
        # Integrate current over each measurement interval.
        #
        # We do NOT assume a voltage.
        #
        # Current * duration is used only as a proportional
        # weight for distributing the session's known
        # kWhDelivered.
        # ----------------------------------------------------

        t_seconds = ts.asi8.astype(np.float64) / 1e9

        dt = np.diff(t_seconds)

        # Weight interval using trapezoidal current.
        avg_current = (
            current[:-1] + current[1:]
        ) / 2.0

        interval_weight = avg_current * dt


        # Remove invalid / extremely long telemetry gaps.
        #
        # ACN samples normally occur every few seconds.
        # A gap > 5 minutes should not be integrated as though
        # current were continuously observed.
        valid_interval = (
            np.isfinite(interval_weight)
            & np.isfinite(dt)
            & (dt > 0)
            & (dt <= 300)
        )

        interval_weight = interval_weight[valid_interval]

        interval_start = ts[:-1][valid_interval]
        interval_end = ts[1:][valid_interval]


        total_weight = interval_weight.sum()

        if total_weight <= 0:

            # If kWh is also essentially zero, nothing needs
            # to be allocated.
            if kwh <= 1e-8:
                processed += 1
                usable += 1
                continue

            zero_current += 1
            continue


        input_energy_usable += kwh


        # ----------------------------------------------------
        # Convert each interval weight into session-energy
        # contribution.
        # ----------------------------------------------------

        interval_energy = (
            kwh * interval_weight / total_weight
        )


        # ----------------------------------------------------
        # Allocate interval energy into hourly bins.
        #
        # Most intervals are only a few seconds long, but this
        # handles intervals crossing an hour boundary exactly.
        # ----------------------------------------------------

        station_idx = station_to_idx[station]

        session_allocated = 0.0


        for start_t, end_t, e in zip(
            interval_start,
            interval_end,
            interval_energy,
        ):

            if e <= 0:
                continue

            if end_t <= START or start_t >= END:
                continue

            start_t = max(start_t, START)
            end_t = min(end_t, END)

            duration = (
                end_t - start_t
            ).total_seconds()

            if duration <= 0:
                continue


            cursor = start_t

            while cursor < end_t:

                hour_start = cursor.floor("h")
                next_hour = hour_start + pd.Timedelta(hours=1)

                segment_end = min(
                    end_t,
                    next_hour,
                )

                segment_seconds = (
                    segment_end - cursor
                ).total_seconds()

                fraction = segment_seconds / duration

                if hour_start in hour_to_idx:

                    hour_idx = hour_to_idx[hour_start]

                    contribution = e * fraction

                    energy[
                        hour_idx,
                        station_idx
                    ] += contribution

                    session_allocated += contribution

                cursor = segment_end


        allocated_energy += session_allocated

        if session_allocated > 0:
            usable += 1
        else:
            outside_period += 1

        processed += 1


    except Exception as exc:

        bad_timeseries += 1

        if bad_timeseries <= 10:
            print(
                "\nWARNING:",
                path.name,
                repr(exc),
            )


    if file_number % 500 == 0:

        print(
            f"files={file_number:5d}/{len(files)} "
            f"usable={usable:5d} "
            f"missing_ts={missing_timeseries:4d} "
            f"bad={bad_timeseries:4d}"
        )


# ============================================================
# SAVE MATRIX
# ============================================================

matrix_df = pd.DataFrame(
    energy,
    index=hours,
    columns=stations,
)

matrix_df.index.name = "timestamp"

CSV_FILE = OUT_DIR / "caltech_hourly_energy.csv"

matrix_df.to_csv(CSV_FILE)


# Save compressed NumPy version for fast model loading.

NPZ_FILE = OUT_DIR / "caltech_hourly_energy.npz"

np.savez_compressed(
    NPZ_FILE,
    energy=energy,
    timestamps=hours.astype(str).to_numpy(),
    stations=np.array(stations),
)


# ============================================================
# QUALITY CONTROL
# ============================================================

hourly_total = energy.sum(axis=1)

station_total = energy.sum(axis=0)

nonzero_hours = int(
    np.count_nonzero(hourly_total > 0)
)

all_zero_hours = int(
    np.count_nonzero(hourly_total == 0)
)

total_matrix_energy = float(
    energy.sum()
)


print()
print("=" * 72)
print("HOURLY MATRIX COMPLETE")
print("=" * 72)

print("Shape:")
print(energy.shape)

print()
print("Stations:")
print(N)

print()
print("Hours:")
print(T)

print()
print("Usable time-series sessions:")
print(usable)

print()
print("Missing chargingCurrent:")
print(missing_timeseries)

print()
print("Bad time series:")
print(bad_timeseries)

print()
print("Zero-current sessions with positive kWh:")
print(zero_current)

print()
print("Sessions outside analysis period:")
print(outside_period)

print()
print("Input kWh from usable sessions:")
print(f"{input_energy_usable:,.3f}")

print()
print("Energy represented in hourly matrix:")
print(f"{total_matrix_energy:,.3f} kWh")

if input_energy_usable > 0:

    retention = (
        100.0
        * total_matrix_energy
        / input_energy_usable
    )

    print()
    print("Energy retention:")
    print(f"{retention:.4f}%")


print()
print("Hours with charging:")
print(nonzero_hours)

print()
print("All-zero hours:")
print(all_zero_hours)

print()
print("Fraction of hours with charging:")
print(
    f"{100 * nonzero_hours / T:.2f}%"
)


print()
print("Per-station matrix energy:")
for station, value in zip(
    stations,
    station_total,
):
    print(
        f"{station:20s} "
        f"{value:10.2f} kWh"
    )


print()
print("Saved:")
print(CSV_FILE)
print(NPZ_FILE)
