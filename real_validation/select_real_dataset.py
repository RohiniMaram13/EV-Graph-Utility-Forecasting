from pathlib import Path
import pandas as pd

SESSIONS_FILE = Path("data/real_acn/caltech_sessions.csv")
OUT_DIR = Path("data/real_acn")
OUT_DIR.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2018-05-01", tz="UTC")
END = pd.Timestamp("2020-03-01", tz="UTC")

MIN_ACTIVE_DAYS = 300

df = pd.read_csv(SESSIONS_FILE)

df["connectionTime"] = pd.to_datetime(
    df["connectionTime"],
    utc=True,
    errors="coerce"
)

df["disconnectTime"] = pd.to_datetime(
    df["disconnectTime"],
    utc=True,
    errors="coerce"
)

df["kWhDelivered"] = pd.to_numeric(
    df["kWhDelivered"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "stationID",
        "connectionTime",
        "disconnectTime",
        "kWhDelivered",
    ]
)

# ------------------------------------------------------------
# Freeze analysis period
# ------------------------------------------------------------

period = df[
    (df["connectionTime"] >= START) &
    (df["connectionTime"] < END)
].copy()

# ------------------------------------------------------------
# Count active charging days by station
# ------------------------------------------------------------

period["active_date"] = period["connectionTime"].dt.date

station_stats = (
    period.groupby("stationID")
    .agg(
        sessions=("stationID", "size"),
        active_days=("active_date", "nunique"),
        total_kwh=("kWhDelivered", "sum"),
        first_session=("connectionTime", "min"),
        last_session=("connectionTime", "max"),
    )
    .reset_index()
)

selected = station_stats[
    station_stats["active_days"] >= MIN_ACTIVE_DAYS
].copy()

selected = selected.sort_values(
    "stationID"
).reset_index(drop=True)

selected["node_id"] = range(len(selected))

selected.to_csv(
    OUT_DIR / "caltech_selected_stations.csv",
    index=False
)

selected_ids = set(selected["stationID"])

filtered = period[
    period["stationID"].isin(selected_ids)
].copy()

filtered.to_csv(
    OUT_DIR / "caltech_selected_sessions.csv",
    index=False
)

print("=" * 70)
print("FROZEN REAL-DATA VALIDATION SET")
print("=" * 70)

print("Period:")
print(START)
print("to")
print(END)

print("\nMinimum active days:")
print(MIN_ACTIVE_DAYS)

print("\nSelected stations:")
print(len(selected))

print("\nSelected sessions:")
print(len(filtered))

print("\nTotal energy:")
print(f"{filtered['kWhDelivered'].sum():,.2f} kWh")

print("\nStations:")
print(
    selected[
        [
            "node_id",
            "stationID",
            "sessions",
            "active_days",
            "total_kwh",
        ]
    ].to_string(index=False)
)

print("\nSaved:")
print(OUT_DIR / "caltech_selected_stations.csv")
print(OUT_DIR / "caltech_selected_sessions.csv")
