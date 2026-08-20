from pathlib import Path

import pandas as pd


FILE = Path("data/real_acn/caltech_sessions.csv")

df = pd.read_csv(FILE)

df["connectionTime"] = pd.to_datetime(
    df["connectionTime"],
    utc=True,
    errors="coerce",
)

df["disconnectTime"] = pd.to_datetime(
    df["disconnectTime"],
    utc=True,
    errors="coerce",
)

df["kWhDelivered"] = pd.to_numeric(
    df["kWhDelivered"],
    errors="coerce",
)

print("=" * 70)
print("ACN CALTECH DATASET")
print("=" * 70)

print("\nTotal sessions:")
print(len(df))

print("\nUnique stations:")
print(df["stationID"].nunique())

print("\nUnique spaces:")
print(df["spaceID"].nunique())

print("\nUnique clusters:")
print(df["clusterID"].nunique())

print("\nCluster IDs:")
print(sorted(df["clusterID"].dropna().astype(str).unique()))

print("\nDataset period:")
print("First:", df["connectionTime"].min())
print("Last :", df["connectionTime"].max())

print("\nTotal delivered energy:")
print(f"{df['kWhDelivered'].sum():,.2f} kWh")


# -------------------------------------------------------
# Station-level statistics
# -------------------------------------------------------

station_stats = (
    df.groupby("stationID")
    .agg(
        sessions=("stationID", "size"),
        first_session=("connectionTime", "min"),
        last_session=("connectionTime", "max"),
        total_kwh=("kWhDelivered", "sum"),
        mean_kwh=("kWhDelivered", "mean"),
        clusterID=("clusterID", "first"),
    )
    .reset_index()
)

station_stats["coverage_days"] = (
    station_stats["last_session"]
    - station_stats["first_session"]
).dt.total_seconds() / 86400


station_stats = station_stats.sort_values(
    "sessions",
    ascending=False,
)


print("\n" + "=" * 70)
print("STATION STATISTICS")
print("=" * 70)

print(
    station_stats[
        [
            "stationID",
            "clusterID",
            "sessions",
            "coverage_days",
            "total_kwh",
            "first_session",
            "last_session",
        ]
    ].to_string(index=False)
)


# -------------------------------------------------------
# Save station summary
# -------------------------------------------------------

OUTPUT = Path(
    "data/real_acn/caltech_station_summary.csv"
)

station_stats.to_csv(
    OUTPUT,
    index=False,
)

print()
print("Station summary saved to:")
print(OUTPUT)
