from pathlib import Path
import pandas as pd

FILE = Path("data/real_acn/caltech_sessions.csv")

df = pd.read_csv(FILE)

df["connectionTime"] = pd.to_datetime(
    df["connectionTime"],
    utc=True,
    errors="coerce"
)

df["kWhDelivered"] = pd.to_numeric(
    df["kWhDelivered"],
    errors="coerce"
)

df = df.dropna(
    subset=["connectionTime", "stationID", "kWhDelivered"]
)

# ---------------------------------------------------------
# Monthly statistics
# ---------------------------------------------------------

df["month"] = (
    df["connectionTime"]
    .dt.to_period("M")
    .astype(str)
)

monthly = (
    df.groupby("month")
    .agg(
        sessions=("stationID", "size"),
        active_stations=("stationID", "nunique"),
        total_kwh=("kWhDelivered", "sum"),
    )
    .reset_index()
)

print("=" * 75)
print("MONTHLY ACN CALTECH COVERAGE")
print("=" * 75)

print(monthly.to_string(index=False))

# ---------------------------------------------------------
# Yearly statistics
# ---------------------------------------------------------

df["year"] = df["connectionTime"].dt.year

yearly = (
    df.groupby("year")
    .agg(
        sessions=("stationID", "size"),
        active_stations=("stationID", "nunique"),
        total_kwh=("kWhDelivered", "sum"),
    )
    .reset_index()
)

print()
print("=" * 75)
print("YEARLY SUMMARY")
print("=" * 75)

print(yearly.to_string(index=False))

# ---------------------------------------------------------
# Station activity BEFORE March 2020
# ---------------------------------------------------------

pre = df[
    (df["connectionTime"] >= "2018-05-01")
    & (df["connectionTime"] < "2020-03-01")
].copy()

pre_stats = (
    pre.groupby("stationID")
    .agg(
        sessions=("stationID", "size"),
        active_days=("connectionTime",
                     lambda x: x.dt.date.nunique()),
        total_kwh=("kWhDelivered", "sum"),
        first_session=("connectionTime", "min"),
        last_session=("connectionTime", "max"),
    )
    .reset_index()
)

pre_stats = pre_stats.sort_values(
    "sessions",
    ascending=False
)

print()
print("=" * 75)
print("PRE-MARCH-2020 STATION ACTIVITY")
print("=" * 75)

print(pre_stats.to_string(index=False))

OUTPUT = Path(
    "data/real_acn/caltech_pre2020_station_summary.csv"
)

pre_stats.to_csv(
    OUTPUT,
    index=False
)

print()
print("Saved:")
print(OUTPUT)
