import pandas as pd
from pathlib import Path
from acnportal import acnsim

SESSIONS = Path(
    "data/real_acn/caltech_selected_sessions.csv"
)

STATIONS = Path(
    "data/real_acn/caltech_selected_stations.csv"
)

OUTPUT = Path(
    "data/real_acn/caltech_station_mapping.csv"
)

# ----------------------------------------------------------
# Load data
# ----------------------------------------------------------

sessions = pd.read_csv(SESSIONS)
selected = pd.read_csv(STATIONS)

sessions["stationID"] = sessions["stationID"].astype(str)
sessions["spaceID"] = sessions["spaceID"].astype(str)

selected["stationID"] = selected["stationID"].astype(str)

# ----------------------------------------------------------
# Check stationID -> spaceID mapping
# ----------------------------------------------------------

mapping_counts = (
    sessions
    .groupby("stationID")["spaceID"]
    .nunique()
)

print("=" * 72)
print("STATION -> SPACE CONSISTENCY")
print("=" * 72)

print("\nSelected stations:")
print(len(selected))

print("\nStations with exactly one spaceID:")
print((mapping_counts == 1).sum())

print("\nStations with multiple spaceIDs:")
print((mapping_counts > 1).sum())

if (mapping_counts > 1).any():

    print("\nWARNING: MULTIPLE SPACE IDs")

    for station in mapping_counts[
        mapping_counts > 1
    ].index:

        values = sorted(
            sessions.loc[
                sessions["stationID"] == station,
                "spaceID"
            ].unique()
        )

        print(station, "->", values)

# ----------------------------------------------------------
# Build mapping table
# ----------------------------------------------------------

mapping = (
    sessions[
        ["stationID", "spaceID"]
    ]
    .drop_duplicates()
)

mapping = selected[
    ["node_id", "stationID"]
].merge(
    mapping,
    on="stationID",
    how="left"
)

# ----------------------------------------------------------
# Compare against actual ACN-Sim network
# ----------------------------------------------------------

network = acnsim.sites.caltech_acn()

sim_ids = set(
    network.station_ids
)

mapping["in_acnsim"] = (
    mapping["spaceID"].isin(sim_ids)
)

print()
print("=" * 72)
print("FINAL MAPPING")
print("=" * 72)

print(
    mapping.to_string(
        index=False
    )
)

print()
print("Mapped into ACN-Sim:")
print(
    f"{mapping['in_acnsim'].sum()} / {len(mapping)}"
)

print()
print("Missing from ACN-Sim:")

missing = mapping[
    ~mapping["in_acnsim"]
]

if len(missing) == 0:
    print("NONE")
else:
    print(
        missing.to_string(
            index=False
        )
    )

# ----------------------------------------------------------
# Save
# ----------------------------------------------------------

mapping.to_csv(
    OUTPUT,
    index=False
)

print()
print("Saved:")
print(OUTPUT)
