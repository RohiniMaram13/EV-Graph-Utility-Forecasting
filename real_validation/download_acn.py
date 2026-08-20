import os
import time
from pathlib import Path

import pandas as pd
import requests


TOKEN = os.environ.get("ACN_API_TOKEN")

if not TOKEN:
    raise RuntimeError(
        "ACN_API_TOKEN is not set.\n"
        "Run: export ACN_API_TOKEN='your-token'"
    )

URL = "https://ev.caltech.edu/api/v1/sessions/caltech"

OUTPUT_DIR = Path("data/real_acn")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "caltech_sessions.csv"

rows = []

page = 1

print("Downloading Caltech ACN session metadata...")

while True:

    response = requests.get(
        URL,
        auth=(TOKEN, ""),
        params={
            "page": page,
            "sort": "connectionTime",
        },
        timeout=60,
    )

    response.raise_for_status()

    data = response.json()

    sessions = data.get("_items", [])

    if not sessions:
        break

    for s in sessions:

        rows.append(
            {
                "stationID": s.get("stationID"),
                "spaceID": s.get("spaceID"),
                "clusterID": s.get("clusterID"),
                "siteID": s.get("siteID"),
                "connectionTime": s.get("connectionTime"),
                "disconnectTime": s.get("disconnectTime"),
                "doneChargingTime": s.get("doneChargingTime"),
                "kWhDelivered": s.get("kWhDelivered"),
                "timezone": s.get("timezone"),
            }
        )

    meta = data.get("_meta", {})

    total = meta.get("total", "?")

    print(
        f"page={page}  downloaded={len(rows)} / {total}"
    )

    if len(rows) >= int(total):
        break

    page += 1

    # Be polite to the public API.
    time.sleep(0.05)


df = pd.DataFrame(rows)

df.to_csv(
    OUTPUT_FILE,
    index=False,
)

print()
print("Download complete.")
print("Sessions:", len(df))
print("Unique stations:", df["stationID"].nunique())
print("Unique clusters:", df["clusterID"].nunique())

print()
print("Saved:")
print(OUTPUT_FILE)
