import os
import json
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

SESSIONS_FILE = Path(
    "data/real_acn/caltech_selected_sessions.csv"
)

OUT_DIR = Path(
    "data/real_acn/timeseries_sessions"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

df = pd.read_csv(SESSIONS_FILE)

selected_station_ids = set(
    df["stationID"].astype(str)
)

START = "Tue, 01 May 2018 00:00:00 GMT"
END = "Sun, 01 Mar 2020 00:00:00 GMT"

BASE_URL = (
    "https://ev.caltech.edu/api/v1/"
    "sessions/caltech/ts"
)

where = (
    f'connectionTime>="{START}" '
    f'and connectionTime<"{END}"'
)

url = BASE_URL

params = {
    "where": where,
    "sort": "-connectionTime",
}

downloaded = 0
saved = 0

print("=" * 70)
print("DOWNLOADING SELECTED ACN TIME SERIES")
print("=" * 70)

while True:

    response = requests.get(
        url,
        auth=(TOKEN, ""),
        params=params,
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    items = data.get("_items", [])

    for session in items:

        downloaded += 1

        station_id = str(
            session.get("stationID")
        )

        if station_id not in selected_station_ids:
            continue

        session_id = session.get(
            "sessionID",
            session.get("_id")
        )

        if not session_id:
            continue

        safe_id = str(session_id).replace(
            "/",
            "_"
        )

        outfile = OUT_DIR / f"{safe_id}.json"

        if not outfile.exists():
            with open(
                outfile,
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    session,
                    f
                )

            saved += 1

        if saved % 100 == 0 and saved > 0:
            print(
                f"saved={saved} "
                f"scanned={downloaded}"
            )

    links = data.get("_links", {})

    next_link = links.get("next")

    if not next_link:
        break

    href = next_link.get("href")

    if not href:
        break

    if href.startswith("http"):
        url = href
    else:
        url = (
            "https://ev.caltech.edu/api/v1/"
            + href.lstrip("/")
        )

    # next URL already contains query arguments
    params = None

    time.sleep(0.05)


print()
print("=" * 70)
print("DOWNLOAD COMPLETE")
print("=" * 70)

print("Sessions scanned:", downloaded)
print("Selected time-series sessions saved:", saved)
print("Output directory:", OUT_DIR)
