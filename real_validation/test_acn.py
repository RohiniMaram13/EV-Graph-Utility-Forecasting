import os
import requests

TOKEN = os.environ.get("ACN_API_TOKEN")

if not TOKEN:
    raise RuntimeError(
        "ACN_API_TOKEN is not set. "
        "Run: export ACN_API_TOKEN='your-token'"
    )

url = "https://ev.caltech.edu/api/v1/sessions/caltech"

response = requests.get(
    url,
    auth=(TOKEN, ""),
    params={"sort": "-connectionTime"},
    timeout=60,
)

print("HTTP status:", response.status_code)

response.raise_for_status()

data = response.json()

print("Metadata:")
print(data.get("_meta"))

sessions = data.get("_items", [])

print("\nSessions on this page:", len(sessions))

if sessions:
    s = sessions[0]

    print("\nExample session:")
    for key in [
        "stationID",
        "spaceID",
        "clusterID",
        "siteID",
        "connectionTime",
        "disconnectTime",
        "doneChargingTime",
        "kWhDelivered",
        "timezone",
    ]:
        print(f"{key}: {s.get(key)}")
