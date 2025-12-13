import requests

URL = "http://localhost:6333/collections/food_recipes_two_step/points/scroll"

seen = set()
offset = None
total = 0

while True:
    body = {
        "limit": 1000,
        "with_payload": ["country"],
        "with_vector": False,
    }
    if offset:
        body["offset"] = offset

    r = requests.post(URL, json=body).json()
    points = r["result"]["points"]

    if not points:
        break

    for p in points:
        country = p.get("payload", {}).get("country")
        if country:
            seen.add(country)

    total += len(points)
    offset = r["result"].get("next_page_offset")
    if not offset:
        break

print("Total points scanned:", total)
print("Unique countries:", sorted(seen))
print("Number of unique countries:", len(seen))
