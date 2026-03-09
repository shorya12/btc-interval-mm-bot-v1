import time
import requests
import json

GAMMA = "https://gamma-api.polymarket.com"

def get_json(url, params=None):
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    time.sleep(0.1)
    return r.json()

print("Exploring NFL market structure...\n")

# Get first 20 NFL events
params = {"tag": "NFL", "limit": 20, "offset": 0}
events = get_json(f"{GAMMA}/events", params=params)

print(f"Found {len(events)} events\n")

# Group by category
categories = {}
for ev in events:
    markets = ev.get("markets", [])
    if markets:
        m = markets[0]
        cat = m.get("category")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append({
            "title": ev.get("title"),
            "question": m.get("question"),
            "slug": m.get("slug"),
        })

print("Events grouped by category:")
for cat, items in categories.items():
    print(f"\n{cat}: {len(items)} events")
    for item in items[:3]:  # Show first 3
        print(f"  - {item['title']}")

# Look for actual NFL team matchup patterns
print("\n" + "="*60)
print("Looking for team vs team matchups...")
print("="*60)

nfl_teams = ["Bills", "Dolphins", "Patriots", "Jets", "Ravens", "Bengals", "Browns", "Steelers", 
             "Texans", "Colts", "Jaguars", "Titans", "Chiefs", "Raiders", "Chargers", "Broncos",
             "Cowboys", "Giants", "Eagles", "Commanders", "Packers", "Bears", "Lions", "Vikings",
             "Buccaneers", "Saints", "Panthers", "Falcons", "49ers", "Seahawks", "Cardinals", "Rams"]

for ev in events[:20]:
    title = ev.get("title", "")
    # Check if title contains team names and looks like a matchup
    if any(team in title for team in nfl_teams):
        markets = ev.get("markets", [])
        if markets:
            m = markets[0]
            print(f"\nEvent: {title}")
            print(f"  Category: {m.get('category')}")
            print(f"  Question: {m.get('question')}")
            print(f"  Outcomes: {m.get('outcomes')}")
            print(f"  Market Type: {m.get('marketType')}")
            print(f"  Slug: {m.get('slug')}")
