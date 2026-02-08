"""
Deep Dive into Kalshi Markets - Why are prices 0?
Let's investigate the actual market structure
"""

import requests
import json
from datetime import datetime

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json"
})

print("="*80)
print("DEEP DIVE: Why are Kalshi market prices 0?")
print("="*80)

# Fetch markets
print("\nStep 1: Fetching markets...")
response = session.get(f"{BASE_URL}/markets", params={"limit": 20})

if not response.ok:
    print(f"Error: {response.status_code}")
    print(response.text)
    exit(1)

markets = response.json().get("markets", [])
print(f"Got {len(markets)} markets\n")

# Analyze different market types
price_analysis = {
    "zero_price": [],
    "non_zero_price": [],
    "no_price_field": []
}

status_counts = {}
type_counts = {}

for market in markets:
    # Status analysis
    status = market.get('status', 'unknown')
    status_counts[status] = status_counts.get(status, 0) + 1
    
    # Type analysis
    ticker = market.get('ticker', '')
    if 'MULTIGAME' in ticker:
        market_type = 'MULTIGAME'
    elif 'KXMV' in ticker:
        market_type = 'KXMV'
    else:
        market_type = 'STANDARD'
    type_counts[market_type] = type_counts.get(market_type, 0) + 1
    
    # Price analysis
    last_price = market.get('last_price')
    
    if last_price is None:
        price_analysis['no_price_field'].append(market)
    elif last_price == 0:
        price_analysis['zero_price'].append(market)
    else:
        price_analysis['non_zero_price'].append(market)

# Report findings
print("="*80)
print("FINDINGS")
print("="*80)

print(f"\n1. STATUS DISTRIBUTION:")
for status, count in sorted(status_counts.items()):
    print(f"   {status}: {count} markets")

print(f"\n2. TYPE DISTRIBUTION:")
for mtype, count in sorted(type_counts.items()):
    print(f"   {mtype}: {count} markets")

print(f"\n3. PRICE DISTRIBUTION:")
print(f"   Zero price (0): {len(price_analysis['zero_price'])} markets")
print(f"   Non-zero price: {len(price_analysis['non_zero_price'])} markets")
print(f"   No price field: {len(price_analysis['no_price_field'])} markets")

# Show examples of each type
print("\n" + "="*80)
print("DETAILED EXAMPLES")
print("="*80)

if price_analysis['non_zero_price']:
    print(f"\n✅ EXAMPLE OF NON-ZERO PRICE MARKET:")
    print("-"*80)
    market = price_analysis['non_zero_price'][0]
    print(json.dumps(market, indent=2))
    print(f"\nKey fields:")
    print(f"  Ticker: {market.get('ticker')}")
    print(f"  Title: {market.get('title', '')[:80]}")
    print(f"  Status: {market.get('status')}")
    print(f"  Last Price: {market.get('last_price')} cents = ${market.get('last_price', 0)/100:.2f}")
    print(f"  Volume 24h: {market.get('volume_24h', 0)}")
    print(f"  Open Interest: {market.get('open_interest', 0)}")
else:
    print("\n⚠️ NO MARKETS WITH NON-ZERO PRICES FOUND!")

if price_analysis['zero_price']:
    print(f"\n⚠️ EXAMPLE OF ZERO PRICE MARKET:")
    print("-"*80)
    market = price_analysis['zero_price'][0]
    print(json.dumps(market, indent=2)[:1000])
    print("...")
    print(f"\nKey fields:")
    print(f"  Ticker: {market.get('ticker')}")
    print(f"  Title: {market.get('title', '')[:80]}")
    print(f"  Status: {market.get('status')}")
    print(f"  Last Price: {market.get('last_price')}")
    print(f"  Created: {market.get('created_time', 'N/A')}")
    print(f"  Close Time: {market.get('close_time', 'N/A')}")

# Check if there are other endpoints with better data
print("\n" + "="*80)
print("ALTERNATIVE APPROACHES")
print("="*80)

print("\n1. Try fetching with different status filters:")
for status_filter in ['active', 'open', 'closed', 'settled']:
    try:
        resp = session.get(f"{BASE_URL}/markets", params={"limit": 5, "status": status_filter})
        if resp.ok:
            data = resp.json()
            count = len(data.get("markets", []))
            print(f"   status='{status_filter}': {count} markets")
            
            # Check if any have prices
            markets_with_status = data.get("markets", [])
            non_zero = [m for m in markets_with_status if m.get('last_price', 0) > 0]
            if non_zero:
                print(f"     → {len(non_zero)} with non-zero prices! ✓")
        else:
            print(f"   status='{status_filter}': Error {resp.status_code}")
    except:
        print(f"   status='{status_filter}': Exception")

print("\n2. Check if there's an 'events' endpoint:")
try:
    resp = session.get(f"{BASE_URL}/events", params={"limit": 5})
    if resp.ok:
        events = resp.json().get("events", [])
        print(f"   ✓ Events endpoint works! Got {len(events)} events")
        if events:
            print(f"   Sample event: {events[0].get('ticker', 'N/A')}")
    else:
        print(f"   ✗ Events endpoint: {resp.status_code}")
except Exception as e:
    print(f"   ✗ Events endpoint error: {e}")

print("\n3. Try looking at series/tickers:")
try:
    resp = session.get(f"{BASE_URL}/series", params={"limit": 5})
    if resp.ok:
        series = resp.json().get("series", [])
        print(f"   ✓ Series endpoint works! Got {len(series)} series")
    else:
        print(f"   ✗ Series endpoint: {resp.status_code}")
except Exception as e:
    print(f"   ✗ Series endpoint error: {e}")

# Final recommendations
print("\n" + "="*80)
print("CONCLUSIONS & RECOMMENDATIONS")
print("="*80)

if len(price_analysis['non_zero_price']) > 0:
    print(f"""
✅ GOOD NEWS: Found {len(price_analysis['non_zero_price'])} markets with prices!

Recommendations:
1. Filter for markets with last_price > 0
2. These markets likely have actual trading activity
3. Focus on these for predictions
""")
elif len(price_analysis['zero_price']) == len(markets):
    print(f"""
⚠️ ISSUE: ALL {len(markets)} markets have price = 0

Possible reasons:
1. These are new/inactive markets that haven't started trading
2. MULTIGAME markets might not have direct prices
3. Need to check individual contract prices within markets
4. May need authentication to see real prices
5. API might be showing only inactive markets

Recommendations:
1. Try filtering by different status (e.g., status='open' vs 'active')
2. Try the /events endpoint for active events
3. Check if authentication provides better data
4. Look for markets with volume_24h > 0
""")

print("\n" + "="*80)
print("END OF ANALYSIS")
print("="*80)
