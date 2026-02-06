Based on your description and the code analysis, the issue stems from two likely causes in `backend/kalshi_service.py`:
1.  **Missing Headers**: The Kalshi API (like many others) likely blocks requests that don't have a valid `User-Agent` header, causing the GET call to fail or return 403 Forbidden.
2.  **Filtering Logic Error**: When `simple_test.py` requests `limit=3` for the "Trump" category, the current code passes `limit=3` directly to the API. The API returns the first 3 *arbitrary* markets (likely not Trump-related), and then the code filters them locally, resulting in an empty list.

I will fix this by modifying `backend/kalshi_service.py`:

1.  **Add User-Agent Header**: Update the `__init__` method to inject a standard browser `User-Agent` into the requests session.
2.  **Fix Pagination/Filtering Logic**: In `get_markets_by_category`:
    *   Fetch a larger batch of markets (e.g., 100) from the API regardless of the requested limit.
    *   Perform the filtering (e.g., for "Trump").
    *   Apply the `limit` *after* finding the matching markets.
3.  **Enhance Error Handling**: Add detailed error logging (status code, response text) to help diagnose any remaining API issues.

After these changes, I will run `backend/simple_test.py` to verify the fix.