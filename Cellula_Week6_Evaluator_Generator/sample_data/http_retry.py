"""Small HTTP client with retry and backoff used by the data team."""

import time

import requests

MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 2


def get_json(url, timeout=10):
    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            return response.json()
        if response.status_code in (429, 500, 502, 503, 504):
            time.sleep(BACKOFF_SECONDS * attempt)
            continue
        response.raise_for_status()
    raise RuntimeError(f"Request failed after {MAX_ATTEMPTS} attempts: {url}")
