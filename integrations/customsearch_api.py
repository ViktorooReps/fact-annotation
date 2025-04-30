import streamlit as st

import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urljoin


def search_urls(query: str, num_results: int = 1) -> list[str]:
    """
    Performs a Google Custom Search for the given query and returns
    a list of result URLs (best matches), using credentials from Streamlit secrets.
    """
    # Load API credentials from Streamlit secrets
    creds = st.secrets["google_customsearch"]
    api_key = creds.get("api_key")
    cse_id = creds.get("cse_id")

    endpoint = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cse_id,
        "q": query,
        "num": num_results
    }

    resp = requests.get(endpoint, params=params)
    resp.raise_for_status()
    data = resp.json()

    # Extract and return the direct link URLs
    return [item["link"] for item in data.get("items", []) if item.get("link")]


def get_organization_info(url, timeout=10, max_retries=1):
    """
    Fetches an organization's logo URL and description from its homepage.

    Args:
        url (str): The homepage URL of the organization.
        timeout (int): Seconds to wait for the HTTP response.
        max_retries (int): How many times to retry on 403/429 status.

    Returns:
        dict:
            {
                'logo':        Absolute URL to the logo image (or None),
                'description': Organization description text (or None)
            }
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/119.0.0.0 Safari/537.36"
        )
    }

    # 1. Fetch with optional retry on 403/429
    attempt = 0
    resp = None
    while attempt <= max_retries:
        try:
            resp = requests.get(url, timeout=timeout, headers=headers)
            resp.raise_for_status()
            break
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response else None
            if code in (403, 429) and attempt < max_retries:
                attempt += 1
                continue
            return {"logo": None, "description": None}
        except requests.exceptions.RequestException:
            return {"logo": None, "description": None}

    base = resp.url
    soup = BeautifulSoup(resp.text, "html.parser")

    description = None
    logo_candidates = []

    # 2. Extract description from JSON-LD
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "{}")
        except json.JSONDecodeError:
            continue
        entries = data if isinstance(data, list) else [data]
        for entry in entries:
            if entry.get("@type") == "Organization":
                if entry.get("description") and not description:
                    description = entry.get("description").strip()
                if entry.get("logo"):
                    logo_candidates.append(urljoin(base, entry.get("logo")))
        if description and logo_candidates:
            break

    # 3. Fallback description to meta tags
    if not description:
        meta = (
            soup.find("meta", attrs={"name": "description"}) or
            soup.find("meta", property="og:description")
        )
        if meta and meta.get("content"):
            description = meta["content"].strip()

    # 4. Add Open Graph image to candidates
    og_img = soup.find("meta", property="og:image")
    if og_img and og_img.get("content"):
        logo_candidates.append(urljoin(base, og_img["content"]))

    # 5. Gather <img> tags with "logo" in alt/class or svg extension as candidates
    for img in soup.find_all("img", src=True):
        src = urljoin(base, img["src"])
        alt = img.get("alt", "").lower()
        classes = " ".join(img.get("class", [])).lower()
        if "logo" in alt or "logo" in classes or src.lower().endswith(".svg"):
            logo_candidates.append(src)

    # 6. Score and pick best candidate
    def score_asset(asset_url: str) -> int:
        score = 0
        u = asset_url.lower()
        if "logo" in u:
            score += 2
        if u.endswith(".svg") or ".svg?" in u:
            score += 1
        return score

    # Remove duplicates while preserving order
    seen = set()
    unique_candidates = []
    for c in logo_candidates:
        if c not in seen:
            seen.add(c)
            unique_candidates.append(c)

    chosen_logo = None
    if unique_candidates:
        # sort by descending score, preserve first order on tie
        unique_candidates.sort(key=lambda u: score_asset(u), reverse=True)
        chosen_logo = unique_candidates[0]

    return {"logo": chosen_logo, "description": description}


if __name__ == "__main__":
    test_url = search_urls("Habitat for Humanity ReStore")[0]
    info = get_organization_info(test_url)
    print("URL:", test_url)
    print("Logo:", info["logo"])
    print("Description:", info["description"])
