from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
import logging
import urllib.parse
import requests
from concurrent.futures import ThreadPoolExecutor

from duckduckgo_search import DDGS
from commons.website_parsing import get_organization_info
import streamlit as st

# ---------------------------------------------------------------------------
# Exceptions & logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

class EntityLookupError(Exception):
    """Base class for all lookup‑related errors."""

class NetworkError(EntityLookupError):
    """Raised when the HTTP request fails or times out."""

class ApiError(EntityLookupError):
    """Raised when the API returns an unexpected payload."""

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def get_file_url(filename: str, size: int | None = None) -> str:
    """Build a direct, CDN‑backed URL for a Wikimedia Commons file."""
    encoded = urllib.parse.quote(filename.replace(" ", "_"))
    url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{encoded}"
    if size:
        url += f"?width={size}"
    return url

@st.cache_data(show_spinner=False)
def verify_logo(logo_url: str) -> bool:
    try:
        response = requests.get(logo_url)
        # Check if the status code is 200 (OK)
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.exceptions.RequestException:
        # Catch any exceptions (e.g., network issues, invalid URL)
        return False

@st.cache_data(show_spinner=False)
def wiki_lookup(title: str) -> Dict[str, Optional[str]]:
    """Return a summary card for *title* from the REST API.

    On failure, an empty card is returned so the caller can still build a
    partial entity result.
    """
    try:
        summary = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
        ).json()
    except requests.exceptions.RequestException as exc:
        logger.warning("wiki_lookup failed for %s: %s", title, exc)
        return {"title": title, "url": None, "description": None, "thumbnail": None}

    return {
        "title": summary.get("title"),
        "url": summary.get("content_urls", {}).get("desktop", {}).get("page"),
        "description": summary.get("extract"),
        "thumbnail": summary.get("thumbnail", {}).get("source"),
    }

def _enrich_entity(result: Dict[str, Any], lang: str) -> Dict[str, Optional[str]]:
    id_ = result["id"]

    entities = requests.get(
        "https://www.wikidata.org/w/api.php",
        params={
            "action": "wbgetentities",
            "ids": id_,
            "props": "claims",
            "languages": lang,
            "sitefilter": "enwiki",
            "format": "json",
        },
    ).json()

    claims = entities.get("entities", {}).get(id_, {}).get("claims", {})

    # logo (P154)
    logo_url = None
    if "P154" in claims:
        try:
            logo_claim = claims["P154"][0]["mainsnak"].get("datavalue", {})
            if logo_claim:
                logo_url = get_file_url(logo_claim["value"], size=200)
                if not verify_logo(logo_url):
                    logo_url = None
        except (KeyError, TypeError) as exc:
            logger.debug("Malformed logo claim for %s: %s", id_, exc)

    # official website (P856)
    official_site = None
    if "P856" in claims:
        try:
            site_claim = claims["P856"][0]["mainsnak"].get("datavalue", {})
            if site_claim:
                official_site = site_claim["value"]
        except (KeyError, TypeError) as exc:
            logger.debug("Malformed official website claim for %s: %s", id_, exc)

    name = result["label"]
    description = result.get("description")

    # Wikipedia summary
    wiki_card = wiki_lookup(name)

    search_engine = DDGS()

    # Fallback: search official URL if missing
    if not official_site:
        ddgs_results = search_engine.text(keywords=name, max_results=1)[0]
        official_url = ddgs_results.get("href")
        if not description:
            description = ddgs_results.get("body")
    else:
        official_url = official_site

    if not logo_url:
        logo_url = wiki_card.get("thumbnail")
        if not verify_logo(logo_url):
            logo_url = None

    if not logo_url:
        ddgs_results = search_engine.images(keywords=name, max_results=1)[0]
        logo_url = ddgs_results.get("image")
        if not verify_logo(logo_url):
            logo_url = None

    # Fetch description and logo from official URL if missing
    if official_url and (not description or not logo_url):
        org_info = asyncio.run(get_organization_info(official_url))
        if not description:
            description = org_info.get("description")

        if not logo_url:
            logo_url = org_info.get("logo")
            if not verify_logo(logo_url):
                logo_url = None

    # Final URL: prefer official, else wiki page
    final_url = official_url or wiki_card.get("url")

    return {
        "name": name,
        "id": id_,
        "description": description,
        "url": final_url,
        "thumbnail": logo_url,
    }

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def entity_lookup(
    term: str,
    *,
    lang: str = "en",
    limit: int = 5,
) -> List[Dict[str, Optional[str]]]:
    """Return up to *limit* candidate entities for *term*.

    All individual enrichment calls run concurrently. If some fail, they are
    logged and silently skipped so the function still returns as many results
    as possible.
    """
    with ThreadPoolExecutor(max_workers=5) as executor:
        try:
            search_resp = requests.get(
                "https://www.wikidata.org/w/api.php",
                params={
                    "action": "wbsearchentities",
                    "search": term,
                    "language": lang,
                    "format": "json",
                    "limit": str(limit),
                },
            ).json()
        except requests.exceptions.RequestException as exc:
            logger.error("Search failed for %s: %s", term, exc)
            return []

        hits = search_resp.get("search", [])
        if not hits:
            return []

        futures = [executor.submit(_enrich_entity, hit, lang) for hit in hits]
        results = []
        for future in futures:
            try:
                results.append(future.result())
            except Exception as exc:
                logger.warning("Enrichment error: %s", exc)

        return results


def entity_lookup_sync(term: str, *, lang: str = "en", limit: int = 5):
    """Blocking convenience wrapper for scripts that are not async‑aware."""
    try:
        return entity_lookup(term, lang=lang, limit=limit)
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("entity_lookup_sync failed: %s", exc)
        return []

# ---------------------------------------------------------------------------
# CLI demo / quick test ------------------------------------------------------
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from pprint import pprint

    for query in ("Johnson and Johnson", "J&J", "Google Inc."):
        print("\n>>>", query)
        pprint(entity_lookup_sync(query, limit=3))
