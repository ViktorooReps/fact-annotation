from __future__ import annotations

from typing import Any, Dict, List, Optional
import asyncio
import logging
import urllib.parse

import aiohttp

__all__ = [
    "EntityLookupError",
    "NetworkError",
    "ApiError",
    "entity_lookup",
    "entity_lookup_sync",
]

from integrations.customsearch_api import get_organization_info, search_urls
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


async def _fetch_json(
    session: aiohttp.ClientSession,
    url: str,
    *,
    timeout: aiohttp.ClientTimeout | None = None,
    **kwargs,
) -> Any:
    """GET *url* and return the parsed JSON body.

    All *aiohttp* network and JSON decoding errors are wrapped into
    :class:`NetworkError` and :class:`ApiError` so callers can catch a single
    family of exceptions.
    """
    try:
        async with session.get(url, timeout=timeout or aiohttp.ClientTimeout(total=5), **kwargs) as resp:
            resp.raise_for_status()
            return await resp.json()
    except asyncio.TimeoutError as exc:
        raise NetworkError(f"Timeout while fetching {url}") from exc
    except aiohttp.ClientError as exc:
        raise NetworkError(f"Network problem while fetching {url}: {exc}") from exc
    except ValueError as exc:  # JSON decoding failed
        raise ApiError(f"Invalid JSON from {url}: {exc}") from exc


# ---------------------------------------------------------------------------
# Wikipedia helpers
# ---------------------------------------------------------------------------

async def wiki_lookup(session: aiohttp.ClientSession, title: str) -> Dict[str, Optional[str]]:
    """Return a summary card for *title* from the REST API.

    On failure, an empty card is returned so the caller can still build a
    partial entity result.
    """
    try:
        summary = await _fetch_json(
            session,
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}",
        )
    except EntityLookupError as exc:
        logger.warning("wiki_lookup failed for %s: %s", title, exc)
        return {"title": title, "url": None, "description": None, "thumbnail": None}

    return {
        "title": summary.get("title"),
        "url": summary.get("content_urls", {}).get("desktop", {}).get("page"),
        "description": summary.get("extract"),
        "thumbnail": summary.get("thumbnail", {}).get("source"),
    }


# ---------------------------------------------------------------------------
# Wikidata helpers
# ---------------------------------------------------------------------------

async def _enrich_entity(
    session: aiohttp.ClientSession,
    result: Dict[str, Any],
    lang: str,
) -> Dict[str, Optional[str]]:
    id_ = result["id"]

    entities = await _fetch_json(
        session,
        "https://www.wikidata.org/w/api.php",
        params={
            "action": "wbgetentities",
            "ids": id_,
            "props": "claims",
            "languages": lang,
            "sitefilter": "enwiki",
            "format": "json",
        },
    )

    claims = entities.get("entities", {}).get(id_, {}).get("claims", {})

    # logo (P154)
    logo_url = None
    if "P154" in claims:
        try:
            logo_claim = claims["P154"][0]["mainsnak"].get("datavalue", {})
            if logo_claim:
                logo_url = get_file_url(logo_claim["value"], size=200)
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

    # Wikipedia summary
    wiki_card = await wiki_lookup(session, name)

    # Fallback: search official URL if missing (not yet implemented)
    if not official_site:
        official_url = (await search_urls(name))[0]
    else:
        official_url = official_site

    # Fetch description and logo from official URL if missing
    if official_url:
        org_info = await get_organization_info(official_url)
        if not result.get("description") and org_info.get("description"):
            description = org_info["description"]
        else:
            description = result.get("description")
        if not logo_url and org_info.get("logo"):
            logo_url = org_info["logo"]
    else:
        description = result.get("description") or wiki_card.get("description")

    # Final URL: prefer official, else wiki page
    final_url = official_url or wiki_card.get("url")

    # Final thumbnail: prefer logo, else wiki thumbnail
    thumbnail = logo_url or wiki_card.get("thumbnail")

    return {
        "name": name,
        "id": id_,
        "description": description,
        "url": final_url,
        "thumbnail": thumbnail,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
async def entity_lookup(
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
    async with aiohttp.ClientSession(headers={"User-Agent": "entity-lookup/1.1"}) as session:
        try:
            search_resp = await _fetch_json(
                session,
                "https://www.wikidata.org/w/api.php",
                params={
                    "action": "wbsearchentities",
                    "search": term,
                    "language": lang,
                    "format": "json",
                    "limit": str(limit),
                },
            )
        except EntityLookupError as exc:
            logger.error("Search failed for %s: %s", term, exc)
            return []

        hits = search_resp.get("search", [])
        if not hits:
            return []

        tasks = [_enrich_entity(session, hit, lang) for hit in hits]
        gathered = await asyncio.gather(*tasks, return_exceptions=True)

        results: List[Dict[str, Optional[str]]] = []
        for res in gathered:
            if isinstance(res, Exception):
                logger.warning("Enrichment error: %s", res)
            else:
                results.append(res)

        return results


def entity_lookup_sync(term: str, *, lang: str = "en", limit: int = 5):
    """Blocking convenience wrapper for scripts that are not async‑aware."""
    try:
        return asyncio.run(entity_lookup(term, lang=lang, limit=limit))
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
