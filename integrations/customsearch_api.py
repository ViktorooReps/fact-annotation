import asyncio
from urllib.parse import urljoin

import aiohttp
from bs4 import BeautifulSoup


async def search_urls(query: str, num_results: int = 1) -> list[str]:
    """
    Performs a Google Custom Search for the given query and returns
    a list of result URLs (best matches), using credentials from Streamlit secrets.
    """
    creds = __import__("streamlit").secrets["google_customsearch"]
    api_key = creds.get("api_key")
    cse_id = creds.get("cse_id")

    endpoint = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cse_id,
        "q": query,
        "num": num_results
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(endpoint, params=params) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return [
                item["link"]
                for item in data.get("items", [])
                if item.get("link")
            ]


async def get_organization_info(url: str, timeout: int = 10, max_retries: int = 1) -> dict:
    """
    Async version: Fetches an organization's logo URL and description from its homepage.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/119.0.0.0 Safari/537.36"
        )
    }

    attempt = 0
    resp_text = None
    final_url = url

    async with aiohttp.ClientSession(headers=headers) as session:
        while attempt <= max_retries:
            try:
                async with session.get(final_url, timeout=timeout) as resp:
                    resp.raise_for_status()
                    final_url = str(resp.url)
                    resp_text = await resp.text()
                    break
            except aiohttp.ClientResponseError as e:
                if e.status in (403, 429) and attempt < max_retries:
                    attempt += 1
                    await asyncio.sleep(1)
                    continue
                return {"logo": None, "description": None}
            except (aiohttp.ClientError, asyncio.TimeoutError):
                return {"logo": None, "description": None}

    soup = BeautifulSoup(resp_text, "html.parser")
    description = None
    logo_candidates = []

    # 1. JSON-LD
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = __import__("json").loads(tag.string or "{}")
        except Exception:
            continue
        entries = data if isinstance(data, list) else [data]
        for entry in entries:
            if entry.get("@type") == "Organization":
                if entry.get("description") and not description:
                    description = entry["description"].strip()
                if entry.get("logo"):
                    logo_candidates.append(urljoin(final_url, entry["logo"]))
        if description and logo_candidates:
            break

    # 2. Meta fallbacks
    if not description:
        meta = (
            soup.find("meta", attrs={"name": "description"}) or
            soup.find("meta", property="og:description")
        )
        if meta and meta.get("content"):
            description = meta["content"].strip()

    # 3. Open Graph image
    og = soup.find("meta", property="og:image")
    if og and og.get("content"):
        logo_candidates.append(urljoin(final_url, og["content"]))

    # 4. <img> candidates
    for img in soup.find_all("img", src=True):
        src = urljoin(final_url, img["src"])
        alt = img.get("alt", "").lower()
        cls = " ".join(img.get("class", [])).lower()
        if "logo" in alt or "logo" in cls or src.lower().endswith(".svg"):
            logo_candidates.append(src)

    # 5. Dedupe & score
    def score(u: str):
        s = 0
        low = u.lower()
        if "logo" in low:
            s += 2
        if low.endswith(".svg") or ".svg?" in low:
            s += 1
        return s

    unique = []
    seen = set()
    for c in logo_candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    if unique:
        unique.sort(key=score, reverse=True)
        chosen = unique[0]
    else:
        chosen = None

    return {"logo": chosen, "description": description}


async def main():
    # Example usage
    results = await search_urls("DataKind", num_results=1)
    if results:
        info = await get_organization_info(results[0])
        print("URL:", results[0])
        print("Logo:", info["logo"])
        print("Description:", info["description"])


if __name__ == "__main__":
    asyncio.run(main())
