import cairosvg
import requests, io
from PIL import Image


USER_AGENT = (
    "Fact-Annotator "
    "(https://github.com/ViktorooReps/fact-annotator ; contact: viktoroo.sch@gmail.com)"
)


def load_from_url(url: str, *, timeout: int = 4) -> Image.Image:
    """
    Fetches an image from `url`.
    - If it’s an SVG, uses cairosvg to convert it to PNG bytes before opening with PIL.
    - Otherwise loads it directly via PIL.

    Raises:
      requests.HTTPError on bad HTTP status.
      cairosvg.exceptions.CairoSVGError on invalid SVG.
    """
    headers = {
        "User-Agent": USER_AGENT
    }
    r = requests.get(url, timeout=timeout, headers=headers)
    r.raise_for_status()

    content_type = r.headers.get("Content-Type", "").lower()
    data = r.content

    # Detect SVG by content-type or file extension
    is_svg = (
        "image/svg+xml" in content_type or
        url.lower().endswith(".svg")
    )

    if is_svg:
        # Convert SVG bytes to PNG bytes at default DPI
        png_bytes = cairosvg.svg2png(
            bytestring=data,
            # you could override size here:
            output_width=200
        )
        return Image.open(io.BytesIO(png_bytes))

    # Otherwise PIL can handle it natively
    return Image.open(io.BytesIO(data))
