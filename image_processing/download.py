import requests, io
from PIL import Image


USER_AGENT = (
    "Fact-Annotator "
    "(https://github.com/ViktorooReps/fact-annotator ; contact: viktoroo.sch@gmail.com)"
)


def load_from_url(url: str, *, timeout: int = 4) -> Image.Image:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()                      # will raise if still forbidden
    return Image.open(io.BytesIO(r.content))
