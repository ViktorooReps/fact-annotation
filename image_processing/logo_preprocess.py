import hashlib
import io, numpy as np
import os
import sys
from functools import lru_cache
from pathlib import Path

import streamlit as st
from PIL import Image
from scipy.ndimage import distance_transform_edt

from image_processing.download import load_from_url


NO_IMG_THUMB = "static/no_img.png"


@lru_cache()
def _app_root() -> Path:
    """
    Return the directory that contains the Streamlit entry-point,
    falling back to CWD in unusual cases (e.g. `streamlit run -`).
    """
    # Try the new, internal ScriptRunContext (Streamlit ≥1.29)
    try:
        from streamlit.runtime.scriptrunner.script_runner import (
            get_script_run_ctx,
        )

        ctx = get_script_run_ctx()
        if ctx is not None and ctx.main_script_path:
            return Path(ctx.main_script_path).resolve().parent
    except Exception:
        pass  # Streamlit version <1.29 or refactor — ignore

    # Fall back to the path that was given on the CLI, if any
    main_file = Path(sys.argv[0])
    if main_file.suffix == ".py" and main_file.exists():
        return main_file.resolve().parent

    # Last resort: current working directory
    return Path.cwd()


def ui_color(light="#FFF", dark="#000", theme=None):
    is_dark = (theme or "light").lower() == "dark"
    return dark if is_dark else light


def add_outline_round(img: Image.Image, width=3, color=None, feather=1, theme=None):
    """Smooth circular halo using Euclidean distance."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    alpha = np.array(img.split()[-1]) > 0
    # distance 0 for border, growing outward
    dist_out = distance_transform_edt(~alpha)
    border = (dist_out > 0) & (dist_out <= width)

    # optional feather at the very edge
    if feather > 0:
        fade_zone = (dist_out > width) & (dist_out <= width + feather)
        fade_alpha = np.clip((width + feather - dist_out) / feather, 0, 1)
    else:
        fade_zone = fade_alpha = None

    *rgb, a = np.array(img).transpose(2, 0, 1)
    halo = np.zeros_like(a, dtype=np.uint8)
    halo[border] = 255
    if fade_zone is not None:
        halo[fade_zone] = (fade_alpha[fade_zone] * 255).astype(np.uint8)

    col = Image.new("RGB", (1, 1), color or ui_color(light="#000", dark="#FFF", theme=theme)).getdata()[0]
    halo_rgb = np.dstack([np.full_like(halo, c) for c in col])

    out = np.dstack([*rgb, a])
    mask = border | (fade_zone if fade_zone is not None else False)
    out[mask] = np.dstack([halo_rgb, halo])[mask]
    return Image.fromarray(out, "RGBA")


def pad(img: Image.Image, pct=10, bg=None, theme=None):
    """Pad on all sides by <pct>% of the largest dimension."""
    pad_px = int(max(img.size) * pct / 100)
    mode  = "RGBA" if img.mode == "RGBA" else "RGB"
    canvas = Image.new(
        mode,
        (img.width + 2*pad_px, img.height + 2*pad_px),
        (0, 0, 0, 0) if mode == "RGBA" else (bg or ui_color(theme=theme))  # invert colors
    )
    canvas.paste(img, (pad_px, pad_px), img.split()[-1] if mode == "RGBA" else None)
    return canvas


@st.cache_data(show_spinner=False)         # avoids re-downloading images
def preprocess_logo(
        src,
        outline_px: int = 10,
        padding_pct: int = 30,
        timeout: int = 4,
        colour: str | None = None,
        theme: str | None = None,
) -> Image.Image:
    """
    Parameters
    ----------
    src          : str | pathlib.Path | BytesIO | PIL.Image
                   URL, local path, byte stream, or already-loaded image.
    outline_px   : thickness of the halo around opaque pixels.
    padding_pct  : extra empty margin so Streamlit doesn’t clip corners.
    timeout      : seconds to wait when fetching a remote image.
    colour       : outline colour; defaults to black in light theme,
                   white in dark theme.
    """
    if isinstance(src, Image.Image):
        img = src
    else:
        if isinstance(src, (bytes, bytearray, io.BytesIO)):
            img = Image.open(src)
        else:
            src = str(src)
            if src.startswith(("http://", "https://")):
                img = load_from_url(src, timeout=timeout)
            else:                                     # assume local path
                img = Image.open(src)

    img = img.convert("RGBA")          # ensure alpha channel exists

    img = pad(img, pct=padding_pct, theme=theme)
    if theme == "dark":
        img = add_outline_round(img, width=outline_px, color=colour, theme=theme)

    return img


@st.cache_data(show_spinner=False)         # avoids re-processing + re-hashing
def logo_to_url(
    src,
    *,
    outline_px: int = 10,
    padding_pct: int = 30,
    timeout: int = 4,
    colour: str | None = None,
    theme: str | None = None,
) -> str:
    """
    Runs preprocess_logo, writes the result once per unique image into
    ./static/, and returns the public URL that Streamlit’s static server
    exposes (e.g.  '/static/<sha256>.png').
    """

    # 1. Location of the static folder (must sit next to your script)
    static_dir = _app_root() / "static"
    static_dir.mkdir(exist_ok=True)               # safe if it already exists

    # 2. Get the processed logo (cached by Streamlit)
    img: Image.Image = preprocess_logo(
        src,
        outline_px=outline_px,
        padding_pct=padding_pct,
        timeout=timeout,
        colour=colour,
        theme=theme,
    )

    # 3. Hash the final PNG bytes ⇒ deterministic, collision-free filename
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    digest = hashlib.sha256(buf.getvalue()).hexdigest()
    fname = f"hosted{digest}.png"
    fpath = static_dir / fname

    # 4. Write once, re-use thereafter
    if not fpath.exists():                      # atomic enough for our use-case
        fpath.write_bytes(buf.getvalue())

    # 5. Return the URL the rest of your app (or a 3rd-party component)
    #    can use.  /static/... works both locally and on Streamlit Cloud.
    base_url = "http://localhost:8501/"
    return f"{base_url}app/static/{fname}"
