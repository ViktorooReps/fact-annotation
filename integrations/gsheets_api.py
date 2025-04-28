import json
import re
import urllib.parse
from typing import List, Callable

import numpy as np
import pandas as pd
import requests
import streamlit as st


def extract_sheet_id(url: str) -> str | None:
    """Return the Google Sheets file ID from a full URL."""
    m = re.search(r"/d/([\w-]+)", url)
    return m.group(1) if m else None


def public_worksheets(sheet_id: str) -> List[str]:
    """List worksheet titles of a *public‑by‑link* sheet (read‑only)."""
    feed_url = (
        f"https://spreadsheets.google.com/feeds/worksheets/{sheet_id}/public/basic?alt=json"
    )
    try:
        data = requests.get(feed_url, timeout=5).json()
        return [e["title"]["$t"] for e in data.get("feed", {}).get("entry", [])]
    except Exception:
        return []


def worksheet_to_df(sheet_id: str, worksheet_name: str) -> pd.DataFrame:
    """Download a worksheet as CSV to DataFrame (public‑to‑web or link‑share)."""
    csv_url = (
        f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?"
        f"tqx=out:csv&sheet={urllib.parse.quote(worksheet_name)}"
    )
    return pd.read_csv(csv_url)


def save_df_to_sheet(sheet_url: str, worksheet_name: str, df: pd.DataFrame):
    """Overwrite *worksheet_name* with the contents of *df* using a service account.

    Requires a JSON service‑account key stored in st.secrets["gcp_service_account"].
    """
    try:
        import gspread
        from google.oauth2 import service_account

        creds_info = st.secrets.get("gcp_service_account")
        if not creds_info:
            st.error("Service‑account credentials missing in secrets; cannot save.")
            return
        creds = service_account.Credentials.from_service_account_info(
            creds_info,
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive",
                "https://www.googleapis.com/auth/drive.file"
            ],
        )

        gc = gspread.authorize(creds)
        sh = gc.open_by_url(sheet_url)
        try:
            ws = sh.worksheet(worksheet_name)
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title=worksheet_name, rows=1, cols=len(df.columns))

        ws.clear()
        ws.update([df.columns.values.tolist()] + df.astype(str).values.tolist())
    except Exception as e:
        st.error(f"✖️ Failed to save: {e}, {type(e)}")


def _dumps_safe(obj):
    """json.dumps that leaves NaN/None as-is so Sheets keeps the cell empty."""
    if obj is None or (isinstance(obj, float) and np.isnan(obj)):
        return None
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def prepare_df_to_export(df: pd.DataFrame, json_columns_filter: Callable[[str], bool]) -> pd.DataFrame:
    """
    Return a copy where `json_columns` are JSON-encoded strings
    so Google Sheets doesn’t coerce lists/dicts to plain text.
    """
    df_out = df.copy(deep=False)    # shallow copy is enough for assignment
    json_columns = filter(json_columns_filter, df_out.columns)
    for col in json_columns:
        if col in df_out.columns:
            df_out[col] = df_out[col].apply(_dumps_safe)
    return df_out


def _loads_safe(val):
    """json.loads that leaves anything non-JSON (or NaN) unchanged."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return val
    try:
        return json.loads(val)
    except (TypeError, json.JSONDecodeError):
        return val        # was never JSON – leave it as the string it is


def postprocess_df(df: pd.DataFrame, json_columns_filter: Callable[[str], bool]) -> pd.DataFrame:
    """
    Decode the JSON-encoded `json_columns` that came back from Sheets
    so Python objects are restored.
    """
    df_out = df.copy(deep=False)
    json_columns = filter(json_columns_filter, df_out.columns)
    for col in json_columns:
        if col in df_out.columns:
            df_out[col] = df_out[col].apply(_loads_safe)
    return df_out

