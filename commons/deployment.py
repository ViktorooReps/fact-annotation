import os

import streamlit


def get_base_url():
    base_url = streamlit.get_option("server.baseUrlPath")
    port = streamlit.get_option("server.port") or 8501
    deployment = os.getenv("DEPLOYMENT")
    app_name = os.getenv("APP_NAME")

    if deployment == "dev":
        base_url = f"http://localhost:{port}/{base_url}" if base_url else f"http://localhost:{port}"
    else:
        # assume to be hosted on streamlit
        if not app_name:
            raise ValueError("Please set up APP_NAME environment variable or set DEPLOYMENT=dev")
        base_url = f"http://{app_name}.streamlit.io"

    return base_url
