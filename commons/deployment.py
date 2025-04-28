import os

import streamlit


def get_base_url():
    base_url = streamlit.get_option("server.baseUrlPath")
    port = streamlit.get_option("server.port")
    deployment = os.getenv("DEPLOYMENT")

    if deployment == "dev":
        base_url = f"http://localhost:{port}"
    else:
        # assume to be hosted on streamlit
        base_url = f"http://{base_url}.streamlit.io"

    return base_url
