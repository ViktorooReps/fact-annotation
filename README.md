# Fact Annotator

Turn any CSV or Google Sheet full of free-text statements into a mini knowledge graph — all from a single Streamlit page (entities, relations, live graph view & Wikipedia look-ups).

---

## Requirements

| Tool                   | Version                                        |
|------------------------|------------------------------------------------|
| **Python**             | 3.11 (confirmed) – 3.12 should also work       |
| **pip**                | 23 or later                                    |
| **Other dependencies** | Installed automatically via `requirements.txt` |

> **Tip:** Replace `python3.11` with `python` or `python3` if those point to 3.11+ on your system.

---

## Setup from source

```bash
git clone https://github.com/ViktorooReps/fact-annotation
cd fact-annotation
```


Create & activate a virtual environment
```bash
python3.11 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
```

Install dependencies
```bash
pip install -r requirements.txt
```

Launch the app
```bash
streamlit run app.py
```

## Optional: enable Google Sheets as a backend

Create a Service Account in the Google Cloud console
* enable the Google Sheets API
* generate a JSON key file.

Share every Sheet you plan to use with the account’s `client_email` (Editor permission) or make the sheet public.

Add the credentials to `secrets.toml` in `.streamlit` directory at the project root:

```toml
[gcp_service_account]
type = "service_account"
project_id = "(...)"
private_key_id = "(...)"
private_key = "-----BEGIN PRIVATE KEY-----(...)-----END PRIVATE KEY-----\n"
client_email = "(...)@(...).iam.gserviceaccount.com"
client_id = "(...)"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/(...)/(...)%40(...).iam.gserviceaccount.com"
universe_domain = "googleapis.com"
```