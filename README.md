# Fact Annotation

Fact Annotation is a lightweight Streamlit-based tool for semi-structured text annotation, designed for tasks that go beyond traditional Named Entity Recognition. Instead of linking entities directly to their mentions, this tool allows annotators to highlight factual information by associating aliases with abstract "facts" present in the text. It's especially useful for building datasets where the goal is to identify and label information without enforcing strict mention-entity ties — such as disambiguated facts or reference points used in downstream reasoning or retrieval tasks.

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
