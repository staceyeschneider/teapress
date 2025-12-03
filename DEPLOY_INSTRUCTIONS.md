# Deployment Instructions (Streamlit Cloud)

Since you are deploying this for Joanna to use via the web, follow these steps:

## 1. Prepare GitHub Repo
1. Create a new repository on GitHub (e.g., `teapress-recruiting`).
2. Upload the following files to the root of the repo:
   - `app.py`
   - `requirements.txt`
   - `packages.txt` (Optional, if we need system tools later)

## 2. Deploy to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in.
2. Click **"New App"**.
3. Select your GitHub repository.
4. **Main file path**: `app.py`

## 3. Configure Secrets (CRITICAL)
**Do not skip this step**, or the app will crash immediately.

1. In the "Deploy an app" screen, click **"Advanced Settings..."**.
2. Copy the content below and paste it into the **"Secrets"** text area.

```toml
[qdrant]
url = "YOUR_QDRANT_CLOUD_URL_HERE" 
api_key = "YOUR_QDRANT_API_KEY_HERE"

[openai]
api_key = "YOUR_OPENAI_API_KEY_HERE"

[auth]
password = "teapress_secret_code"
```

> **Note:** You must use the **Qdrant Cloud URL** (ending in `qdrant.tech`), NOT `localhost`.
> **Note:** Change `teapress_secret_code` to whatever password you want Joanna to use.

## 4. Launch!
Click **"Deploy"**. The app will build (takes 2-3 minutes) and then be live.

## 5. Share
Send the URL (e.g., `teapress-recruiting.streamlit.app`) to Joanna. She can use it from any browser!
