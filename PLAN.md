# Resume Search Dashboard Plan

## 1. Project Structure
```
resume_search_dashboard/
├── app.py                 # Main Streamlit application
├── utils.py               # Helper functions (PDF parsing, Embeddings, Qdrant client)
├── requirements.txt       # Python dependencies
├── .streamlit/
│   └── secrets.toml       # Secrets (OpenAI Key, Qdrant URL/Key)
└── PLAN.md                # This file
```

## 2. Dependencies
- `streamlit`
- `qdrant-client`
- `openai`
- `fastembed`
- `pypdf`
- `python-dotenv`

## 3. Implementation Details

### `utils.py`
- **PDF Parsing**: Use `pypdf` to extract text from uploaded files.
- **Qdrant Client**: Initialize `QdrantClient` using provided Cloud credentials (or localhost fallback).
- **Embeddings**:
    - Dense: `OpenAI` client for `text-embedding-3-small`.
    - Sparse: `fastembed` for BM25 generation.
- **Indexing**: Function to process text -> generate vectors -> upload to Qdrant.
- **Search**: Function to perform Hybrid Search (Dense + Sparse) with RRF or weighting.

### `app.py`
- **Sidebar**:
    - File Uploader (PDFs).
    - "Process & Index" button.
    - Status messages.
- **Main Area**:
    - Title & Description.
    - Search Input.
    - "Hybrid Search" button.
    - Results Display:
        - Cards with Candidate Name, Score, Skills.
        - Expander for full text.

## 4. Configuration
- **Collection Name**: `resumes`
- **Dense Vector**: `resume_embedding`
- **Sparse Vector**: `resume_keywords`
- **Qdrant Endpoint**: `https://f58fe292-5793-4cdf-93b1-eac7b3a63416.us-east4-0.gcp.cloud.qdrant.io`
- **Qdrant Key**: Provided in prompt.

## 5. Next Steps
1. Create `requirements.txt`.
2. Create `utils.py` with core logic.
3. Create `app.py` with UI.
4. Verify connection and functionality.
