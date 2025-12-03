# Resume Search Dashboard

This is a Streamlit application for searching resumes using Hybrid Search (Dense + Sparse embeddings) with Qdrant.

## Prerequisites

## Prerequisites

1.  **Install Python 3.9+**:
    -   Download from [python.org](https://www.python.org/downloads/).
    -   **CRITICAL STEP**: During installation, you **MUST** check the box that says **"Add Python to PATH"**.
    -   If you don't do this, the script will not work.

2.  **OpenAI API Key**: You will need a valid OpenAI API Key in the `secrets.toml` file.

## Installation (Windows)

1.  **Unzip the project** to a folder on your computer.

2.  **Open Command Prompt (cmd)** or PowerShell and navigate to the project folder:
    ```cmd
    cd path\to\resume_search_dashboard
    ```

3.  **Create a Virtual Environment** (Recommended):
    ```cmd
    python -m venv venv
    venv\Scripts\activate
    ```

4.  **Install Dependencies**:
    ```cmd
    pip install -r requirements.txt
    ```

## Configuration

1.  Navigate to the `.streamlit` folder inside the project.
2.  Open `secrets.toml` with a text editor (Notepad).
3.  Ensure your OpenAI API Key is set correctly:
    ```toml
    [openai]
    api_key = "sk-..."
    ```
    (The Qdrant credentials should already be pre-filled).

## Running the App

**Option 1: Double-click `run_app.bat`**
We have included a batch script. Just double-click `run_app.bat` in the folder.

**Option 2: Command Line**
Make sure your virtual environment is activated, then run:
```cmd
streamlit run app.py
```

The application should open automatically in your default web browser at `http://localhost:8501`.

## Bulk Ingestion

To ingest a folder of PDF resumes:
```cmd
python ingest_bulk.py "C:\path\to\your\resumes"
```
