import os
import toml
import glob
import argparse
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
from qdrant_client import QdrantClient, models
from openai import OpenAI
from fastembed import SparseTextEmbedding
from pypdf import PdfReader
import docx
import uuid

# Configuration
SECRETS_PATH = ".streamlit/secrets.toml"
COLLECTION_NAME = "resumes"
DENSE_VECTOR_NAME = "resume_embedding"
SPARSE_VECTOR_NAME = "resume_keywords"
DENSE_MODEL_NAME = "text-embedding-3-small"
SPARSE_MODEL_NAME = "Qdrant/bm25"

def load_secrets():
    """Loads secrets from .streamlit/secrets.toml"""
    if not os.path.exists(SECRETS_PATH):
        raise FileNotFoundError(f"Secrets file not found at {SECRETS_PATH}")
    return toml.load(SECRETS_PATH)

def parse_pdf(file_path):
    """Extracts text from a PDF file."""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error parsing PDF {file_path}: {e}")
        return None

def parse_docx(file_path):
    """Extracts text from a DOCX file."""
    try:
        doc = docx.Document(file_path)
        text = []
        for para in doc.paragraphs:
            text.append(para.text)
        return "\n".join(text)
    except Exception as e:
        print(f"Error parsing DOCX {file_path}: {e}")
        return None

def parse_text_file(file_path):
    """Extracts text from TXT or MD file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
        except Exception as e:
            print(f"Error parsing text file {file_path}: {e}")
            return None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def extract_text(file_path):
    """Extracts text based on file extension."""
    lower_path = file_path.lower()
    if lower_path.endswith('.pdf'):
        return parse_pdf(file_path)
    elif lower_path.endswith('.docx'):
        return parse_docx(file_path)
    elif lower_path.endswith('.txt') or lower_path.endswith('.md'):
        return parse_text_file(file_path)
    elif lower_path.endswith('.doc'):
        print(f"Skipping {file_path}: Legacy .doc format not supported. Convert to .docx")
        return None
    else:
        return None

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_dense_embedding(client, text):
    """Generates dense embedding with retry logic."""
    response = client.embeddings.create(
        input=text,
        model=DENSE_MODEL_NAME
    )
    return response.data[0].embedding

def generate_sparse_embedding(model, text):
    """Generates sparse embedding."""
    embeddings = list(model.embed([text]))
    if not embeddings:
        return None
    sparse_vec = embeddings[0]
    return models.SparseVector(
        indices=sparse_vec.indices.tolist(),
        values=sparse_vec.values.tolist()
    )

def main():
    parser = argparse.ArgumentParser(description="Bulk ingest resumes into Qdrant.")
    parser.add_argument("directory", help="Directory containing PDF, DOCX, TXT, or MD resumes")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist.")
        return

    # Load Secrets & Initialize Clients
    print("Loading secrets and initializing clients...")
    secrets = load_secrets()
    
    qdrant_client = QdrantClient(
        url=secrets["qdrant"]["url"],
        api_key=secrets["qdrant"]["api_key"]
    )
    
    openai_client = OpenAI(api_key=secrets["openai"]["api_key"])
    
    # Initialize Sparse Model (Local)
    print("Loading sparse embedding model...")
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)

    # Find files
    extensions = ['*.pdf', '*.docx', '*.txt', '*.md', '*.doc']
    all_files = []
    for ext in extensions:
        all_files.extend(glob.glob(os.path.join(args.directory, ext)))
    
    print(f"Found {len(all_files)} files in {args.directory}")

    # Process Loop
    success_count = 0
    fail_count = 0

    for file_path in tqdm(all_files, desc="Processing Resumes"):
        try:
            filename = os.path.basename(file_path)
            candidate_name = os.path.splitext(filename)[0].replace("_", " ").title()
            
            # 1. Parse Text
            text = extract_text(file_path)
            if not text or len(text.strip()) < 50:
                # Error already logged in extract_text for .doc
                fail_count += 1
                continue

            # 2. Generate Embeddings
            dense_vec = generate_dense_embedding(openai_client, text)
            sparse_vec = generate_sparse_embedding(sparse_model, text)

            # 3. Prepare Point
            point_id = str(uuid.uuid4())
            payload = {
                "candidate_name": candidate_name,
                "text": text,
                "source_file": filename,
                "skills": [],
                "years_experience": 0,
                "location": "Unknown"
            }

            point = models.PointStruct(
                id=point_id,
                vector={
                    DENSE_VECTOR_NAME: dense_vec,
                    SPARSE_VECTOR_NAME: sparse_vec
                },
                payload=payload
            )

            # 4. Upsert
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=[point]
            )
            success_count += 1

        except Exception as e:
            print(f"Failed to process {file_path}: {e}")
            fail_count += 1

    print(f"\nIngestion Complete!")
    print(f"Successfully indexed: {success_count}")
    print(f"Failed: {fail_count}")

if __name__ == "__main__":
    main()
