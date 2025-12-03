import streamlit as st
import os
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from openai import OpenAI
from fastembed import SparseTextEmbedding
from pypdf import PdfReader
import docx
import uuid
import io

# --- UTILS SECTION (Merged to avoid import errors) ---

# Constants
COLLECTION_NAME = "resumes"
DENSE_VECTOR_NAME = "resume_embedding"
SPARSE_VECTOR_NAME = "resume_keywords"
DENSE_MODEL_NAME = "text-embedding-3-small"
SPARSE_MODEL_NAME = "Qdrant/bm25"

@st.cache_resource
def get_qdrant_client():
    """Initializes and returns the Qdrant Client."""
    if "qdrant" in st.secrets:
        url = st.secrets["qdrant"]["url"]
        api_key = st.secrets["qdrant"]["api_key"]
    else:
        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")
    
    if not url:
        raise ValueError("Qdrant URL not found in secrets or environment variables.")

    return QdrantClient(url=url, api_key=api_key)

@st.cache_resource
def get_openai_client():
    """Initializes and returns the OpenAI Client."""
    if "openai" in st.secrets:
        api_key = st.secrets["openai"]["api_key"]
    else:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        return None
        
    return OpenAI(api_key=api_key)

@st.cache_resource
def get_sparse_embedding_model():
    """Initializes and returns the Sparse Embedding Model."""
    return SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)

def parse_pdf(file):
    # Let exceptions bubble up
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def parse_docx(file):
    # Let exceptions bubble up
    doc = docx.Document(file)
    text = []
    for para in doc.paragraphs:
        text.append(para.text)
    return "\n".join(text)

def parse_text_file(file):
    try:
        return file.read().decode("utf-8")
    except UnicodeDecodeError:
        file.seek(0)
        return file.read().decode("latin-1")

def get_resume_text(file, filename):
    filename_lower = filename.lower()
    if filename_lower.endswith('.pdf'):
        return parse_pdf(file)
    elif filename_lower.endswith('.docx'):
        return parse_docx(file)
    elif filename_lower.endswith('.txt') or filename_lower.endswith('.md'):
        return parse_text_file(file)
    elif filename_lower.endswith('.doc'):
        raise ValueError("Legacy .doc format not supported. Please convert to .docx")
    else:
        raise ValueError("Unsupported file format")

def generate_dense_embedding(text, client):
    if not client:
        raise ValueError("OpenAI Client not initialized. Check API Key.")
    
    # Truncate to ~8000 tokens (approx 32k chars) to avoid 400 errors
    truncated_text = text[:32000]
    
    response = client.embeddings.create(input=truncated_text, model=DENSE_MODEL_NAME)
    return response.data[0].embedding

def generate_sparse_embedding(text, model):
    embeddings = list(model.embed([text]))
    if not embeddings:
        return None
    sparse_vec = embeddings[0]
    return models.SparseVector(indices=sparse_vec.indices.tolist(), values=sparse_vec.values.tolist())

def index_resume(text, metadata, qdrant_client, openai_client, sparse_model):
    dense_vec = generate_dense_embedding(text, openai_client)
    sparse_vec = generate_sparse_embedding(text, sparse_model)
    point_id = str(uuid.uuid4())
    point = models.PointStruct(
        id=point_id,
        vector={DENSE_VECTOR_NAME: dense_vec, SPARSE_VECTOR_NAME: sparse_vec},
        payload={"text": text, **metadata}
    )
    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=[point])
    return point_id

def hybrid_search(query_text, qdrant_client, openai_client, sparse_model, limit=5):
    dense_vec = generate_dense_embedding(query_text, openai_client)
    sparse_vec = generate_sparse_embedding(query_text, sparse_model)
    
    prefetch_dense = models.Prefetch(query=dense_vec, using=DENSE_VECTOR_NAME, limit=limit * 2)
    prefetch_sparse = models.Prefetch(query=sparse_vec, using=SPARSE_VECTOR_NAME, limit=limit * 2)
    
    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[prefetch_dense, prefetch_sparse],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True
    )
    return results.points

def extract_metadata(text, client):
    if not client: return {}
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts structured data from resumes. Return a JSON object with keys: 'candidate_name' (string), 'skills' (list of strings), 'years_experience' (integer, total years), and 'location' (string, city/state)."},
                {"role": "user", "content": f"Extract metadata from this resume:\n\n{text[:4000]}"}
            ],
            response_format={"type": "json_object"}
        )
        import json
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"Error extracting metadata: {e}")
        return {}

def generate_search_reasoning(query, candidate_data, client):
    if not client: return "AI Reasoning unavailable."
    try:
        resume_snippet = candidate_data.get('text', '')[:1000]
        prompt = f"""
        Query: "{query}"
        Candidate: {candidate_data.get('candidate_name')}
        Skills: {candidate_data.get('skills')}
        Resume Snippet: {resume_snippet}...
        
        Explain in 1 short sentence why this candidate is a good match for the query. Highlight key matching skills or experience.
        """
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a recruiter assistant. Be concise and professional."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=60
        )
        return response.choices[0].message.content
    except Exception:
        return "Could not generate reasoning."

# --- AUTHENTICATION ---
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["auth"]["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Please enter the Teapress access code:", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input + error.
        st.text_input(
            "Please enter the Teapress access code:", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Access code incorrect")
        return False
    else:
        # Password correct.
        return True

if not check_password():
    st.stop()

# --- APP UI ---

st.set_page_config(page_title="Teapress Recruiting", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .reportview-container { background: #FAFAF9; }
    .main-header { 
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 3rem; 
        color: #002147; 
        font-weight: 700; 
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #57534E;
        margin-bottom: 30px;
    }
    .card { 
        background-color: white; 
        padding: 25px; 
        border-radius: 15px; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.05); 
        margin-bottom: 20px; 
        border-left: 6px solid #002147; 
        transition: transform 0.2s;
    }
    .card:hover {
        transform: translateY(-2px);
    }
    .score-badge { 
        background-color: #E0E7FF; 
        color: #002147; 
        padding: 6px 12px; 
        border-radius: 20px; 
        font-weight: bold; 
        float: right; 
        font-size: 0.9rem;
    }
    .reasoning-box { 
        background-color: #F8FAFC; 
        padding: 15px; 
        border-radius: 8px; 
        margin-top: 15px; 
        font-style: italic; 
        color: #334155; 
        border: 1px solid #E2E8F0;
    }
    .stButton>button {
        background-color: #002147 !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 10px 20px !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Clients
try:
    qdrant_client = get_qdrant_client()
    openai_client = get_openai_client()
    sparse_model = get_sparse_embedding_model()
except Exception as e:
    st.error(f"⚠️ Configuration Error: {e}")
    st.markdown("""
    ### How to fix this on Streamlit Cloud:
    1. Go to your app dashboard at **share.streamlit.io**.
    2. Click the **three dots (⋮)** next to your app -> **Settings**.
    3. Go to the **Secrets** tab.
    4. Paste your configuration (from your local `.streamlit/secrets.toml`) into the box:
    
    ```toml
    [qdrant]
    url = "YOUR_QDRANT_URL"
    api_key = "YOUR_QDRANT_KEY"

    [openai]
    api_key = "YOUR_OPENAI_KEY"
    ```
    5. Click **Save**. The app will restart automatically.
    """)
    st.stop()

# --- SIDEBAR: BULK UPLOAD ---
st.sidebar.title("Teapress Upload")

# Database Stats & Connection Check
try:
    if not qdrant_client.collection_exists(collection_name="resumes"):
        st.sidebar.error("Collection 'resumes' not found!")
        if st.sidebar.button("Create Collection"):
            try:
                qdrant_client.create_collection(
                    collection_name="resumes",
                    vectors_config={
                        "resume_embedding": models.VectorParams(size=1536, distance=models.Distance.COSINE)
                    },
                    sparse_vectors_config={
                        "resume_keywords": models.SparseVectorParams()
                    }
                )
                st.sidebar.success("Collection created!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Failed to create: {e}")
    else:
        count_result = qdrant_client.count(collection_name="resumes")
        st.sidebar.markdown(f"**Total Talent Pool:** {count_result.count} candidates")
except Exception as e:
    st.sidebar.error(f"Connection Error: {e}")

st.sidebar.info("Hi Joanna! Drag & drop resumes here to add them to your Teapress talent pool.")

uploaded_files = st.sidebar.file_uploader(
    "Upload Resumes (PDF, DOCX, TXT)", 
    type=["pdf", "docx", "txt", "md"], 
    accept_multiple_files=True
)

if uploaded_files:
    if st.sidebar.button(f"Process {len(uploaded_files)} Resumes"):
        if not openai_client:
            st.sidebar.error("OpenAI API Key is missing.")
        else:
            progress_bar = st.sidebar.progress(0)
            status_text = st.sidebar.empty()
            success_count = 0
            errors = []
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Reading {uploaded_file.name}...")
                
                # Explicitly check for legacy .doc files
                if uploaded_file.name.lower().endswith('.doc'):
                    errors.append(f"{uploaded_file.name}: Legacy .doc format. Please save as .docx")
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    continue

                try:
                    # 1. Extract Text
                    text = get_resume_text(uploaded_file, uploaded_file.name)
                    
                    if text:
                        # 2. AI Extraction
                        ai_metadata = extract_metadata(text, openai_client)
                        
                        # 3. Prepare Metadata
                        c_name = ai_metadata.get("candidate_name") or uploaded_file.name.rsplit('.', 1)[0].replace("_", " ").replace("-", " ").title()
                        metadata = {
                            "candidate_name": c_name,
                            "skills": ai_metadata.get("skills", []),
                            "years_experience": ai_metadata.get("years_experience", 0),
                            "location": ai_metadata.get("location", "Unknown"),
                            "source_filename": uploaded_file.name
                        }
                        
                        # 4. Index
                        index_resume(text, metadata, qdrant_client, openai_client, sparse_model)
                        success_count += 1
                    else:
                        errors.append(f"{uploaded_file.name}: Could not read file content")
                except Exception as e:
                    errors.append(f"{uploaded_file.name}: {str(e)}")
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.text("Processing Complete")
            if success_count > 0:
                st.sidebar.success(f"Added {success_count} new candidates.")
            
            if errors:
                st.sidebar.error(f"Had trouble with {len(errors)} files.")
                with st.sidebar.expander("See details"):
                    for err in errors:
                        st.write(f"- {err}")

# --- MAIN AREA: RECRUITER SEARCH ---
st.markdown('<div class="main-header">Teapress Talent Search</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Welcome back, Joanna. Let\'s find your next great hire.</div>', unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])
with col1:
    search_query = st.text_input("Who are you looking for today?", placeholder="e.g. Friendly Customer Success Manager who loves tea and has startup experience")
with col2:
    st.write("") # Spacer
    st.write("")
    search_btn = st.button("Find Matches", type="primary", use_container_width=True)

if search_btn and search_query:
    if not openai_client:
        st.error("OpenAI API Key is missing.")
    else:
        with st.spinner("Searching candidates..."):
            try:
                results = hybrid_search(search_query, qdrant_client, openai_client, sparse_model, limit=10)
                
                if not results:
                    st.warning("No perfect matches found yet. Try a broader search?")
                else:
                    st.success(f"Found {len(results)} great candidates for Teapress.")
                    
                    for point in results:
                        payload = point.payload
                        score = point.score
                        
                        # Generate AI Reasoning on the fly
                        reasoning = generate_search_reasoning(search_query, payload, openai_client)
                        
                        # Render Card
                        st.markdown(f"""
                        <div class="card">
                            <span class="score-badge">Match: {int(score * 100)}%</span>
                            <h3 style="color: #002147; margin-top:0;">{payload.get('candidate_name', 'Unknown Candidate')}</h3>
                            <p style="color: #666;"><strong>Location: {payload.get('location', 'Unknown')}</strong> &nbsp;|&nbsp; <strong>Experience: {payload.get('years_experience', 0)} Years</strong></p>
                            <p><strong>Key Skills:</strong> {', '.join(payload.get('skills', [])[:8])}...</p>
                            <div class="reasoning-box">
                                <strong>Why they fit Teapress:</strong> {reasoning}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col_actions1, col_actions2 = st.columns([1, 1])
                        
                        with col_actions1:
                            with st.expander(f"Read Resume"):
                                st.markdown("### Full Resume Content")
                                st.text_area("Resume Content", payload.get('text', ''), height=400, label_visibility="collapsed")
                        
                        with col_actions2:
                            with st.expander(f"Draft Email"):
                                if st.button(f"Generate Email for {payload.get('candidate_name').split()[0]}", key=f"email_{point.id}"):
                                    with st.spinner("Drafting email..."):
                                        try:
                                            email_prompt = f"""
                                            Write a warm, professional recruiting email from Joanna at Teapress to {payload.get('candidate_name')}.
                                            
                                            Context:
                                            - We are Teapress, a modern tea company.
                                            - We are looking for: {search_query}
                                            - Why we like them: {reasoning}
                                            
                                            Keep it short, friendly, and mention specifically why their background stands out based on the query.
                                            """
                                            
                                            email_response = openai_client.chat.completions.create(
                                                model="gpt-4o-mini",
                                                messages=[{"role": "user", "content": email_prompt}]
                                            )
                                            email_draft = email_response.choices[0].message.content
                                            st.text_area("Draft", email_draft, height=250)
                                            st.markdown(f"[Open in Mail App](mailto:?subject=Interview%20with%20Teapress&body={email_draft.replace(' ', '%20').replace(chr(10), '%0A')})")
                                        except Exception as e:
                                            st.error("Could not generate email.")


            except Exception as e:
                st.error(f"Search Error: {e}")
