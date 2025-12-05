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
import datetime

# --- APP UI CONFIG ---
st.set_page_config(page_title="Teapress Recruiting", layout="wide", initial_sidebar_state="expanded")

# --- CONSTANTS ---
COLLECTION_NAME = "resumes"
JOBS_COLLECTION_NAME = "jobs"
FEEDBACK_COLLECTION_NAME = "feedback"
DENSE_VECTOR_NAME = "resume_embedding"
SPARSE_VECTOR_NAME = "resume_keywords"
DENSE_MODEL_NAME = "text-embedding-3-small"
SPARSE_MODEL_NAME = "Qdrant/bm25"

# --- CLIENT INITIALIZATION ---
@st.cache_resource
def get_qdrant_client():
    if "qdrant" in st.secrets:
        url = st.secrets["qdrant"]["url"]
        api_key = st.secrets["qdrant"]["api_key"]
    else:
        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")
    
    if not url:
        raise ValueError("Qdrant URL not found.")
    return QdrantClient(url=url, api_key=api_key)

@st.cache_resource
def get_openai_client():
    if "openai" in st.secrets:
        api_key = st.secrets["openai"]["api_key"]
    else:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key: return None
    return OpenAI(api_key=api_key)

@st.cache_resource
def get_sparse_embedding_model():
    return SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)

# --- PARSING FUNCTIONS ---
def parse_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def parse_docx(file):
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
    if filename_lower.endswith('.pdf'): return parse_pdf(file)
    elif filename_lower.endswith('.docx'): return parse_docx(file)
    elif filename_lower.endswith('.txt') or filename_lower.endswith('.md'): return parse_text_file(file)
    elif filename_lower.endswith('.doc'): raise ValueError("Legacy .doc format not supported. Please convert to .docx")
    else: raise ValueError("Unsupported file format")

# --- EMBEDDING & INDEXING ---
def generate_dense_embedding(text, client):
    if not client: raise ValueError("OpenAI Client not initialized.")
    truncated_text = text[:32000]
    response = client.embeddings.create(input=truncated_text, model=DENSE_MODEL_NAME)
    return response.data[0].embedding

def generate_sparse_embedding(text, model):
    embeddings = list(model.embed([text]))
    if not embeddings: return None
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

# --- SEARCH & AI ---
def hybrid_search(query_text, qdrant_client, openai_client, sparse_model, limit=20):
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
        
        Analyze this candidate against the query.
        Provide 2 bullet points on why they are a good fit.
        Provide 1 bullet point on a potential concern or missing skill (if any).
        Format as HTML: 
        <b>Why they fit:</b><br>- Point 1<br>- Point 2<br><br>
        <b>Potential Concerns:</b><br>- Concern 1
        """
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a critical recruiter assistant. Be concise."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )
        content = response.choices[0].message.content
        return content.replace("```html", "").replace("```", "").strip()
    except Exception:
        return "Could not generate reasoning."

# --- JOB MANAGEMENT ---
def ensure_collections_exist(qdrant_client):
    try:
        if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={DENSE_VECTOR_NAME: models.VectorParams(size=1536, distance=models.Distance.COSINE)},
                sparse_vectors_config={SPARSE_VECTOR_NAME: models.SparseVectorParams()}
            )
        if not qdrant_client.collection_exists(collection_name=JOBS_COLLECTION_NAME):
            qdrant_client.create_collection(
                collection_name=JOBS_COLLECTION_NAME,
                vectors_config={"job_embedding": models.VectorParams(size=1536, distance=models.Distance.COSINE)}
            )
        if not qdrant_client.collection_exists(collection_name=FEEDBACK_COLLECTION_NAME):
            qdrant_client.create_collection(
                collection_name=FEEDBACK_COLLECTION_NAME,
                vectors_config={} # No vectors needed for feedback log yet, or maybe just simple payload storage
            )
        return True
    except Exception as e:
        st.error(f"Collection Init Error: {e}")
        return False

def create_job(title, description, qdrant_client, openai_client):
    try:
        embedding = generate_dense_embedding(description, openai_client)
        point_id = str(uuid.uuid4())
        point = models.PointStruct(
            id=point_id,
            vector={"job_embedding": embedding},
            payload={"title": title, "description": description, "created_at": str(uuid.uuid1()), "status": "Open"}
        )
        qdrant_client.upsert(collection_name=JOBS_COLLECTION_NAME, points=[point])
        return True
    except Exception as e:
        st.error(f"Error creating job: {e}")
        return False

def update_job(job_id, title, description, status, qdrant_client, openai_client):
    try:
        embedding = generate_dense_embedding(description, openai_client)
        point = models.PointStruct(
            id=job_id,
            vector={"job_embedding": embedding},
            payload={"title": title, "description": description, "updated_at": str(uuid.uuid1()), "status": status}
        )
        qdrant_client.upsert(collection_name=JOBS_COLLECTION_NAME, points=[point])
        return True
    except Exception as e:
        st.error(f"Error updating job: {e}")
        return False

def save_feedback(query, job_id, rating, comment, qdrant_client):
    try:
        point_id = str(uuid.uuid4())
        point = models.PointStruct(
            id=point_id,
            vector={},
            payload={
                "query": query,
                "job_id": job_id,
                "rating": rating, # "Good" or "Bad"
                "comment": comment,
                "timestamp": datetime.datetime.now().isoformat()
            }
        )
        qdrant_client.upsert(collection_name=FEEDBACK_COLLECTION_NAME, points=[point])
        return True
    except Exception as e:
        print(f"Error saving feedback: {e}")
        return False

def get_all_feedback(qdrant_client):
    try:
        result, _ = qdrant_client.scroll(collection_name=FEEDBACK_COLLECTION_NAME, limit=100, with_payload=True)
        return sorted([p.payload for p in result], key=lambda x: x['timestamp'], reverse=True)
    except Exception:
        return []

def delete_job(job_id, qdrant_client):
    try:
        qdrant_client.delete(
            collection_name=JOBS_COLLECTION_NAME,
            points_selector=models.PointIdsList(points=[job_id])
        )
        return True
    except Exception as e:
        st.error(f"Error deleting job: {e}")
        return False

def get_all_jobs(qdrant_client):
    try:
        result, _ = qdrant_client.scroll(collection_name=JOBS_COLLECTION_NAME, limit=100, with_payload=True, with_vectors=False)
        return result
    except Exception:
        return []

def update_candidate_metadata(point_id, update_dict):
    try:
        qdrant_client.set_payload(collection_name=COLLECTION_NAME, payload=update_dict, points=[point_id])
        return True
    except Exception as e:
        st.error(f"Error updating candidate: {e}")
        return False

def add_note(point_id, current_notes, new_note):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    note_entry = f"[{timestamp}] {new_note}"
    updated_notes = current_notes + [note_entry]
    success = update_candidate_metadata(point_id, {"notes": updated_notes})
    return success, updated_notes

def shortlist_candidate(candidate_id, job_id, job_title, current_shortlists):
    new_entry = {"job_id": job_id, "job_title": job_title}
    if any(entry['job_id'] == job_id for entry in current_shortlists): return False
    updated_list = current_shortlists + [new_entry]
    return update_candidate_metadata(candidate_id, {"shortlists": updated_list})

# --- AUTHENTICATION ---
import hashlib

def check_password():
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False
    
    # Simple hash for the token (in production, use something more secure)
    secret_pass = st.secrets["auth"]["password"]
    token_hash = hashlib.sha256(secret_pass.encode()).hexdigest()
    
    # Check query param
    params = st.query_params
    if params.get("auth_token") == token_hash:
        st.session_state.password_correct = True
    
    def password_entered():
        if st.session_state["password"] == secret_pass:
            st.session_state.password_correct = True
            st.query_params["auth_token"] = token_hash
            del st.session_state["password"]
        else:
            st.session_state.password_correct = False

    if st.session_state.password_correct: return True

    st.markdown("""<style>.stTextInput > div > div > input {text-align: center;}</style>""", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("<h1 style='text-align: center; color: #002147;'>Teapress Recruiting</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Welcome back, Joanna! Please enter your secret tea code.</p>", unsafe_allow_html=True)
        st.text_input("Access Code", type="password", on_change=password_entered, key="password", label_visibility="collapsed", placeholder="Enter Access Code")
        if "password_correct" in st.session_state and not st.session_state["password_correct"]:
            st.error("That code didn't steep correctly. Try again?")
    return False

if not check_password(): st.stop()

# --- APP UI ---
st.markdown("""
<style>
    .reportview-container { background: #FAFAF9; }
    .main-header { font-family: 'Helvetica Neue', sans-serif; font-size: 3rem; color: #002147; font-weight: 700; margin-bottom: 0px; }
    .sub-header { font-size: 1.2rem; color: #57534E; margin-bottom: 30px; }
    .card { background-color: white; padding: 25px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); margin-bottom: 20px; border-left: 6px solid #002147; transition: transform 0.2s; }
    .card:hover { transform: translateY(-2px); }
    .score-badge { background-color: #E0E7FF; color: #002147; padding: 6px 12px; border-radius: 20px; font-weight: bold; float: right; font-size: 0.9rem; }
    .reasoning-box { background-color: #F8FAFC; padding: 15px; border-radius: 8px; margin-top: 15px; font-style: italic; color: #334155; border: 1px solid #E2E8F0; }
    .stButton>button { background-color: #002147 !important; color: white !important; border-radius: 8px !important; border: none !important; padding: 10px 20px !important; }
</style>
""", unsafe_allow_html=True)

try:
    qdrant_client = get_qdrant_client()
    openai_client = get_openai_client()
    sparse_model = get_sparse_embedding_model()
    ensure_collections_exist(qdrant_client)
except Exception as e:
    st.error(f"Configuration Error: {e}")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    try:
        st.image("assets/teapress_logo.png", use_container_width=True)
    except:
        st.title("Teapress Recruiting")

# Job Management Section
st.sidebar.subheader("Job Requisitions")

jobs = get_all_jobs(qdrant_client)
# Sort jobs: Open first, then by name
job_list = []
for job in jobs:
    title = job.payload['title']
    status = job.payload.get('status', 'Open')
    display_title = title if status == 'Open' else f"{title} (Closed)"
    job_list.append({"display": display_title, "data": job, "status": status})

# Sort: Open jobs first, then alphabetical
job_list.sort(key=lambda x: (0 if x['status'] == 'Open' else 1, x['display']))

job_options = {j['display']: j['data'] for j in job_list}
job_dropdown_options = ["Create New Job"] + list(job_options.keys())

# Ensure session state selector is valid
if "job_selector" not in st.session_state:
    st.session_state.job_selector = job_dropdown_options[0] if len(job_options) == 0 else job_dropdown_options[1]

# Handle case where selected job was deleted
if st.session_state.job_selector not in job_dropdown_options:
    st.session_state.job_selector = job_dropdown_options[0]

selected_option = st.sidebar.selectbox(
    "Select Active Job:", 
    job_dropdown_options, 
    key="job_selector"
)
active_job_data = None

# Render UI based on selection
if selected_option == "Create New Job":
    st.session_state.active_job_id = None
    st.session_state.active_job_title = None
    
    st.sidebar.markdown("### Define New Role")
    with st.sidebar.form("create_job_form", clear_on_submit=True):
        new_job_title = st.text_input("Job Title")
        new_job_desc = st.text_area("Job Description")
        submitted = st.form_submit_button("Save Job")
        
        if submitted:
            if new_job_title and new_job_desc:
                if create_job(new_job_title, new_job_desc, qdrant_client, openai_client):
                    st.sidebar.success("Job Created!")
                    st.session_state.job_selector = new_job_title
                    st.rerun()
            else:
                st.sidebar.error("Please fill in both fields.")

else:
    # Existing Job Selected
    active_job_data = job_options[selected_option]
    st.session_state.active_job_id = active_job_data.id
    st.session_state.active_job_title = active_job_data.payload['title']
    
    # Condensed Job Info / Edit
    current_status = active_job_data.payload.get('status', 'Open')
    with st.sidebar.expander("Manage Job Settings", expanded=False):
        edit_title = st.text_input("Edit Title", value=active_job_data.payload['title'])
        edit_desc = st.text_area("Edit Description", value=active_job_data.payload['description'])
        edit_status = st.selectbox("Job Status", ["Open", "Closed"], index=0 if current_status == "Open" else 1)
        
        col_save, col_del = st.columns(2)
        with col_save:
            if st.button("Update Job"):
                # We need to update the update_job function to handle status, 
                # but since we are replacing the call here, we must ensure the function exists or update it too.
                # Since I cannot update the function definition in this specific block without replacing the whole file,
                # I will handle the update strictly here or ensure I update the function definition in valid scope.
                # Ideally, I should update the function definition in the 'JOB MANAGEMENT' section first. 
                # HOWEVER, I can just update the payload directly here using qdrant_client if I want to be quick, 
                # but better to update the helper function in a separate call or same call if possible.
                # I will assume I will update the function definition in the next tool call or usage.
                # WAIT: I will update the function definition in the SAME file replacement if possible or separate.
                # Since this tool call targets lines 258->381, I CANNOT update the function definitions (lines 190+).
                # So I will update the call here to pass status, and THEN I will immediately update the function definition in the next step.
                update_job(active_job_data.id, edit_title, edit_desc, edit_status, qdrant_client, openai_client)
                
                # If title/status changed, update selector
                new_display = edit_title if edit_status == 'Open' else f"{edit_title} (Closed)"
                if new_display != selected_option:
                    st.session_state.job_selector = new_display
                st.success("Updated!")
                st.rerun()
        with col_del:
            if st.button("Delete Job"):
                delete_job(active_job_data.id, qdrant_client)
                st.session_state.job_selector = "Create New Job"
                st.success("Deleted!")
                st.rerun()

st.sidebar.divider()
st.sidebar.subheader("Upload Resumes")
uploaded_files = st.sidebar.file_uploader("Upload Resumes (PDF, DOCX, TXT)", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True)

if uploaded_files:
    if st.sidebar.button(f"Process {len(uploaded_files)} Resumes"):
        if not openai_client: st.sidebar.error("OpenAI API Key is missing.")
        else:
            progress_bar = st.sidebar.progress(0)
            status_text = st.sidebar.empty()
            success_count = 0
            errors = []
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Reading {uploaded_file.name}...")
                if uploaded_file.name.lower().endswith('.doc'):
                    errors.append(f"{uploaded_file.name}: Legacy .doc format. Please save as .docx")
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    continue
                try:
                    text = get_resume_text(uploaded_file, uploaded_file.name)
                    if text:
                        ai_metadata = extract_metadata(text, openai_client)
                        c_name = ai_metadata.get("candidate_name") or uploaded_file.name.rsplit('.', 1)[0].replace("_", " ").replace("-", " ").title()
                        metadata = {
                            "candidate_name": c_name,
                            "skills": ai_metadata.get("skills", []),
                            "years_experience": ai_metadata.get("years_experience", 0),
                            "location": ai_metadata.get("location", "Unknown"),
                            "source_filename": uploaded_file.name,
                            "shortlists": []
                        }
                        index_resume(text, metadata, qdrant_client, openai_client, sparse_model)
                        success_count += 1
                    else: errors.append(f"{uploaded_file.name}: Could not read file content")
                except Exception as e: errors.append(f"{uploaded_file.name}: {str(e)}")
                progress_bar.progress((i + 1) / len(uploaded_files))
            status_text.text("Processing Complete")
            if success_count > 0: st.sidebar.success(f"Added {success_count} new candidates.")
            if errors:
                st.sidebar.error(f"Had trouble with {len(errors)} files.")
                with st.sidebar.expander("See details"):
                    for err in errors: st.write(f"- {err}")

# --- REFINEMENT LOGIC ---
def refine_query_with_feedback(current_query, feedback_list, client):
    if not feedback_list: return current_query
    
    # Format feedback for LLM
    feedback_text = "\n".join([f"- Candidate: {f['name']} | Rating: {f['rating']} | Reason: {f['reason']}" for f in feedback_list])
    
    prompt = f"""
    Refine this search query based on recruiter feedback on specific candidates.
    
    Original Query: "{current_query}"
    
    Recruiter Feedback:
    {feedback_text}
    
    Task: Rewrite the query to better capture the recruiter's intent. 
    - If they liked a candidate,emphasize those skills/traits.
    - If they disliked a candidate, explicitly exclude or deprioritize those traits.
    - Keep the query concise (under 50 words) but specific.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return current_query

# --- SEARCH STATE ---
if "search_results" not in st.session_state: st.session_state.search_results = []
if "last_query" not in st.session_state: st.session_state.last_query = ""
if "feedback_buffer" not in st.session_state: st.session_state.feedback_buffer = []

# --- MAIN AREA ---
st.markdown('<div class="main-header">Teapress Talent Search</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Welcome back, Joanna. Let\'s find your next great hire.</div>', unsafe_allow_html=True)

# ACTIVE JOB BANNER
if st.session_state.active_job_title:
    st.markdown(f"""
    <div style="background-color: #E0E7FF; padding: 15px; border-radius: 8px; border-left: 5px solid #002147; margin-bottom: 20px;">
        <h3 style="margin:0; color: #002147;">Working on: {st.session_state.active_job_title}</h3>
        <p style="margin:0; color: #4B5563; font-size: 0.9rem;">Search results and shortlists will be linked to this job.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("Please select a **Job Requisition** from the sidebar to start searching.")

st.markdown("### Search Candidates")

# Search Controls
with st.expander("Search Filters", expanded=False):
    min_score = st.slider("Minimum Match Score (%)", 0, 100, 60, step=5) / 100.0

search_tab1, search_tab2, search_tab3 = st.tabs(["Quick Search", "Match with Job Description", "View Shortlist"])

with search_tab1:
    col_q1, col_q2 = st.columns([4, 1])
    with col_q1:
        # If we have refined the query, pre-fill it.
        # We use a temp key if needed, or update session state logic
        default_query = st.session_state.get("refined_query", "")
        search_query = st.text_input("Describe the perfect candidate:", value=default_query, placeholder="e.g. Friendly Customer Success Manager...")
    with col_q2:
        st.write("")
        st.write("")
        search_btn_1 = st.button("Find Matches", type="primary", key="btn1")

with search_tab2:
    default_jd = active_job_data.payload['description'] if active_job_data else ""
    col_jd1, col_jd2 = st.columns([1, 1])
    with col_jd1:
        job_description = st.text_area("Paste the Job Description here:", value=default_jd, height=250, placeholder="Paste the full JD text here...")
    with col_jd2:
        st.info("**Pro Tip:** Pasting the full JD helps our AI find candidates with the exact skills.")
        st.write("")
        search_btn_2 = st.button("Find Matches based on JD", type="primary", key="btn2")

# Refinement Action (Visible if feedback exists)
if st.session_state.feedback_buffer:
    st.info(f"You have provided feedback on {len(st.session_state.feedback_buffer)} candidates.")
    if st.button("✨ Refine & Rerun Search"):
        with st.spinner("Refining your search strategy..."):
            base_query = search_query if search_query else (job_description if job_description else st.session_state.last_query)
            refined_query = refine_query_with_feedback(base_query, st.session_state.feedback_buffer, openai_client)
            st.session_state.refined_query = refined_query # Store for the text input
            st.session_state.last_query = refined_query
            # Rerun search immediately
            results = hybrid_search(refined_query, qdrant_client, openai_client, sparse_model, limit=20)
            st.session_state.search_results = results
            st.session_state.feedback_buffer = [] # Clear buffer after refinement
            st.rerun()

active_query = ""
if search_btn_1 and search_query: active_query = search_query
elif search_btn_2 and job_description: active_query = f"Job Description:\n{job_description}"

if active_query:
    st.session_state.last_query = active_query
    if not openai_client: st.error("OpenAI API Key is missing.")
    else:
        with st.spinner("Searching candidates..."):
            try:
                results = hybrid_search(active_query, qdrant_client, openai_client, sparse_model, limit=20)
                st.session_state.search_results = results
            except Exception as e: st.error(f"Search Error: {e}")

with search_tab3:
    if st.session_state.active_job_id:
        if st.button("Refresh Shortlist"):
            shortlist_filter = models.Filter(must=[models.FieldCondition(key="shortlists.job_id", match=models.MatchValue(value=st.session_state.active_job_id))])
            shortlisted_points, _ = qdrant_client.scroll(collection_name=COLLECTION_NAME, scroll_filter=shortlist_filter, limit=100, with_payload=True)
            st.session_state.search_results = shortlisted_points
            st.session_state.last_query = f"Shortlist for {st.session_state.active_job_title}"
            st.rerun()
        st.write(f"Candidates shortlisted for **{st.session_state.active_job_title}**:")
    else: st.warning("Please select an Active Job from the sidebar to view its shortlist.")

# --- RENDER RESULTS ---
if st.session_state.search_results:
    results = st.session_state.search_results
    col_filter1, col_filter2 = st.columns([3, 1])
    with col_filter2: show_ignored = st.checkbox("Show Ignored Candidates", value=False)
    
    visible_results = []
    for point in results:
        status = point.payload.get("status", "New")
        if status == "Ignored" and not show_ignored: continue
        # Score Filter
        if point.score < min_score: continue
        
        visible_results.append(point)

    if not visible_results: st.warning("No active candidates found matching criteria.")
    else:
        st.success(f"Showing {len(visible_results)} candidates for: '{st.session_state.last_query[:50]}...'")

        for point in visible_results:
            payload = point.payload
            score = point.score
            status = payload.get("status", "New")
            rating = payload.get("rating", 0)
            notes = payload.get("notes", [])
            
            # --- CARD CONTAINER ---
            with st.container(border=True):
                # Header Row
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"<h3 style='margin:0; color:#002147;'>{payload.get('candidate_name', 'Unknown')}</h3>", unsafe_allow_html=True)
                    st.markdown(f"**{payload.get('location', 'Unknown')}** • {payload.get('years_experience', 0)} Years Exp")
                with c2:
                    st.markdown(f"""
                        <div style="text-align:right;">
                            <span style="background-color:#E0E7FF; padding:4px 10px; border-radius:12px; font-weight:bold; color:#002147;">
                                {int(score*100)}% Match
                            </span>
                            <div style="margin-top:5px; font-size:0.8rem; color:#666;">Status: {status}</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Tabs for content
                tab_overview, tab_resume, tab_review, tab_email = st.tabs(["Overview", "Resume", "Review & Notes", "Draft Email"])
                
                with tab_overview:
                    st.markdown(f"**Key Skills:** {', '.join(payload.get('skills', [])[:10])}")
                    reasoning = generate_search_reasoning(st.session_state.last_query, payload, openai_client)
                    st.markdown(f"""
                    <div style="background-color:#F8FAFC; padding:15px; border-radius:8px; border:1px solid #E2E8F0; margin-top:10px;">
                        {reasoning}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # --- NEW: Candidate Feedback for Refinement ---
                    with st.expander("Improve Future Searches", expanded=False):
                        st.caption("Help the system learn. Why is this candidate a Good or Bad match for this job?")
                        fb_col1, fb_col2 = st.columns([1, 3])
                        with fb_col1:
                            fb_type = st.radio("Verdict", ["Good Match", "Bad Match"], key=f"fb_type_{point.id}", label_visibility="collapsed")
                        with fb_col2:
                            fb_reason = st.text_input("Reasoning (e.g. 'Expert in Python', 'Too Junior')", key=f"fb_reason_{point.id}")
                        
                        if st.button("Log Feedback", key=f"log_fb_{point.id}"):
                            # Add to session buffer
                            st.session_state.feedback_buffer.append({
                                "name": payload.get('candidate_name'),
                                "rating": fb_type,
                                "reason": fb_reason
                            })
                            st.toast("Feedback logged! Click 'Refine & Rerun' above when ready.")
                
                with tab_resume:
                    st.text_area("Resume Content", payload.get('text', ''), height=400, label_visibility="collapsed", key=f"resume_{point.id}")
                
                with tab_review:
                    r1, r2 = st.columns(2)
                    with r1:
                        # Status & Rating
                        new_status = st.selectbox("Application Status", ["New", "Reviewed", "Contacted", "Ignored"], index=["New", "Reviewed", "Contacted", "Ignored"].index(status), key=f"status_{point.id}")
                        if new_status != status:
                            update_candidate_metadata(point.id, {"status": new_status})
                            point.payload["status"] = new_status
                            st.rerun()
                        
                        new_rating = st.slider("Star Rating", 1, 5, int(rating) if rating else 0, key=f"rating_{point.id}")
                        if new_rating != rating:
                            update_candidate_metadata(point.id, {"rating": new_rating})
                            point.payload["rating"] = new_rating
                            st.rerun()
                            
                        # Shortlist Logic
                        if st.session_state.active_job_id:
                            current_shortlists = payload.get("shortlists", [])
                            is_shortlisted = any(entry['job_id'] == st.session_state.active_job_id for entry in current_shortlists)
                            
                            if is_shortlisted:
                                st.success(f"Shortlisted for {st.session_state.active_job_title}")
                                if st.button("Remove from Shortlist", key=f"rem_sl_{point.id}"):
                                    updated_list = [entry for entry in current_shortlists if entry['job_id'] != st.session_state.active_job_id]
                                    update_candidate_metadata(point.id, {"shortlists": updated_list})
                                    st.rerun()
                            else:
                                if st.button(f"Add to Shortlist", key=f"add_sl_{point.id}"):
                                    shortlist_candidate(point.id, st.session_state.active_job_id, st.session_state.active_job_title, current_shortlists)
                                    st.rerun()
                        else:
                            st.info("Select a Job to enable shortlisting.")

                    with r2:
                        # Notes
                        st.markdown("**Notes History**")
                        for note in notes:
                            st.caption(note)
                        
                        new_note_text = st.text_input("Add new note", key=f"note_in_{point.id}")
                        if st.button("Add Note", key=f"btn_note_{point.id}"):
                            if new_note_text:
                                success, updated_notes = add_note(point.id, notes, new_note_text)
                                if success:
                                    point.payload["notes"] = updated_notes
                                    st.rerun()
                
                with tab_email:
                    st.write("Generate a personalized outreach email.")
                    if st.button(f"Draft Email to {payload.get('candidate_name')}", key=f"email_gen_{point.id}"):
                        with st.spinner("Writing..."):
                            email_prompt = f"""
                            Write a warm, professional recruiting email from Joanna at Teapress to {payload.get('candidate_name')}.
                            Context:
                            - We are Teapress, a modern tea company.
                            - We are looking for: {st.session_state.last_query}
                            - Mention these skills: {', '.join(payload.get('skills', [])[:5])}
                            Keep it short and friendly.
                            """
                            try:
                                email_response = openai_client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": email_prompt}])
                                st.session_state[f"email_{point.id}"] = email_response.choices[0].message.content
                            except Exception as e: st.error("Failed to generate email.")
                    
                    if f"email_{point.id}" in st.session_state:
                         st.text_area("Subject: Invitation to Interview", value=st.session_state[f"email_{point.id}"], height=200)
                         link_body = st.session_state[f"email_{point.id}"].replace('\n', '%0A')
                         st.markdown(f"[Open in Mail Client](mailto:?subject=Teapress%20Opportunity&body={link_body})")

# --- FEEDBACK UI ---
if st.session_state.search_results and st.session_state.last_query:
    st.divider()
    st.caption("How was the search quality for this query?")
    col_fb1, col_fb2 = st.columns([1, 4])
    with col_fb1:
        feedback_sentiment = st.radio("Search Quality", ["Good", "Needs Improvement"], label_visibility="collapsed", horizontal=True, key="fb_sentiment")
    
    with col_fb2:
        if feedback_sentiment == "Needs Improvement":
            feedback_comment = st.text_input("What was missing or incorrect?", placeholder="e.g. Too many junior candidates...")
            if st.button("Submit Feedback"):
                job_id = st.session_state.active_job_id if st.session_state.active_job_id else "general"
                save_feedback(st.session_state.last_query, job_id, "Bad", feedback_comment, qdrant_client)
                st.toast("Thanks! We'll use this to improve.")
        else:
             if st.button("Mark as Good"):
                job_id = st.session_state.active_job_id if st.session_state.active_job_id else "general"
                save_feedback(st.session_state.last_query, job_id, "Good", "", qdrant_client)
                st.toast("Thanks for the feedback!")

# --- ADMIN VIEW ---
with st.sidebar:
    st.divider()
    show_admin = st.toggle("Admin: Search Quality")

if show_admin:
    st.markdown("---")
    st.title("Admin: Search Quality Review")
    feedback_data = get_all_feedback(qdrant_client)
    if feedback_data:
        st.dataframe(feedback_data, use_container_width=True)
    else:
        st.info("No feedback collected yet.")
