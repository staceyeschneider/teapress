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

# --- APP UI CONFIG ---
st.set_page_config(page_title="Teapress Recruiting", layout="wide", initial_sidebar_state="expanded")

# --- UTILS SECTION (Merged to avoid import errors) ---

# Constants
COLLECTION_NAME = "resumes"
JOBS_COLLECTION_NAME = "jobs"
DENSE_VECTOR_NAME = "resume_embedding"
SPARSE_VECTOR_NAME = "resume_keywords"
DENSE_MODEL_NAME = "text-embedding-3-small"
SPARSE_MODEL_NAME = "Qdrant/bm25"

# ... (Previous imports and client getters remain the same) ...

# --- JOB MANAGEMENT FUNCTIONS ---
def ensure_collections_exist(qdrant_client):
    """Ensures both resumes and jobs collections exist."""
    try:
        # Resumes Collection
        if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={
                    DENSE_VECTOR_NAME: models.VectorParams(size=1536, distance=models.Distance.COSINE)
                },
                sparse_vectors_config={
                    SPARSE_VECTOR_NAME: models.SparseVectorParams()
                }
            )
        
        # Jobs Collection
        if not qdrant_client.collection_exists(collection_name=JOBS_COLLECTION_NAME):
            qdrant_client.create_collection(
                collection_name=JOBS_COLLECTION_NAME,
                vectors_config={
                    "job_embedding": models.VectorParams(size=1536, distance=models.Distance.COSINE)
                }
            )
        return True
    except Exception as e:
        st.error(f"Collection Init Error: {e}")
        return False

def create_job(title, description, qdrant_client, openai_client):
    """Creates a new job requisition."""
    try:
        # Embed the description for future matching
        embedding = generate_dense_embedding(description, openai_client)
        
        point_id = str(uuid.uuid4())
        point = models.PointStruct(
            id=point_id,
            vector={"job_embedding": embedding},
            payload={
                "title": title,
                "description": description,
                "created_at": str(uuid.uuid1()) # Simple timestamp
            }
        )
        qdrant_client.upsert(collection_name=JOBS_COLLECTION_NAME, points=[point])
        return True
    except Exception as e:
        st.error(f"Error creating job: {e}")
        return False

def get_all_jobs(qdrant_client):
    """Fetches all jobs."""
    try:
        # Scroll to get all (limit 100 for now)
        result, _ = qdrant_client.scroll(
            collection_name=JOBS_COLLECTION_NAME,
            limit=100,
            with_payload=True,
            with_vectors=False
        )
        return result
    except Exception:
        return []

def shortlist_candidate(candidate_id, job_id, job_title, current_shortlists):
    """Adds a job to the candidate's shortlist."""
    # current_shortlists is a list of dicts: [{"job_id": "...", "job_title": "..."}]
    new_entry = {"job_id": job_id, "job_title": job_title}
    
    # Check for duplicates
    if any(entry['job_id'] == job_id for entry in current_shortlists):
        return False # Already shortlisted
        
    updated_list = current_shortlists + [new_entry]
    return update_candidate_metadata(candidate_id, {"shortlists": updated_list})

# ... (Previous parsing/embedding functions remain the same) ...

# --- SIDEBAR: JOBS & UPLOAD ---
st.sidebar.title("Teapress Recruiting")

# 1. Job Management
st.sidebar.header("📂 Job Requisitions")
jobs = get_all_jobs(qdrant_client)
job_options = {job.payload['title']: job for job in jobs}
job_titles = ["-- None --"] + list(job_options.keys())

selected_job_title = st.sidebar.selectbox("Active Job:", job_titles)
active_job_data = None
if selected_job_title != "-- None --":
    active_job_data = job_options[selected_job_title]
    st.session_state.active_job_id = active_job_data.id
    st.session_state.active_job_title = selected_job_title
else:
    st.session_state.active_job_id = None
    st.session_state.active_job_title = None

with st.sidebar.expander("➕ Create New Job"):
    new_job_title = st.text_input("Job Title")
    new_job_desc = st.text_area("Job Description")
    if st.button("Save Job"):
        if new_job_title and new_job_desc:
            if create_job(new_job_title, new_job_desc, qdrant_client, openai_client):
                st.sidebar.success("Job Created!")
                st.rerun()
        else:
            st.sidebar.error("Please fill in both fields.")

st.sidebar.divider()

# 2. Bulk Upload (Existing Logic)
st.sidebar.header("📥 Upload Resumes")
# ... (Existing upload logic) ...
uploaded_files = st.sidebar.file_uploader(
    "Upload Resumes (PDF, DOCX, TXT)", 
    type=["pdf", "docx", "txt", "md"], 
    accept_multiple_files=True
)

if uploaded_files:
    if st.sidebar.button(f"Process {len(uploaded_files)} Resumes"):
        # ... (Existing processing logic, ensure ensure_collections_exist is called first) ...
        ensure_collections_exist(qdrant_client)
        # ... (Rest of processing loop) ...
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
                            "source_filename": uploaded_file.name,
                            "shortlists": [] # Initialize empty shortlist
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


# --- MAIN AREA ---
# ... (Header) ...
st.markdown('<div class="main-header">Teapress Talent Search</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Welcome back, Joanna. Let\'s find your next great hire.</div>', unsafe_allow_html=True)

# Show Active Job Context
if st.session_state.active_job_title:
    st.info(f"📂 **Active Job:** {st.session_state.active_job_title}")

st.markdown("### 🔎 How do you want to search?")
search_tab1, search_tab2, search_tab3 = st.tabs(["Quick Search", "Match with Job Description", "📂 View Shortlist"])

with search_tab1:
    search_query = st.text_input("Describe the perfect candidate:", placeholder="e.g. Friendly Customer Success Manager...")
    search_btn_1 = st.button("Find Matches", type="primary", key="btn1")

with search_tab2:
    # Auto-fill if active job
    default_jd = active_job_data.payload['description'] if active_job_data else ""
    col_jd1, col_jd2 = st.columns([1, 1])
    with col_jd1:
        job_description = st.text_area("Paste the Job Description here:", value=default_jd, height=250, placeholder="Paste the full JD text here...")
    with col_jd2:
        st.info("💡 **Pro Tip:** Pasting the full JD helps our AI find candidates with the exact skills.")
        st.write("")
        search_btn_2 = st.button("Find Matches based on JD", type="primary", key="btn2")

with search_tab3:
    if st.session_state.active_job_id:
        if st.button("Refresh Shortlist"):
            # We need to filter candidates who have this job_id in their 'shortlists'
            # Qdrant filtering
            shortlist_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="shortlists.job_id",
                        match=models.MatchValue(value=st.session_state.active_job_id)
                    )
                ]
            )
            # Fetch them
            shortlisted_points, _ = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=shortlist_filter,
                limit=100,
                with_payload=True
            )
            st.session_state.search_results = shortlisted_points
            st.session_state.last_query = f"Shortlist for {st.session_state.active_job_title}"
            st.rerun()
        st.write("Click Refresh to see candidates saved to this job.")
    else:
        st.warning("Please select an Active Job from the sidebar to view its shortlist.")

# ... (Search Logic & Result Rendering) ...
# ... (Inside Result Loop) ...
            
            # --- ACTIONS & ANNOTATIONS ---
            with st.expander("📝 Review & Annotate", expanded=False):
                col_review1, col_review2 = st.columns([1, 1])
                
                with col_review1:
                    # Shortlist Button (If Job Active)
                    if st.session_state.active_job_id:
                        current_shortlists = payload.get("shortlists", [])
                        is_shortlisted = any(entry['job_id'] == st.session_state.active_job_id for entry in current_shortlists)
                        
                        if is_shortlisted:
                            st.success(f"✅ Shortlisted for {st.session_state.active_job_title}")
                        else:
                            if st.button(f"⭐ Add to {st.session_state.active_job_title}", key=f"shortlist_{point.id}"):
                                shortlist_candidate(point.id, st.session_state.active_job_id, st.session_state.active_job_title, current_shortlists)
                                st.rerun()
                    else:
                        st.caption("Select a Job in the sidebar to shortlist candidates.")

                    # Status Toggle (Existing)
                    # ...

```
