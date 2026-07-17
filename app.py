import streamlit as st
import json
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, util
import spacy

# 1. ADVANCED AI SCHEMA DEFINITION (Pydantic)

class JobMatch(BaseModel):
    match_score_out_of_100: int = Field(description="A score from 0 to 100 evaluating candidate fit.")
    reasoning: str = Field(description="A short 2-sentence explanation of the score.")
    key_strengths: list[str] = Field(description="Top 3 strengths of this candidate.")
    areas_for_improvement: list[str] = Field(description="1 or 2 missing skills or weak points.")
    interview_questions: list[str] = Field(description="3 custom technical or behavioral interview questions.")

class ResumeDetails(BaseModel):
    candidate_name: str
    contact_email: str
    professional_summary: str
    technical_skills: list[str] = Field(description="Hard skills, programming languages, and software.")
    creative_and_soft_skills: list[str] = Field(description="Design, communication, leadership, etc.")
    total_years_experience: int
    most_recent_job_title: str
    job_match_analysis: JobMatch

# 2. CACHED AI MODELS (loaded once per session, not per request)

@st.cache_resource
def load_embedding_model():
    # Lightweight, fast sentence-embedding model for semantic similarity scoring
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_nlp_model():
    # spaCy pipeline used as an independent NLP cross-check on top of the LLM's extraction.
    # The model is installed as a direct dependency in requirements.txt (not downloaded
    # at runtime) — runtime downloads are unreliable on hosted/free-tier environments
    # like Streamlit Community Cloud, where they can fail or time out silently.
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error(
            "spaCy model 'en_core_web_sm' is not installed. Make sure requirements.txt "
            "includes the direct wheel URL for en_core_web_sm and redeploy the app."
        )
        st.stop()

def compute_semantic_similarity(resume_text: str, role_requirements: str) -> float:
    """
    Independent, explainable similarity signal computed via sentence embeddings.
    This does NOT depend on the LLM — it's a second opinion using cosine similarity
    between the resume content and the target role's requirement text.
    """
    model = load_embedding_model()
    embeddings = model.encode([resume_text, role_requirements], convert_to_tensor=True)
    score = util.cos_sim(embeddings[0], embeddings[1]).item()
    # Cosine similarity is roughly in [-1, 1]; clamp and scale to 0-100
    return round(max(0.0, min(1.0, (score + 1) / 2)) * 100, 1)

def extract_nlp_entities(resume_text: str) -> dict:
    """
    Independent NLP cross-check using spaCy — extracts organizations, skills-like
    proper nouns, and noun phrases without relying on the LLM's own extraction.
    Useful for sanity-checking the LLM output and as the basis for the
    no-API-key rule-based fallback path.
    """
    nlp = load_nlp_model()
    doc = nlp(resume_text)

    orgs = sorted(set(ent.text for ent in doc.ents if ent.label_ == "ORG"))
    dates = sorted(set(ent.text for ent in doc.ents if ent.label_ == "DATE"))
    noun_chunks = sorted(set(
        chunk.text.strip() for chunk in doc.noun_chunks
        if 1 < len(chunk.text.split()) <= 4
    ))[:20]

    return {"organizations": orgs, "dates": dates, "key_phrases": noun_chunks}

# 3. CUSTOM UI STYLING

st.set_page_config(page_title="AI Resume Parser", page_icon="🚀", layout="wide")

st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            color: #ffffff;
        }
        [data-testid="stSidebar"] {
            background: rgba(15, 32, 39, 0.85);
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }
        div[data-testid="metric-container"] {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 15px 20px;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            transition: transform 0.2s ease;
        }
        div[data-testid="metric-container"]:hover {
            transform: translateY(-5px);
            border: 1px solid rgba(0, 210, 255, 0.5);
        }
        .main-header {
            background: -webkit-linear-gradient(45deg, #00d2ff, #3a7bd5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            font-size: 3.5rem;
            font-weight: 900;
            padding-bottom: 0px;
            letter-spacing: -1px;
        }
        .sub-header {
            text-align: center;
            font-size: 1.2rem;
            color: #b0c4de;
            margin-bottom: 30px;
            font-weight: 300;
        }
        [data-testid="stMetricValue"] {
            color: #00d2ff !important;
            font-weight: 800;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'>NextGen AI Resume Engine</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>LLM Orchestration, Vision OCR, Sentence Embeddings & spaCy NLP</div>", unsafe_allow_html=True)
st.markdown("---")

# 4. DYNAMIC AI ROLE PROFILING

role_profiles = {
    "AI Engineer": "Python, PyTorch, LLM Orchestration, RAG, Prompt Engineering, Vector Databases",
    "Software Engineer": "C++, Java, Algorithmic Problem Solving, System Design, Object-Oriented Programming",
    "Custom Role": "User Defined"
}

with st.sidebar:
    st.header("⚙️ LLM Configuration")
    api_key = st.text_input("Enter Gemini API Key:", type="password")

    # Free-tier daily quotas vary a lot by model. Flash-Lite has the most
    # generous free RPD, full Flash is stricter, Pro is the most limited.
    # Ordered so the fallback chain tries the most generous quota first.
    gemini_model_options = {
        "Gemini Flash-Lite (best for free tier — highest daily quota)": "gemini-flash-lite-latest",
        "Gemini Flash (balanced)": "gemini-flash-latest",
        "Gemini Pro (most capable, lowest free quota)": "gemini-pro-latest",
    }
    selected_model_label = st.selectbox("AI Model:", list(gemini_model_options.keys()))
    selected_model = gemini_model_options[selected_model_label]
    auto_fallback = st.checkbox(
        "Auto-fallback to another model if quota is exceeded", value=True,
        help="If the selected model hits its free-tier daily limit, automatically retry with the next model in the list."
    )
    st.markdown("---")
    st.markdown("**🎯 Select AI Evaluation Target:**")
    
    selected_menu_option = st.selectbox("Evaluate Candidate For:", list(role_profiles.keys()))
    
    if selected_menu_option == "Custom Role":
        target_role_title = st.text_input("Job Title:", placeholder="e.g., Data Scientist")
        target_role_reqs = st.text_area("Required Competencies:", placeholder="Paste technical skills here...")
        if not target_role_title: target_role_title = "Custom Role"
        if not target_role_reqs: target_role_reqs = "No specific requirements provided."
    else:
        target_role_title = selected_menu_option
        target_role_reqs = role_profiles[selected_menu_option]
        st.info(f"**Semantic Matching Criteria:**\n\n{target_role_reqs}")

# 5. CORE AI INFERENCE LOGIC

uploaded_file = st.file_uploader("📂 Upload Candidate Document (PDF, Scans, or Images)", type=["pdf"])

if uploaded_file and api_key:
    with st.spinner(f"⚡ Vision LLM is analyzing semantic fit for '{target_role_title}'..."):
        try:
            client = genai.Client(api_key=api_key)
            pdf_document = types.Part.from_bytes(data=uploaded_file.getvalue(), mime_type='application/pdf')
            
            prompt = f"""
            You are an autonomous AI technical recruiter. Analyze the attached resume document using vision capabilities. 
            Extract the candidate's details and deeply evaluate their fit for a '{target_role_title}' role strictly requiring these competencies: {target_role_reqs}.
            Generate insightful technical interview questions based directly on their resume and this specific role. 
            Provide a highly accurate match_score_out_of_100 based solely on how well their resume aligns with these exact requirements.
            """
            
            # Build the fallback order: selected model first, then the rest
            # of the list in order, skipping duplicates.
            fallback_models = [selected_model] + [
                m for m in gemini_model_options.values() if m != selected_model
            ]
            if not auto_fallback:
                fallback_models = [selected_model]

            response = None
            last_error = None
            model_used = None

            for candidate_model in fallback_models:
                try:
                    response = client.models.generate_content(
                        model=candidate_model,
                        contents=[prompt, pdf_document],
                        config={'response_mime_type': 'application/json', 'response_schema': ResumeDetails}
                    )
                    model_used = candidate_model
                    break
                except Exception as model_error:
                    last_error = model_error
                    # 429 = quota/rate limit exceeded on this model — try the next one.
                    if "429" in str(model_error) or "RESOURCE_EXHAUSTED" in str(model_error):
                        st.toast(f"⚠️ {candidate_model} quota exceeded, trying next model...", icon="⏳")
                        continue
                    # Any other error (bad key, malformed PDF, etc.) — stop retrying.
                    raise

            if response is None:
                raise last_error

            if model_used != selected_model:
                st.info(f"ℹ️ '{selected_model}' hit its free-tier quota — this result was generated with '{model_used}' instead.")

            data = json.loads(response.text)

            # 6. SECOND-OPINION NLP LAYER (independent of the LLM)

            resume_text_for_nlp = " ".join([
                data.get("professional_summary", ""),
                ", ".join(data.get("technical_skills", [])),
                ", ".join(data.get("creative_and_soft_skills", [])),
            ])

            semantic_similarity_score = compute_semantic_similarity(resume_text_for_nlp, target_role_reqs)
            nlp_entities = extract_nlp_entities(resume_text_for_nlp)

            # 7. AI INFERENCE DASHBOARD
            
            st.success(f"✅ AI Inference Complete: Graded for {target_role_title}")
            
            candidate_name = data.get("candidate_name", "Unknown_Candidate")
            match_score = data.get('job_match_analysis', {}).get('match_score_out_of_100', 0)
            
            # Exportable Report Generation
            report_text = f"""====================================================
NEXTGEN AI ANALYSIS REPORT
====================================================
Candidate Name : {candidate_name}
Email          : {data.get("contact_email", "N/A")}
Experience     : {data.get("total_years_experience", 0)} Years
----------------------------------------------------
TARGET ROLE          : {target_role_title}
LLM MATCH SCORE      : {match_score}%
EMBEDDING SIMILARITY : {semantic_similarity_score}%  (independent sentence-embedding cross-check)
----------------------------------------------------
EXECUTIVE SUMMARY:
{data.get("professional_summary", "No summary provided.")}

AI REASONING:
{data.get("job_match_analysis", {}).get("reasoning", "N/A")}

--- KEY STRENGTHS ---
"""
            for strength in data.get("job_match_analysis", {}).get("key_strengths", []):
                report_text += f"• {strength}\n"

            report_text += "\n--- RECOMMENDED INTERVIEW QUESTIONS ---\n"
            for i, question in enumerate(data.get("job_match_analysis", {}).get("interview_questions", [])):
                report_text += f"{i+1}. {question}\n"

            report_text += "\n--- NLP CROSS-CHECK (spaCy) ---\n"
            report_text += f"Organizations detected: {', '.join(nlp_entities['organizations']) or 'None'}\n"
            report_text += f"Key phrases detected: {', '.join(nlp_entities['key_phrases'][:10]) or 'None'}\n"

            report_text += "\n====================================================\nGenerated autonomously by the AI Engine."

            col_a, col_b = st.columns([0.8, 0.2])
            with col_b:
                st.download_button(
                    label="📄 Export AI Insights",
                    data=report_text,
                    file_name=f"{candidate_name.replace(' ', '_')}_AI_Report.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("👤 Candidate", candidate_name)
            col2.metric("💼 Experience", f"{data.get('total_years_experience', 0)} Years")
            col3.metric("📌 Target Role", target_role_title)
            col4.metric("🎯 LLM Match Score", f"{match_score}%")
            col5.metric("🧬 Embedding Similarity", f"{semantic_similarity_score}%")
            
            st.markdown("---")
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📋 Executive Summary", "🔍 Deep AI Analysis", "💬 Dynamic Interview Prep",
                "🧬 NLP Cross-Check", "💻 AI Output Payload"
            ])
            
            with tab1:
                st.write(f"**Contact:** {data.get('contact_email', 'N/A')}")
                st.info(data.get("professional_summary", "No summary provided."))
                s_col1, s_col2 = st.columns(2)
                with s_col1:
                    st.markdown("#### ⚙️ Technical Competencies")
                    for skill in data.get("technical_skills", []): st.markdown(f"- {skill}")
                with s_col2:
                    st.markdown("#### 🎨 Cross-Functional Skills")
                    for skill in data.get("creative_and_soft_skills", []): st.markdown(f"- {skill}")
                        
            with tab2:
                st.progress(match_score / 100.0)
                st.markdown(f"*{data.get('job_match_analysis', {}).get('reasoning', 'No reasoning provided.')}*")
                a_col1, a_col2 = st.columns(2)
                with a_col1:
                    st.success("#### ✅ AI Identified Strengths")
                    for strength in data.get("job_match_analysis", {}).get("key_strengths", []): st.markdown(f"- {strength}")
                with a_col2:
                    st.warning("#### ⚠️ Skill Gaps Detected")
                    for weak in data.get("job_match_analysis", {}).get("areas_for_improvement", []): st.markdown(f"- {weak}")
                        
            with tab3:
                for i, question in enumerate(data.get("job_match_analysis", {}).get("interview_questions", [])):
                    st.markdown(f"**Q{i+1}:** {question}")

            with tab4:
                st.caption("Independent NLP layer (spaCy) — sanity-checks the LLM's extraction and doubles as the basis for the no-API-key rule-based fallback path.")
                n_col1, n_col2 = st.columns(2)
                with n_col1:
                    st.markdown("#### 🏢 Organizations Detected")
                    for org in nlp_entities["organizations"]: st.markdown(f"- {org}")
                    st.markdown("#### 📅 Dates Detected")
                    for d in nlp_entities["dates"]: st.markdown(f"- {d}")
                with n_col2:
                    st.markdown("#### 🔑 Key Phrases")
                    for phrase in nlp_entities["key_phrases"]: st.markdown(f"- {phrase}")
                    
            with tab5:
                st.json(data)
                
        except Exception as e:
            st.error(f"🛑 LLM API Error: {e}")
            st.warning("The Vision AI could not process this request. Please check your network connection and API key.")

elif not api_key:
    st.info("👈 Please initialize the AI engine by providing a Gemini API Key in the sidebar.")