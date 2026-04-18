"""
Multi-Modal Chatbot — Extended Internship Project
==================================================
Builds on the training project (End-To-End-Gemini-Project + customer_service_chatbot_LLM)
and adds:
  1. Text Q&A  (Gemini 1.5 Flash — replaces deprecated gemini-pro)
  2. Image Understanding (Gemini 1.5 Flash vision — replaces deprecated gemini-pro-vision)
  3. Customer Service RAG (Google Palm + FAISS + CSV knowledge base)
  4. [Extra] Chat History Export
  5. [Extra] Sentiment Analysis of user queries (TextBlob)
  6. [Extra] Auto-suggest follow-up questions via Gemini
"""

import os
import io
import csv
import datetime
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

# ── Load environment ──────────────────────────────────────────────────────────
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# ── Conditional imports (graceful degradation if packages missing) ─────────────
try:
    import google.generativeai as genai
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
    GEMINI_AVAILABLE = bool(GOOGLE_API_KEY)
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.llms import GooglePalm
    from langchain_community.document_loaders.csv_loader import CSVLoader
    from langchain_community.embeddings import HuggingFaceInstructEmbeddings
    from langchain.prompts import PromptTemplate
    from langchain.chains import RetrievalQA
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MultiModal AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

:root {
    --bg: #0d0f14;
    --surface: #161b22;
    --border: #30363d;
    --accent: #58a6ff;
    --accent2: #3fb950;
    --warn: #f0883e;
    --text: #e6edf3;
    --muted: #8b949e;
}

html, body, .stApp { background: var(--bg); color: var(--text); font-family: 'DM Sans', sans-serif; }

h1, h2, h3 { font-family: 'Space Mono', monospace; }

.stSidebar { background: var(--surface) !important; border-right: 1px solid var(--border); }

.chat-bubble-user {
    background: linear-gradient(135deg, #1f4068 0%, #1b262c 100%);
    border: 1px solid var(--accent);
    border-radius: 12px 12px 2px 12px;
    padding: 12px 16px;
    margin: 8px 0;
    max-width: 80%;
    margin-left: auto;
}

.chat-bubble-bot {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px 12px 12px 2px;
    padding: 12px 16px;
    margin: 8px 0;
    max-width: 80%;
}

.sentiment-positive { color: var(--accent2); font-size: 0.8em; }
.sentiment-negative { color: #f85149; font-size: 0.8em; }
.sentiment-neutral  { color: var(--muted); font-size: 0.8em; }

.mode-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75em;
    font-family: 'Space Mono', monospace;
    margin-bottom: 8px;
}
.badge-text  { background: #1f4068; color: var(--accent); border: 1px solid var(--accent); }
.badge-vision { background: #2d1b69; color: #d2a8ff; border: 1px solid #d2a8ff; }
.badge-cs    { background: #1a2f1a; color: var(--accent2); border: 1px solid var(--accent2); }
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode" not in st.session_state:
    st.session_state.mode = "Text Q&A"

# ── Helper: Sentiment Analysis ────────────────────────────────────────────────
def analyze_sentiment(text: str) -> tuple[str, float]:
    """Returns (label, polarity) for a text string."""
    if not TEXTBLOB_AVAILABLE:
        return "neutral", 0.0
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return "positive", polarity
    elif polarity < -0.1:
        return "negative", polarity
    return "neutral", polarity

# ── Helper: Gemini Text Q&A ───────────────────────────────────────────────────
@st.cache_resource
def get_text_chat():
    """Returns a Gemini chat session (cached)."""
    if not GEMINI_AVAILABLE:
        return None
    model = genai.GenerativeModel("gemini-1.5-flash")
    return model.start_chat(history=[])

def gemini_text_response(question: str) -> str:
    chat = get_text_chat()
    if chat is None:
        return "⚠️ Gemini API not configured. Add your GOOGLE_API_KEY to .env"
    response = chat.send_message(question, stream=False)
    return response.text

# ── Helper: Gemini Vision ─────────────────────────────────────────────────────
def gemini_vision_response(prompt: str, image: Image.Image) -> str:
    if not GEMINI_AVAILABLE:
        return "⚠️ Gemini API not configured."
    model = genai.GenerativeModel("gemini-1.5-flash")
    contents = [prompt, image] if prompt.strip() else [image]
    response = model.generate_content(contents)
    return response.text

# ── Helper: Auto Follow-up Questions ─────────────────────────────────────────
def suggest_followups(question: str, answer: str) -> list[str]:
    if not GEMINI_AVAILABLE:
        return []
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            f"Given this Q&A:\nQ: {question}\nA: {answer[:300]}\n\n"
            "Suggest exactly 3 short follow-up questions the user might ask next. "
            "Return only the questions, one per line, no numbering."
        )
        resp = model.generate_content(prompt)
        lines = [l.strip() for l in resp.text.strip().split("\n") if l.strip()]
        return lines[:3]
    except Exception:
        return []

# ── Helper: Customer Service RAG ─────────────────────────────────────────────
VECTORDB_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "faiss_index")
DATASET_PATH  = os.path.join(os.path.dirname(__file__), "..", "dataset", "dataset.csv")

@st.cache_resource
def load_rag_chain():
    if not LANGCHAIN_AVAILABLE or not GOOGLE_API_KEY:
        return None
    try:
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
        if os.path.exists(VECTORDB_PATH):
            vectordb = FAISS.load_local(VECTORDB_PATH, embeddings, allow_dangerous_deserialization=True)
        else:
            loader = CSVLoader(file_path=DATASET_PATH, source_column="prompt")
            data = loader.load()
            vectordb = FAISS.from_documents(documents=data, embedding=embeddings)
            vectordb.save_local(VECTORDB_PATH)

        llm = GooglePalm(google_api_key=GOOGLE_API_KEY, temperature=0.1)
        retriever = vectordb.as_retriever(score_threshold=0.7)
        prompt_template = """Given the following context, answer the question.
If the answer is not in the context, say "I don't know."

CONTEXT: {context}
QUESTION: {question}"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=retriever,
            input_key="query", return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT},
        )
        return chain
    except Exception as e:
        st.warning(f"RAG chain error: {e}")
        return None

def cs_response(question: str) -> str:
    chain = load_rag_chain()
    if chain is None:
        return "⚠️ Customer Service RAG unavailable. Ensure LangChain + API key are configured."
    result = chain({"query": question})
    return result.get("result", "No answer found.")

# ── Helper: Export Chat ───────────────────────────────────────────────────────
def export_chat_csv(messages: list) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["role", "content", "mode", "timestamp"])
    for m in messages:
        writer.writerow([m.get("role"), m.get("content", ""), m.get("mode", ""), m.get("ts", "")])
    return buf.getvalue().encode()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 MultiModal AI\n*Internship Project*")
    st.divider()

    st.markdown("### Mode")
    mode = st.radio(
        "Select chatbot mode:",
        ["Text Q&A", "Image Understanding", "Customer Service RAG"],
        index=["Text Q&A", "Image Understanding", "Customer Service RAG"].index(st.session_state.mode),
        label_visibility="collapsed",
    )
    st.session_state.mode = mode

    st.divider()
    st.markdown("### API Status")
    st.markdown(f"Gemini: {'✅' if GEMINI_AVAILABLE else '❌ Add GOOGLE_API_KEY'}")
    st.markdown(f"LangChain RAG: {'✅' if LANGCHAIN_AVAILABLE else '⚠️ Optional'}")
    st.markdown(f"Sentiment: {'✅' if TEXTBLOB_AVAILABLE else '⚠️ Optional'}")

    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    if st.session_state.messages:
        csv_bytes = export_chat_csv(st.session_state.messages)
        st.download_button(
            "⬇️ Export Chat (CSV)",
            data=csv_bytes,
            file_name=f"chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

    st.divider()
    st.caption("Built on: Gemini 1.5 Flash · Google Palm · FAISS · LangChain")

# ── Main Header ───────────────────────────────────────────────────────────────
badge_map = {
    "Text Q&A":               ("badge-text",   "💬 Text Q&A — Gemini 1.5 Flash"),
    "Image Understanding":    ("badge-vision", "🖼️ Vision — Gemini 1.5 Flash"),
    "Customer Service RAG":   ("badge-cs",     "🏢 Customer Service — Palm + FAISS RAG"),
}
badge_cls, badge_label = badge_map[mode]

st.markdown(f"# 🤖 MultiModal AI Chatbot")
st.markdown(f'<span class="mode-badge {badge_cls}">{badge_label}</span>', unsafe_allow_html=True)

if mode == "Text Q&A":
    st.caption("Ask anything — powered by Gemini 1.5 Flash with multi-turn conversation memory.")
elif mode == "Image Understanding":
    st.caption("Upload an image and ask questions about it. Gemini will analyse and explain it.")
else:
    st.caption("Customer FAQ chatbot powered by Google Palm + FAISS vector retrieval (RAG).")

st.divider()

# ── Chat History Display ──────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        sentiment, polarity = analyze_sentiment(msg.get("content", ""))
        sem_class = f"sentiment-{sentiment}"
        sem_icon = {"positive": "😊", "negative": "😟", "neutral": "😐"}[sentiment]
        st.markdown(
            f'<div class="chat-bubble-user">'
            f'<strong>You</strong><br>{msg["content"]}'
            f'<br><span class="{sem_class}">{sem_icon} {sentiment} ({polarity:+.2f})</span>'
            f'</div>', unsafe_allow_html=True
        )
        if msg.get("image_caption"):
            st.caption(f"📎 {msg['image_caption']}")
    else:
        st.markdown(
            f'<div class="chat-bubble-bot">'
            f'<strong>🤖 Assistant</strong><br>{msg["content"]}'
            f'</div>', unsafe_allow_html=True
        )
        if msg.get("followups"):
            st.markdown("**💡 Follow-up suggestions:**")
            cols = st.columns(len(msg["followups"]))
            for i, fq in enumerate(msg["followups"]):
                if cols[i].button(fq, key=f"fq_{msg.get('ts','')}_{i}"):
                    # Inject follow-up as new user message
                    st.session_state._pending_followup = fq
                    st.rerun()

# ── Handle pending follow-up ──────────────────────────────────────────────────
pending = getattr(st.session_state, "_pending_followup", None)

# ── Input Area ────────────────────────────────────────────────────────────────
uploaded_image = None
if mode == "Image Understanding":
    uploaded_image = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_image:
        img = Image.open(uploaded_image)
        st.image(img, caption="Uploaded image", use_container_width=False, width=400)

col1, col2 = st.columns([5, 1])
with col1:
    user_input = st.text_input(
        "Message",
        value=pending or "",
        placeholder={
            "Text Q&A": "Ask me anything…",
            "Image Understanding": "Describe what you want to know about the image…",
            "Customer Service RAG": "Ask a customer service question…",
        }[mode],
        label_visibility="collapsed",
        key="user_input_field",
    )
with col2:
    send = st.button("Send ➤", use_container_width=True)

if pending:
    st.session_state._pending_followup = None

# ── Process Input ─────────────────────────────────────────────────────────────
if (send or pending) and (user_input or pending):
    query = pending or user_input
    ts = datetime.datetime.now().strftime("%H:%M:%S")

    # Record user message
    user_msg = {"role": "user", "content": query, "mode": mode, "ts": ts}

    # Get bot response
    if mode == "Text Q&A":
        answer = gemini_text_response(query)
        followups = suggest_followups(query, answer)
        st.session_state.messages.append(user_msg)
        st.session_state.messages.append({
            "role": "assistant", "content": answer,
            "mode": mode, "ts": ts, "followups": followups
        })

    elif mode == "Image Understanding":
        if uploaded_image is None:
            st.warning("Please upload an image first.")
        else:
            img = Image.open(uploaded_image)
            answer = gemini_vision_response(query, img)
            user_msg["image_caption"] = uploaded_image.name
            followups = suggest_followups(query, answer)
            st.session_state.messages.append(user_msg)
            st.session_state.messages.append({
                "role": "assistant", "content": answer,
                "mode": mode, "ts": ts, "followups": followups
            })

    else:  # Customer Service RAG
        answer = cs_response(query)
        st.session_state.messages.append(user_msg)
        st.session_state.messages.append({
            "role": "assistant", "content": answer, "mode": mode, "ts": ts
        })

    st.rerun()
