# 🤖 Multi-Modal AI Chatbot

**Internship Project | Domain: Generative AI**  
*Built on the ElevanceSkills training project — extended with multi-modal capabilities*

---

## Problem Statement

Extend the training chatbot (Gemini Q&A + Customer Service RAG) into a **unified multi-modal chatbot** that handles both text and image inputs, integrates retrieval-augmented generation, and adds NLP-powered extras.

---

## What Was Built

### Foundation (Training Project)
| File | Purpose |
|---|---|
| `QA.py` | Gemini Pro text Q&A |
| `img_model.py` | Gemini Pro Vision image understanding |
| `langchain_helper.py` + `main.py` | Google Palm + FAISS customer service RAG |

### Extensions (Internship)
| Feature | Description |
|---|---|
| **Unified App** | Single Streamlit app combining all 3 modes |
| **Model Upgrade** | `gemini-pro` → `gemini-1.5-flash` (multimodal) |
| **Multi-turn Memory** | Chat sessions persist context across turns |
| **Sentiment Analysis** | TextBlob polarity classification per query |
| **Auto Follow-ups** | Gemini suggests 3 relevant follow-up questions |
| **Chat Export** | Download full conversation history as CSV |

---

## System Architecture

```
User Input (Text / Image)
        │
        ▼
   Mode Router (Streamlit)
   ┌────┬────┬─────────────────┐
   │    │    │                 │
   ▼    ▼    ▼                 │
Text  Vision  Customer Svc      │
Q&A   Gemini  Palm+FAISS RAG   │
   │    │    │                 │
   └────┴────┘                 │
        │                      │
        ▼                      │
  Post-Processing               │
  (Sentiment + Follow-ups)      │
        │                      │
        ▼                      │
   Response + Chat History ◄───┘
```

---

## Dataset

- **Source:** ElevanceSkills customer service FAQ CSV (training project)
- **Size:** 76 Q&A pairs
- **Avg prompt length:** 85 characters
- **Avg response length:** 318 characters
- **Sentiment:** 68% Neutral, 20% Positive, 12% Negative

---

## Methodology

1. **Data Exploration** — Analysed FAQ dataset with sentiment scoring and length statistics
2. **Text Q&A** — Multi-turn Gemini 1.5 Flash chat with session history
3. **Vision** — Gemini 1.5 Flash processes uploaded images with optional text prompts
4. **RAG Pipeline** — FAISS vector index + Google Palm for grounded customer service answers
5. **Sentiment** — TextBlob polarity on every user query (shown in UI)
6. **Follow-up Gen** — Gemini generates 3 contextual follow-up suggestions post-response
7. **Export** — Chat log (role, content, mode, timestamp) downloadable as CSV

---

## Model Comparison

| Aspect | Baseline | Extended |
|---|---|---|
| Text model | `gemini-pro` | `gemini-1.5-flash` |
| Vision model | `gemini-pro-vision` | `gemini-1.5-flash` |
| Chat memory | Single-turn | Multi-turn history |
| Sentiment | ✗ | ✓ TextBlob |
| Follow-ups | ✗ | ✓ Gemini-generated |
| Export | ✗ | ✓ CSV |
| UI | 3 separate apps | 1 unified app |

---

## Visualisations

- `assets/eda_plots.png` — Sentiment distribution, prompt/response lengths
- `assets/architecture.png` — System architecture diagram
- `notebooks/multimodal_chatbot.ipynb` — Full walkthrough with interactive Plotly charts

---

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API key
echo "GOOGLE_API_KEY=your_key_here" > .env

# 3. Launch app
streamlit run src/app.py
```

**Requirements:**
- Python 3.10+
- Google API key (Gemini + Palm access)
- 4GB RAM (for instructor-large embeddings)

---

## Project Structure

```
multimodal_chatbot/
├── src/
│   └── app.py                      # Unified Streamlit chatbot
├── dataset/
│   ├── dataset.csv                 # FAQ knowledge base
│   └── faiss_index/                # Auto-generated vector store
├── notebooks/
│   └── multimodal_chatbot.ipynb    # Full walkthrough notebook
├── assets/
│   ├── eda_plots.png
│   ├── architecture.png
│   └── chat_export.csv
├── requirements.txt
└── README.md
```

---

## Contact

For queries: training@elevanceskills.com  
Include: Name · Domain (Generative AI) · GitHub Link
