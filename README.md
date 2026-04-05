# 📄 PaperLens — AI Research Assistant (RAG App)

PaperLens is an AI-powered Research Assistant built using Retrieval-Augmented Generation (RAG). It allows you to upload academic papers (PDF or ArXiv URL) and ask questions, get summaries, comparisons, critiques, and more — all grounded strictly in the uploaded documents.

---

## Features

- **PDF Upload & ArXiv URL Support** — Load documents from local files or directly from ArXiv
- **RAG Pipeline** — Retrieves relevant chunks before generating answers
- **Multi-turn Conversation** — Maintains chat history for context-aware follow-up questions
- **Streaming Responses** — Answers stream word by word for faster perceived response
- **Persistent Vector Store** — Embeddings are saved and reused to avoid recomputation
- **RAG Evaluation** — Integrated RAGAS framework for pipeline evaluation
- **Streamlit UI** — Clean interface for PDF upload and Q&A

---

## Tech Stack

| Component | Tool |
|-----------|------|
| LLM | `Qwen/Qwen3-Coder-Next` via HuggingFace Endpoint |
| Embeddings | HuggingFace Embeddings |
| Vector Store | FAISS |
| Framework | LangChain |
| Evaluation | RAGAS |
| UI | Streamlit |

---

## Workflow

```
 Upload PDF / ArXiv URL
        ↓
 Load Documents          ← runs once on file load
        ↓
  Chunk Documents         ← runs once on file load
        ↓
 Create & Save           ← runs once, reused on next load
   Vector Store
        ↓
 User asks a Question
        ↓
 Retrieve Relevant Chunks
        ↓
 Create Prompt
   (context + chat history)
        ↓
 LLM Generates Answer
   (streamed)
        ↓
 Update Chat History
        ↓
 Next Question...
```

---

## Auto-Detect Modes

PaperLens automatically detects the type of query and responds accordingly:

| Trigger | Mode |
|---------|------|
| General question | Q&A mode |
| "summarize" | Problem / Contributions / Method / Results / Limitations |
| "compare" | Side-by-side comparison table |
| "extract" | Structured list (datasets, metrics, models, etc.) |
| "critique" | Gaps, reproducibility, unsupported claims |
| "literature review" | Synthesize themes across all papers |

---

## Improvements Log

| # | Improvement | Commit |
|---|-------------|--------|
| 1 | Replaced direct text input with document loader | `feat: integrate document loader to replace hardcoded text in RAG app` |
| 2 | Converted random code into functions | `refactor: convert components into functions` |
| 3 | Moved chunking & vector store creation to one-time initialization, added multi-turn conversation | `feat: add multi-turn conversation with one-time initialization` |
| 4 | Enabled streaming for faster perceived LLM response | `feat: enable streaming for faster LLM response` |
| 5 | Increased batch size and persist vector store to reduce embedding overhead | `feat: increase batch size and persist vector store to reduce embedding overhead` |
| 6 | Integrated RAGAS framework for RAG evaluation | `feat: integrate RAGAS framework for RAG pipeline evaluation` |
| 7 | Added Streamlit UI for PDF upload and Q&A | `feat: add Streamlit UI for PDF upload and Q&A` |
| 8 | Added support for ArXiv URL as input source | `feat: add ArXiv URL support for document ingestion` |

---

## Getting Started
To use PaperLens:
1. Clone the repository to your local machine.
```bash
git clone https://github.com/TarunSehgal27/PaperLens.git
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Set your HuggingFace API key.
```bash
export HUGGINGFACEHUB_API_TOKEN=your_token_here
```

4. Run the application.
```bash
streamlit run main.py
```

---

## RAG Evaluation (RAGAS)

The pipeline is evaluated using the RAGAS framework with the following metrics:

- **Faithfulness** — Is the answer grounded in the context?
- **Answer Relevancy** — Is the answer relevant to the question?
- **Context Precision** — Are retrieved chunks relevant?
- **Context Recall** — Are all relevant chunks retrieved?

---

## Project Structure

```
paperlens/
│
├── main.py            # Streamlit UI
├── app.py             # Core RAG functions
│   ├── load_documents()
│   ├── get_chunks()
│   ├── create_vectorstore()
│   ├── retrieve()
│   ├── create_prompt()
│   ├── get_result()
│   └── ask()
├── evaluation.py           # RAGAS evaluation
├── vector_store/           # Saved vector store
└── requirements.txt
```

---

## Response Format

Every answer follows this structure:

```
Answer: <direct answer grounded in context>
Source: <bullet points with inline citations>
Confidence: High | Medium | Low
```

---

*Built with LangChain, HuggingFace, and RAGAS*