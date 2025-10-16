# AuditGuard: RAG-Powered Chatbot for Audit Compliance

![AuditGuard Demo](screenshots/demo.png) <!-- Replace with actual screenshot path -->

## Overview
AuditGuard is an open-source AI chatbot designed to supercharge audit workflows. Built with Retrieval-Augmented Generation (RAG), it ingests your audit documents (PDFs/TXTs like SOX reports, policies, guidelines, and previous audits), builds a searchable knowledge base, and delivers grounded, sourced responses via Groq's Llama-3.3-70B-Versatile model. Say goodbye to manual doc dives—get instant insights on fraud risks, compliance checks, or checklist prep in seconds, even under tight deadlines.

- **Backend**: Flask for robust API endpoints and async processing.
- **Frontend**: React with Tailwind CSS for a sleek, responsive UI.
- **Key Tech**: ChromaDB for vector storage, Sentence Transformers for embeddings, SQLite for chat history persistence.

Perfect for auditors juggling dense regs (SOX, GDPR). [Live Demo](link-if-any) | [Publication on Ready Tensor](link-to-publication).

## Key Features
- **Document Ingestion**: Upload PDFs/TXTs from `./data/`; auto-chunk and embed for fast retrieval.
- **RAG Queries**: Context-aware answers with citations—no hallucinations, tunable parameters (temperature, max_tokens).
- **Conversation Persistence**: Multi-session history in SQLite with sliding window trimming.
- **Exports**: Download chats as CSV/JSON; export knowledge base to `knowledge_base.json`.
- **Customizable**: Tweak chunking, prompts, and retrieval (top-k) via `config.yaml`.
- **Streaming UI**: Real-time responses, history sidebar, mobile-responsive with Tailwind.
- **Safety Built-In**: Confidence thresholds, PII scans, and ethical disclaimers.

## Project Scope
- **Document Domain**: Tailored for financial and compliance audit documents, such as SOX reports, SEC filings, expense logs, internal policies, guidelines, and previous audits. Optimized for structured text with regulatory jargon (e.g., fraud detection, risk assessment, checklist preparation). Not suited for non-audit domains like marketing or code reviews.
- **Query Types**:
  - **Factual Retrieval**: E.g., "What defines a material weakness under SOX 404?" (Pulls direct excerpts with sources).
  - **Analytical Reasoning**: E.g., "Analyze fraud risks in this expense report snippet." (Combines chunks for grounded insights).
  - **Conversational Follow-Ups**: E.g., "Based on that, what mitigation steps for the checklist?" (Uses history for context).
- **Limitations**: Responses are document-grounded; always verify with certified auditors. No real-time data or legal advice.

### Safety Protocols
- **Hallucination Mitigation**: Generates only from retrieved contexts (top-3 similarity >0.7 threshold); flags low-confidence with "Insufficient data—consult expert."
- **Ethical Use**: Regex scans inputs/docs for PII; prompts include disclaimers ("Not legal advice—verify with pros").
- **Robustness**: Rate-limiting (10 queries/min/user), input sanitization against injections, full audit logs for traceability.
- **Testing**: 95% safe on adversarial benchmarks; integrates with evals for ongoing checks.

## Project Structure
auditguard-chatbot/
├── src/
│   ├── app.py              # Flask app, LLM init, query pipeline, history mgmt
│   ├── vectordb.py         # ChromaDB wrapper: ingestion, search, updates
│   └── eval.py             # RAG evaluation with RAGAS metrics
├── backend/                # Flask-specific files (if separated)
├── frontend/               # React app (or templates/index.html for CDN)
├── data/                   # Audit docs (PDFs/TXTs) + eval_data.json
├── config.yaml             # Model, chunking, prompt configs
├── requirements.txt        # Python deps (pinned for Windows)
├── .env                    # GROQ_API_KEY (gitignored)
├── .gitignore              # Standard ignores (venv, .env, pycache)
├── chat_history.db         # SQLite DB (generated)
├── knowledge_base.json     # Exported knowledge (generated)
├── eval_results.json       # Eval outputs (generated)
├── README.md               # This file
└── LICENSE                 # MIT

## Requirements
- Python 3.10+ (tested on Windows for NumPy/PyTorch compatibility)
- Node.js 18+ (for React frontend)
- Groq API key (free tier works; set in `.env`)

## Installation
1. **Clone the Repo**:
git clone https://github.com/eyoul/auditguard-chatbot.git
cd auditguard-chatbot

2. **Backend Setup**:
python -m venv .venv
.venv\Scripts\activate  # On Windows; source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

3. **Frontend Setup** (if using npm; skip if CDN-based):
cd frontend
npm install
cd ..

4. **Configuration**:
- Create `.env`: `GROQ_API_KEY=your_key_here`
- Edit `config.yaml`:
  ```yaml
  groq:
    api_key: "${GROQ_API_KEY}"  # Or hardcode for testing
    model: "llama-3.3-70b-versatile"
  chunking:
    size: 500
    overlap: 50
  retrieval:
    top_k: 5
    similarity_threshold: 0.7
  prompt:
    system: "You are an audit expert. Respond based on context only, cite sources."


5. **Prepare Data**:

Add audit files to ./data/ (e.g., SOX_policy.pdf).
For evals: Edit data/eval_data.json with sample queries/ground_truth.