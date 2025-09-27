# Audit Agentic AI – RAG Assistant

> Production-ready RAG assistant with Flask backend, React front-end (vanilla + CDN), Groq LLM, Chroma vector DB, and SQLite chat memory. Configurable via `config.yaml`. Supports knowledge export to JSON.

## Key Features

- **Document ingestion** from `data/` (PDF and TXT)
- **Configurable chunking** via `config.yaml` (`chunk_size`, `chunk_overlap`)
- **Vector search** using Chroma with SentenceTransformer embeddings
- **RAG querying** powered by Groq LLM (`ChatGroq`)
- **Chat memory** persisted to SQLite (`chat_history.db`)
- **Knowledge export** endpoint to `knowledge_base.json`
- **Front-end UX**: history sidebar, sliding window trimming, auto-scroll, session persistence, Markdown rendering, Tailwind UI

## Project Structure

```
Audit agentic AI/
├── src/
│   ├── app.py           # RAG app (Groq, memory, config, export)
│   └── vectordb.py      # Chroma wrapper (chunking, search, export)
├── data/                # PDF/TXT files (add your documents here)
├── templates/
│   └── index.html       # React UI (CDN, history sidebar, markdown)
├── app.py               # Flask API and static serving
├── config.yaml          # Model, chunking, prompt config
├── requirements.txt     # Python deps (Windows-friendly pins)
├── .env                 # GROQ_API_KEY, optional GROQ_MODEL
├── chat_history.db      # SQLite database
├── knowledge_base.json  # Exported knowledge base
```

## Requirements

- Python 3.10.x (recommended on Windows for PyTorch + NumPy compatibility)
- A Groq API key: set `GROQ_API_KEY` in `.env`

## Setup

1. Create and activate a virtual environment (Python 3.10):
```bash
py -3.10 -m venv .venv
.\.venv\Scripts\activate
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Configure environment and model:
```env
# .env
GROQ_API_KEY=your_groq_key
# Optional – if set, ensure it matches config.yaml
GROQ_MODEL=llama-3.3-70b-versatile
```
4. Adjust settings in `config.yaml` (model, chunking, prompt):
```yaml
model:
  name: "llama-3.3-70b-versatile"
  temperature: 0.7
  max_tokens: 4096
vector_db:
  collection_name: "rag_collection"
  chunk_size: 500
  chunk_overlap: 50
prompt:
  system: |
    You are a helpful AI assistant specialized in providing accurate and concise answers based on the provided context.
```
5. Add documents to `data/` (PDF or TXT)

## Run

```bash
python app.py
```
Open http://localhost:5000

## API Endpoints
- `GET /` – serves `templates/index.html`
- `POST /api/query`
  - body: `{ question, user_id?, session_id?, n_results? }`
  - returns: `{ answer, context, metadata, session_id }`
- `POST /api/history`
  - body: `{ user_id, session_id }`
  - returns: `{ history: [{ role, content }] }`
- `GET /api/export_knowledge`
  - exports PDFs and vector DB contents to `knowledge_base.json`

## Front-end Highlights (`templates/index.html`)

- **History sidebar** lists recent Q/A pairs (sliding window)
- **Session persistence** via `localStorage` (session_id reuse)
- **Auto-scroll** to latest message
- **Markdown rendering** with `marked.js` + `DOMPurify`
- **Tailwind**-styled layout; CDN is fine for dev (use CLI/PostCSS in prod)

## Notes for Windows

- The project pins versions to avoid NumPy 2/PyTorch issues on Python 3.10.
- `uvloop` is disabled on Windows via environment markers.

## Exporting Knowledge

- Hit `GET /api/export_knowledge` to write a consolidated `knowledge_base.json` containing:
  - `pdfs`: parsed text and metadata from `data/`
  - `vectordb`: all chunks + metadata + ids from Chroma

## Troubleshooting
- If the page is blank, check browser Console for JSX/JS errors and ensure the CDN scripts (React, ReactDOM, Babel, marked, DOMPurify) load (status 200).
- If server crashes at startup, verify `.env` contains a valid `GROQ_API_KEY`.
- If you see NumPy/Torch issues, ensure you are using Python 3.10 in the venv.

---

## License

This project is licensed. See the [LICENSE](LICENSE) file for details.
