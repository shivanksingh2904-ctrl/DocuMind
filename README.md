# 🧠 DocuMind

Ask questions about your PDFs and get answers with **numbered citations** that point to the exact page.

## Features
- **Hybrid search**: dense semantic vectors + BM25 keywords, merged with Reciprocal Rank Fusion
- **Diversity re-ranking (MMR)** so answers don't rely on near-duplicate passages
- **Sentence-aware chunking** that never cuts a sentence and tracks page ranges
- **One-click document summaries** with suggested questions
- **Private per-session library**, so visitors never see each other's files
- Answer engines: Claude, local Ollama, or a no-AI extractive mode
- Save/load the index, export the chat as Markdown

## How it works
```
PDF -> pypdf -> sentence-aware chunks (+ page range)
     -> BGE-small embeddings (ONNX)  +  BM25 index
question -> dense ranking + keyword ranking -> RRF fusion -> MMR -> top-k passages
         -> LLM answers using only those passages, citing [1], [2]...
```

## Project layout
| File | Role |
|---|---|
| `app.py` | Streamlit interface |
| `rag.py` | BM25, embeddings, RRF, MMR, LLM calls |
| `pdf_processor.py` | PDF text extraction and chunking |
| `documents/` | Optional folder of PDFs to index locally |
| `vector_store/` | Saved index |

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # add ANTHROPIC_API_KEY
streamlit run app.py
```

## Deploy on Hugging Face Spaces
1. Create a new Space, choose **Docker** (blank), then push this repo to it.
2. Optional: add `ANTHROPIC_API_KEY` under Settings, then Secrets. Without it, each visitor enters their own key.

## Limitations
Scanned (image-only) PDFs have no extractable text; OCR is not included.

## License
MIT
