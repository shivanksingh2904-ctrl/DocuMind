"""DocuMind - ask questions about your PDFs, with numbered source citations.
Run locally:  streamlit run app.py
"""
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from pdf_processor import process_pdf  # noqa: E402
from rag import Library, stream_answer, summarize  # noqa: E402

st.set_page_config(page_title="DocuMind", page_icon="🧠", layout="wide")
st.markdown("""<style>
.block-container{padding-top:2rem;max-width:1100px}
.dm-title{font-size:2.1rem;font-weight:700;letter-spacing:-.5px;margin:0}
.dm-sub{opacity:.7;margin-bottom:1rem}
</style>""", unsafe_allow_html=True)

# Each visitor gets a private library (safe for shared hosting).
if "lib" not in st.session_state:
    st.session_state.lib = Library()
    st.session_state.messages = []
lib: Library = st.session_state.lib


def pages_label(h):
    return f"p.{h['page']}" if h["page"] == h["page_end"] else f"pp.{h['page']}-{h['page_end']}"


def show_sources(hits):
    for n, h in enumerate(hits, 1):
        with st.expander(f"[{n}] {h['doc']} · {pages_label(h)} · relevance {h['relevance']:.0%}"):
            st.write(h["text"])


def ingest(items):
    bar = st.progress(0.0, text="Reading and embedding...")
    for n, (name, path) in enumerate(items, 1):
        try:
            chunks, info = process_pdf(path, name)
            if chunks:
                lib.add(chunks, info, name)
            else:
                st.warning(f"{name}: no extractable text (scanned PDF?).")
        except Exception as e:
            st.error(f"{name}: {e}")
        bar.progress(n / len(items))
    bar.empty()


# ------------------------------ sidebar ------------------------------
with st.sidebar:
    st.markdown("## 🧠 DocuMind")
    provider = st.selectbox("Answer engine", ["Claude", "Ollama", "Extractive (no AI)"])
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if provider == "Claude" and not api_key:
        api_key = st.text_input("Anthropic API key", type="password", key="api_key_input",
                                help="Kept only in your browser session.")
    if provider == "Claude":
        st.caption("Retrieved passages are sent to the Claude API.")

    st.markdown("### Retrieval")
    mode = st.radio("Search mode", ["Hybrid", "Semantic", "Keyword"], horizontal=True)
    top_k = st.slider("Passages to use", 2, 10, 5)
    diversity = st.slider("Diversity", 0.0, 0.8, 0.25, 0.05,
                          help="Higher = avoid near-duplicate passages.")
    scope = st.multiselect("Limit to documents", list(lib.docs))

    st.markdown("### Session")
    c1, c2 = st.columns(2)
    if c1.button("💾 Save"):
        lib.save()
        st.toast("Saved to vector_store/")
    if c2.button("📂 Load"):
        st.toast("Loaded." if lib.load() else "Nothing saved yet.")
        st.rerun()
    if st.button("🧹 Clear everything"):
        lib.clear()
        st.session_state.messages = []
        st.rerun()
    if st.session_state.messages:
        md = "\n\n".join(f"**{m['role'].title()}:** {m['content']}" for m in st.session_state.messages)
        st.download_button("⬇️ Export chat (.md)", md, "documind_chat.md")

st.markdown('<p class="dm-title">DocuMind</p><p class="dm-sub">Ask your documents anything. '
            'Every answer points back to its source.</p>', unsafe_allow_html=True)

tab_chat, tab_lib = st.tabs(["💬 Chat", "📚 Library"])
prompt = st.chat_input("Ask a question about your PDFs...")

# ------------------------------ library ------------------------------
with tab_lib:
    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    a, b = st.columns(2)
    if files and a.button("Index uploads", type="primary"):
        items = []
        for f in files:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp.write(f.getbuffer())
            tmp.close()
            items.append((f.name, tmp.name))
        ingest(items)
        for _, p in items:
            os.remove(p)
        st.rerun()
    local = [f for f in os.listdir("documents") if f.lower().endswith(".pdf")] if os.path.isdir("documents") else []
    if local and b.button(f"Index documents/ folder ({len(local)})"):
        ingest([(f, os.path.join("documents", f)) for f in local])
        st.rerun()

    if lib.docs:
        st.markdown("#### Loaded documents")
        for name, info in list(lib.docs.items()):
            n_chunks = sum(1 for c in lib.chunks if c["doc"] == name)
            r1, r2, r3 = st.columns([5, 2, 1])
            r1.markdown(f"**{name}**  \n{info['pages']} pages · {info['words']:,} words · {n_chunks} passages")
            if r2.button("Summarize", key=f"sum_{name}"):
                if provider == "Claude" and not api_key:
                    st.error("Add your Anthropic API key in the sidebar.")
                else:
                    with st.spinner("Reading..."):
                        try:
                            st.session_state[f"summary_{name}"] = summarize(lib, name, provider, api_key)
                        except Exception as e:
                            st.error(f"{provider} error: {e}")
            if r3.button("✕", key=f"del_{name}"):
                lib.remove(name)
                st.rerun()
            if f"summary_{name}" in st.session_state:
                st.info(st.session_state[f"summary_{name}"])
    else:
        st.caption("No documents yet. Upload a PDF to begin.")

# ------------------------------ chat ------------------------------
with tab_chat:
    if not lib.chunks:
        st.info("Add PDFs in the **Library** tab to get started.")
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            show_sources(m.get("sources", []))

    if prompt and lib.chunks:
        if provider == "Claude" and not api_key:
            st.error("Add your Anthropic API key in the sidebar, or pick another engine.")
            st.stop()
        with st.chat_message("user"):
            st.markdown(prompt)
        hits = lib.search(prompt, k=top_k, mode=mode, diversity=diversity, docs=scope or None)
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        with st.chat_message("assistant"):
            try:
                answer = st.write_stream(stream_answer(prompt, hits, provider, api_key, history))
            except Exception as e:
                answer = f"⚠️ {provider} error: {e}"
                st.error(answer)
            show_sources(hits)
        st.session_state.messages += [{"role": "user", "content": prompt},
                                      {"role": "assistant", "content": answer, "sources": hits}]
