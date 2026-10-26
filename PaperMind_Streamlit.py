import json
import os
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agentic_rag.core.QdrantVector_Store import add_research_paper, list_research_papers
from agentic_rag.core.Agentic_rag1 import build_graph
from agentic_rag.core.rag_loaders import load_arxiv, load_document

st.set_page_config(page_title="PaperMind", page_icon="🧠", layout="wide")

DATA_DIR = ROOT / ".papermind"
DATA_DIR.mkdir(exist_ok=True)
CHAT_FILE = DATA_DIR / "chats.json"


def read_chats() -> dict:
    """Load locally persisted chat display data; graph state lives in SQLite."""
    try:
        data = json.loads(CHAT_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def persist_chats() -> None:
    """Write to a temporary file first to avoid a partially written JSON file."""
    pending = CHAT_FILE.with_suffix(".tmp")
    pending.write_text(json.dumps(st.session_state.chats, indent=2), encoding="utf-8")
    os.replace(pending, CHAT_FILE)


def create_chat() -> str:
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {
        "title": "New research chat",
        "created": datetime.now(timezone.utc).isoformat(),
        "messages": [],
    }
    st.session_state.active_chat = chat_id
    persist_chats()
    return chat_id


@st.cache_resource(show_spinner="Initializing PaperMind...")
def graph():
    return build_graph(str(ROOT / "checkpoints.db"))


def ingest_documents(docs: list, chat_id: str) -> int:
    if not docs:
        raise ValueError("The source contains no extractable text.")
    add_research_paper(docs, chat_id)
    return len(docs)


def upload_file(file, chat_id: str) -> int:
    """Pass a real path to the existing loader, then restore display metadata."""
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=Path(file.name).suffix.lower(), delete=False
        ) as temporary:
            temporary.write(file.getvalue())
            temporary_path = Path(temporary.name)
        docs = load_document(str(temporary_path))
        for doc in docs:
            doc.metadata["title"] = Path(file.name).stem
            doc.metadata["source"] = file.name
        return ingest_documents(docs, chat_id)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def ask_papermind(question: str, chat_id: str) -> tuple[str, dict]:
    """Reuse a stable LangGraph thread for each research chat."""
    config = {"configurable": {"thread_id": chat_id}}
    result = graph().invoke(
        {"messages": [HumanMessage(content=question)], "session_id": chat_id},
        config=config,
    )
    answer = result.get("final_answer") or "No answer was generated."
    details = {
        "route": result.get("selected_route"),
        "retrieved_passages": len(result.get("retrieved_docs") or []),
        "query_rewrites": result.get("rewrite_count", 0),
        "relevant": result.get("documents_relevant"),
    }
    return answer, details


if "chats" not in st.session_state:
    st.session_state.chats = read_chats()
if "active_chat" not in st.session_state:
    if st.session_state.chats:
        st.session_state.active_chat = max(
            st.session_state.chats,
            key=lambda cid: st.session_state.chats[cid].get("created", ""),
        )
    else:
        create_chat()

chat_id = st.session_state.active_chat

with st.sidebar:
    st.header("📚 PaperMind")
    if st.button("＋ New research chat", use_container_width=True):
        create_chat()
        st.rerun()

    st.subheader("Conversations")
    for cid, info in sorted(
        st.session_state.chats.items(),
        key=lambda entry: entry[1].get("created", ""),
        reverse=True,
    ):
        if st.button(
            info.get("title", "Research chat"),
            key=f"chat_{cid}",
            type="primary" if cid == chat_id else "secondary",
            use_container_width=True,
        ):
            st.session_state.active_chat = cid
            st.rerun()

    st.divider()
    st.subheader("Add research material")
    with st.expander("Upload files", expanded=True):
        files = st.file_uploader(
            "PDF, TXT, or Markdown",
            type=["pdf", "txt", "md", "markdown"],
            accept_multiple_files=True,
            key=f"upload_{chat_id}",
        )
        if st.button("Index selected files", key=f"index_{chat_id}", use_container_width=True):
            if not files:
                st.warning("Choose at least one file.")
            else:
                for file in files:
                    try:
                        with st.spinner(f"Indexing {file.name}..."):
                            count = upload_file(file, chat_id)
                        st.success(f"{file.name}: {count} chunks indexed")
                    except Exception as exc:
                        st.error(f"{file.name}: {exc}")

    with st.expander("Web page or PDF link"):
        url = st.text_input("Public URL", key=f"url_{chat_id}", placeholder="https://example.com/article")
        if st.button("Index URL", key=f"index_url_{chat_id}", use_container_width=True):
            if not url.strip():
                st.warning("Enter a URL.")
            else:
                try:
                    with st.spinner("Loading source..."):
                        count = ingest_documents(load_document(url.strip()), chat_id)
                    st.success(f"Indexed {count} chunks")
                except Exception as exc:
                    st.error(f"Could not load URL: {exc}")

    with st.expander("Find an arXiv paper"):
        arxiv_query = st.text_input(
            "Paper title, ID, or arXiv link",
            key=f"arxiv_{chat_id}",
            placeholder="1706.03762",
        )
        if st.button("Index arXiv paper", key=f"index_arxiv_{chat_id}", use_container_width=True):
            if not arxiv_query.strip():
                st.warning("Enter an arXiv title, ID, or link.")
            else:
                try:
                    with st.spinner("Downloading paper..."):
                        count = ingest_documents(load_arxiv(arxiv_query.strip()), chat_id)
                    st.success(f"Indexed {count} chunks")
                except Exception as exc:
                    st.error(f"Could not load arXiv paper: {exc}")

    st.divider()
    st.subheader("Papers in this chat")
    try:
        titles = list_research_papers(chat_id)
        if titles:
            for title in titles:
                st.caption(f"📄 {title}")
        else:
            st.caption("No indexed papers yet.")
    except Exception as exc:
        st.warning(f"Unable to list papers: {exc}")

st.title("PaperMind")
st.caption("Research paper Q&A · Live web research · Claim verification")

current_chat = st.session_state.chats[chat_id]
if not current_chat["messages"]:
    st.info("Upload research material in the sidebar, or ask a question to begin.")

for message in current_chat["messages"]:
    with st.chat_message(
        message["role"],
        avatar=":material/person:" if message["role"] == "user" else ":material/psychology:",
    ):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("details"):
            with st.expander("How this answer was produced"):
                st.json(message["details"])

if question := st.chat_input("Ask about a paper, recent research, or a claim..."):
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(question)

    with st.chat_message("assistant", avatar=":material/psychology:"):
        try:
            with st.spinner("Researching..."):
                answer, details = ask_papermind(question, chat_id)
            st.markdown(answer)
            with st.expander("How this answer was produced"):
                st.json(details)
        except Exception as exc:
            st.error(f"PaperMind could not complete the request: {exc}")
        else:
            current_chat["messages"].extend(
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer, "details": details},
                ]
            )
            if current_chat["title"] == "New research chat":
                current_chat["title"] = " ".join(question.split()[:6])[:60]
            persist_chats()
            st.rerun()
