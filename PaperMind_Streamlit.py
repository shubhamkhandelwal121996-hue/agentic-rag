import json
import os
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessageChunk, HumanMessage

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agentic_rag.core.QdrantVector_Store import add_research_paper, list_research_papers

try:
    from agentic_rag.core.QdrantVector_Store import delete_session
except ImportError:
    # Add delete_session(session_id) to your Qdrant module to also remove a
    # deleted chat's indexed papers. Until then they stay in Qdrant.
    delete_session = None

from agentic_rag.core.Agentic_rag1 import build_graph
from agentic_rag.core.rag_loaders import load_arxiv, load_document

# "centered" keeps the chat column and the input box aligned and readable.
st.set_page_config(page_title="🧠PaperMind", page_icon="🧠", layout="centered")

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
    :root {
        --pm-accent: #4F46E5;
        --pm-border: rgba(255, 255, 255, 0.09);
        color-scheme: dark;
    }

    /* Theme: dark navy page, slightly lighter sidebar, transparent top bar.
       Built for Streamlit's Dark theme (menu > Settings > Theme). */
    .stApp { background: #0B1020; color: #E5E7EB; }
    [data-testid="stSidebar"] { background: #131A2E; }
    [data-testid="stHeader"] { background: transparent; }

    /* Hide the "Deploy" button; the ⋮ menu stays. */
    [data-testid="stAppDeployButton"], .stDeployButton { display: none !important; }

    /* Slightly wider than Streamlit's default centered column. The input box
       gets the same width so the two stay aligned. */
    [data-testid="stMainBlockContainer"],
    [data-testid="stBottomBlockContainer"] {
        max-width: 880px !important;
    }
    [data-testid="stMainBlockContainer"] { padding-top: 2.5rem; }

    /* Title */
    .pm-title {
        font-size: 2.1rem; font-weight: 800; line-height: 1.2;
        letter-spacing: -0.02em; margin: 0 0 0.2rem 0;
    }
    .pm-gradient {
        background: linear-gradient(90deg, #818CF8, #C084FC, #F472B6);
        -webkit-background-clip: text; background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .pm-tagline {
        color: rgba(250, 250, 250, 0.6); font-size: 0.95rem; margin-bottom: 1.5rem;
    }

    /* Chat bubbles: rounded cards, with the user's messages tinted */
    [data-testid="stChatMessage"] {
        border: 1px solid var(--pm-border);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.9rem;
        background: rgba(255, 255, 255, 0.02);
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: rgba(79, 70, 229, 0.10);
        border-color: rgba(99, 102, 241, 0.30);
    }

    /* Rounded buttons in the sidebar */
    [data-testid="stSidebar"] .stButton > button { border-radius: 10px; }

    /* Active chat (Streamlit "primary" buttons): indigo instead of default red.
       Both selectors are listed because Streamlit renamed the attribute
       between versions. */
    button[kind="primary"],
    button[data-testid="stBaseButton-primary"] {
        background-color: #4F46E5; border-color: #4F46E5; color: #FFFFFF;
    }
    button[kind="primary"]:hover,
    button[data-testid="stBaseButton-primary"]:hover {
        background-color: #4338CA; border-color: #4338CA; color: #FFFFFF;
    }
    button[kind="primary"]:focus:not(:active),
    button[data-testid="stBaseButton-primary"]:focus:not(:active) {
        border-color: #6366F1; box-shadow: 0 0 0 0.2rem rgba(99, 102, 241, 0.5);
        color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_DIR = ROOT / ".papermind"
DATA_DIR.mkdir(exist_ok=True)
CHAT_FILE = DATA_DIR / "chats.json"

DEFAULT_TITLE = "New Chat"
LEGACY_TITLE = "New research chat"  # title used by chats saved before the rename

# Shown as one-click starters on an empty chat.
EXAMPLE_PROMPTS = [
    ("📝 Summarize", "Summarize the paper in 50 words"),
    ("🔬 Key results", "What are the key contributions and results of this paper?"),
    ("🔎 Still current?", "Has the main claim of this paper been superseded by newer research?"),
]

# Progress messages, one per graph node (reported after the node finishes).
STEP_LABELS = {
    "router": "Understood your question",
    "agent": "Chose a search tool",
    "tools": "Searched your sources",
    "relevancy_check": "Checked how relevant the results are",
    "query_rewrite": "Rewrote the query and searched again",
    "verify_claim": "Checked for newer research",
    "generate_answer": "Wrote the answer",
}

# -----------------------------------Chat storage---------------------------------------- #
def read_chats() -> dict:
    """Load locally persisted chat display data; graph state lives in SQLite."""
    try:
        data = json.loads(CHAT_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}

    chats = data if isinstance(data, dict) else {}

    # Old saved chats still carry the previous default title; migrate them so
    # the sidebar and the auto-rename on first question both use the new name.
    for info in chats.values():
        if isinstance(info, dict) and info.get("title") == LEGACY_TITLE:
            info["title"] = DEFAULT_TITLE
    return chats


def persist_chats() -> None:
    """Write to a temporary file first to avoid a partially written JSON file."""
    pending = CHAT_FILE.with_suffix(".tmp")
    pending.write_text(json.dumps(st.session_state.chats, indent=2), encoding="utf-8")
    os.replace(pending, CHAT_FILE)


def create_chat() -> str:
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {
        "title": DEFAULT_TITLE,
        "created": datetime.now(timezone.utc).isoformat(),
        "messages": [],
    }
    st.session_state.active_chat = chat_id
    persist_chats()
    return chat_id


@st.cache_resource(show_spinner="Initializing PaperMind...")
def graph():
    return build_graph(str(ROOT / "checkpoints.db"))


def rename_chat(chat_to_rename: str, new_title: str) -> None:
    title = " ".join(new_title.split())[:60]
    if title and chat_to_rename in st.session_state.chats:
        st.session_state.chats[chat_to_rename]["title"] = title
        persist_chats()


def delete_chat(chat_to_delete: str) -> None:
    """Remove a chat's indexed papers, LangGraph history and saved messages."""
    problems = []

    if delete_session is not None:
        try:
            delete_session(chat_to_delete)
        except Exception as exc:
            problems.append(f"indexed papers could not be removed ({exc})")

    try:
        graph().checkpointer.delete_thread(chat_to_delete)
    except Exception as exc:
        problems.append(f"conversation memory could not be cleared ({exc})")

    st.session_state.chats.pop(chat_to_delete, None)

    if st.session_state.active_chat == chat_to_delete:
        remaining = st.session_state.chats
        if remaining:
            st.session_state.active_chat = max(
                remaining, key=lambda cid: remaining[cid].get("created", "")
            )
        else:
            create_chat()  # never leave the app without an active chat

    persist_chats()

    if problems:
        st.session_state.notice = "Chat deleted, but " + "; ".join(problems) + "."


# ------------------------------------Ingestion and question answering--------------------------------------- #
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


def _chunk_text(chunk) -> str:
    """Text of one streamed chunk (some models send a list of content parts)."""
    content = chunk.content
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict))


def stream_answer(compiled_graph, question: str, chat_id: str, on_step=None, on_first_token=None):
    """Run the graph and yield the answer text as the model writes it.

    Two stream modes are read together:
      "updates"  -> one event per finished node (drives the progress box)
      "messages" -> LLM tokens; only those from the answer node are yielded,
                    so the router, agent and relevancy check never leak in.
    """
    config = {"configurable": {"thread_id": chat_id}}
    inputs = {"messages": [HumanMessage(content=question)], "session_id": chat_id}

    started = False
    for mode, data in compiled_graph.stream(
        inputs, config=config, stream_mode=["updates", "messages"]
    ):
        if mode == "updates":
            if on_step is not None:
                for node in data:
                    on_step(STEP_LABELS.get(node, node))

        elif mode == "messages":
            chunk, metadata = data
            # LangGraph also re-emits the finished AIMessage the node returns;
            # only real streamed chunks (AIMessageChunk) are shown, so the
            # answer is never written twice.
            if not isinstance(chunk, AIMessageChunk):
                continue
            if metadata.get("langgraph_node") != "generate_answer":
                continue

            text = _chunk_text(chunk)
            if text:
                if not started:
                    started = True
                    if on_first_token is not None:
                        on_first_token()
                yield text


def get_final_answer(chat_id: str) -> str:
    """The saved answer for the chat's latest turn, read from the checkpoint."""
    config = {"configurable": {"thread_id": chat_id}}
    return graph().get_state(config).values.get("final_answer") or ""


# ------------------------------------UI helpers--------------------------------------- #
def welcome_card(has_papers: bool) -> None:
    """Empty-chat starter: one-click example questions."""
    with st.container(border=True):
        st.markdown("### Ask your papers anything")
        if has_papers:
            st.caption("Your papers are indexed. Try one of these, or type your own question below.")
        else:
            st.caption("Add a paper from the sidebar first, then try one of these.")

        columns = st.columns(len(EXAMPLE_PROMPTS))
        for column, (label, prompt) in zip(columns, EXAMPLE_PROMPTS):
            if column.button(
                label,
                key=f"example_{label}",
                help=prompt,
                disabled=not has_papers,
                use_container_width=True,
            ):
                st.session_state.pending_question = prompt
                st.rerun()


# ----------------------------------Session setup----------------------------------------- #
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

# ---------------------------------Sidebar------------------------------------------ #
with st.sidebar:
    st.header("🧠 PaperMind")
    if st.button("＋ New Chat", use_container_width=True):
        create_chat()
        st.rerun()

    st.subheader("Conversations")

    notice = st.session_state.pop("notice", None)
    if notice:
        st.warning(notice)

    for cid, info in sorted(
        st.session_state.chats.items(),
        key=lambda entry: entry[1].get("created", ""),
        reverse=True,
    ):
        title = info.get("title", DEFAULT_TITLE)
        select_col, menu_col = st.columns([5, 1], gap="small")

        if select_col.button(
            title,
            key=f"chat_{cid}",
            type="primary" if cid == chat_id else "secondary",
            use_container_width=True,
        ):
            st.session_state.active_chat = cid
            st.rerun()

        # Opening this menu and then clicking Delete is the confirmation step.
        with menu_col.popover("⋯", use_container_width=True):
            new_title = st.text_input(
                "Rename chat",
                value=title,
                max_chars=60,
                # Title in the key so the box refreshes when the chat is auto-named.
                key=f"rename_{cid}_{title}",
            )
            if st.button("Save name", key=f"save_{cid}", use_container_width=True):
                rename_chat(cid, new_title)
                st.rerun()

            st.divider()
            st.caption("Deleting also removes this chat's messages and indexed papers.")
            if st.button("🗑️ Delete chat", key=f"delete_{cid}", use_container_width=True):
                delete_chat(cid)
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
    has_papers = False
    try:
        titles = list_research_papers(chat_id)
        has_papers = bool(titles)
        if titles:
            for title in titles:
                st.caption(f"📄 {title}")
        else:
            st.caption("No indexed papers yet.")
    except Exception as exc:
        st.warning(f"Unable to list papers: {exc}")

# --------------------------------------------------------------------------- #
# Main area
# --------------------------------------------------------------------------- #
st.markdown(
    '<div class="pm-title">🧠 <span class="pm-gradient">PaperMind</span></div>'
    '<div class="pm-tagline">Research paper Q&amp;A · Live web research · Claim verification</div>',
    unsafe_allow_html=True,
)

current_chat = st.session_state.chats[chat_id]

# The input is read up front so the welcome card can hide as soon as a question
# arrives (typed, or clicked from an example). The box itself stays pinned to
# the bottom of the page wherever it is called.
typed_question = st.chat_input("Ask about a paper, recent research, or a claim...")
question = st.session_state.pop("pending_question", None) or typed_question

if not current_chat["messages"] and not question:
    welcome_card(has_papers)

for message in current_chat["messages"]:
    with st.chat_message(
        message["role"],
        avatar=":material/person:" if message["role"] == "user" else ":material/psychology:",
    ):
        st.markdown(message["content"])

if question:
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(question)

    with st.chat_message("assistant", avatar=":material/psychology:"):
        status = st.status("Researching…", expanded=True)

        def collapse_status() -> None:
            status.update(label="Done", state="complete", expanded=False)

        try:
            # Steps tick off in the status box; once the answer starts
            # streaming, the box collapses so the text takes the spotlight.
            streamed = st.write_stream(
                stream_answer(
                    graph(),
                    question,
                    chat_id,
                    on_step=lambda label: status.write(f"✓ {label}"),
                    on_first_token=collapse_status,
                )
            )
            was_streamed = isinstance(streamed, str) and bool(streamed.strip())

            # The saved answer is the source of truth for the chat history.
            answer = get_final_answer(chat_id) or (streamed if was_streamed else "")
            answer = answer or "No answer was generated."

            # Claim checks and "nothing relevant found" replies are built in
            # code rather than written by the model, so nothing streamed.
            if not was_streamed:
                st.markdown(answer)
            collapse_status()
        except Exception as exc:
            status.update(label="Something went wrong", state="error", expanded=False)
            st.error(f"PaperMind could not complete the request: {exc}")
        else:
            current_chat["messages"].extend(
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
            )
            if current_chat["title"] == DEFAULT_TITLE:
                current_chat["title"] = " ".join(question.split()[:6])[:60]
            persist_chats()
            st.rerun()