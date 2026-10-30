import os
import sqlite3
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Annotated, Literal

warnings.filterwarnings("ignore", message="The default value of `allowed_objects`")

from dotenv import load_dotenv
from tavily import TavilyClient
from langgraph.types import Command
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.graph import END, MessagesState, StateGraph, START
from langgraph.prebuilt import InjectedState, ToolNode
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from agentic_rag.core.QdrantVector_Store import retrieve_documents as VectorStore_search

load_dotenv()


# -------------------------------Structured-output models-------------------------------------------- #
class RouterDecision(BaseModel):
    route: Literal["retrieve", "verify_claim", "direct_answer"]


class RelevancyDecision(BaseModel):
    is_relevant: bool


class SupersedingPaper(BaseModel):
    title: str
    url: str
    summary: str


class ClaimVerificationResult(BaseModel):
    is_superseded: bool
    verdict_summary: str
    superseding_papers: list[SupersedingPaper]


# --------------------------------LLM's------------------------------------------- #
llm = ChatOpenAI(model="gpt-5.4-mini")
fast_llm = ChatOpenAI(model="gpt-5.4-mini")

router_llm = fast_llm.with_structured_output(RouterDecision)
relevancy_llm = fast_llm.with_structured_output(RelevancyDecision)
claim_llm = llm.with_structured_output(ClaimVerificationResult)


@lru_cache(maxsize=1)
def get_tavily() -> TavilyClient:
    return TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


MAX_REWRITES = 1
MAX_EVIDENCE_DOCS = 10
MAX_HISTORY_MESSAGES = 12  # messages sent to the agent; keeps token use flat


class RAGState(MessagesState):
    # Request info
    session_id: str
    query: str                      # the user's ORIGINAL question (never overwritten)
    search_query: str | None        # rewritten query used for retrieval
    selected_route: str | None

    # Retrieval
    retrieved_docs: list[Document]
    rewrite_count: int
    documents_relevant: bool | None

    # Claim verification
    verification_status: str | None
    verification_source: str | None
    related_papers: list[dict] | None

    # Output
    final_answer: str | None


def merge_docs(existing: list[Document] | None, new: list[Document]) -> list[Document]:
    """Append new docs, skipping exact-duplicate chunks."""
    merged = list(existing or [])
    seen = {d.page_content for d in merged}
    for d in new:
        if d.page_content not in seen:
            seen.add(d.page_content)
            merged.append(d)
    return merged


# ---------------------------------Router------------------------------------------ #
ROUTER_SYSTEM_PROMPT = """
<role>
You are a query classification assistant for a research paper question-answering system.
</role>

<task>
Classify the user's query into exactly one route:
retrieve, verify_claim, or direct_answer.
</task>

<routing_rules>
<retrieve>
Choose when the query requires information from uploaded research papers
or external sources, including current events, recent research, live prices,
weather, and other time-sensitive information.
</retrieve>

<verify_claim>
Choose when the user wants to check whether a specific claim or finding
from a research paper remains valid or has been challenged or superseded
by newer evidence.
</verify_claim>

<direct_answer>
Choose for stable general-knowledge questions that can be answered
without document retrieval or external searches.
</direct_answer>
</routing_rules>

<decision_rules>
Prioritize verify_claim when checking the validity of a research claim.
Otherwise, prefer retrieve over direct_answer whenever retrieval may be needed.
</decision_rules>

<output_constraints>
Return only the structured output required by RouterDecision.
Select exactly one route: retrieve, verify_claim, or direct_answer.
Do not answer the user's question.
</output_constraints>
"""

router_chain = ChatPromptTemplate.from_messages(
    [("system", ROUTER_SYSTEM_PROMPT), ("human", "{query}")]
) | router_llm


def router_node(state: RAGState) -> dict:
    query = state["messages"][-1].content
    decision = router_chain.invoke({"query": query})
    # The checkpointer persists state across turns in a thread, so every
    # per-turn field is reset here.
    return {
        "query": query,
        "search_query": query,
        "selected_route": decision.route,
        "retrieved_docs": [],
        "rewrite_count": 0,
        "documents_relevant": None,
        "verification_status": None,
        "verification_source": None,
        "related_papers": None,
        "final_answer": None,
    }


# ---------------------------------Tools------------------------------------------ #
@tool
def ResearchPaper_VectorStore(
    query: Annotated[str, Field(description="Question or topic to look for in the research paper collection")],
    session_id: Annotated[str, InjectedState("session_id")],
    existing_docs: Annotated[list, InjectedState("retrieved_docs")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Research paper vector store containing the uploaded papers.

    Use this tool for any question about the content of those papers: their
    methods, results, conclusions, datasets, authors, or specific claims.
    Do NOT use it for current events, recent developments, or anything that
    would not appear in the uploaded papers; use the web_search tool for those."""

    documents = VectorStore_search(query=query, session_id=session_id, k=4)

    if not documents:
        msg = "No relevant passages were found in the uploaded papers."
        return Command(update={"messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)]})

    msg = f"Found {len(documents)} relevant passage(s) in the research papers."
    return Command(
        update={
            "retrieved_docs": merge_docs(existing_docs, documents),
            "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)],
        }
    )


@tool
def web_search(
    query_optimized: Annotated[str, Field(description="Query rewritten and optimized for web search")],
    current_docs: Annotated[list, InjectedState("retrieved_docs")],
    tool_call_id: Annotated[str, InjectedToolCallId],
    max_results: Annotated[int, Field(ge=1, le=8, description="Maximum number of search results to return")] = 3,
) -> Command:
    """Web search. Use for current events, recent developments, or
    information that is not in the uploaded papers."""

    results = get_tavily().search(query_optimized, max_results=max_results).get("results", [])

    if not results:
        msg = "The online search did not return any results."
        return Command(update={"messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)]})

    documents = [
        Document(
            page_content=item["content"],
            metadata={"url": item["url"], "title": item.get("title") or "Untitled web page"},
        )
        for item in results
    ]
    msg = f"Added {len(documents)} online result(s) to the retrieved documents."
    return Command(
        update={
            "retrieved_docs": merge_docs(current_docs, documents),
            "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)],
        }
    )


TOOLS = [ResearchPaper_VectorStore, web_search]
llm_with_tools = llm.bind_tools(TOOLS, parallel_tool_calls=False)
tool_node = ToolNode(TOOLS)

RETRIEVE_SYSTEM_PROMPT = """
<role>
You are a context-gathering agent for a research paper Q&A system.
Your only job is to collect the information needed to answer the user's question.
</role>

<tools>
<ResearchPaper_VectorStore>
Searches the research papers uploaded in this session.
Parameters you control:
- query: a search phrase that matches how the papers would word the answer
</ResearchPaper_VectorStore>

<web_search>
Searches the live web.
Parameters you control:
- query_optimized: the question rewritten as a short, keyword-rich search query
- max_results: number of results to return (1-8)
</web_search>
</tools>

<tool_selection_rules>
- Questions about the content of the uploaded papers: use ResearchPaper_VectorStore.
- Questions about current events, recent developments, or information outside the papers: use web_search.
- Call exactly one tool, then stop.
</tool_selection_rules>

<output_constraints>
Do not write a final answer.
Only call a tool to gather context.
</output_constraints>
"""


def agent_node(state: RAGState) -> dict:
    """Select a retrieval tool using the conversation context."""
    history = []
    for message in state["messages"]:
        if isinstance(message, HumanMessage):
            history.append(message)
        elif isinstance(message, AIMessage) and not message.tool_calls and message.content:
            # Completed assistant answers are safe to include as context.
            history.append(message)

    history = history[-MAX_HISTORY_MESSAGES:]
    search_query = state.get("search_query") or state["query"]

     # Use the rewritten query without changing the stored conversation.
    if search_query != state["query"]:
        history.append(HumanMessage(content=f"Search for: {search_query}"))

     # Handle cases where no conversation history is available.
    if not history:
        history = [HumanMessage(content=search_query)]

    result = llm_with_tools.invoke(
        [{"role": "system", "content": RETRIEVE_SYSTEM_PROMPT}, *history]
    )
    return {"messages": [result]}


# ----------------------------------Relevancy Check----------------------------------------- #
RELEVANCY_CHECK_PROMPT = """
<role>
You evaluate whether retrieved document chunks are relevant to a user's question about research papers.
</role>

<task>
Determine whether the retrieved chunks contain useful information for answering the question.
</task>

<evaluation_rules>
- Return is_relevant=true if the chunks meaningfully address the question, even partially.
- Return is_relevant=false if the chunks are clearly unrelated or contain no useful information.
- Be lenient when evaluating relevance. Any substantive connection to the question is sufficient to return true.
</evaluation_rules>

<output_constraints>
Return the structured output required by RelevancyDecision.
</output_constraints>
"""


def relevancy_check_node(state: RAGState) -> dict:
    documents = state.get("retrieved_docs") or []
    if not documents:  # no LLM call needed
        return {"documents_relevant": False}

    context = "\n\n---\n\n".join(d.page_content[:400] for d in documents[-4:])
    result = relevancy_llm.invoke([
        {"role": "system", "content": RELEVANCY_CHECK_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question: {state['query']}\n\n"
                f"Retrieved chunks:\n{context}\n\n"
                "Are the retrieved chunks relevant to answering the question?"
            ),
        },
    ])
    return {"documents_relevant": result.is_relevant}


# -------------------------------Query rewrite-------------------------------------------- #

QUERY_REWRITE_PROMPT = """
<role>
You are a query rewriting assistant for a research paper retrieval system.
</role>

<task>
Improve the user's query when the previous search failed to find relevant document chunks.
</task>

<guidelines>
- Use more specific terminology and relevant keywords.
- Include alternative terms related to the research topic.
- Narrow the query to a particular concept or question when appropriate.
- Preserve the original meaning and intent.
</guidelines>

<output_constraints>
Return only the rewritten query as plain text.
Do not include explanations or introductory text.
</output_constraints>
"""


def query_rewrite_node(state: RAGState) -> dict:
    response = fast_llm.invoke([
        {"role": "system", "content": QUERY_REWRITE_PROMPT},
        {
            "role": "user",
            "content": f"Original query: {state['query']}\n\nRewrite it as a better search query.",
        },
    ])
    new_query = response.content.strip()

    return {
        # `query` stays as the user's original question for the final answer.
        "search_query": new_query,
        "retrieved_docs": [],
        "documents_relevant": None,
        "rewrite_count": state.get("rewrite_count", 0) + 1,
    }


# ------------------------------Claim Verification--------------------------------------------- #
CLAIM_ANALYSIS_PROMPT = """
<role>
You are a literature-review assistant who checks whether a research finding
is still current.
</role>

<task>
You will receive one claim taken from a research paper, plus search results
from the web and from arXiv. Decide whether newer work has overturned,
seriously challenged, or refined the claim.
</task>

<evidence_rules>
- Use only the titles and URLs that appear exactly as written in the search results.
- Never invent, guess, or alter a title or link.
- If an arXiv link and a general web link cover the same work, choose the arXiv link.
- Select at most 3 papers.
</evidence_rules>

<output_rules>
- For each selected paper, write one sentence explaining how it affects the claim.
- If the claim still stands, set is_superseded to false and leave superseding_papers empty.
- Write verdict_summary as 1-2 plain sentences that can be shown directly to the user.
</output_rules>
"""


def _tavily_search(kwargs: dict) -> list[dict]:
    return get_tavily().search(**kwargs).get("results", [])


def verify_claim_node(state: RAGState) -> dict:
    claim = state["query"]
    snippet = claim[:200]

    searches = [
        {"query": f"recent studies challenging or updating: {snippet}", "max_results": 5},
        {"query": snippet, "max_results": 5, "include_domains": ["arxiv.org"]},
    ]

    # The two searches are independent, so run them concurrently.
    with ThreadPoolExecutor(max_workers=2) as pool:
        general, arxiv = pool.map(_tavily_search, searches)

    # Drop web results that duplicate an arXiv hit (saves prompt tokens).
    arxiv_urls = {r.get("url") for r in arxiv}
    general = [r for r in general if r.get("url") not in arxiv_urls]

    sections = []
    for title, results in (
        ("General Research and Web Results", general),
        ("arXiv Research Papers", arxiv),
    ):
        sections.append(f"=== {title} ===")
        sections.extend(
            f"Title: {r.get('title', '')}\nURL: {r.get('url', '')}\nContent: {r.get('content', '')[:300]}"
            for r in results
        )
    search_context = "\n\n".join(sections)

    prompt = f"""
{CLAIM_ANALYSIS_PROMPT}

<claim_to_verify>
{claim}
</claim_to_verify>

<search_evidence>
{search_context}
</search_evidence>
"""
    response = claim_llm.invoke([{"role": "user", "content": prompt}])
    papers = [p.model_dump() for p in response.superseding_papers[:3]]

    return {
        "verification_status": response.verdict_summary,
        "verification_source": papers[0].get("url") if papers else None,
        "related_papers": papers,
    }


# -------------------------------Answer Generation-------------------------------- #
ANSWER_SYSTEM_PROMPT = (
    "<role>You answer using only the evidence provided.</role>\n"
    "<rules>\n"
    "- Base every statement on the passages in <evidence>.\n"
    "- If the evidence does not answer the question, say so plainly. Do not guess.\n"
    "- Cite the passage number, like [2], after each claim.\n"
    "</rules>"
)


def _format_evidence(docs: list[Document]) -> str:
    lines = []
    for i, d in enumerate(docs[:MAX_EVIDENCE_DOCS], 1):
        src = d.metadata.get("title") or d.metadata.get("source") or ""
        header = f"[{i}] ({src})" if src else f"[{i}]"
        lines.append(f"{header} {d.page_content}")
    return "\n".join(lines)


def generate_answer_node(state: RAGState) -> dict:
    route = state.get("selected_route")
    question = state["query"]

    if route == "retrieve":
        docs = state.get("retrieved_docs") or []
        if not docs or state.get("documents_relevant") is False:
            answer = (
                "I couldn't find anything relevant to that question. "
                "Try rephrasing it or uploading more papers."
            )
        else:
            answer = llm.invoke([
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"<evidence>\n{_format_evidence(docs)}\n</evidence>\n\n"
                        f"<question>{question}</question>"
                    ),
                },
            ]).content

    elif route == "verify_claim":
        papers = state.get("related_papers") or []
        lines = [
            "## Claim check",
            f"Claim: {question}",
            f"Verdict: {state.get('verification_status') or ''}",
        ]
        if papers:
            lines.append("\nNewer work:")
            lines += [f"- {p['title']}: {p['summary']} ({p['url']})" for p in papers]
        else:
            lines.append("\nNo newer work that challenges this claim was found.")
        answer = "\n".join(lines)

    else:  # direct_answer
        answer = llm.invoke([{"role": "user", "content": question}]).content

    return {"final_answer": answer, "messages": [AIMessage(content=answer)]}


# ----------------------------Edges----------------------------------------------- #
def route_by_intent(state: RAGState) -> str:
    return state.get("selected_route") or "direct_answer"


def after_agent(state: RAGState) -> str:
    last = state["messages"][-1]
    # A pending tool call must run first, otherwise the saved history would
    # contain a tool call with no matching ToolMessage.
    if getattr(last, "tool_calls", None):
        return "run_tools"
    return "judge_docs"


def after_relevancy(state: RAGState) -> str:
    if state.get("documents_relevant"):
        return "answer"
    if state.get("rewrite_count", 0) < MAX_REWRITES:
        return "rewrite"
    return "answer"


def build_graph(db_path: str = "checkpoints.db"):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")      # faster checkpoint writes
    conn.execute("PRAGMA synchronous=NORMAL")

    builder = StateGraph(RAGState)

    builder.add_node("router", router_node)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)
    builder.add_node("relevancy_check", relevancy_check_node)
    builder.add_node("query_rewrite", query_rewrite_node)
    builder.add_node("verify_claim", verify_claim_node)
    builder.add_node("generate_answer", generate_answer_node)

    builder.add_edge(START, "router")

    builder.add_conditional_edges(
        "router",
        route_by_intent,
        {
            "retrieve": "agent",
            "verify_claim": "verify_claim",
            "direct_answer": "generate_answer",
        },
    )
    builder.add_conditional_edges(
        "agent",
        after_agent,
        {"run_tools": "tools", "judge_docs": "relevancy_check"},
    )
    builder.add_edge("tools", "relevancy_check")
    builder.add_conditional_edges(
        "relevancy_check",
        after_relevancy,
        {"rewrite": "query_rewrite", "answer": "generate_answer"},
    )
    builder.add_edge("query_rewrite", "agent")
    builder.add_edge("verify_claim", "generate_answer")
    builder.add_edge("generate_answer", END)

    return builder.compile(checkpointer=SqliteSaver(conn))