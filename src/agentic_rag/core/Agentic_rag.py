import os
import sqlite3
import warnings
from typing import Annotated

warnings.filterwarnings("ignore", message="The default value of `allowed_objects`")

from dotenv import load_dotenv
from tavily import TavilyClient
from langgraph.types import Command
from pydantic import BaseModel,Field
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.graph import END, MessagesState, StateGraph,START
from langgraph.prebuilt import InjectedState, ToolNode, tools_condition
from agentic_rag.core.vector_store import search as vs_search
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from agentic_rag.core.models import ClaimVerificationResult, RelevancyDecision, RouterDecision

load_dotenv()

llm = ChatOpenAI(model="gpt-5.4-mini")

class RAGState(MessagesState):
    # Basic information about the current request
    session_id: str
    query: str
    selected_route: str | None

    # Information collected during retrieval
    retrieved_docs: list[Document]
    retrieval_count: int
    rewrite_count: int
    documents_relevant: bool | None

    # Results from checking research claims
    verification_status: str | None
    verification_source: str | None
    related_papers: list[dict] | None

    # Response generated for the user
    final_answer: str | None

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

ROUTER_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system",ROUTER_SYSTEM_PROMPT),
    ("human","{query}")
])

def router_node(state: RAGState) -> dict:
    query = state["messages"][-1].content
    routing_decision = (ROUTER_PROMPT_TEMPLATE | llm.with_structured_output(RouterDecision)).invoke({"query": query})
    return {"selected_route": routing_decision.route}

class PaperSearchInput(BaseModel):
    query: str = Field(description="Question or topic to look for in the research paper collection")
    k: int = Field(default=3,ge=1,le=8,description="Maximum number of relevant text chunks to return")

class WebSearchInput(BaseModel):
    query_optimized : str = Field(description=" Do query rewritten and optimized for web search")
    max_results: int = Field(default=3, ge=1, le=8, description="Maximum number of search results to return")


@tool(args_schema=PaperSearchInput)
def ResearchPaper_VectorStore(
    query: str,
    k: int,
    session_id: Annotated[str, InjectedState("session_id")],
    existing_docs: Annotated[list, InjectedState("retrieved_docs")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> list:
    """This is Research paper vector store tool that contains research papers.

    Use this tool for any question about the content of those papers: their
    methods, results, conclusions, datasets, authors, or specific claims.
    Do NOT use it for current events, recent developments, or anything that
    would not appear in the uploaded papers; use the web_search tool for those."""

    documents = vs_search(
        query = query,
        session_id=session_id,
        k = k,
    )

    if not documents:
        return [
            ToolMessage(
                content="No relevant passages were found in the uploaded papers.",
                tool_call_id=tool_call_id,
            )
        ]

    status_message = ToolMessage(
        content=f"Found {len(documents)} relevant passage(s) in the research papers.",
        tool_call_id=tool_call_id,
    )

    updated_docs = (existing_docs or []) + documents

    return [
        status_message,
        Command(update={"retrieved_docs": updated_docs}),
    ]

@tool(args_schema = WebSearchInput)
def web_search(
    query_optimized: str,
    max_results: int,
    current_docs: Annotated[list, InjectedState("retrieved_docs")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> list:
    """This is web_search tool. Use this tool for current events, recent developments, or
       information that is not in the uploaded papers."""

    tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    response = tavily.search(query_optimized, max_results=max_results)
    results = response.get("results", [])

    if not results:
        return [
            ToolMessage(
                content="The online search did not return any results.",
                tool_call_id=tool_call_id,
            )
        ]

    documents = [
        Document(
            page_content=item["content"],
            metadata={
                "url": item["url"],
                "title": item.get("title") or "Untitled web page",
            },
        )
        for item in results
    ]

    status = ToolMessage(
        content=f"Added {len(documents)} online result(s) to the retrieved documents.",
        tool_call_id=tool_call_id,
    )

    updated_docs = (current_docs or []) + documents

    return [
        status,
        Command(update={"retrieved_docs": updated_docs}),
    ]


TOOLS = [ResearchPaper_VectorStore, web_search]
llm_with_tools = llm.bind_tools(TOOLS, parallel_tool_calls=False)

RETRIEVE_SYSTEM_PROMPT= """
<role>
You are a context-gathering agent for a research paper Q&A system.
Your only job is to collect the information needed to answer the user's question.
</role>

<tools>
<ResearchPaper_VectorStore>
Searches the research papers uploaded in this session.
Parameters you control:
- query: a search phrase that matches how the papers would word the answer
- k: number of passages to return (1-8). Use 2-3 for narrow questions, 5-8 for broad ones.
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
- Call only one tool per turn.
</tool_selection_rules>

<output_constraints>
Do not write a final answer.
Only call tools to gather context.
</output_constraints>
"""

MAX_RETRIEVAL_ATTEMPTS = 3
def agent_node(state: RAGState) -> dict:
    attempts = state.get("retrieval_attempts", 0)

    if attempts >= MAX_RETRIEVAL_ATTEMPTS:
        model = llm
    else:
        model = llm_with_tools
    messages = [
        {"role": "system", "content": RETRIEVE_SYSTEM_PROMPT},
        *state["messages"],
    ]

    result = model.invoke(messages)

    state_updates = {"messages": [result]}

    if result.tool_calls:
        state_updates["retrieval_count"] = attempts + 1

    return state_updates

tool_node = ToolNode(TOOLS)

RELEVANCY_CHECK_PROMPT = """
<role>
You evaluate whether retrieved document chunks are relevant to a user's question about research papers.
</role>

<task>
Determine whether the retrieved chunks contain useful information for answering the question.
</task>

<evaluation_rules>
- Return documents_relevant=true if the chunks meaningfully address the question, even partially.
- Return documents_relevant=false if the chunks are clearly unrelated or contain no useful information.
- Be lenient when evaluating relevance. Any substantive connection to the question is sufficient to return true.
</evaluation_rules>

<output_constraints>
Return the structured output required by RelevancyDecision.
</output_constraints>
"""

def relevancy_check_node(state: RAGState) -> dict:
    query = state["query"]
    documents = state.get("retrieved_docs", [])

    context = "\n\n---\n\n".join(
        doc.page_content[:400] for doc in documents[:4]
    )

    if not context:
        return {"documents_relevant": False}

    messages = [
        {"role": "system", "content": RELEVANCY_CHECK_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question: {query}\n\n"
                f"Retrieved chunks:\n{context}\n\n"
                "Are the retrieved chunks relevant to answering the question?"
            ),
        },
    ]

    result = (llm.with_structured_output(RelevancyDecision)).invoke(messages)

    return {"documents_relevant": result.documents_relevant}

 
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
    query = state["query"]
    rewrite_count = state.get("rewrite_count", 0)

    messages = [
        {"role": "system", "content": QUERY_REWRITE_PROMPT},
        {
            "role": "user",
            "content": f"Original query: {query}\n\nRewrite it as a better search query.",
        },
    ]

    response = llm.invoke(messages)
    new_query = response.content.strip()

    return {
        "messages": [HumanMessage(content=new_query)],
        "query": new_query,
        "retrieved_docs": [],
        "retrieval_count": 0,
        "rewrite_count": rewrite_count + 1,
        "documents_relevant": None,
    }

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
def verify_claim_node(state: RAGState) -> dict:
    claim = state["messages"][-1].content
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    search_queries = [
        (f"recent studies challenging or updating: {claim[:200]}", 5),
        (f"site:arxiv.org {claim[:200]}", 5),
    ]

    search_results = []

    for query, limit in search_queries:
        results = client.search(query, max_results=limit).get("results", [])
        search_results.append(results)

    context_sections = []

    section_titles = [
        "General Research and Web Results",
        "arXiv Research Papers",
    ]

    for title, results in zip(section_titles, search_results):
        context_sections.append(f"=== {title} ===")

        for item in results:
            context_sections.append(
                f"Title: {item.get('title', '')}\n"
                f"URL: {item.get('url', '')}\n"
                f"Content: {item.get('content', '')[:300]}"
            )

    search_context = "\n\n".join(context_sections)

    prompt = f"""
{CLAIM_ANALYSIS_PROMPT}

<claim_to_verify>
{claim}
</claim_to_verify>

<search_evidence>
{search_context}
</search_evidence>
"""
    response =  (llm.with_structured_output(ClaimVerificationResult)).invoke([
        {"role": "user", "content": prompt}
    ])

    papers = [
        paper.model_dump()
        for paper in response.superseding_papers[:3]
    ]

    return {
        "verification_status": response.verdict_summary,
        "verification_source": papers[0].get("url") if papers else None,
        "related_papers": papers,
    }

def generate_answer_node(state: RAGState) -> dict:
    route = state.get("selected_route")
    question = state["user_query"]

    if route == "retrieve":
        docs = state.get("retrieved_docs") or []
        if not docs or state.get("documents_relevant") is False:
            answer = (
                "I couldn't find anything relevant to that question in the "
                "uploaded papers. Try rephrasing it or uploading more papers."
            )
        else:
            evidence = "\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs, 1))
            answer = llm.invoke([
                {"role": "system", "content": (
                    "<role>You answer using only the evidence provided.</role>\n"
                    "<rules>\n"
                    "- Base every statement on the passages in <evidence>.\n"
                    "- If the evidence lacks the answer, say so. Do not guess.\n"
                    "- Cite the passage number, like [2], after each claim.\n"
                    "</rules>"
                )},
                {"role": "user", "content": (
                    f"<evidence>\n{evidence}\n</evidence>\n\n"
                    f"<question>{question}</question>"
                )},
            ]).content

    elif route == "verify_claim":
        papers = state.get("related_papers") or []
        lines = [
            "## Claim check",
            f"Claim: {question}",
            f"Verdict: {state.get('verification_status', '')}",
        ]
        if papers:
            lines.append("\nNewer work:")
            lines += [f"- {p['title']}: {p['summary']} ({p['url']})" for p in papers]
        else:
            lines.append("\nNo newer work that challenges this claim was found.")
        answer = "\n".join(lines)

    else:  # direct_answer, or any unknown route
        answer = llm.invoke([{"role": "user", "content": question}]).content

    return {"final_answer": answer, "messages": [AIMessage(content=answer)]}

MAX_REWRITES = 1


def route_by_intent(state: RAGState) -> str:
    # unknown or missing route falls back to answering directly
    return state.get("selected_route") or "direct_answer"


def after_agent(state: RAGState) -> str:
    last = state["messages"][-1]
    # A pending tool call must run first. Otherwise the saved history would
    # hold a tool call with no matching ToolMessage.
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
    builder.add_edge("tools", "agent")

    builder.add_conditional_edges(
        "relevancy_check",
        after_relevancy,
        {"rewrite": "query_rewrite", "answer": "generate_answer"},
    )
    builder.add_edge("query_rewrite", "agent")

    builder.add_edge("verify_claim", "generate_answer")
    builder.add_edge("generate_answer", END)

    return builder.compile(checkpointer=SqliteSaver(conn))