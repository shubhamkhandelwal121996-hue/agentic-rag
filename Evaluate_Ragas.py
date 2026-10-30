
import json
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel

from ragas import EvaluationDataset, evaluate
from ragas.metrics import (
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
    Faithfulness,
    NoiseSensitivity,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from agentic_rag.core.QdrantVector_Store import add_research_paper
from agentic_rag.core.Agentic_rag1 import build_graph
from agentic_rag.core.rag_loaders import load_document

load_dotenv()

PDF_PATH = "documents/Openclaw_Research_Report.pdf"
TEST_DATASET_FILE = Path("ragas_test_dataset.json")
RESULTS_FILE = Path("ragas_evaluation_results.json")

MAX_CONTEXTS = 5
TEST_CASES_PER_CONTEXT = 2
EVALUATOR_MODEL = "gpt-4o-mini"


# ---------------- TEST DATASET ----------------

class TestCase(BaseModel):
    input: str
    reference_answer: str


class TestCases(BaseModel):
    test_cases: list[TestCase]


def generate_test_dataset() -> list[dict]:
    generator = ChatOpenAI(
        model=EVALUATOR_MODEL,
        temperature=0,
    ).with_structured_output(TestCases)

    test_dataset = []

    for doc in load_document(PDF_PATH)[:MAX_CONTEXTS]:
        result = generator.invoke(
            f"Generate exactly {TEST_CASES_PER_CONTEXT} distinct "
            "question-answer pairs from the research text below. "
            "Questions must require information from this text. "
            "Answers must be supported by the text. "
            "Do not invent facts.\n\n"
            f"{doc.page_content}"
        )

        test_dataset.extend(
            case.model_dump()
            for case in result.test_cases
            if case.input.strip() and case.reference_answer.strip()
        )

    if not test_dataset:
        raise ValueError("No evaluation test cases were generated.")

    TEST_DATASET_FILE.write_text(
        json.dumps(test_dataset, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return test_dataset


# ---------------- RUN AGENTIC RAG ----------------

def run_rag_query(graph, query: str, session_id: str):
    state = graph.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "session_id": session_id,
        },
        config={"configurable": {"thread_id": str(uuid4())}},
    )

    return (
        state.get("final_answer") or "",
        [doc.page_content for doc in state.get("retrieved_docs") or []],
        state.get("selected_route"),
    )


# ---------------- MAIN EVALUATION ----------------

def main():
    test_dataset = (
        json.loads(TEST_DATASET_FILE.read_text(encoding="utf-8"))
        if TEST_DATASET_FILE.exists()
        else generate_test_dataset()
    )

    docs = load_document(PDF_PATH)
    if not docs:
        raise ValueError("No document chunks were loaded.")

    graph = build_graph(db_path="eval_checkpoints.db")

    # Index the PDF once and reuse it across all test questions.
    session_id = f"ragas_evaluation_{uuid4()}"
    add_research_paper(docs, session_id)

    samples = []

    for index, case in enumerate(test_dataset, start=1):
        query = (
            case["input"]
            + "\nAnswer using the uploaded research report."
        )

        answer, contexts, route = run_rag_query(
            graph, query, session_id
        )

        print(
            f"Question {index}/{len(test_dataset)} "
            f"| Route: {route} | Contexts: {len(contexts)}"
        )

        if route != "retrieve":
            print("Skipped: question did not use retrieval.")
            continue

        samples.append({
            "user_input": query,
            "response": answer,
            "retrieved_contexts": contexts,
            "reference": case["reference_answer"],
        })

    if not samples:
        raise ValueError(
            "No questions followed the retrieve route."
        )

    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(model=EVALUATOR_MODEL, temperature=0)
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small")
    )

    metrics = [
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
        ResponseRelevancy(),
        Faithfulness(),
        NoiseSensitivity(),
    ]

    results = evaluate(
        dataset=EvaluationDataset.from_list(samples),
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        raise_exceptions=True,
        show_progress=True,
    )

    results_df = results.to_pandas()

    RESULTS_FILE.write_text(
        results_df.to_json(
            orient="records",
            indent=2,
            force_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n========== RAGAS EVALUATION RESULTS ==========")
    print(results_df.select_dtypes(include="number").mean().to_string())
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
