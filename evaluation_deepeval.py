
import json
import sys
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig
from deepeval.test_case import LLMTestCase

from agentic_rag.core.QdrantVector_Store import add_research_paper
from agentic_rag.core.Agentic_rag1 import build_graph
from agentic_rag.core.rag_loaders import load_document


# ---------------- Configuration ----------------

load_dotenv()

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")


# ---------------- Research Papers ----------------

DOCUMENT_PATHS = [
    r"C:\Users\sonia\agentic-rag\src\doc\NIPS-2017-attention-is-all-you-need-Paper.pdf",
    r"C:\Users\sonia\agentic-rag\src\doc\bert.pdf",
    r"C:\Users\sonia\agentic-rag\src\doc\Lora.pdf",
    r"C:\Users\sonia\agentic-rag\src\doc\rag123.pdf",
    r"C:\Users\sonia\agentic-rag\src\doc\REACT.pdf",
    r"C:\Users\sonia\agentic-rag\src\doc\self_rag.pdf",
]

DATASET_PATH = Path("dataset_6papers.json")
OUTPUT_PATH = Path("evaluation_results_6papers.json")

EVALUATION_MODEL = "gpt-5.4-mini"
PASS_THRESHOLD = 0.7

# Target: approximately 48 golden questions
CONTEXT_LIMIT = 4
QUESTIONS_PER_CONTEXT = 2


# ---------------- Validate PDF Paths ----------------

def validate_documents():
    for pdf_path in DOCUMENT_PATHS:
        if not Path(pdf_path).is_file():
            raise FileNotFoundError(
                f"Research paper not found: {pdf_path}"
            )

    print(f"All {len(DOCUMENT_PATHS)} research papers found.")


# ---------------- Dataset Preparation ----------------

def prepare_dataset():

    if DATASET_PATH.exists():
        dataset = json.loads(
            DATASET_PATH.read_text(encoding="utf-8")
        )

        print(
            f"Loaded {len(dataset)} existing golden questions."
        )

        return dataset

    print("\nGenerating golden questions from research papers...")

    generator = Synthesizer()

    generated = generator.generate_goldens_from_docs(
        document_paths=DOCUMENT_PATHS,
        include_expected_output=True,
        max_goldens_per_context=QUESTIONS_PER_CONTEXT,
        context_construction_config=ContextConstructionConfig(
            max_contexts_per_document=CONTEXT_LIMIT
        ),
    )

    dataset = [
        {
            "input": item.input,
            "expected_output": item.expected_output,
        }
        for item in generated
        if item.input and item.expected_output
    ]

    if not dataset:
        raise ValueError(
            "DeepEval did not generate any valid golden questions."
        )

    DATASET_PATH.write_text(
        json.dumps(
            dataset,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Generated {len(dataset)} golden questions.")
    print(f"Dataset saved to: {DATASET_PATH}")

    return dataset


# ---------------- Metric Configuration ----------------

def create_metrics():

    metric_classes = [
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
        AnswerRelevancyMetric,
        FaithfulnessMetric,
    ]

    return [
        metric(
            threshold=PASS_THRESHOLD,
            model=EVALUATION_MODEL,
        )
        for metric in metric_classes
    ]


# ---------------- Load Research Papers ----------------

def load_research_papers():

    documents = []

    for index, pdf_path in enumerate(DOCUMENT_PATHS, start=1):

        print(
            f"\nLoading paper {index}/{len(DOCUMENT_PATHS)}: "
            f"{Path(pdf_path).name}"
        )

        docs = load_document(pdf_path)

        if not docs:
            print("Warning: No readable content found.")
            continue

        documents.extend(docs)

        print(f"Loaded {len(docs)} document objects.")

    if not documents:
        raise ValueError(
            "No readable content found in the research papers."
        )

    print(
        f"\nTotal loaded document objects: {len(documents)}"
    )

    return documents


# ---------------- Agentic RAG Execution ----------------

def execute_rag(graph, question, session_id):

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content=question)
            ],
            "session_id": session_id,
        },
        config={
            "configurable": {
                "thread_id": str(uuid4())
            }
        },
    )

    answer = result.get("final_answer") or ""

    contexts = [
        document.page_content
        for document in (
            result.get("retrieved_docs") or []
        )
    ]

    route = result.get("selected_route")

    return answer, contexts, route


# ---------------- Build Evaluation Test Cases ----------------

def build_test_cases(graph, dataset, session_id):

    test_cases = []
    skipped_questions = []

    for index, item in enumerate(dataset, start=1):

        question = item["input"]

        rag_question = (
            f"{question}\n"
            "Answer using the uploaded research papers."
        )

        answer, contexts, route = execute_rag(
            graph=graph,
            question=rag_question,
            session_id=session_id,
        )

        print(
            f"\nTest {index}/{len(dataset)} "
            f"| Route: {route} "
            f"| Contexts: {len(contexts)}"
        )

        if route != "retrieve":
            print("Skipped: Retrieval route not selected.")

            skipped_questions.append({
                "question": question,
                "route": route,
            })

            continue

        test_cases.append(
            LLMTestCase(
                input=rag_question,
                actual_output=answer,
                expected_output=item["expected_output"],
                retrieval_context=contexts,
            )
        )

    print("\n---------------- Evaluation Summary ----------------")
    print(f"Total golden questions: {len(dataset)}")
    print(f"Retrieval test cases: {len(test_cases)}")
    print(f"Skipped questions: {len(skipped_questions)}")
    print("----------------------------------------------------")

    return test_cases, skipped_questions


# ---------------- Save Evaluation Results ----------------

def export_results(evaluation, skipped_questions):

    records = []

    for test in evaluation.test_results:

        records.append({
            "question": test.input,
            "generated_answer": test.actual_output,
            "passed": test.success,
            "metric_scores": {
                metric.name: {
                    "score": metric.score,
                    "passed": metric.success,
                    "reason": metric.reason,
                }
                for metric in test.metrics_data
            },
        })

    output = {
        "total_evaluated": len(records),
        "total_skipped": len(skipped_questions),
        "evaluation_results": records,
        "skipped_questions": skipped_questions,
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"\nEvaluation completed. Results saved to {OUTPUT_PATH}"
    )


# ---------------- Main Pipeline ----------------

def main():

    print("\n========== PaperMind RAG Evaluation ==========")

    # Step 1: Validate all six PDF files
    validate_documents()

    # Step 2: Generate or load golden questions
    dataset = prepare_dataset()

    # Step 3: Load all six research papers
    documents = load_research_papers()

    # Step 4: Create a unique Qdrant session
    session_id = f"deepeval_{uuid4()}"

    print(f"\nQdrant Session ID: {session_id}")

    # Step 5: Upload all research papers
    # into the same Qdrant collection
    print("\nUploading research papers to Qdrant...")

    add_research_paper(
        documents,
        session_id,
    )

    print("Research papers uploaded successfully.")

    # Step 6: Build LangGraph agent
    graph = build_graph(
        db_path="eval_checkpoints.db"
    )

    # Step 7: Run questions through PaperMind
    test_cases, skipped_questions = build_test_cases(
        graph=graph,
        dataset=dataset,
        session_id=session_id,
    )

    if not test_cases:
        raise ValueError(
            "No valid retrieval test cases were collected."
        )

    # Step 8: Evaluate all five metrics
    print("\nStarting DeepEval evaluation...")

    evaluation = evaluate(
        test_cases=test_cases,
        metrics=create_metrics(),
        async_config=AsyncConfig(
            max_concurrent=3,
            throttle_value=5,
        ),
    )

    # Step 9: Save results
    export_results(
        evaluation,
        skipped_questions,
    )

    print("\n========== Evaluation Finished ==========")


if __name__ == "__main__":
    main()
