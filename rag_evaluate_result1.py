# rag_evaluate_from_dataset.py

import json
from pathlib import Path

from datasets import Dataset

from ragas import evaluate

# OLD STABLE METRICS API
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_groq import ChatGroq

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from query_data import query_rag


# =========================================================
# PATHS
# =========================================================

CHROMA_PATH = "chroma"

EVALUATION_DATASET = "cleaned_evaluation_dataset.json"


# =========================================================
# LOAD API KEY FROM query_data.py
# =========================================================

def get_api_key():

    text = Path("query_data.py").read_text(encoding="utf-8")

    start = text.find('api_key="') + len('api_key="')
    end = text.find('"', start)

    return text[start:end]


# =========================================================
# EMBEDDING FUNCTION
# =========================================================

def get_embedding_function():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings


# =========================================================
# LOAD CHROMA DATABASE
# =========================================================

def load_db():

    embedding_function = get_embedding_function()

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )

    return db


# =========================================================
# LOAD EVALUATION DATASET
# =========================================================

def load_evaluation_dataset():

    with open(EVALUATION_DATASET, "r", encoding="utf-8") as f:

        data = json.load(f)

    return data


# =========================================================
# MAIN
# =========================================================

def main():

    print("\nLoading evaluation dataset...")

    evaluation_data = load_evaluation_dataset()

    print(f"Loaded {len(evaluation_data)} evaluation questions")

    print("\nLoading Chroma database...")

    db = load_db()

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    print("\nRunning RAG pipeline on evaluation dataset...\n")

    for idx, item in enumerate(evaluation_data):

        question = item["question"]

        ground_truth = item["ground_truth"]

        try:

            response_text, retrieved_chunks, sources = query_rag(
                query_text=question,
                top_k=3,
                db=db
            )

            questions.append(question)

            answers.append(response_text)

            contexts.append(retrieved_chunks)

            ground_truths.append(ground_truth)

            print("=" * 80)
            print(f"QUESTION {idx+1}")

            print("QUESTION:", question)

            print("GROUND TRUTH:", ground_truth)

            print("GENERATED ANSWER:", response_text)

            print("SOURCES:", sources)

        except Exception as e:

            print("\nFAILED QUESTION:")

            print(question)

            print(e)

    # =====================================================
    # BUILD DATASET FOR RAGAS
    # =====================================================

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    # =====================================================
    # CREATE LLM JUDGE
    # =====================================================

    groq_api_key = get_api_key()

    evaluator_llm = LangchainLLMWrapper(
        ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=groq_api_key
        )
    )

    # =====================================================
    # CREATE EMBEDDING WRAPPER
    # =====================================================

    evaluator_embeddings = LangchainEmbeddingsWrapper(
        get_embedding_function()
    )

    # =====================================================
    # RAGAS EVALUATION
    # =====================================================

    print("\nRunning RAGAS evaluation...\n")

    result = evaluate(
        dataset=dataset,

        metrics=[

            faithfulness,

            answer_relevancy,

            context_precision,

            context_recall
        ],

        llm=evaluator_llm,

        embeddings=evaluator_embeddings
    )

    # =====================================================
    # FINAL RESULTS
    # =====================================================

    print("\n" + "=" * 80)
    print("FINAL RAG EVALUATION RESULTS")
    print("=" * 80)

    print(result)

    # =====================================================
    # SAVE RESULTS
    # =====================================================

    with open("ragas_results.txt", "w", encoding="utf-8") as f:

        f.write(str(result))

    print("\nSaved results to:")
    print("ragas_results.txt")


if __name__ == "__main__":
    main()