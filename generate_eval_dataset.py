import json
from pathlib import Path

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_groq import ChatGroq


DATA_PATH = "raw_data"

OUTPUT_FILE = "evaluation_dataset.json"


# =========================================================
# LOAD API KEY FROM query_data.py
# =========================================================

def get_api_key():

    text = Path("query_data.py").read_text(encoding="utf-8")

    start = text.find('api_key="') + len('api_key="')
    end = text.find('"', start)

    return text[start:end]


# =========================================================
# LOAD PDF FILES
# =========================================================

def load_documents():

    loader = PyPDFDirectoryLoader(DATA_PATH)

    documents = loader.load()

    return documents


# =========================================================
# SPLIT DOCUMENTS
# =========================================================

def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(documents)

    return chunks


# =========================================================
# GENERATE QUESTIONS
# =========================================================

def generate_question_answer(llm, chunk_text):

    prompt = f"""
You are creating a professional RAG evaluation dataset.

From the following document chunk:

-------------------
{chunk_text}
-------------------

Generate:

1. One important factual question
2. One concise ground-truth answer

Rules:
- The question must be answerable ONLY from this chunk.
- The question should test important information.
- Avoid vague questions.
- Keep the answer concise and factual.

Return ONLY valid JSON:

{{
  "question": "...",
  "ground_truth": "..."
}}
"""

    response = llm.invoke(prompt)

    return response.content


# =========================================================
# MAIN
# =========================================================

def main():

    api_key = get_api_key()

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=api_key
    )

    print("\nLoading PDFs...")
    documents = load_documents()

    print("Splitting documents...")
    chunks = split_documents(documents)

    evaluation_dataset = []

    used_sources = set()

    print("\nGenerating evaluation questions...\n")

    for chunk in chunks:

        source = chunk.metadata.get("source", "")

        # avoid generating too many from same file section
        source_key = f"{source}_{chunk.metadata.get('page', 0)}"

        if source_key in used_sources:
            continue

        used_sources.add(source_key)

        chunk_text = chunk.page_content.strip()

        if len(chunk_text) < 300:
            continue

        try:

            result = generate_question_answer(
                llm=llm,
                chunk_text=chunk_text
            )

            parsed = json.loads(result)

            evaluation_dataset.append({
                "question": parsed["question"],
                "ground_truth": parsed["ground_truth"],
                "source": source,
                "page": chunk.metadata.get("page", 0)
            })

            print("=" * 80)
            print("SOURCE:", source)
            print("QUESTION:", parsed["question"])
            print("ANSWER:", parsed["ground_truth"])

        except Exception as e:

            print("\nSkipped chunk بسبب:")
            print(e)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

        json.dump(
            evaluation_dataset,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 80)
    print(f"Saved evaluation dataset to: {OUTPUT_FILE}")
    print(f"Total questions generated: {len(evaluation_dataset)}")


if __name__ == "__main__":
    main()