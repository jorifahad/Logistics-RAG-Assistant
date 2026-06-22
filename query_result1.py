import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Question: {question}
"""


def query_rag(query_text: str, top_k: int, db):

    results = db.similarity_search_with_score(
        query_text,
        k=top_k
    )

    sources = [
        f"{doc.metadata.get('source')} | Page {doc.metadata.get('page')}"
        for doc, score in results
    ]

    retrieved_chunks = [
        doc.page_content
        for doc, score in results
    ]

    context_text = "\n\n---\n\n".join(retrieved_chunks)

    prompt = ChatPromptTemplate.from_template(
        PROMPT_TEMPLATE
    ).format(
        context=context_text,
        question=query_text
    )

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing from the environment."
        )

    model = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=api_key
    )

    response = model.invoke(prompt).content

    return response, retrieved_chunks, sources