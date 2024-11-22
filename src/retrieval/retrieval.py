"""Retrieve relevant chunks from a vector store."""

import chromadb


def get_relevant_chunks(retriever, query: str, k: int
) -> list[str]:
    """Retrieve k most relevant chunks for the query"""
    # results = collection.query(query_texts=[query], n_results=k)

    results  = retriever.single_query(query, k=50, n=5, rrf_k=80, hybrid=False)['results']

    evaluation = retriever.evaluate_retrieval(query, results)
    print(f'An LLM evaluated this retrieval as : {evaluation}')

    return results
