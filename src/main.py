"""FastAPI app creation, main API routes."""

from fastapi import FastAPI

from src.constants import SETTINGS, chroma_client, openai_client
from src.llm.completions import create_prompt, get_completion
from src.llm.embeddings import openai_ef
from src.loading.document_loader import (
    add_documents,
    build_docs,
    get_json_data,
    split_docs,
)
import redis
import json
from src.models import ChatOutput, ChatQuery, HealthRouteOutput, LoadDocumentsOutput
from src.retrieval.retrieval import get_relevant_chunks
from src.retrieval.vector_store import create_collection
from src.retrieval.hybrid_retriever import ParallelHybridRetriever


app = FastAPI()

collection = create_collection(chroma_client, openai_ef, SETTINGS.collection_name)
retriever = ParallelHybridRetriever(collection)
retriever.load_bm25_from_cache()
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)


@app.get("/health")
def health_check_route() -> HealthRouteOutput:
    """Health check route to check that the API is up."""
    return HealthRouteOutput(status="ok")


@app.get("/load")
async def load_docs_route() -> LoadDocumentsOutput:
    """Route to load documents into vector store."""
    api_specs_json = get_json_data()

    for json_data in api_specs_json:
        if isinstance(json_data, str):
            try:
                json_data = json.loads(json_data)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON string: {json_data}") from e
        if isinstance(json_data, dict):
            documents = build_docs(json_data)
        else:
            raise TypeError(f"Expected a dictionary, but got {type(json_data)} after processing.")

        # split docs
        documents = split_docs(documents)

        # load documents into vector store
        retriever.index_data(documents)
        # add_documents(collection, documents)

        # check the number of documents in the collection
        print(f"Number of documents in collection: {collection.count()}")

    return LoadDocumentsOutput(status="ok")


@app.post("/chat")
def chat_route(chat_query: ChatQuery) -> ChatOutput:
    """Chat route to chat with the API."""
    # Get relevant chunks from the collection
    relevant_chunks = get_relevant_chunks(query=chat_query.query, retriever=retriever, k=SETTINGS.k_neighbors
    )

    # Create prompt with context
    prompt = create_prompt(query=chat_query.query, context=relevant_chunks)

    print(f"Prompt: {prompt}")

    # Get completion from LLM
    result = get_completion(
        client=openai_client,
        prompt=prompt,
        model=SETTINGS.openai_model,
    )

    return ChatOutput(message=result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=80, reload=True)
