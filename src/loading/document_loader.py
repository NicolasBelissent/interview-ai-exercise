"""Document loader for the RAG example."""

import json
from typing import Any

import chromadb
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.constants import SETTINGS
from src.loading.chunk_json import chunk_json_schema_with_metadata
from src.models import Document


def get_json_data() -> dict[str, Any]:
    # Send a GET request to the URL specified in SETTINGS.DOCS_URL
    api_specs = []
    docs_url: list = [
        "https://api.eu1.stackone.com/oas/stackone.json",
        "https://api.eu1.stackone.com/oas/hris.json",
        "https://api.eu1.stackone.com/oas/ats.json",
        "https://api.eu1.stackone.com/oas/lms.json",
        "https://api.eu1.stackone.com/oas/iam.json",
        "https://api.eu1.stackone.com/oas/crm.json",
        "https://api.eu1.stackone.com/oas/marketing.json"
    ]


    for url in docs_url:
        print(url)
        response = requests.get(url)

        json_data = response.json()
        print('json_data:', json_data)
        response.raise_for_status()
        api_specs.append(json_data)

    return api_specs


def document_json_array(data: list[dict[str, Any]]) -> list[Document]:
    """Converts an array of JSON chunks into a list of Document objects."""
    return [
        Document(page_content=json.dumps(item["chunk"]), metadata={"source": item['attribute']})
        for item in data
    ]


def build_docs(data: dict[str, Any]) -> list[Document]:
    """Chunk (badly) and convert the JSON data into a list of Document objects."""
    docs = []
    chunks = chunk_json_schema_with_metadata(data)
    docs.extend(document_json_array(chunks))
    return docs


def split_docs(docs_array: list[Document]) -> list[Document]:
    """Some may still be too long, so we split them."""
    splitter = RecursiveCharacterTextSplitter(
        separators=["}],", "},", "}", "]", " ", ""], chunk_size=SETTINGS.chunk_size
    )
    return splitter.split_documents(docs_array)


def add_documents(collection: chromadb.Collection, docs: list[Document]) -> None:
    """Add documents to the collection"""
    collection.add(
        documents=[doc.page_content for doc in docs],
        metadatas=[doc.metadata or {} for doc in docs],
        ids=[f"doc_{i}" for i in range(len(docs))],
    )
