"""Basic function to chunk JSON data by key."""

from typing import Any
import json 
from textwrap import wrap
import hashlib 

def chunk_data(data: dict[str, Any], key: str) -> list[dict[str, Any]]:
    info = data.get(key, {})
    return [{sub_key: sub_info} for sub_key, sub_info in info.items()]

def chunk_json_schema_with_metadata(json_data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Processes and chunks a large OpenAPI JSON schema into manageable pieces for RAG, 
    ensuring chunks are <= max_chunk_size, and adds an 'attribute' to each chunk.

    Args:
        json_data (dict): The full JSON schema to chunk.
        max_chunk_size (int): Maximum character size for a single chunk.

    Returns:
        list: A list of chunks with metadata and attributes.
    """
    chunks = []
    print("Starting ingestion and chunking process.")



    
    # Helper function to add chunks
    def add_chunk(section: str, content: dict[str, Any]):


        chunks.append({
            "attribute": section,
            "chunk": content
            })
        
    # Chunk `info` Section
    if "info" in json_data:
        print("Processing 'info' section.")
        add_chunk("info", json_data["info"])

    # Chunk `tags` Section
    if "tags" in json_data:
        print("Processing 'tags' section.")
        for tag in json_data["tags"]:
            add_chunk("tags", tag)

    # Chunk `servers` Section
    if "servers" in json_data:
        print("Processing 'servers' section.")
        for server in json_data["servers"]:
            add_chunk("servers", server)

    # Chunk `paths` Section
    if "paths" in json_data:
        print("Processing 'paths' section.")
        for path, methods in json_data["paths"].items():
            for method, details in methods.items():
                add_chunk("paths", {
                    "path": path,
                    "method": method.upper(),
                    "details": details
                })

    # Chunk `components` Section
    if "components" in json_data:
        print("Processing 'components' section.")
        components = json_data["components"]

        # Chunk Schemas
        if "schemas" in components:
            print("Processing schemas in 'components'.")
            for schema_name, schema_details in components["schemas"].items():
                add_chunk("components.schemas", {
                    "schemaName": schema_name,
                    "details": schema_details
                })

        # Chunk Security Schemes
        if "securitySchemes" in components:
            print("Processing securitySchemes in 'components'.")
            for scheme_name, scheme_details in components["securitySchemes"].items():
                add_chunk("components.securitySchemes", {
                    "schemeName": scheme_name,
                    "details": scheme_details
                })

    # Chunk `openapi` Section
    if "openapi" in json_data:
        print("Processing 'openapi' section.")
        add_chunk("openapi", {"version": json_data["openapi"]})

    print("Chunking process completed successfully.")
    return chunks

