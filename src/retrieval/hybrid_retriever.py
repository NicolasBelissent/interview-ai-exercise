from langchain.docstore.document import Document
from src.loading.document_loader import add_documents
from src.retrieval.bm25 import BM25
import json
from typing import List
from concurrent.futures import ThreadPoolExecutor
import redis
import numpy as np
from src.constants import openai_client

class ParallelHybridRetriever:
    def __init__(self, chroma_collection, redis_host="localhost", redis_port=6379, redis_db=0):
        """
        Initialize the retriever with ChromaDB and Redis.

        Args:
            chroma_collection: ChromaDB collection instance.
            redis_host (str): Redis server hostname.
            redis_port (int): Redis server port.
            redis_db (int): Redis database index.
        """
        self.chroma_collection = chroma_collection
        self.redis_client = redis.StrictRedis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)
        self.bm25 = None


    def index_data(self, docs):
        """
        Index documents by caching them in Redis, adding to ChromaDB, and preparing BM25.

        Args:
            docs (list): List of Document objects to index.
        """
        print("Caching documents in Redis...")
        doc_texts = []
        for idx, doc in enumerate(docs):
            doc_key = f"doc:{idx}"
            doc_texts.append(doc.page_content)
            self.redis_client.set(doc_key, json.dumps({"page_content": doc.page_content, "metadata": doc.metadata}))

        # Save document texts for BM25
        print("Caching BM25 document texts...")
        self.redis_client.set("bm25_doc_texts", json.dumps(doc_texts))

        # Add documents to ChromaDB
        print("Adding documents to ChromaDB...")
        add_documents(self.chroma_collection, docs)

        # Prepare BM25
        print("Preparing BM25...")
        self.bm25 = BM25(doc_texts)

        print("Indexing complete.")

    def load_bm25_from_cache(self):
        """
        Load the BM25 model from cached document texts in Redis.
        """
        print("Loading BM25 model from cache...")
        cached_texts = self.redis_client.get("bm25_doc_texts")
        if cached_texts:
            doc_texts = json.loads(cached_texts)
            self.bm25 = BM25(doc_texts)
            print("BM25 model successfully loaded.")
        else:
            print("No cached BM25 document texts found. Please index documents first.")

    def get_cached_documents(self):
        """
        Retrieve all documents from Redis.

        Returns:
            list: List of Document objects.
        """
        print("Retrieving documents from Redis...")
        doc_keys = [key for key in self.redis_client.keys("doc:*")]  # Keys are already strings
        doc_keys = sorted(doc_keys, key=lambda x: int(x.split(":")[1]))  # Sort by document ID
        documents = []
        for doc_key in doc_keys:
            cached_doc = self.redis_client.get(doc_key)
            if cached_doc:
                doc_data = json.loads(cached_doc)
                documents.append(Document(page_content=doc_data["page_content"], metadata=doc_data["metadata"]))
        return documents
    
    def reciprocal_rank_fusion(self, doc_id, bm25_ranks, chroma_ranks, rrf_k, alpha=0.5):
        """
        Calculate the RRF score for a given document ID based on BM25 and Chroma ranks.

        Args:
            doc_id (int): The document ID.
            bm25_ranks (dict): Ranks of documents from BM25 retrieval.
            chroma_ranks (dict): Ranks of documents from ChromaDB retrieval.
            rrf_k (int): RRF hyperparameter controlling the rank decay.
            alpha (float): Weighting factor for BM25 and Chroma RRF scores (0 <= alpha <= 1).

        Returns:
            float: The combined RRF score for the document.
        """
        bm25_rrf_score = 1 / (rrf_k + bm25_ranks.get(doc_id, float('inf')))
        chroma_rrf_score = 1 / (rrf_k + chroma_ranks.get(doc_id, float('inf')))
        return alpha * bm25_rrf_score + (1 - alpha) * chroma_rrf_score
    

    
    def single_query(self, query, k=20, n=5, rrf_k=60, hybrid = False):
        """
        Perform hybrid retrieval (or just semantic) for a single query using Reciprocal Rank Fusion (RRF).

        Args:
            query (str): The search query.
            k (int): Number of top results to retrieve.
            n (int): Number of contexts to share to the model.
            rrf_k (int): RRF hyperparameter controlling the rank decay.

        Returns:
            dict: Top-k results with precomputed RRF scores.
        """
        if not self.bm25:
            raise ValueError("BM25 is not initialized. Please cache documents first.")
        
        if hybrid:

            # BM25 Retrieval
            bm25_results = self.bm25.query(query, k=k)
            bm25_ranks = {doc_id: rank for rank, (doc_id, _) in enumerate(bm25_results, start=1)}
            print('bm25_ranks: ', bm25_ranks)
            # ChromaDB Retrieval
            embedding_results = self.chroma_collection.query(
                query_texts=[query],
                n_results=k
            )
            chroma_ranks = {
                int(result_id.split("_")[1]): rank
                for rank, result_id in enumerate(embedding_results["ids"][0], start=1)
            }
            print('chroma_ranks: ', chroma_ranks)
            # RRF Scoring
            combined_scores = {}
            for doc_id in set(bm25_ranks.keys()).union(chroma_ranks.keys()):
                combined_scores[doc_id] = self.reciprocal_rank_fusion(doc_id, bm25_ranks, chroma_ranks, rrf_k)

        else:
            # ChromaDB Retrieval
            embedding_results = self.chroma_collection.query(
                query_texts=[query],
                n_results=k
            )
            chroma_ranks = {
                int(result_id.split("_")[1]): rank
                for rank, result_id in enumerate(embedding_results["ids"][0], start=1)
            }
            print('chroma_ranks: ', chroma_ranks)

            combined_scores = {}
            for doc_id in set(chroma_ranks.keys()).union(chroma_ranks.keys()):
                combined_scores[doc_id] = self.reciprocal_rank_fusion(doc_id, chroma_ranks, chroma_ranks, rrf_k)

        # Sort by combined RRF scores
        final_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:n]

        # Retrieve cached documents
        documents = {f"doc:{idx}": doc for idx, doc in enumerate(self.get_cached_documents())}

        # Format results
        final_output = []
        for doc_id, score in final_results:
            doc_key = f"doc:{doc_id}"
            if doc_key not in documents:
                print(f"WARNING: Document ID {doc_id} not found in cached documents. Skipping.")
                continue
            doc = documents[doc_key]
            final_output.append({
                "document": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            })

        return {"results": final_output, "total_rrf_score": sum(score for _, score in final_results)}

    def multi_query_parallel(self, query, num_instances=5, k=50, max_workers=4):
        """
        Perform the same query multiple times in parallel using Reciprocal Rank Fusion (RRF).

        Args:
            query (str): The search query.
            num_instances (int): Number of instances of the query to execute.
            k (int): Number of top results to retrieve for each query.
            max_workers (int): Maximum number of threads for parallel execution.

        Returns:
            dict: The results from the `single_query` call with the highest combined RRF score.
        """
        if not self.bm25:
            raise ValueError("BM25 is not initialized. Please cache documents first.")

        # Generate different rrf_k values for each call
        rrf_k_range = np.linspace(60, 100, num_instances)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit multiple calls to `single_query` with different rrf_k values
            futures = {
                executor.submit(self.single_query, query, k, n=5, rrf_k=rrf_k_range[i]): rrf_k_range[i]
                for i in range(num_instances)
            }

            # Store the results and total RRF scores for each query instance
            results_with_scores = []

            for future in futures:
                try:
                    # Retrieve results from `single_query`
                    result = future.result()  # Already includes total RRF score
                    results_with_scores.append((result["results"], result["total_rrf_score"], futures[future]))
                except Exception as e:
                    print(f"ERROR: Failed to process query instance: {e}")

            # Select the results with the highest total RRF score
            best_results = max(results_with_scores, key=lambda x: x[1])  # Select based on RRF score

            # Extract and return the best results
            best_output = {
                "rrf_k": best_results[2],  # rrf_k value used for this query
                "total_rrf_score": best_results[1],  # Total RRF score
                "results": best_results[0],  # List of top-k results
            }

            print("This is the best output: ", best_output)

            # Extract contexts from the best results
            contexts = best_output["results"]

            return contexts
    
    def evaluate_retrieval(self, query, context):
        """
        Evaluates the relevance of a given context to a query.

        Args:
            query (str): The query to be evaluated.
            context (str): The context to be evaluated.

        Returns:
            list: A list of related questions that concerns compliance and cybersecurity.
                  Each question is represented as a dictionary with a 'question' key and its evaluation as the value.
                  The evaluation can be either 'Good' or 'Bad'.

        Raises:
            Exception: If an error occurs during the evaluation process.
        """
        try:
            result = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": f"""You are an evaluation tool. Your task is to judge the relevance of a given context to a query. 
                                    Respond with "Good" if the context appropriately answers or relates to the query, and "Bad" otherwise. 
                                    Be precise and do not provide any additional commentary.

                                    Context: {context}

                                    Query: {query}"""
                    }
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "retrievalQuality",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "retrievalQuality": {
                                    "type": "string",
                                    "description": "Evaluate this retrieval.",
                                    "enum": ["Good", "Bad"]
                                }
                            },
                            "required": ["retrievalQuality"],
                            "additionalProperties": False
                        }
                    }
                }
            )

            response_content = result.choices[0].message.content
            return json.loads(response_content).get("retrievalQuality", 0)
        
        except Exception as e:
            raise e
        return
