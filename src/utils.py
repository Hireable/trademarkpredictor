import os
import logging
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

def get_pinecone_index(index_name: str = "case-json", dimension: int = 768, metric: str = "cosine", cloud: str = "aws", region: str = "eu-west-1"):
    """
    Creates or connects to a Pinecone index using the updated API.
    :param index_name: Name of the Pinecone index
    :param dimension: Dimension of the vectors (default: 768 for many NLP models)
    :param metric: Metric for similarity calculation (default: cosine)
    :param cloud: Cloud provider (default: "aws")
    :param region: Region for the Pinecone instance (default: "eu-west-1")
    :return: Pinecone index
    """
    # Create the ServerlessSpec configuration
    spec = ServerlessSpec(
        cloud=cloud,
        region=region
    )
    
    # Initialize Pinecone instance
    pinecone = Pinecone(
        api_key=PINECONE_API_KEY,
        spec=spec
    )

    # Check if the index already exists
    if index_name not in pinecone.list_indexes().names():
        # Create the index with the required specification
        pinecone.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=spec  # Add the spec parameter here
        )

    # Connect to the index
    return pinecone.Index(index_name)

def upsert_chunks(index, chunks: List[Dict[str, Any]], namespace: str):
    """Upserts chunks into the specified Pinecone index with namespace."""
    try:
        upsert_response = index.upsert(vectors=chunks, namespace=namespace)
        logging.info(f"Upserted {len(chunks)} chunks: {upsert_response}")
        return upsert_response

    except Exception as e:
        logging.error(f"Error upserting chunks: {e}")
        return None

def delete_namespace(index, namespace: str):
    """Deletes all data in a specific namespace from the Pinecone index."""
    try:
        delete_response = index.delete(delete_all=True, namespace=namespace)
        logging.info(f"Deleted namespace '{namespace}': {delete_response}")
        return delete_response

    except Exception as e:
        logging.error(f"Error deleting namespace: {e}")
        return None

def query_pinecone(index, query_embedding: List[float], top_k: int = 10, namespace: str = None, metadata_filter: dict = None) -> List[Dict[str, Any]]:
    """Queries a Pinecone index."""
    try:
        query_response = index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace,
            filter=metadata_filter,
            include_metadata=True,
        )

        results = []
        for match in query_response.matches:
            results.append({
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata,
            })
        return results

    except Exception as e:
        logging.error(f"Error querying Pinecone: {e}")
        return []

def fetch_data_by_id(index, chunk_ids: list) -> List[Dict[str, Any]]:
    """Fetches data from a Pinecone index given a list of vector IDs."""
    try:
        fetch_response = index.fetch(ids=chunk_ids)

        if fetch_response and fetch_response['vectors']:
            return [{'id': vec_id, 'metadata': fetch_response['vectors'][vec_id]['metadata']} for vec_id in chunk_ids if vec_id in fetch_response['vectors']]
        else:
            return []

    except Exception as e:
        logging.error(f"Error fetching data by ID: {e}")
        return []
