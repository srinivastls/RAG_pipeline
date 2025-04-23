from pymilvus import Collection, connections
from database import connect_to_database, embed_text
import requests
import json

def search_database(query, tokenizer, model):
    collection = Collection("Chunked_Docs")
    query_embedding = embed_text([query], tokenizer, model)
    search_params = {"metric_type": "IP", "params": {"nprobe": 10}}

    results = collection.search(
        data=query_embedding,
        anns_field="embedding",
        param=search_params,
        limit=10,
        output_fields=["text", "filename"],
    )
    
    return results


def search_cloud_database(query, tokenizer, model):
    query_embedding = embed_text([query], tokenizer, model)

    url = "url"
    headers = {
    "Authorization": "Bearer <API-BEARER_TOKEN>",
    "Accept": "application/json",
    "Content-Type": "application/json"
    }

    payload = {
        "collectionName": "RAG_legal",
        "data": [query_embedding[0]],
        "limit": 10,
        "outputFields": ["primary_key", "text", "filename", 'metadata'],
        "searchParams": {
            "metric_type": "COSINE",
            "params": {
                "nprobe": 10
            }
        }
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    results = response.json()
    
    return results