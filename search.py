from pymilvus import Collection, connections
from database import connect_to_database, embed_text
import requests
import json
import streamlit as st

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
    #st.write("Query Embedding:", query_embedding)    
    url = "https://in03-505c80f9dc0263a.serverless.gcp-us-west1.cloud.zilliz.com/v2/vectordb/entities/search"
    headers = {
    "Authorization": "Bearer 1831fe961113fd3aa8af9c99b8a7d2f3e87b34bc2ac4d0f5bc701b1b09049be2aaba097631a7c8943701cb39b7f625597eb16bae",
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
    st.write(response)
    results = response.json()
    
    return results