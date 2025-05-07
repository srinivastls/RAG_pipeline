import streamlit as st
import fitz
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import torch
import torch.nn.functional as F
import requests
import json

def extract_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return "".join(page.get_text() for page in doc)

def make_chunks(text, chunk_size=512, overlap=50):
    words = text.split()   
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
        if i + chunk_size >= len(words):
            break
    return chunks

def embed_text(chunks, tokenizer, model):
    # Add 'query: ' prefix if embedding for queries (optional for docs)
    chunks = ["query: " + text for text in chunks]

    # Tokenize input
    inputs = tokenizer(
        chunks,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]

    # Attention-aware mean pooling
    attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(hidden_states.size())
    masked_hidden = hidden_states * attention_mask
    summed = masked_hidden.sum(dim=1)
    counts = attention_mask.sum(dim=1)
    mean_pooled = summed / counts

    # Normalize embeddings (recommended for cosine similarity)
    normalized_embeddings = F.normalize(mean_pooled, p=2, dim=1)

    # print("Embedding shape:", len(normalized_embeddings), "x", len(normalized_embeddings[0]))

    return normalized_embeddings.numpy().tolist()

def connect_to_database():
    if not connections.has_connection("default"):
        connections.connect(
            alias="default",
            uri="milvus_lite.db"
        )
    # return connections.get_connection("default")

def create_collection():
    id_field = FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64)
    vector_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024)
    text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=20000)
    file_name_field = FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=256)

    schema = CollectionSchema(fields=[id_field, vector_field, text_field, file_name_field], description="Document collection")

    collection_name = "Chunked_Docs"
    if not utility.has_collection(collection_name):
        collection = Collection(name=collection_name, schema=schema)
        collection.create_index(field_name="embedding", index_params={"index_type": "IVF_FLAT", "params": {"nprobe": 64}})
        collection.load()
    else:
        collection = Collection(collection_name)
        collection.load()

    return collection

def upload_to_milvus(collection, uploaded_file, tokenizer, model):
    text = extract_from_pdf(uploaded_file)
    # st.text_area("Extracted Text Preview", text[:1000], height=300)
    if not text.strip():
        st.error("No extractable text found in the PDF.")
        return
    
    chunks = make_chunks(text)
    
    if not chunks:
        st.error("Text extraction failed or resulted in empty chunks.")
        return

    embeddings = embed_text(chunks, tokenizer, model)
    ids = [f"{uploaded_file.name}-{i}" for i in range(len(chunks))]
    print(ids)
    file_names = [uploaded_file.name] * len(chunks)
    print(file_names)
    try:
        collection.insert([ids, embeddings, chunks, file_names])  
    except Exception as e:
        st.error(f"Error occurred: {str(e)}")
        return

    st.write(f"Total documents in collection: {collection.num_entities}")
    st.success(f"{uploaded_file.name} uploaded and stored successfully!")


def upload_to_milvus_cloud(uploaded_file, tokenizer, model):
    
    url = "https://in03-505c80f9dc0263a.serverless.gcp-us-west1.cloud.zilliz.com/v2/vectordb/entities/insert"
    headers = {
    "Authorization": "Bearer 1831fe961113fd3aa8af9c99b8a7d2f3e87b34bc2ac4d0f5bc701b1b09049be2aaba097631a7c8943701cb39b7f625597eb16bae",
    "Accept": "application/json",
    "Content-Type": "application/json"
    }

    text = extract_from_pdf(uploaded_file)
    chunks = make_chunks(text)
    embedding = embed_text(chunks, tokenizer, model)
    
    for i in range(len(chunks)):
        pk = uploaded_file.name + "-" + str(i)
        filename = uploaded_file.name

        payload = {
            "collectionName": "RAG_legal",
            "data": [
                {
                    "primary_key": pk,
                    "text": chunks[i],
                    "filename": filename,
                    "embedding": embedding[i],
                    "metadata": filename
                }
            ]
        }

        response = requests.post(url, data=json.dumps(payload), headers=headers)

        if response.status_code == 200:
            print(f"Inserted: {pk}")
        else:
            print(f"Failed ({response.status_code}) - {pk}")
            print(response.text)