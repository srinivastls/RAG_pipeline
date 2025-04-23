from database import embed_text, make_chunks
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import requests
import json
import numpy as np

embedding_model_name = "BAAI/bge-large-en"
embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
embedding_model = AutoModelForCausalLM.from_pretrained(embedding_model_name)
embedding_model.eval()

df = pd.read_csv('data.csv')

url = "url"
headers = {
  "Authorization": "Bearer <API-BEARER_TOKEN>",
  "Accept": "application/json",
  "Content-Type": "application/json"
}

for index, row in df.iterrows():
    chunks = make_chunks(str(row['judgment']))
    metadata = row["meta_data"]
    embedding = embed_text(chunks, embedding_tokenizer, embedding_model)
    # print("x: ", len(row['judgment']))
    # print(len(chunks))
    for i, chunk in enumerate(chunks):
        pk = row['file_name'] + "-" + str(i)
        filename = row['file_name']

        payload = {
            "collectionName": "RAG",
            "data": [
                {
                    "primary_key": pk,
                    "text": chunk,
                    "filename": filename,
                    "embedding": embedding[i],
                    "metadata": metadata
                }
            ]
        }

        response = requests.post(url, data=json.dumps(payload), headers=headers)

        if response.status_code == 200:
            print(f"Inserted: {pk}")
        else:
            print(f"Failed ({response.status_code}) - {pk}")
            print(response.text)

