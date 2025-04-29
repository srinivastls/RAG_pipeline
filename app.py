import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from database import connect_to_database, create_collection, upload_to_milvus, upload_to_milvus_cloud
from search import search_database, search_cloud_database
from prompts import finetune_prompt, final_prompt
from output import generate_response
import requests


url = "http://34.72.217.0:8000/summarize"  # Make sure the URL is correct
url1= "http://34.72.217.0:8000/generate"  # Make sure the URL is correct


embedding_model_name = "BAAI/bge-large-en"

embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name,trust_remote_code=True)
embedding_model = AutoModelForCausalLM.from_pretrained(embedding_model_name,device_map="auto", trust_remote_code=True)

embedding_model.eval()

#Streamlit Interface
st.set_page_config(page_title="NyayaMitra", layout="wide")
st.title("NyayaMitra:AI-Powered Legal Assistant")
st.write("Welcome to NyayaMitra, your AI-powered legal assistant. Upload your legal documents, search through your database, or chat with our AI for legal advice.")

# Tabs
tab1, tab2, tab3,tab4 = st.tabs(["📤 Upload Document", "🔍 Search Database", "⚖️ Chat with Legal AI","Fake news Detector"])
connect_to_database()

# ---------- Tab 1: Upload ----------
with tab1:
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if st.button("Upload File"):
        if uploaded_file:
            collection = create_collection()
            upload_to_milvus_cloud(uploaded_file, embedding_tokenizer, embedding_model)

            st.success("File uploaded and stored in Milvus.")
            st.write("File type:", uploaded_file.type)
            st.write("File size (bytes):", uploaded_file.size)
        else:
            st.warning("Please select a file before uploading.")

# ---------- Tab 2: Search ----------
with tab2:
    query = st.text_input("Enter your search query:")
    

    if st.button("Search") and query:
        try:
            results = search_cloud_database(query, embedding_tokenizer, embedding_model)
            
            
            for i, hit in enumerate(results['data']):
                st.markdown(f"**Result {i+1}**")
                st.write(hit.get("text"))
                st.write("File Name:", hit.get("filename"))
                st.caption(f"Distance: {hit.get('distance')}")
        
        except Exception as e:
            st.error(f"Search failed: {str(e)}")

with tab3:
    query = st.text_input("Enter your query:")
    if st.button("Ask Legal AI") and query:
        finetuned_query = finetune_prompt(query, url1)
        #st.write("Finetuned Query:", finetuned_query)
        # chunks = search_database(finetuned_query, embedding_tokenizer, embedding_model)
        st.write("searching for chunks...")
        chunks = search_cloud_database(query, embedding_tokenizer, embedding_model)
        st.write("Chunks:", chunks)
        # print(chunks)
        context = ""
        # for i, hit in enumerate(chunks[0]):
        #     context += hit.entity.get("text") + "\n"

        for i, hit in enumerate(chunks['data']):
            context += hit['text'] + "\n"
        print("Context:", context)
        
        #prompt = final_prompt(query, context)
        st.write("generating response...")
        response = generate_response(query,context, url)
        st.markdown("### AI Response:")
        st.markdown(response)

with tab4:
    st.write("Check whether the given query is real news  or not(in the section level)")
    query1 = st.text_input("Enter your search query:")
    

    if st.button("fake news detect") and query:
        try:
            result = search_cloud_database(query1, embedding_tokenizer, embedding_model)
            
            
            for i, hit in enumerate(result['data']):
                st.markdown(f"**Result {i+1}**")
                st.write(hit.get("text"))
                st.write("File Name:", hit.get("filename"))
                st.caption(f"Distance: {hit.get('distance')}")
        
        except Exception as e:
            st.error(f"Search failed: {str(e)}")