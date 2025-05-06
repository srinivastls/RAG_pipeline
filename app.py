import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from database import connect_to_database, create_collection, upload_to_milvus, upload_to_milvus_cloud
from search import search_database, search_cloud_database
from prompts import finetune_prompt, final_prompt
from output import generate_response
import requests


url = "http://34.72.217.0:8000/summarize" 
url1= "http://34.72.217.0:8000/generate"  
fake_news_api_url = "http://34.72.217.0:8000/fake-news-detection" 
summarization_api_url = "http://34.72.217.0:8000/summarize" 
embedding_model_name = "BAAI/bge-large-en"

embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name,trust_remote_code=True)
embedding_model = AutoModelForCausalLM.from_pretrained(embedding_model_name,device_map="auto", trust_remote_code=True)

embedding_model.eval()

#Streamlit Interface
st.set_page_config(page_title="NyayaMitra", layout="wide")
st.title("NyayaMitra:AI-Powered Legal Assistant")
st.write("Welcome to NyayaMitra, your AI-powered legal assistant. Upload your legal documents, search through your database, or chat with our AI for legal advice.")

# Tabs
tab1, tab2, tab3, tab4,tab5 = st.tabs(["📤 Upload Document", "🔍 Search Database", "⚖️ Chat with Legal AI","📰 Fake News Detection","Summarization"])
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
    st.subheader("Fake News Detection")
    news_text = st.text_area("Enter the news article or text to verify:")

    if st.button("Check if Fake News"):
        if news_text:
            # Replace the URL with your fake news detection API endpoint
            fake_news_api_url = fake_news_api_url
            payload = {'text': news_text}
            
            try:
                response = requests.post(fake_news_api_url, json=payload)
                response_data = response.json()

                if response_data.get("is_fake"):
                    st.markdown("### ❌ This is likely Fake News.")
                else:
                    st.markdown("### ✅ This seems to be Real News.")
            except requests.exceptions.RequestException as e:
                st.error(f"Error checking fake news: {e}")
        else:
            st.warning("Please enter some text to check.")


with tab5:
    st.subheader("Summarization")
    text_to_summarize = st.text_area("Enter the text to summarize:")

    if st.button("Summarize"):
        if text_to_summarize:
            payload = {'text': text_to_summarize}
            
            try:
                response = requests.post(summarization_api_url, json=payload)
                response_data = response.json()

                if "summary" in response_data:
                    st.markdown("### Summary:")
                    st.write(response_data["summary"])
                else:
                    st.error("Error: Summary not found in the response.")
            except requests.exceptions.RequestException as e:
                st.error(f"Error summarizing text: {e}")
        else:
            st.warning("Please enter some text to summarize.")

