import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from database import connect_to_database, create_collection, upload_to_milvus, upload_to_milvus_cloud
from search import search_database, search_cloud_database
from prompts import finetune_prompt, final_prompt
from output import generate_response

embedding_model_name = "BAAI/bge-large-en"

embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
from transformers import AutoModel

embedding_model = AutoModel.from_pretrained(embedding_model_name)

embedding_model.eval()

embedding_model.to("meta")  # or "cuda"

llm_model_name = "microsoft/Phi-3-mini-128k-instruct"

llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_tokenizer.pad_token = llm_tokenizer.eos_token
llm_tokenizer.padding_side = "right"

llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name)

llm_model.eval()

llm_model.to("meta")  # or "cuda"

#Streamlit Interface
st.set_page_config(page_title="RAG Pipeline", layout="wide")
st.title("RAG Pipeline: Upload & Search")

# Tabs
tab1, tab2, tab3 = st.tabs(["📤 Upload Document", "🔍 Search Database", "⚖️ Chat with Legal AI"])
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
    for name, param in embedding_model.named_parameters():
        st.write(f"{name} -> {param.device}")

    if st.button("Search") and query:
        try:
            results = search_cloud_database(query, embedding_tokenizer, embedding_model)
            if results and len(results[0]) > 0:
                for i, hit in enumerate(results[0]):
                    st.markdown(f"**Result {i+1}**")
                    st.write(hit.entity.get("text"))
                    st.write("File Name:", hit.entity.get("filename"))
                    st.caption(f"Distance: {hit.distance:.4f}")
            else:
                st.info("No results found.")
        except Exception as e:
            st.error(f"Search failed: {str(e)}")

with tab3:
    query = st.text_input("Enter your query:")
    if st.button("Ask Legal AI") and query:
        finetuned_query = finetune_prompt(query, llm_model, llm_tokenizer)
        print(f"Finetuned Query: {finetuned_query}")
        # chunks = search_database(finetuned_query, embedding_tokenizer, embedding_model)
        chunks = search_cloud_database(query, embedding_tokenizer, embedding_model)
        # print(chunks)
        context = ""
        # for i, hit in enumerate(chunks[0]):
        #     context += hit.entity.get("text") + "\n"

        for i, hit in enumerate(chunks['data']):
            context += hit['metadata'] + "\n" + hit['text'] + "\n"
        # print("Context:", context)
        
        prompt = final_prompt(query, context)

        response = generate_response(prompt, llm_model, llm_tokenizer)
        st.markdown("### AI Response:")
        st.markdown(response)