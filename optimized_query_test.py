from prompts import finetune_prompt
from transformers import AutoTokenizer, AutoModelForCausalLM
import streamlit as st

model_name = "microsoft/Phi-3-mini-128k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side =  "right"

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
model.eval()

st.set_page_config(page_title="RAG Pipeline", layout="wide")
st.title("Chat with Legal AI")
input_query = st.text_input("Enter your query: ")

if st.button("Submit"):
    if input_query:
        with st.spinner("Generating response..."):
            # Finetune the prompt
            optimized_query = finetune_prompt(input_query, model, tokenizer)
            print(optimized_query)
            st.write("Optimized Query:", optimized_query)

            
            # st.write("AI Response:", optimized_query)
    else:
        st.warning("Please enter a query before submitting.")