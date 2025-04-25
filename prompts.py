import re
import requests
import streamlit as st


  # Make sure the URL is correct



def finetune_prompt(query,url):
    prompt_template = """
        You are an intelligent search assistant. Convert the following user query into a concise, well-formed search query that captures the core intent and essential information. Avoid unnecessary details. Your output must include the optimized query wrapped between <|startquery|> and <|endquery|>.

        User Query: {query}

        Respond only with the formatted search query.

        Format:
        <|startquery|>optimized search query here<|endquery|>
    """
    
    prompt = prompt_template.format(query=query)
    payload = {
    "prompt": prompt
    }

    response= requests.post(url, json=payload)
    st.write("Decoded Response:", response)
    decoded=response.json()
    st.write("Decoded Response:", decoded['response'])

    
    
    return re.findall(r"<\|startquery\|>(.*?)<\|endquery\|>", decoded["response"], re.DOTALL)[-1]

def final_prompt(query, context):
    prompt_tempplate = """
        You are a legal assistant specialized in Indian law. Based on the context provided from legal case documents,
        answer the user's question precisely and factually. Always quote relevant case numbers, parties involved,
        and final judgments whenever possible.

        The output should be concise, clear and under 200-300 words.

        ### Context:
        {context}

        ### User Question:
        {question}

        ### Answer:
    """

    return prompt_tempplate.format(context=context, question=query)