import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
from PIL import Image
import io
import base64

# API Configuration
API_URL = "https://api.nyayamitra.org"
API_TOKEN = None

# Authentication
def authenticate(username, password):
    response = requests.post(
        f"{API_URL}/token",
        data={"username": username, "password": password}
    )
    if response.status_code == 200:
        global API_TOKEN
        API_TOKEN = response.json()["access_token"]
        return True
    return False

# API Calls
def call_api(endpoint, data):
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    response = requests.post(f"{API_URL}/{endpoint}", json=data, headers=headers)
    return response.json()

# Sidebar for navigation
st.sidebar.title("NyayaMitra")
page = st.sidebar.selectbox("Choose a feature", ["Login", "Chat", "Case Search", "Document Summarization", "News Verification", "Dashboard"])

# Login Page
if page == "Login":
    st.title("Welcome to NyayaMitra")
    st.write("The AI-powered legal assistant for the Indian judiciary system")
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if authenticate(username, password):
            st.success("Successfully logged in!")
            st.session_state.logged_in = True
        else:
            st.error("Invalid credentials")

# Chat Page
elif page == "Chat":
    st.title("Legal Assistant Chat")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # User input
    user_input = st.chat_input("Ask a legal question...")
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = call_api("api/chat", {"message": user_input})
                st.write(response["response"])
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response["response"]})

# Case Search Page
elif page == "Case Search":
    st.title("Case Precedent Search")
    
    query = st.text_area("Describe the legal issue or cite a case")
    max_results = st.slider("Maximum results", 1, 20, 5)
    
    if st.button("Search"):
        with st.spinner("Searching for relevant cases..."):
            results = call_api("api/search", {"query": query, "max_results": max_results})
            
            for i, result in enumerate(results["results"]):
                with st.expander(f"{i+1}. {result['title']}"):
                    st.write(f"**Court:** {result['court']}")
                    st.write(f"**Date:** {result['date']}")
                    st.write(f"**Relevance:** {result['relevance_score']:.2f}")
                    st.write("**Summary:**")
                    st.write(result['summary'])
                    st.write("**Key Extract:**")
                    st.write(result['extract'])
                    st.write(f"[View Full Judgment]({result['url']})")

# Document Summarization Page
elif page == "Document Summarization":
    st.title("Legal Document Summarization")
    
    uploaded_file = st.file_uploader("Upload a legal document", type=["pdf", "txt", "docx"])
    summary_length = st.select_slider("Summary Length", options=["Short", "Medium", "Long"])
    
    if uploaded_file and st.button("Summarize"):
        with st.spinner("Generating summary..."):
            # Read and encode file
            bytes_data = uploaded_file.getvalue()
            encoded = base64.b64encode(bytes_data).decode()
            
            # Call API
            summary_response = call_api("api/summarize", {
                "document": encoded,
                "length": summary_length.lower()
            })
            
            # Display results
            st.subheader("Document Summary")
            st.write(summary_response["summary"])
            
            # Option to download
            st.download_button(
                "Download Summary",
                summary_response["summary"],
                file_name="document_summary.txt"
            )

# News Verification Page
elif page == "News Verification":
    st.title("Legal News Verification")
    
    news_text = st.text_area("Paste the legal news text to verify")
    
    if st.button("Verify"):
        with st.spinner("Analyzing news..."):
            verification = call_api("api/verify", {"news_text": news_text})
            
            # Display result with color coding
            if verification["is_reliable"]:
                st.success("This news appears to be reliable.")
            else:
                st.error("This news appears to be misleading or false.")
            
            # Display confidence and explanation
            st.write(f"**Confidence:** {verification['confidence']:.2f}")
            st.write("**Analysis:**")
            st.write(verification["explanation"])

# Dashboard Page
elif page == "Dashboard":
    st.title("NyayaMitra Analytics Dashboard")
    
    # Sample data (in a real app, this would come from the API)
    query_data = pd.DataFrame({
        "Category": ["Constitutional", "Criminal", "Civil", "Tax", "Other"],
        "Count": [145, 210, 180, 75, 60]
    })
    
    # Create charts
    st.subheader("Query Categories")
    fig1 = px.pie(query_data, values="Count", names="Category")
    st.plotly_chart(fig1)