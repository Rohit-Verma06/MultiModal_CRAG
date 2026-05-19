import streamlit as st
import requests

# Set up the page
st.set_page_config(page_title="AI Research Assistant", page_icon="🧠", layout="wide")
st.title("🧠 AI Research Assistant (CRAG)")

# The address of your new FastAPI server!
API_URL = "http://127.0.0.1:8000"

# --- SIDEBAR: FILE UPLOADER ---
with st.sidebar:
    st.header("📄 Upload Documents")
    st.write("Upload PDFs or Images to build your custom knowledge base.")
    
    uploaded_files = st.file_uploader("Upload Files", type=["pdf" ,"jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()

    if uploaded_files:
        for file in uploaded_files:
            # Only process if we haven't sent this file to the server yet
            if file.name not in st.session_state.processed_files:
                with st.spinner(f"Sending {file.name} to server..."):
                    
                    # Package the file to send over HTTP
                    files = {"file": (file.name, file.getvalue(), file.type)}
                    
                    # POST the file to your FastAPI /upload endpoint
                    response = requests.post(f"{API_URL}/upload", files=files)
                    
                    if response.status_code == 200:
                        st.session_state.processed_files.add(file.name)
                        st.success(f"✅ {file.name} added to Knowledge Base!")
                    else:
                        st.error(f"❌ Failed to upload {file.name}. Server responded with error.")
            else:
                st.success(f"✅ {file.name} is already in the Knowledge Base!")

# --- MAIN CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history on screen
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input field
if prompt := st.chat_input("Ask a question about your documents, or search the web..."):
    
    # 1. Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Call the FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Server is thinking..."):
            
            # Package the query into a JSON dictionary
            payload = {
                "query": prompt, 
                "thread_id": "session_1"
            }
            
            try:
                # POST the question to your FastAPI /chat endpoint
                response = requests.post(f"{API_URL}/chat", json=payload)
                
                if response.status_code == 200:
                    # Extract the answer from the FastAPI JSON response
                    # (This matches the {"answer": output.get("ans")} you just wrote!)
                    bot_answer = response.json().get("answer", "Error: No answer key found.")
                    
                    st.markdown(bot_answer)
                    st.session_state.messages.append({"role": "assistant", "content": bot_answer})
                else:
                    st.error(f"❌ The server encountered an error: {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Could not connect to the FastAPI server. Is it running?") 