import streamlit as st
import requests

# Set up the page
st.set_page_config(page_title="AI Research Assistant", page_icon="🧠", layout="wide")
st.title("🧠 AI Research Assistant (CRAG)")

# The address of your FastAPI server
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

    # 2. Call the FastAPI backend with streaming enabled
    with st.chat_message("assistant"):
        payload = {
            "query": prompt, 
            "thread_id": "session_1"
        }
        
        try:
            # Send the request with stream=True
            response = requests.post(f"{API_URL}/chat", json=payload, stream=True)
            
            if response.status_code == 200:
                # Helper generator function to read chunks from the stream
                def response_generator():
                    # Read incoming chunks as decoded UTF-8 text
                    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                        if chunk:
                            yield chunk

                # st.write_stream consumes the generator and animates text naturally
                bot_answer = st.write_stream(response_generator())
                
                # Append the completed answer to history once streaming finishes
                st.session_state.messages.append({"role": "assistant", "content": bot_answer})
            else:
                st.error(f"❌ The server encountered an error: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Could not connect to the FastAPI server. Is it running?")