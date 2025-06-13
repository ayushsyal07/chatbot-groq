import streamlit as st
import requests
import os
from dotenv import load_dotenv
from summary-logic.yt-pdf import extract_text_from_pdf, get_youtube_transcript

# Load env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Groq Chatbot Plus", layout="wide")
st.title("💬 Enhanced Groq Chatbot")

# Sidebar
st.sidebar.header("🛠️ Model Configuration")
model = st.sidebar.selectbox("Select Model", ["llama3-8b-8192", "mistral-saba-24b", "gemma2-9b-it"])
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
max_tokens = st.sidebar.slider("Max Tokens", 50, 1024, 300, 50)

# Call Groq API
def ask_groq(prompt, model, temperature, max_tokens):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    res = requests.post(url, headers=headers, json=payload)
    if res.status_code == 200:
        return res.json()["choices"][0]["message"]["content"]
    return f"❌ Error: {res.status_code} - {res.text}"

# Tab Layout
tab1, tab2, tab3 = st.tabs(["🧠 Ask Chatbot", "📄 Summarize Document", "📺 Summarize YouTube"])

# 1. Chat Tab
with tab1:
    st.subheader("Ask Anything")
    user_input = st.text_input("You:")
    if user_input:
        with st.spinner("Thinking..."):
            answer = ask_groq(user_input, model, temperature, max_tokens)
            st.markdown(f"**🤖 Answer:** {answer}")

# 2. Document Summary Tab
with tab2:
    st.subheader("Upload and Summarize PDF")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file:
        with st.spinner("Reading file..."):
            text = extract_text_from_pdf(uploaded_file)
            summary = ask_groq(f"Summarize this document:\n{text[:4000]}", model, temperature, max_tokens)
            st.markdown("**📄 Summary:**")
            st.write(summary)

# 3. YouTube Summary Tab
with tab3:
    st.subheader("Paste YouTube URL")
    yt_url = st.text_input("YouTube Link:")
    if yt_url:
        with st.spinner("Fetching transcript..."):
            transcript = get_youtube_transcript(yt_url)
            if "❌" not in transcript:
                summary = ask_groq(f"Summarize this YouTube transcript:\n{transcript[:4000]}", model, temperature, max_tokens)
                st.markdown("**📺 Summary:**")
                st.write(summary)
            else:
                st.error(transcript)
