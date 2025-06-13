import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Set page title
st.title("💬 Q&A Chatbot using Groq LLMs")

# Sidebar configuration
st.sidebar.header("🛠️ Model Configuration")

# Select model
model = st.sidebar.selectbox(
    "Select Groq Model",
    options=["llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"],
    index=0
)

# Temperature slider
temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.05,
)
st.sidebar.caption("🔹 Controls creativity:\n- Low = focused\n- High = creative")

# Max tokens slider
max_tokens = st.sidebar.slider(
    "Max Tokens",
    min_value=50,
    max_value=1024,
    value=300,
    step=50,
)
st.sidebar.caption("🔸 Limits response length")

# Function to call Groq API
def generate_response_groq(question, model, temperature, max_tokens):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"❌ Error: {response.status_code} - {response.text}"

# User input
st.write("Type your question below:")
user_question = st.text_input("You:")

# Show response
if user_question:
    with st.spinner("Thinking..."):
        answer = generate_response_groq(user_question, model, temperature, max_tokens)
        st.markdown(f"**🤖 Answer:** {answer}")
