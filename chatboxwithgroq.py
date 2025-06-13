import os
import re
import torch
import tempfile
from PIL import Image
import fitz  # PyMuPDF
import streamlit as st
import asyncio
import nest_asyncio
nest_asyncio.apply()
from dotenv import load_dotenv
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- PDF Extraction ---
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# --- Image Captioning ---
def describe_image(image):
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

        img = Image.open(image).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=20)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        return f"⚠️ Error describing image: {e}"

# --- Chatbot (Groq + Mixtral/LLaMA) ---
def get_llm(temperature=0.7, max_tokens=512):
    return ChatGroq(model="llama3-70b-8192", api_key=GROQ_API_KEY, temperature=temperature, max_tokens=max_tokens)

def run_chatbot(user_query, temperature=0.7, max_tokens=512):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("user", "{input}")
    ])
    chain = prompt | get_llm(temperature, max_tokens) | StrOutputParser()
    return chain.invoke({"input": user_query})

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="AI Assistant: Chat + PDF + Image", layout="wide")
    st.title("🤖 AI Assistant: Chatbot | PDF Summarizer | Image Identifier")

    # Sidebar controls
    st.sidebar.title("⚙️ Chat Settings")
    temperature = st.sidebar.slider("Temperature (Creativity)", 0.0, 1.0, 0.7, step=0.1)
    max_tokens = st.sidebar.slider("Max Tokens (Response Length)", 64, 2048, 512, step=64)

    st.sidebar.markdown("""
    **What These Do:**
    - 🔥 **Temperature** controls randomness in replies.  
      - Low = factual  
      - High = creative  
    - ✏️ **Max Tokens** controls how long the model's response can be.
    """)

    tabs = st.tabs(["💬 Chatbot", "📄 PDF Summary", "🖼️ Image Identifier"])

    # --- Tab 1: Chatbot ---
    with tabs[0]:
        st.subheader("Ask Anything!")
        user_input = st.text_input("Type your message:")
        if user_input:
            with st.spinner("Generating response..."):
                response = run_chatbot(user_input, temperature, max_tokens)
            st.write("🧠 Response:")
            st.markdown(response)

    # --- Tab 2: PDF Summarizer ---
    with tabs[1]:
        st.subheader("Upload a PDF")
        pdf_file = st.file_uploader("Choose a PDF file", type="pdf")
        if pdf_file is not None:
            with st.spinner("Extracting text from PDF..."):
                extracted_text = extract_text_from_pdf(pdf_file)
            st.write("📜 Extracted Text:")
            st.text_area("Content", extracted_text, height=300)

    # --- Tab 3: Image Caption Generator ---
    with tabs[2]:
        st.subheader("Upload an Image")
        uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            with st.spinner("Describing image..."):
                caption = describe_image(uploaded_image)
            st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
            st.write("📝 Description:")
            st.success(caption)
