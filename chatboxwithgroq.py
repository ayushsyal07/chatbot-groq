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

# --- Chatbot (Groq + Mixtral) ---
def get_llm():
    return ChatGroq(model="mixtral-8x7b-32768", api_key=GROQ_API_KEY)

def run_chatbot(user_query):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("user", "{input}")
    ])
    chain = prompt | get_llm() | StrOutputParser()
    return chain.invoke({"input": user_query})

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="AI Assistant: Chat + PDF + Image", layout="wide")
    st.title("🤖 AI Assistant: Chatbot | PDF Summarizer | Image Identifier")

    tabs = st.tabs(["💬 Chatbot", "📄 PDF Summary", "🖼️ Image Identifier"])

    # --- Tab 1: Chatbot ---
    with tabs[0]:
        st.subheader("Ask Anything!")
        user_input = st.text_input("Type your message:")
        if user_input:
            with st.spinner("Generating response..."):
                response = run_chatbot(user_input)
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
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
            st.write("📝 Description:")
            st.success(caption)

if __name__ == "__main__":
    main()
