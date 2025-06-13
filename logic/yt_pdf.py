from PIL import Image
import fitz  # PyMuPDF for PDF extraction
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# PDF extraction logic
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Image captioning logic using BLIP
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
