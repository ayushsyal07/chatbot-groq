import fitz  # PyMuPDF
import re
from pytube import YouTube

# PDF extraction
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# YouTube ID extractor
def extract_video_id(url):
    patterns = [r"v=([a-zA-Z0-9_-]{11})", r"youtu\.be/([a-zA-Z0-9_-]{11})"]
    for p in patterns:
        match = re.search(p, url)
        if match:
            return match.group(1)
    return None

# Captions extraction from YouTube using pytube
def get_youtube_captions(video_url):
    try:
        yt = YouTube(video_url)
        caption = yt.captions.get_by_language_code("en")  # English
        if not caption:
            return "❌ No English captions available for this video."
        srt_captions = caption.generate_srt_captions()
        # Remove timestamps and numbers from SRT
        lines = srt_captions.split('\n')
        text_lines = []
        for line in lines:
            if line.strip().isdigit() or '-->' in line:
                continue
            text_lines.append(line.strip())
        return " ".join(text_lines)
    except Exception as e:
        return f"❌ Error fetching captions: {e}"
