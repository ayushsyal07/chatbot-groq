import fitz  # PyMuPDF
from pytube import YouTube
import re

# PDF Text Extraction
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# YouTube ID Extractor
def extract_video_id(url):
    patterns = [r"v=([a-zA-Z0-9_-]{11})", r"youtu\.be/([a-zA-Z0-9_-]{11})"]
    for p in patterns:
        match = re.search(p, url)
        if match:
            return match.group(1)
    return None

# YouTube Transcript Extractor (from captions)
def get_youtube_transcript(url):
    try:
        yt = YouTube(url)
        caption = yt.captions.get_by_language_code("en")
        if not caption:
            return "❌ No English subtitles found."
        srt = caption.generate_srt_captions()
        return srt_to_text(srt)
    except Exception as e:
        return f"❌ Could not retrieve captions: {str(e)}"

# Clean up SRT format
def srt_to_text(srt):
    lines = srt.strip().split('\n')
    text_lines = []
    for line in lines:
        if re.match(r'^\d+$', line): continue
        if re.match(r'\d{2}:\d{2}:\d{2},\d{3}', line): continue
        text_lines.append(line)
    return ' '.join(text_lines)
