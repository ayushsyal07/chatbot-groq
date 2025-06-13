import fitz  # PyMuPDF
from youtube_transcript_api import YouTubeTranscriptApi
import re

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_video_id(url):
    # Handles https://www.youtube.com/watch?v=abc123 or youtu.be/abc123
    patterns = [r"v=([a-zA-Z0-9_-]{11})", r"youtu\.be/([a-zA-Z0-9_-]{11})"]
    for p in patterns:
        match = re.search(p, url)
        if match:
            return match.group(1)
    return None

def get_youtube_transcript(video_url):
    video_id = extract_video_id(video_url)
    if not video_id:
        return "❌ Invalid YouTube URL"
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return " ".join([entry['text'] for entry in transcript])
