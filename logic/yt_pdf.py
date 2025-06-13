from pytube import YouTube
import re
import fitz  # PyMuPDF for PDF extraction
import tempfile
import os

# PDF extraction logic
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# YouTube video ID extractor
def extract_video_id(url):
    patterns = [r"v=([a-zA-Z0-9_-]{11})", r"youtu\.be/([a-zA-Z0-9_-]{11})"]
    for p in patterns:
        match = re.search(p, url)
        if match:
            return match.group(1)
    return None

# Extract captions with fallback to Whisper
def get_youtube_captions(video_url, preferred_lang='en'):
    try:
        yt = YouTube(video_url)
        captions = yt.captions

        if captions:
            caption = captions.get_by_language_code(preferred_lang)
            if not caption and captions.all():
                caption = next(iter(captions.all()))

            if caption:
                srt_captions = caption.generate_srt_captions()
                lines = srt_captions.split('\n')
                clean_lines = [
                    line.strip() for line in lines
                    if line.strip() and not line.strip().isdigit() and '-->' not in line
                ]
                return " ".join(clean_lines)

        # If no captions, fallback to Whisper
        return transcribe_with_whisper(yt)

    except Exception as e:
        return f"❌ Error fetching captions: {str(e)}"

# Whisper ASR transcription
def transcribe_with_whisper(yt):
    try:
        import whisper  # 👈 Lazy import here

        audio_stream = yt.streams.filter(only_audio=True).first()
        if not audio_stream:
            return "❌ No audio stream available for transcription."

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            audio_path = tmp_file.name
            audio_stream.download(filename=audio_path)

        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        os.remove(audio_path)

        return result["text"]

    except Exception as e:
        return f"❌ Whisper transcription failed: {str(e)}"

# List available caption languages
def get_available_caption_languages(video_url):
    try:
        yt = YouTube(video_url)
        return [c.code for c in yt.captions.all()]
    except Exception as e:
        return []
