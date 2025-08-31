import streamlit as st
import PyPDF2
import tempfile
import os
import io
import re
import warnings
from pathlib import Path
from real_voice_cloning import TTSSTSConverter

# --- Suppress Torchaudio Warning ---
warnings.filterwarnings("ignore", message=".*Torchaudio's I/O functions.*")

# --- Page Config ---
st.set_page_config(page_title="PDF to Speech", page_icon="📘", layout="wide")

# --- Custom Styling ---
st.markdown("""
<style>
.main { background: linear-gradient(to right, #f0f4f8, #e8f0fe); padding: 1rem; border-radius: 12px; }
.stButton>button { background-color: #4A90E2; color: white; border-radius: 10px; padding: 0.6em 1.2em; font-weight: bold; }
.stButton>button:hover { background-color: #357ABD; color: #ffffff; }
</style>
""", unsafe_allow_html=True)

# --- Helpers ---
def clean_pdf_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n{2,}', '\n', text)
    return text.strip()

def save_upload_to_temp(uploaded_file):
    suffix = Path(uploaded_file.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    return tmp.name

def extract_text_from_pdf(pdf_bytes: bytes):
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        for p in reader.pages:
            text += (p.extract_text() or "") + "\n"
        return clean_pdf_text(text), len(reader.pages)
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF: {e}")

def stream_and_download_audio(file_path: str, label="Download WAV"):
    if not file_path or not os.path.exists(file_path):
        st.error("No audio file generated.")
        return
    with open(file_path, "rb") as f:
        audio_bytes = f.read()
    st.audio(audio_bytes, format="audio/wav")
    st.download_button(label, data=audio_bytes, file_name=os.path.basename(file_path), mime="audio/wav")

# --- Session State ---
if "tts_converter" not in st.session_state:
    st.session_state.tts_converter = TTSSTSConverter()
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""
if "voice_samples" not in st.session_state:
    st.session_state.voice_samples = []
if "model_ready" not in st.session_state:
    st.session_state.model_ready = False

# --- Sidebar ---
st.sidebar.title("Settings")
uploaded_pdf = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
voice_mode = st.sidebar.radio("Voice mode", ["System (pyttsx3)", "TTS + STS (Your voice)"])
rate = st.sidebar.slider("Speech rate (WPM)", 100, 250, 170)
volume = st.sidebar.slider("Volume", 0.0, 1.0, 0.9)

if voice_mode == "TTS + STS (Your voice)":
    st.sidebar.subheader("Voice Samples")
    samples = st.sidebar.file_uploader("Upload Samples", type=["wav", "mp3", "m4a", "flac"], accept_multiple_files=True)
    if samples:
        for s in samples:
            path = save_upload_to_temp(s)
            st.session_state.tts_converter.add_voice_sample(path)
            st.session_state.voice_samples.append(path)
        st.sidebar.success(f"Added {len(samples)} samples")
    if st.sidebar.button("Prepare Voice Model"):
        if not st.session_state.voice_samples:
            st.sidebar.error("Upload at least one sample.")
        else:
            try:
                st.sidebar.info("Preparing voice model...")
                st.session_state.tts_converter.prepare_voice_samples()
                st.session_state.tts_converter.load_model()
                st.session_state.model_ready = True
                st.sidebar.success("Voice model ready!")
            except Exception as e:
                st.sidebar.error(f"Setup failed: {e}")

# --- Main ---
st.title("📘 PDF Text-to-Speech Reader")
if uploaded_pdf:
    try:
        pdf_bytes = uploaded_pdf.read()
        text, pages = extract_text_from_pdf(pdf_bytes)
        st.session_state.pdf_text = text
        st.success(f"Loaded PDF ({pages} pages)")
    except Exception as e:
        st.error(str(e))

preview_text = st.text_area("Preview & Edit Text", value=st.session_state.pdf_text, height=300)

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("▶ Generate Speech"):
        if not preview_text.strip():
            st.warning("Enter or upload text first.")
        else:
            st.info("Generating speech...")
            try:
                if voice_mode == "System (pyttsx3)":
                    path = st.session_state.tts_converter._generate_base_audio_pyttsx3(preview_text)
                    if path:
                        stream_and_download_audio(path)
                else:
                    if not st.session_state.model_ready:
                        st.warning("Prepare voice model first.")
                    else:
                        path = st.session_state.tts_converter.convert_text_to_speech(preview_text)
                        if isinstance(path, str) and os.path.exists(path):
                            stream_and_download_audio(path)
                        else:
                            st.warning("Speech generated but no file available.")
            except Exception as e:
                st.error(f"Error: {e}")

with col2:
    if st.button("💾 Save Speech"):
        if not preview_text.strip():
            st.warning("Enter or upload text first.")
        else:
            try:
                path = (
                    st.session_state.tts_converter._generate_base_audio_pyttsx3(preview_text)
                    if voice_mode == "System (pyttsx3)"
                    else st.session_state.tts_converter.convert_text_to_speech(preview_text)
                )
                if path and os.path.exists(path):
                    with open(path, "rb") as f:
                        st.download_button("Download", f.read(), file_name="speech.wav", mime="audio/wav")
                else:
                    st.error("Failed to save audio.")
            except Exception as e:
                st.error(f"Save failed: {e}")

st.markdown("---")
st.caption("Styled PDF to Speech app using Streamlit • Supports basic and cloned voices")
