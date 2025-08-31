# streamlit_app.py
import streamlit as st
import PyPDF2
import tempfile
import os
import io
import threading
from pathlib import Path

# Import your backend converter (from your uploaded file)
from real_voice_cloning import TTSSTSConverter  # uses logic from your real_voice_cloning.py
# (See real_voice_cloning.py for implementation details). :contentReference[oaicite:2]{index=2}

st.set_page_config(page_title="PDF Text-to-Speech Reader", layout="centered")

# --- Helpers -----------------------------------------------------------------
def save_upload_to_temp(uploaded_file):
    suffix = Path(uploaded_file.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    return tmp.name

def extract_text_from_pdf_bytes(pdf_bytes: bytes):
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for p in reader.pages:
            text = p.extract_text() or ""
            pages.append(text)
        return "\n\n".join(pages), len(pages)
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF: {e}")

def stream_and_download_audio(file_path: str, button_label="Download WAV"):
    if not file_path or not os.path.exists(file_path):
        st.error("No audio file generated.")
        return
    with open(file_path, "rb") as f:
        audio_bytes = f.read()
    st.audio(audio_bytes, format="audio/wav")
    st.download_button(button_label, data=audio_bytes, file_name=os.path.basename(file_path), mime="audio/wav")

# --- Session state initialization -------------------------------------------
if "tts_converter" not in st.session_state:
    st.session_state.tts_converter = TTSSTSConverter()

if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""
if "pdf_pages" not in st.session_state:
    st.session_state.pdf_pages = 0
if "voice_sample_paths" not in st.session_state:
    st.session_state.voice_sample_paths = []
if "model_ready" not in st.session_state:
    st.session_state.model_ready = False

# --- Header ------------------------------------------------------------------
st.title("📘 PDF Text-to-Speech Reader (Streamlit)")
st.markdown("Upload a PDF, preview text, and convert it to speech. Optionally clone your voice with TTS+STS.")

# --- Requirements check (from your converter) --------------------------------
with st.expander("Requirements / Status", expanded=False):
    missing = st.session_state.tts_converter.check_requirements()
    if missing:
        st.error("Missing dependencies for full functionality.")
        st.write("Install at least one TTS engine (pyttsx3 or edge-tts). For full OpenVoice STS: torch, torchaudio, librosa, soundfile, transformers.")
        st.code("pip install pyttsx3 edge-tts PyPDF2 streamlit", language="bash")
        st.write("Optional (OpenVoice heavy):")
        st.code("pip install torch torchaudio librosa soundfile transformers huggingface_hub", language="bash")
    else:
        st.success("All basic requirements available.")

# --- PDF upload / preview ---------------------------------------------------
st.subheader("1) Load PDF")
uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
col1, col2 = st.columns([2,1])

if uploaded_pdf:
    try:
        pdf_bytes = uploaded_pdf.read()
        text, pages = extract_text_from_pdf_bytes(pdf_bytes)
        st.session_state.pdf_text = text
        st.session_state.pdf_pages = pages
        st.success(f"Loaded PDF — {pages} pages")
    except Exception as e:
        st.session_state.pdf_text = ""
        st.session_state.pdf_pages = 0
        st.error(f"Failed to load PDF: {e}")

preview = st.text_area("PDF Text Preview (editable)", value=st.session_state.pdf_text[:10000], height=300)
# Allow user to edit preview; the actual text used for TTS will be `preview` variable.

# --- Voice mode --------------------------------------------------------------
st.subheader("2) Voice mode & settings")
voice_mode = st.radio("Choose voice mode", ["System (pyttsx3)", "TTS + STS (Your voice)"], index=0)

rate = st.slider("Speech rate (WPM)", min_value=50, max_value=300, value=180)
volume = st.slider("Volume (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=0.95)
st.caption("System voice uses pyttsx3; if not available the app will try Edge TTS if installed.")

# --- TTS+STS voice sample management ----------------------------------------
if voice_mode == "TTS + STS (Your voice)":
    st.subheader("3) Voice samples (for cloning)")
    uploaded_samples = st.file_uploader("Upload voice sample files (10s - 5min recommended)", type=["wav","mp3","m4a","flac","aac"], accept_multiple_files=True)
    if uploaded_samples:
        added = 0
        for f in uploaded_samples:
            try:
                tmp_path = save_upload_to_temp(f)
                st.session_state.tts_converter.add_voice_sample(tmp_path)
                st.session_state.voice_sample_paths.append(tmp_path)
                added += 1
            except Exception as e:
                st.error(f"Failed to add {f.name}: {e}")
        if added:
            st.success(f"Added {added} sample(s).")

    if st.session_state.voice_sample_paths:
        st.write("Current samples:")
        for p in st.session_state.voice_sample_paths:
            st.write("-", os.path.basename(p))

    cola, colb = st.columns(2)
    with cola:
        if st.button("Clear all samples"):
            st.session_state.tts_converter.clear_voice_samples()
            st.session_state.voice_sample_paths = []
            st.session_state.model_ready = False
            st.success("Cleared samples.")
    with colb:
        if st.button("Prepare & Clone Voice (Setup TTS+STS)"):
            if not st.session_state.voice_sample_paths:
                st.error("Add at least one voice sample before setup.")
            else:
                # Run prepare / clone steps (blocking but shown with spinner)
                try:
                    with st.spinner("Preparing voice samples and loading STS model..."):
                        st.session_state.tts_converter.prepare_voice_samples(progress_callback=lambda m: st.info(m))
                        # if OpenVoice available attempt to load model & create embedding
                        if st.session_state.tts_converter.check_requirements() == []:
                            # still try to load openvoice model if dependencies exist
                            try:
                                st.session_state.tts_converter.load_openvoice_model(progress_callback=lambda m: st.info(m))
                                st.session_state.tts_converter.create_speaker_embedding(progress_callback=lambda m: st.info(m))
                            except Exception as e:
                                st.warning(f"OpenVoice model or embedding skipped/failed: {e}")
                        # finally load TTS+STS system (Edge / pyttsx3)
                        st.session_state.tts_converter.load_model(progress_callback=lambda m: st.info(m))
                        st.session_state.model_ready = True
                        st.success("TTS+STS setup done — you can now generate speech in your voice.")
                except Exception as e:
                    st.exception(e)

# --- Generate / Play / Save -------------------------------------------------
st.subheader("4) Generate speech")
text_to_speak = st.text_area("Text to speak (you can edit the preview above)", value=preview[:20000], height=200)

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("▶️ Generate & Play"):
        if not text_to_speak.strip():
            st.error("Please provide text to speak.")
        else:
            try:
                with st.spinner("Generating speech..."):
                    if voice_mode == "System (pyttsx3)":
                        # Use internal pyttsx3-based generator from the converter for uniform behavior
                        tmp = st.session_state.tts_converter._generate_base_audio_pyttsx3(text_to_speak)
                        if tmp and os.path.exists(tmp):
                            stream_and_download_audio(tmp, button_label="Download System TTS WAV")
                        else:
                            st.error("System TTS generation failed.")
                    else:
                        # TTS + STS: ensure model loaded
                        if not st.session_state.model_ready:
                            st.warning("TTS+STS not prepared — clicking Prepare & Clone Voice is recommended.")
                        converted = st.session_state.tts_converter.convert_text_to_speech(text_to_speak)
                        # convert_text_to_speech may return a path or a string indicating completion
                        if isinstance(converted, str) and os.path.exists(converted):
                            stream_and_download_audio(converted, button_label="Download Cloned WAV")
                        elif isinstance(converted, str) and not os.path.exists(converted):
                            # In some codepaths the converter plays audio itself (system player). Tell user.
                            st.info("Speech generation completed. Audio may have been opened via system player.")
                        else:
                            st.error("Failed to generate TTS+STS audio.")
            except Exception as e:
                st.exception(e)

with col2:
    if st.button("💾 Save audio (WAV)"):
        if not text_to_speak.strip():
            st.error("Please provide text to save.")
        else:
            try:
                with st.spinner("Generating & saving..."):
                    if voice_mode == "System (pyttsx3)":
                        tmp = st.session_state.tts_converter._generate_base_audio_pyttsx3(text_to_speak)
                        if tmp and os.path.exists(tmp):
                            with open(tmp, "rb") as f:
                                bytes_data = f.read()
                            st.download_button("Download WAV", data=bytes_data, file_name="speech.wav", mime="audio/wav")
                        else:
                            st.error("System TTS generation failed.")
                    else:
                        if not st.session_state.model_ready:
                            st.warning("TTS+STS not prepared — run Prepare & Clone Voice first.")
                        converted = st.session_state.tts_converter.convert_text_to_speech(text_to_speak)
                        if isinstance(converted, str) and os.path.exists(converted):
                            with open(converted, "rb") as f:
                                bytes_data = f.read()
                            st.download_button("Download cloned WAV", data=bytes_data, file_name="cloned_speech.wav", mime="audio/wav")
                        else:
                            st.info("Speech generated but file could not be retrieved. Check console output.")
            except Exception as e:
                st.exception(e)

with col3:
    if st.button("⏹ Stop playback (if any)"):
        try:
            st.session_state.tts_converter._play_openvoice_audio  # no-op to ensure converter present
            # Attempt to stop system TTS if running by creating an engine and stopping it
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.stop()
            except Exception:
                pass
            st.success("Stop attempted (if system player was used).")
        except Exception as e:
            st.warning("Stop may not be supported in this environment.")

# --- Footer / tips ----------------------------------------------------------
st.markdown("---")
st.caption("This Streamlit UI uses your project's TTS+STS logic (TTSSTSConverter) defined in `real_voice_cloning.py`. :contentReference[oaicite:3]{index=3}")
st.markdown("**Notes / Troubleshooting**:")
st.markdown(
    "- If `pyttsx3` is not working on your system, install it: `pip install pyttsx3`.\n"
    "- Edge TTS requires network access and may produce MP3 that will be converted to WAV (ffmpeg helps).\n"
    "- For full OpenVoice STS (deep voice cloning) install the heavy dependencies listed in the README file."
)
