# 📚 Interactive PDF Reader with TTS + STS Voice Conversion

**Advanced PDF Reader with Real Voice Cloning using OpenVoice**

A sophisticated GenAI project that converts PDF documents to speech using **YOUR OWN VOICE** through advanced TTS + STS technology.

## Features

- **PDF Support**: Reads text from PDF documents
- **Multiple Voice Options**:
  - System voice (offline using `pyttsx3`)
  - Edge TTS (high-quality neural voices)
  - **TTS + STS (Your Voice)** - Real voice cloning with OpenVoice
- **Voice Controls**: Adjustable speech rate and volume (for system voice)
- **Resume Functionality**: Continue from where you stopped
- **Audio Export**: Save generated speech as audio files
- **GUI Interface**: Simple and intuitive graphical interface
- **Text Preview**: Preview PDF content before conversion
- **Progress Tracking**: See reading progress and word count

## Installation

1. **Clone or download this project**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Additional system requirements**:
   - For Windows: No additional setup needed
   - For Linux: You might need to install espeak
     ```bash
     sudo apt-get install espeak espeak-data libespeak1 libespeak-dev
     ```
   - For macOS: No additional setup needed

## Usage

### GUI Application

Run the interactive GUI:
```bash
python interactive_reader.py
```

**How to use:**
1. Click "Browse" to select your PDF file
2. Choose voice mode:
   - **System Voice**: Use computer's built-in voice
   - **My Voice**: Use your own cloned voice (requires setup)
3. For custom voice: Click "Setup My Voice" and follow the guide
4. Click "Load PDF" to load and preview the text
5. Click "Start Speaking" to begin text-to-speech
6. Use "Stop Speaking" to pause (resumes from same position)
7. Use "Restart" to start over from the beginning
8. Use "Save Audio" to export as audio file

## Supported File Format

- **PDF**: `.pdf` files (text-based PDFs work best)
  - Note: Image-based or scanned PDFs may not work properly
  - For best results, use PDFs with selectable text

## Voice Settings

- **Speech Rate**: 50-300 words per minute (default: 150)
- **Volume**: 0-100% (default: 90%)
- **Voice**: Uses system default voice (Windows SAPI, macOS Speech, Linux espeak)

## Custom Voice Setup

To use your own voice instead of the computer voice:

1. **Get an API key** from [ElevenLabs](https://elevenlabs.io) (recommended) or [OpenAI](https://platform.openai.com)
2. **Record voice samples** (3-5 audio files, 5 seconds to 5 minutes each)
3. **Run the app** and click "Setup My Voice"
4. **Follow the setup wizard** to create your voice clone

📖 **Detailed instructions**: See [VOICE_SETUP_GUIDE.md](VOICE_SETUP_GUIDE.md)

## Project Structure

```
├── pdf_reader.py          # Core PDF reading and TTS functionality
├── interactive_reader.py  # GUI application
├── voice_cloning.py       # Voice cloning functionality
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── VOICE_SETUP_GUIDE.md  # Detailed voice setup instructions
```

## Example Usage

1. **Basic PDF to speech conversion**:
   ```bash
   python interactive_reader.py
   ```
   Then select your PDF file and click "Start Speaking"

2. **Create an audiobook file**:
   - Load your PDF in the GUI
   - Click "Save Audio" to export as WAV file
   - Choose your desired location and filename

## Troubleshooting

### Common Issues

1. **"No module named 'pyttsx3'"**:
   - Run: `pip install -r requirements.txt`

2. **PDF text extraction issues**:
   - Ensure your PDF contains selectable text (not scanned images)
   - Try converting scanned PDFs to text first using OCR tools

3. **Audio playback issues**:
   - Make sure your system audio is working
   - Try different TTS engines

4. **Permission errors**:
   - Run terminal/command prompt as administrator (Windows)
   - Check file permissions for the book file

### Performance Tips

- For large books, consider breaking them into chapters
- Use pyttsx3 for faster processing
- Use Google TTS for better voice quality
- Save audio files for repeated listening

## Extending the Project

You can enhance this project by adding:

- Chapter detection and navigation
- Multiple voice options
- Reading speed controls
- Bookmark functionality
- Web interface
- Mobile app version
- OCR support for scanned PDFs

## Educational Value

This project demonstrates several GenAI and programming concepts:

- **Text Processing**: Extracting and cleaning text from various formats
- **Speech Synthesis**: Converting text to natural-sounding speech
- **File I/O**: Reading different file formats
- **GUI Development**: Creating user-friendly interfaces
- **API Integration**: Using cloud-based AI services
- **Error Handling**: Robust error management
- **Threading**: Non-blocking UI operations

## License

This project is for educational purposes. Feel free to modify and extend it for your learning!

## Contributing

This is a learning project, but suggestions and improvements are welcome!
