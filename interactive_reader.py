"""
Interactive PDF Text-to-Speech Reader
Simple GUI for PDF documents only
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import PyPDF2
import os
from real_voice_cloning import TTSSTSConverter, TTSSTSSetupGUI


class PDFReader:
    """PDF reader with TTS functionality"""

    def __init__(self):
        self.pdf_path = None
        self.pages = []
        self.current_page = 0
        self.text_content = ""
        self.current_position = 0
        self.speech_rate = 180
        self.volume = 0.9
        self.voice_mode = "system"
        self.tts_sts_converter = None

        # Initialize TTS+STS converter
        try:
            self.tts_sts_converter = TTSSTSConverter()
        except:
            self.tts_sts_converter = None

    def load_pdf(self, file_path):
        """Load PDF file and extract text"""
        try:
            self.pdf_path = file_path
            self.pages = []

            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)

                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    self.pages.append(text)

            # Set text content for TTS
            self.text_content = self.get_all_text()
            self.current_position = 0

            print(f"✅ Loaded PDF with {len(self.pages)} pages")
            return True

        except Exception as e:
            print(f"❌ Error loading PDF: {e}")
            return False

    def get_page_text(self, page_num):
        """Get text from specific page"""
        if 0 <= page_num < len(self.pages):
            return self.pages[page_num]
        return ""

    def get_total_pages(self):
        """Get total number of pages"""
        return len(self.pages)

    def get_all_text(self):
        """Get all text from PDF"""
        return "\n\n".join(self.pages)

    def get_text_preview(self, max_chars=1000):
        """Get preview of PDF text (first max_chars characters)"""
        all_text = self.get_all_text()
        if len(all_text) <= max_chars:
            return all_text
        else:
            return all_text[:max_chars] + "..."

    def set_speech_rate(self, rate):
        """Set speech rate"""
        self.speech_rate = rate

    def set_volume(self, volume):
        """Set volume"""
        self.volume = volume

    def set_voice_mode(self, mode):
        """Set voice mode"""
        self.voice_mode = mode

    def get_progress_info(self):
        """Get speaking progress information"""
        if not self.text_content:
            return {"percentage": 0, "position": 0, "total": 0}

        words = self.text_content.split()
        total_words = len(words)
        current_word = min(self.current_position, total_words)
        percentage = int((current_word / total_words) * 100) if total_words > 0 else 0

        return {
            "percentage": percentage,
            "position": current_word,
            "total": total_words
        }

    def speak_text(self, resume=False, save_audio=False, output_file=None):
        """Speak the PDF text"""
        if not self.text_content:
            return

        try:
            if self.voice_mode == "tts_sts" and self.tts_sts_converter and self.tts_sts_converter.is_model_loaded:
                # Use TTS+STS voice cloning
                words = self.text_content.split()
                if resume:
                    text_to_speak = " ".join(words[self.current_position:])
                else:
                    text_to_speak = self.text_content
                    self.current_position = 0

                if text_to_speak.strip():
                    result = self.tts_sts_converter.convert_text_to_speech(text_to_speak)
                    if result:
                        self.current_position = len(words)  # Mark as completed

            else:
                # Use system TTS
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty('rate', self.speech_rate)
                engine.setProperty('volume', self.volume)

                words = self.text_content.split()
                if resume:
                    text_to_speak = " ".join(words[self.current_position:])
                else:
                    text_to_speak = self.text_content
                    self.current_position = 0

                if save_audio and output_file:
                    engine.save_to_file(text_to_speak, output_file)
                    engine.runAndWait()
                else:
                    engine.say(text_to_speak)
                    engine.runAndWait()

                self.current_position = len(words)  # Mark as completed

        except Exception as e:
            print(f"❌ Speaking error: {e}")
            raise

    def stop_speaking(self):
        """Stop speaking"""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.stop()
        except:
            pass


class PDFReaderGUI:
    """GUI interface for the PDF reader"""

    def __init__(self, root):
        self.root = root
        self.root.title("PDF Text-to-Speech Reader")
        self.root.geometry("700x600")
        
        self.reader = None
        self.is_speaking = False
        self.speak_thread = None
        self.is_paused = False
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="PDF Text-to-Speech Reader",
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # File selection
        ttk.Label(main_frame, text="Select PDF File:").grid(row=1, column=0, sticky=tk.W, pady=5)
        
        file_frame = ttk.Frame(main_frame)
        file_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.file_path_var = tk.StringVar()
        self.file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, width=50)
        self.file_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        ttk.Button(file_frame, text="Browse", command=self.browse_file).grid(row=0, column=1)

        # Voice settings
        settings_frame = ttk.LabelFrame(main_frame, text="Voice Settings", padding="10")
        settings_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(20, 10))

        # Voice mode selection
        voice_mode_frame = ttk.Frame(settings_frame)
        voice_mode_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(voice_mode_frame, text="Voice Mode:").grid(row=0, column=0, sticky=tk.W)
        self.voice_mode_var = tk.StringVar(value="system")

        system_radio = ttk.Radiobutton(voice_mode_frame, text="System Voice",
                                     variable=self.voice_mode_var, value="system",
                                     command=self.update_voice_mode)
        system_radio.grid(row=0, column=1, padx=(10, 0))

        tts_sts_radio = ttk.Radiobutton(voice_mode_frame, text="TTS + STS (Your Voice)",
                                       variable=self.voice_mode_var, value="tts_sts",
                                       command=self.update_voice_mode)
        tts_sts_radio.grid(row=0, column=2, padx=(10, 0))

        self.setup_voice_button = ttk.Button(voice_mode_frame, text="Setup TTS+STS",
                                           command=self.setup_tts_sts_voice, state=tk.NORMAL)
        self.setup_voice_button.grid(row=0, column=3, padx=(10, 0))

        # Speech rate (only for system voice)
        ttk.Label(settings_frame, text="Speech Rate:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.rate_var = tk.IntVar(value=150)
        self.rate_scale = ttk.Scale(settings_frame, from_=50, to=300, variable=self.rate_var,
                              orient=tk.HORIZONTAL, length=200, command=self.update_speech_rate)
        self.rate_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(10, 0))
        self.rate_label = ttk.Label(settings_frame, text="150 WPM")
        self.rate_label.grid(row=1, column=2, padx=(10, 0))

        # Volume (only for system voice)
        ttk.Label(settings_frame, text="Volume:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.volume_var = tk.DoubleVar(value=1.0)
        self.volume_scale = ttk.Scale(settings_frame, from_=0.0, to=1.0, variable=self.volume_var,
                                orient=tk.HORIZONTAL, length=200, command=self.update_volume)
        self.volume_scale.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(10, 0))
        self.volume_label = ttk.Label(settings_frame, text="100%")
        self.volume_label.grid(row=2, column=2, padx=(10, 0))

        # Voice status
        self.voice_status_var = tk.StringVar(value="Using system voice")
        voice_status_label = ttk.Label(settings_frame, textvariable=self.voice_status_var,
                                     font=("Arial", 9), foreground="blue")
        voice_status_label.grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=(10, 0))
        
        # Text preview
        ttk.Label(main_frame, text="PDF Text Preview:").grid(row=4, column=0, sticky=tk.W, pady=(20, 5))

        preview_frame = ttk.Frame(main_frame)
        preview_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.preview_text = tk.Text(preview_frame, height=10, width=70, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=self.preview_text.yview)
        self.preview_text.configure(yscrollcommand=scrollbar.set)
        
        self.preview_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=6, column=0, columnspan=2, pady=20)

        self.load_button = ttk.Button(button_frame, text="Load PDF", command=self.load_pdf)
        self.load_button.grid(row=0, column=0, padx=5)

        self.speak_button = ttk.Button(button_frame, text="Start Speaking",
                                     command=self.toggle_speaking, state=tk.DISABLED)
        self.speak_button.grid(row=0, column=1, padx=5)

        self.restart_button = ttk.Button(button_frame, text="Restart",
                                       command=self.restart_speaking, state=tk.DISABLED)
        self.restart_button.grid(row=0, column=2, padx=5)

        self.save_button = ttk.Button(button_frame, text="Save Audio",
                                    command=self.save_audio, state=tk.DISABLED)
        self.save_button.grid(row=0, column=3, padx=5)

        # Progress info
        self.progress_var = tk.StringVar(value="Progress: 0%")
        progress_label = ttk.Label(button_frame, textvariable=self.progress_var)
        progress_label.grid(row=1, column=0, columnspan=4, pady=(10, 0))

        # Status bar
        self.status_var = tk.StringVar(value="Ready - Please select a PDF file")
        status_label = ttk.Label(main_frame, textvariable=self.status_var,
                               relief=tk.SUNKEN, anchor=tk.W)
        status_label.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(5, weight=1)
        file_frame.columnconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        settings_frame.columnconfigure(1, weight=1)

    def update_speech_rate(self, value):
        """Update speech rate display and setting"""
        rate = int(float(value))
        self.rate_label.config(text=f"{rate} WPM")
        if self.reader:
            self.reader.set_speech_rate(rate)

    def update_volume(self, value):
        """Update volume display and setting"""
        volume = float(value)
        self.volume_label.config(text=f"{int(volume * 100)}%")
        if self.reader:
            self.reader.set_volume(volume)

    def update_voice_mode(self):
        """Update voice mode and UI accordingly"""
        mode = self.voice_mode_var.get()

        if self.reader:
            self.reader.set_voice_mode(mode)

        # Enable/disable system voice controls
        if mode == "system":
            self.rate_scale.config(state=tk.NORMAL)
            self.volume_scale.config(state=tk.NORMAL)
            self.voice_status_var.set("Using system voice")
        elif mode == "tts_sts":
            self.rate_scale.config(state=tk.DISABLED)
            self.volume_scale.config(state=tk.DISABLED)
            if self.reader and self.reader.tts_sts_converter and self.reader.tts_sts_converter.is_model_loaded:
                sample_count = len(self.reader.tts_sts_converter.voice_samples)
                self.voice_status_var.set(f"🔄 TTS+STS: Your voice ({sample_count} samples)")
            else:
                self.voice_status_var.set("🔄 TTS+STS not setup - click 'Setup TTS+STS'")

    def setup_tts_sts_voice(self):
        """Open TTS+STS setup dialog"""
        if not self.reader:
            messagebox.showwarning("Warning", "Please load a PDF first, then setup TTS+STS.")
            return

        try:
            # Check if TTS+STS is available
            if not self.reader.tts_sts_converter:
                messagebox.showerror("Error",
                    "TTS+STS system not available.\n\n"
                    "Please install: pip install edge-tts pyttsx3\n"
                    "Then restart the application.")
                return

            # Open TTS+STS setup
            from real_voice_cloning import TTSSTSSetupGUI
            setup_gui = TTSSTSSetupGUI(self.root, self.reader.tts_sts_converter)
            setup_gui.show_setup_dialog()

            # Update status after dialog closes
            self.root.after(1000, self.update_voice_mode)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to open TTS+STS setup: {e}")

    def show_free_voice_selection(self):
        """Show simple voice selection dialog for free voices"""
        # Create a simple dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("🆓 Free Voice Selection")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()

        # Main frame
        main_frame = tk.Frame(dialog, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = tk.Label(main_frame, text="🆓 Select Free Voice",
                              font=("Arial", 14, "bold"), fg="green")
        title_label.pack(pady=(0, 20))

        # Info
        info_label = tk.Label(main_frame,
            text="Choose a high-quality AI voice from Microsoft Edge TTS.\n"
                 "These voices are completely free and work offline after setup.",
            justify=tk.CENTER, wraplength=350)
        info_label.pack(pady=(0, 20))

        # Voice selection
        tk.Label(main_frame, text="Select Voice:", font=("Arial", 10, "bold")).pack(anchor=tk.W)

        voice_var = tk.StringVar(value="en-US-AriaNeural (Female, Natural)")
        voices = [
            "en-US-AriaNeural (Female, Natural)",
            "en-US-DavisNeural (Male, Natural)",
            "en-US-JennyNeural (Female, Natural)",
            "en-US-JasonNeural (Male, Natural)",
            "en-GB-SoniaNeural (Female, British)",
            "en-GB-RyanNeural (Male, British)",
            "en-AU-NatashaNeural (Female, Australian)",
            "en-AU-WilliamNeural (Male, Australian)"
        ]

        for voice in voices:
            tk.Radiobutton(main_frame, text=voice, variable=voice_var,
                          value=voice, anchor=tk.W).pack(fill=tk.X, pady=2)

        # Buttons
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))

        def setup_voice():
            selected = voice_var.get()
            voice_name = selected.split(" (")[0]  # Extract voice name

            try:
                # Setup the voice
                self.reader.free_voice_cloner.add_voice_sample(voice_name)
                self.reader.free_voice_cloner.load_model()

                messagebox.showinfo("Success",
                    f"🎉 Free voice setup completed!\n\n"
                    f"Selected voice: {selected}\n\n"
                    f"You can now use 'My Voice (FREE)' mode to read PDFs "
                    f"with this high-quality AI voice.")

                dialog.destroy()
                self.update_voice_mode()

            except Exception as e:
                messagebox.showerror("Error", f"Failed to setup voice: {e}")

        tk.Button(button_frame, text="🚀 Setup This Voice", command=setup_voice,
                 bg="green", fg="white", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(0, 10))
        tk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT)

    def update_speech_rate(self, value):
        """Update speech rate display and setting"""
        rate = int(float(value))
        self.rate_label.config(text=f"{rate} WPM")
        if self.reader:
            self.reader.set_speech_rate(rate)

    def update_volume(self, value):
        """Update volume display and setting"""
        volume = float(value)
        percentage = int(volume * 100)
        self.volume_label.config(text=f"{percentage}%")
        if self.reader:
            self.reader.set_volume(volume)

    def browse_file(self):
        """Open file dialog to select PDF file"""
        file_types = [
            ("PDF files", "*.pdf"),
            ("All files", "*.*")
        ]

        filename = filedialog.askopenfilename(
            title="Select PDF File",
            filetypes=file_types
        )

        if filename:
            self.file_path_var.set(filename)
    
    def load_pdf(self):
        """Load the selected PDF file"""
        file_path = self.file_path_var.get()

        if not file_path:
            messagebox.showerror("Error", "Please select a PDF file first!")
            return

        try:
            self.status_var.set("Loading PDF...")
            self.root.update()

            # Create reader instance
            self.reader = PDFReader()

            # Load the PDF
            if self.reader.load_pdf(file_path):
                # Show preview
                preview = self.reader.get_text_preview(1000)
                self.preview_text.delete(1.0, tk.END)
                self.preview_text.insert(1.0, preview)
                
                # Enable buttons
                self.speak_button.config(state=tk.NORMAL)
                self.restart_button.config(state=tk.NORMAL)
                self.save_button.config(state=tk.NORMAL)

                # Apply current settings to the reader
                self.reader.set_speech_rate(self.rate_var.get())
                self.reader.set_volume(self.volume_var.get())
                self.reader.set_voice_mode(self.voice_mode_var.get())

                # Update voice mode UI
                self.update_voice_mode()

                self.status_var.set(f"PDF loaded successfully - {len(self.reader.text_content)} characters")
            else:
                messagebox.showerror("Error", "Failed to load the PDF file!")
                self.status_var.set("Failed to load PDF")
                
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.status_var.set("Error loading PDF")

    def toggle_speaking(self):
        """Start, resume, or stop speaking"""
        if not self.is_speaking:
            if self.is_paused:
                self.resume_speaking()
            else:
                self.start_speaking()
        else:
            self.stop_speaking()

    def start_speaking(self):
        """Start text-to-speech from current position"""
        if not self.reader or not self.reader.text_content:
            messagebox.showerror("Error", "Please load a PDF first!")
            return

        self.is_speaking = True
        self.is_paused = False
        self.speak_button.config(text="Stop Speaking")

        # Update status with progress
        progress = self.reader.get_progress_info()
        self.status_var.set(f"Speaking... ({progress['percentage']}%)")
        self.progress_var.set(f"Progress: {progress['percentage']}% ({progress['position']}/{progress['total']} words)")

        # Start speaking in a separate thread to avoid blocking the UI
        self.speak_thread = threading.Thread(target=self._speak_worker)
        self.speak_thread.daemon = True
        self.speak_thread.start()

    def resume_speaking(self):
        """Resume speaking from where it was stopped"""
        if not self.reader or not self.reader.text_content:
            messagebox.showerror("Error", "Please load a PDF first!")
            return

        self.start_speaking()  # This will resume from current position
    
    def _speak_worker(self):
        """Worker function for speaking (runs in separate thread)"""
        try:
            # Check if we should still be speaking before starting
            if self.is_speaking:
                # Use resume functionality
                self.reader.speak_text(resume=True)
        except Exception as e:
            # Only show error if we're still supposed to be speaking
            if self.is_speaking:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Speaking error: {str(e)}"))
        finally:
            # Reset UI state
            self.root.after(0, self._speaking_finished)
    
    def _speaking_finished(self):
        """Called when speaking is finished"""
        self.is_speaking = False
        self.is_paused = False

        # Update progress
        if self.reader:
            progress = self.reader.get_progress_info()
            self.progress_var.set(f"Progress: {progress['percentage']}% ({progress['position']}/{progress['total']} words)")

            if progress['percentage'] >= 100:
                self.speak_button.config(text="Start Speaking")
                self.status_var.set("Speaking completed")
            else:
                self.speak_button.config(text="Resume Speaking")
                self.status_var.set("Speaking paused - click Resume to continue")
        else:
            self.speak_button.config(text="Start Speaking")
            self.status_var.set("Speaking completed")

    def stop_speaking(self):
        """Stop speaking and save position for resume"""
        self.is_speaking = False
        self.is_paused = True

        # Try to stop the TTS engine using the reader's stop method
        if self.reader:
            try:
                self.reader.stop_speaking()
            except:
                pass

        # Update UI
        self.speak_button.config(text="Resume Speaking")
        self.status_var.set("Speaking stopped - click Resume to continue from this position")

    def restart_speaking(self):
        """Restart speaking from the beginning"""
        if not self.reader or not self.reader.text_content:
            messagebox.showerror("Error", "Please load a PDF first!")
            return

        # Reset position and start from beginning
        self.reader.current_position = 0
        self.is_paused = False

        if not self.is_speaking:
            self.start_speaking()
        else:
            # Stop current speech and restart
            self.stop_speaking()
            self.root.after(100, self.start_speaking)  # Small delay to ensure stop completes
    
    def save_audio(self):
        """Save audio to file"""
        if not self.reader or not self.reader.text_content:
            messagebox.showerror("Error", "Please load a PDF first!")
            return

        # Ask for save location
        filename = filedialog.asksaveasfilename(
            title="Save Audio File",
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.status_var.set("Saving audio...")
                self.root.update()
                
                self.reader.speak_text(save_audio=True, output_file=filename)
                
                messagebox.showinfo("Success", f"Audio saved to: {filename}")
                self.status_var.set("Audio saved successfully")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save audio: {str(e)}")
                self.status_var.set("Failed to save audio")


def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = PDFReaderGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
