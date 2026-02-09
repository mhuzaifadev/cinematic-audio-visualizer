# Cinematic Audio Visualizer

Generate stunning 1920×1080 cinematic videos where music energy drives speed, motion, and chaos.

## 🎯 Features

- **BPM → Speed Lines**: High tempo creates aggressive forward motion
- **Bass → Screen Shake**: Low-frequency energy drives camera instability
- **Drops → White Flash/Zoom**: Onset detection triggers dramatic flashes
- **Orchestra Swell → Cinematic Zoom**: Spectral centroid controls emotional zoom

## 🎵 Audio Features Used

- **RMS Energy**: Overall power and intensity
- **Tempo (BPM)**: Beat detection for rhythm-driven visuals
- **Spectral Centroid**: Brightness/chaos measurement
- **Onset Strength**: Detects drops, hits, and musical events
- **Bass Energy**: Low-frequency content for shake effects

## 📋 Requirements

- Python 3.7+
- See `requirements.txt` for dependencies

## 🚀 Installation

```bash
pip install -r requirements.txt
```

## 💻 Usage

### Basic Usage

```bash
# Local audio file
python cinematic_visualizer.py your_audio_file.mp3

# YouTube URL (extracts highest quality audio)
python cinematic_visualizer.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Direct audio URL
python cinematic_visualizer.py "https://example.com/audio.mp3"
```

### Specify Output Path

```bash
python cinematic_visualizer.py your_audio_file.wav -o output_video.mp4
```

### Examples

```bash
# Process a local WAV file
python cinematic_visualizer.py lose_my_mind.wav

# Process a local MP3 file with custom output
python cinematic_visualizer.py track.mp3 -o cinematic_output.mp4

# Download and process from YouTube (highest quality audio)
python cinematic_visualizer.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Process audio from direct URL
python cinematic_visualizer.py "https://example.com/music/track.mp3"
```

## 📤 Output

- **Format**: MP4 (H.264 video, AAC audio)
- **Resolution**: 1920×1080
- **Frame Rate**: 30 FPS
- **Duration**: Matches input audio (60-120 seconds recommended)

## 🎨 Visual Elements

- **Speed Lines**: Dynamic lines that move based on BPM and RMS energy
- **Horizon Gradient**: Cinematic depth with gradient from horizon to bottom
- **Screen Shake**: Bass-driven camera instability
- **Zoom Effects**: Spectral centroid and RMS control zoom intensity
- **Flash Effects**: White flash on detected musical drops/hits

## 🧠 How It Works

1. **Load Audio**: 
   - Supports local files (MP3/WAV)
   - Downloads from YouTube URLs (extracts highest quality audio)
   - Downloads from direct audio URLs
   - Uses librosa for audio processing
2. **Extract Features**: Analyzes audio at 30 FPS intervals
   - RMS energy per frame
   - Global tempo (BPM)
   - Spectral centroid per frame
   - Onset strength per frame
   - Bass energy per frame
3. **Map to Visuals**: Converts audio features to visual parameters
4. **Render Frames**: Generates 1920×1080 frames with all effects
5. **Encode Video**: Combines frames with original audio into MP4
6. **Cleanup**: Automatically removes temporary downloaded files

## 📝 Notes

- **Minimal UI**: Pure visuals - no text overlays
- **Cinematic Feel**: Designed for high-energy music (EDM, rock, orchestral)
- **Performance**: Processing time depends on audio length (~1-2x audio duration)

## 🔥 Tips

- Use **WAV** files for best quality (MP3 works fine too)
- **YouTube URLs**: Automatically extracts highest quality audio available
- Works best with **60-120 second** audio clips
- High-energy tracks produce the most dramatic visuals
- The visualizer automatically syncs to your audio's tempo
- Temporary files from URL downloads are automatically cleaned up

## 🏁 Example Workflow

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run visualizer (local file)
python cinematic_visualizer.py your_track.mp3

# Or use YouTube URL
python cinematic_visualizer.py "https://www.youtube.com/watch?v=VIDEO_ID"

# 3. Enjoy your cinematic video!
# Output: your_track_cinematic_visualizer.mp4
```

## 🌐 URL Support

The visualizer supports multiple input types:

- **Local Files**: `track.mp3`, `audio.wav`
- **YouTube URLs**: Automatically downloads and extracts highest quality audio
  - Supports: `youtube.com/watch?v=...`, `youtu.be/...`
  - Extracts audio in WAV format for best quality
- **Direct Audio URLs**: Downloads MP3/WAV files from any URL
  - Example: `https://example.com/music/track.mp3`

All temporary files are automatically cleaned up after processing.

---

**Built for pure cinematic energy. No text. No cringe. Just visuals.**
