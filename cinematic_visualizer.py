#!/usr/bin/env python3
"""
Cinematic Audio Visualizer

Generates a 1920×1080 cinematic video where music energy drives speed, motion, and chaos.
Features:
- BPM → speed lines
- Bass → screen shake
- Drops → white flash / zoom
- Orchestra swell → slow cinematic zoom
"""

import argparse
import librosa
import numpy as np
import cv2
try:
    # MoviePy 2.0+ API
    from moviepy import ImageSequenceClip
except ImportError:
    # Fallback for MoviePy 1.x
    from moviepy.editor import ImageSequenceClip
import os
import re
import shutil
from pathlib import Path
from urllib.parse import urlparse
import yt_dlp
import requests


# Constants
WIDTH, HEIGHT = 1920, 1080
FPS = 60


def get_project_temp_dir():
    """Get or create temp directory in project root."""
    # Get the script's directory (project root)
    script_dir = Path(__file__).parent.absolute()
    temp_dir = script_dir / ".temp"
    temp_dir.mkdir(exist_ok=True)
    return str(temp_dir)


def clamp(x, a=0, b=1):
    """Clamp value between a and b."""
    return max(a, min(b, x))


def is_url(path):
    """Check if the input is a URL."""
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def is_youtube_url(url):
    """Check if URL is a YouTube URL."""
    youtube_patterns = [
        r'(?:https?://)?(?:www\.)?(?:youtube\.com|youtu\.be)',
        r'youtube\.com/watch',
        r'youtu\.be/'
    ]
    return any(re.search(pattern, url, re.IGNORECASE) for pattern in youtube_patterns)


def download_youtube_audio(url, output_dir=None):
    """
    Download/extract audio from YouTube URL in highest possible quality.
    
    Returns:
        Path to downloaded audio file (temporary file)
    """
    if output_dir is None:
        output_dir = get_project_temp_dir()
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    # Configure yt-dlp for best audio quality
    ydl_opts = {
        'format': 'bestaudio/best',  # Best audio quality available
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',  # Use WAV for best quality
            'preferredquality': '0',  # Best quality (0 = best)
        }],
        'quiet': False,
        'no_warnings': False,
        'extract_flat': False,
    }
    
    print(f"Downloading audio from YouTube: {url}")
    print("Extracting highest quality audio...")
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract info and download
            info = ydl.extract_info(url, download=True)
            title = info.get('title', 'audio')
            
            # Clean filename
            safe_title = re.sub(r'[^\w\s-]', '', title).strip()
            safe_title = re.sub(r'[-\s]+', '-', safe_title)
            
            # Find the downloaded file (check for WAV first, then other formats)
            audio_extensions = ['.wav', '.m4a', '.webm', '.opus', '.mp3', '.ogg']
            downloaded_file = None
            
            for ext in audio_extensions:
                potential_file = os.path.join(output_dir, f"{safe_title}{ext}")
                if os.path.exists(potential_file):
                    downloaded_file = potential_file
                    break
            
            # If not found with title, search all files in directory
            if not downloaded_file:
                all_files = [f for f in os.listdir(output_dir) 
                           if any(f.lower().endswith(ext) for ext in audio_extensions)]
                if all_files:
                    downloaded_file = os.path.join(output_dir, all_files[0])
            
            if downloaded_file:
                # Ensure it's WAV format for best quality (convert if needed)
                if not downloaded_file.endswith('.wav'):
                    final_path = os.path.join(output_dir, f"{safe_title}.wav")
                    # Use FFmpeg to convert if available, otherwise use as-is
                    try:
                        import subprocess
                        subprocess.run([
                            'ffmpeg', '-i', downloaded_file, '-y',
                            '-acodec', 'pcm_s16le', final_path
                        ], check=True, capture_output=True)
                        if os.path.exists(final_path):
                            os.remove(downloaded_file)
                            return final_path
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        # FFmpeg not available or conversion failed, use original
                        pass
                
                return downloaded_file
            else:
                raise FileNotFoundError("Audio file not found after download")
                
    except Exception as e:
        raise Exception(f"Failed to download YouTube audio: {e}")


def download_audio_from_url(url, output_dir=None):
    """
    Download audio file from direct URL.
    
    Returns:
        Path to downloaded audio file (temporary file)
    """
    if output_dir is None:
        output_dir = get_project_temp_dir()
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading audio from URL: {url}")
    
    try:
        # Get filename from URL or use default
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        if not filename or '.' not in filename:
            filename = "audio.mp3"
        
        output_path = os.path.join(output_dir, filename)
        
        # Download with streaming for large files
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rDownloading: {percent:.1f}%", end='', flush=True)
        
        print()  # New line after progress
        return output_path
        
    except Exception as e:
        raise Exception(f"Failed to download audio from URL: {e}")


def prepare_audio_input(input_path, temp_dir=None):
    """
    Prepare audio input - handles local files, YouTube URLs, and direct audio URLs.
    
    Returns:
        tuple: (audio_file_path, is_temporary, temp_dir)
    """
    input_path_str = str(input_path)
    
    # Check if it's a URL
    if is_url(input_path_str):
        if is_youtube_url(input_path_str):
            print("Detected YouTube URL")
            if temp_dir is None:
                temp_dir = get_project_temp_dir()
            audio_path = download_youtube_audio(input_path_str, temp_dir)
            return Path(audio_path), True, temp_dir
        else:
            print("Detected direct audio URL")
            if temp_dir is None:
                temp_dir = get_project_temp_dir()
            audio_path = download_audio_from_url(input_path_str, temp_dir)
            return Path(audio_path), True, temp_dir
    else:
        # Local file
        audio_path = Path(input_path_str)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        return audio_path, False, None


def extract_audio_features(audio_path, sr=None):
    """
    Extract all audio features needed for visualization.
    
    Returns:
        dict with keys: y, sr, duration, rms, tempo, centroid, onset_env, bass_energy,
        frequency_bands, spectral_rolloff, zero_crossing_rate, beats
    """
    print(f"Loading audio: {audio_path}")
    y, sr = librosa.load(audio_path, sr=sr)
    
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"Duration: {duration:.2f} seconds")
    
    # Calculate hop length for frame-perfect sync
    hop_length = int(sr / FPS)
    
    # RMS Energy (overall power)
    print("Extracting RMS energy...")
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms = librosa.util.normalize(rms)
    
    # Tempo (global BPM)
    print("Detecting tempo...")
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    # Convert tempo to scalar if it's an array
    if isinstance(tempo, np.ndarray):
        tempo = tempo.item() if tempo.size > 0 else float(tempo[0])
    elif isinstance(tempo, list):
        tempo = float(tempo[0]) if tempo else 0.0
    else:
        tempo = float(tempo)
    print(f"Detected BPM: {tempo:.1f}")
    
    # Convert beat frames to frame indices
    beat_frames = librosa.frames_to_samples(beats, hop_length=hop_length)
    beat_frame_indices = (beat_frames / (sr / FPS)).astype(int)
    
    # Spectral Centroid (brightness / chaos)
    print("Extracting spectral centroid...")
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    centroid = librosa.util.normalize(centroid)
    
    # Spectral Rolloff (high frequency content)
    print("Extracting spectral rolloff...")
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
    rolloff = librosa.util.normalize(rolloff)
    
    # Zero Crossing Rate (noisiness/percussiveness)
    print("Extracting zero crossing rate...")
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
    zcr = librosa.util.normalize(zcr)
    
    # Onset Strength (drops / hits)
    print("Extracting onset strength...")
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_env = librosa.util.normalize(onset_env)
    
    # Frequency bands for color mapping
    print("Extracting frequency bands...")
    stft = librosa.stft(y, hop_length=hop_length)
    magnitude = np.abs(stft)
    
    # Define frequency bands
    freqs = librosa.fft_frequencies(sr=sr, n_fft=stft.shape[0] * 2 - 1)
    
    # Low (0-200 Hz) - Red
    low_mask = freqs <= 200
    low_energy = np.abs(stft[low_mask, :]).mean(axis=0)
    low_energy = librosa.util.normalize(low_energy)
    
    # Mid-Low (200-800 Hz) - Orange/Yellow
    mid_low_mask = (freqs > 200) & (freqs <= 800)
    mid_low_energy = np.abs(stft[mid_low_mask, :]).mean(axis=0) if np.any(mid_low_mask) else low_energy
    mid_low_energy = librosa.util.normalize(mid_low_energy)
    
    # Mid (800-3000 Hz) - Green/Cyan
    mid_mask = (freqs > 800) & (freqs <= 3000)
    mid_energy = np.abs(stft[mid_mask, :]).mean(axis=0) if np.any(mid_mask) else low_energy
    mid_energy = librosa.util.normalize(mid_energy)
    
    # High (3000+ Hz) - Blue/Purple
    high_mask = freqs > 3000
    high_energy = np.abs(stft[high_mask, :]).mean(axis=0) if np.any(high_mask) else low_energy
    high_energy = librosa.util.normalize(high_energy)
    
    # Bass energy (for shake)
    bass_energy = low_energy
    
    return {
        'y': y,
        'sr': sr,
        'duration': duration,
        'rms': rms,
        'tempo': tempo,
        'centroid': centroid,
        'rolloff': rolloff,
        'zcr': zcr,
        'onset_env': onset_env,
        'bass_energy': bass_energy,
        'low_energy': low_energy,
        'mid_low_energy': mid_low_energy,
        'mid_energy': mid_energy,
        'high_energy': high_energy,
        'beat_frames': beat_frame_indices,
        'hop_length': hop_length
    }


# ============================================================
# CINEMATIC PALETTES — locked per scene, changed only on drops
# ============================================================

SCENE_PALETTES = [
    ((255, 60, 60),   (255, 120, 80)),   # red / warm orange
    ((60, 180, 255),  (100, 220, 255)),   # cyan / ice blue
    ((255, 180, 40),  (255, 220, 100)),   # amber / gold
    ((160, 80, 255),  (200, 140, 255)),   # purple / lavender
    ((40, 255, 180),  (100, 255, 220)),   # teal / mint
    ((255, 60, 160),  (255, 120, 200)),   # magenta / pink
]


# ============================================================
# PRE-COMPUTE: generate stable random seeds for depth lines
# so every frame is deterministic without per-frame np.random
# ============================================================

NUM_DEPTH_LINES = 180
_line_seeds = np.random.RandomState(42).randn(NUM_DEPTH_LINES, 2)  # (dx, dy) offsets


# ============================================================
# SCENE STATE — mutable state passed through frames
# ============================================================

def make_scene_state():
    """Create the initial mutable scene state dict."""
    return {
        'palette_idx': 0,
        'prev_zoom': 1.0,
        'prev_shake_x': 0.0,
        'prev_shake_y': 0.0,
        'flash_decay': 0.0,        # 1.0 on drop, decays to 0
        'frames_since_drop': 999,   # large initial value
    }


# ============================================================
# CORE ILLUSION: depth-based speed lines (vanishing point)
# ============================================================

def draw_depth_lines(frame, speed, intensity, color, accent, frame_idx, width, height):
    """
    Radiating tunnel lines from a TRUE center vanishing point.
    Lines point OUTWARD from center — creates forward velocity illusion.
    """
    cx, cy = width // 2, height // 2  # TRUE center vanishing point

    max_len = int(400 + intensity * 600)

    for i in range(NUM_DEPTH_LINES):
        # Angle around the vanishing point (full 360 degrees)
        angle = (_line_seeds[i, 0] * 2 - 1) * np.pi

        # z cycles forward — gives the "rushing through tunnel" feel
        z = (i * 25 + frame_idx * speed * 6) % 2000
        depth = max(0.05, 1.0 - z / 2000)  # 1 = near camera, 0 = far

        # Direction vector from center outward
        dx = np.cos(angle)
        dy = np.sin(angle)

        # Start near vanishing point, end far from it
        start_x = int(cx + dx * 50 * depth)
        start_y = int(cy + dy * 50 * depth)
        end_x = int(cx + dx * max_len * depth)
        end_y = int(cy + dy * max_len * depth)

        # Near lines are thicker, brighter
        thickness = max(1, int(5 * depth))
        brightness = depth * (0.4 + intensity * 0.6)

        # Mix primary and accent color
        c = color if i % 4 else accent
        line_color = (
            int(c[0] * brightness),
            int(c[1] * brightness),
            int(c[2] * brightness),
        )

        cv2.line(frame, (start_x, start_y), (end_x, end_y), line_color, thickness)


# ============================================================
# SECONDARY: subtle dust / ember particles (background layer)
# ============================================================

NUM_EMBERS = 60
_ember_seeds = np.random.RandomState(99).rand(NUM_EMBERS, 3)  # x_frac, y_frac, phase

def draw_embers(frame, intensity, color, frame_idx, width, height):
    """Tiny floating embers that drift upward — adds life without competing."""
    for i in range(NUM_EMBERS):
        sx, sy, phase = _ember_seeds[i]
        x = int(sx * width)
        y = int((sy * height - frame_idx * (0.3 + intensity * 0.5) + phase * 500) % height)
        size = 1 if intensity < 0.5 else 2
        alpha = 0.15 + intensity * 0.25
        ember_color = (
            int(color[0] * alpha),
            int(color[1] * alpha),
            int(color[2] * alpha),
        )
        cv2.circle(frame, (x, y), size, ember_color, -1)


# ============================================================
# VIGNETTE — darkens edges, focuses the eye
# ============================================================

_vignette_cache = {}

def get_vignette(width, height, strength=0.7):
    """Create (and cache) a vignette mask."""
    key = (width, height, int(strength * 100))
    if key not in _vignette_cache:
        X = np.linspace(-1, 1, width)
        Y = np.linspace(-1, 1, height)
        xx, yy = np.meshgrid(X, Y)
        dist = np.sqrt(xx ** 2 + yy ** 2)
        vignette = 1.0 - np.clip(dist * strength, 0, 1)
        # Expand to 3 channels
        _vignette_cache[key] = np.stack([vignette] * 3, axis=-1).astype(np.float32)
    return _vignette_cache[key]


# ============================================================
# POST-FX: smooth zoom, smooth shake, white flash
# ============================================================

def apply_smooth_zoom(frame, zoom_factor, width, height):
    """Apply zoom via center crop + resize."""
    if abs(zoom_factor - 1.0) < 0.005:
        return frame
    new_w = max(1, int(width / zoom_factor))
    new_h = max(1, int(height / zoom_factor))
    x_off = (width - new_w) // 2
    y_off = (height - new_h) // 2
    cropped = frame[y_off:y_off + new_h, x_off:x_off + new_w]
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)


def apply_smooth_shake(frame, dx, dy, width, height):
    """Apply sub-pixel camera shake."""
    if abs(dx) < 0.5 and abs(dy) < 0.5:
        return frame
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, M, (width, height), borderMode=cv2.BORDER_REPLICATE)


# ============================================================
# RENDER ONE FRAME — the single, clean pipeline
# ============================================================

def render_frame(features, frame_idx, width, height, state):
    """
    Render a single cinematic frame.

    Philosophy: ONE dominant illusion (depth speed lines) + camera breathing.
    Scene cuts ONLY on rare major drops. Palette is locked per scene.
    """
    fi = min(frame_idx, len(features['rms']) - 1)

    # --- read audio features for this frame ---
    rms       = features['rms'][fi]
    onset     = features['onset_env'][fi]
    bass      = features['bass_energy'][fi]
    centroid  = features['centroid'][fi]
    low       = features['low_energy'][fi]
    high      = features['high_energy'][fi]

    # --- detect major drop (rare!) ---
    is_major_drop = onset > 0.9 and rms > 0.7 and state['frames_since_drop'] > 20
    state['frames_since_drop'] += 1

    if is_major_drop:
        state['frames_since_drop'] = 0
        state['palette_idx'] = (state['palette_idx'] + 1) % len(SCENE_PALETTES)
        state['flash_decay'] = 1.0

    # --- current palette ---
    color, accent = SCENE_PALETTES[state['palette_idx']]

    # --- smoothed speed (RMS drives it) ---
    speed = 3 + rms * 45

    # --- layer 0: tinted background (NOT black — warm, with headroom) ---
    bg_boost = 0.12 + rms * 0.18  # brighter when loud, never pitch black
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (
        int(color[0] * bg_boost * 0.15),
        int(color[1] * bg_boost * 0.12),
        int(color[2] * bg_boost * 0.18),
    )

    # --- layer 1: floating embers (background, subtle) ---
    draw_embers(frame, rms, accent, frame_idx, width, height)

    # --- layer 2: THE MAIN EVENT — depth speed lines ---
    draw_depth_lines(frame, speed, rms, color, accent, frame_idx, width, height)

    # --- post-fx 1: smooth zoom (CAMERA — applied first) ---
    zoom_target = clamp(1.0 + centroid * 0.05 + rms * 0.03, 1.0, 1.12)
    state['prev_zoom'] = state['prev_zoom'] * 0.92 + zoom_target * 0.08
    frame = apply_smooth_zoom(frame, state['prev_zoom'], width, height)

    # --- post-fx 2: smooth shake (CAMERA — bass-driven, low-pass filtered) ---
    shake_mag = bass * 18
    raw_dx = np.random.uniform(-shake_mag, shake_mag)
    raw_dy = np.random.uniform(-shake_mag, shake_mag)
    state['prev_shake_x'] = state['prev_shake_x'] * 0.7 + raw_dx * 0.3
    state['prev_shake_y'] = state['prev_shake_y'] * 0.7 + raw_dy * 0.3
    frame = apply_smooth_shake(frame, state['prev_shake_x'], state['prev_shake_y'], width, height)

    # --- post-fx 3: vignette (LENS — after camera movement) ---
    vignette_strength = 0.55 + (1.0 - rms) * 0.25  # lighter vignette overall
    vignette = get_vignette(width, height, vignette_strength)
    frame = (frame.astype(np.float32) * vignette).astype(np.uint8)

    # --- post-fx 4: white flash on drop (EXPOSURE — last, sits on top) ---
    if state['flash_decay'] > 0.05:
        flash_alpha = state['flash_decay'] * 0.85
        white = np.full_like(frame, 255)
        frame = cv2.addWeighted(frame, 1 - flash_alpha, white, flash_alpha, 0)
        state['flash_decay'] *= 0.55  # exponential decay

    return frame


def create_visualization(audio_input, output_path=None):
    """
    Main function to create the cinematic visualization.
    
    Args:
        audio_input: Path to local audio file, YouTube URL, or direct audio URL
        output_path: Optional output path for the video
    """
    temp_dir = None
    is_temp = False
    
    try:
        # Prepare audio input (handles URLs and local files)
        audio_path, is_temp, temp_dir = prepare_audio_input(audio_input)
        
        # Set output path - always in project root, not temp directory
        if output_path is None:
            # Use project root, not temp directory
            project_root = Path(__file__).parent.absolute()
            output_path = project_root / f"{audio_path.stem}_cinematic_visualizer.mp4"
        else:
            output_path = Path(output_path)
            # Ensure output is absolute path
            if not output_path.is_absolute():
                project_root = Path(__file__).parent.absolute()
                output_path = project_root / output_path
        
        print(f"\n{'='*60}")
        print("CINEMATIC AUDIO VISUALIZER")
        print(f"{'='*60}\n")
        
        # Extract audio features
        features = extract_audio_features(str(audio_path))
        
        # Calculate total frames
        total_frames = int(features['duration'] * FPS)
        print(f"\nRendering {total_frames} frames at {FPS} FPS...")
        print(f"Resolution: {WIDTH}×{HEIGHT}\n")
        
        # Render all frames with cinematic state tracking
        frames = []
        state = make_scene_state()
        
        for i in range(total_frames):
            if (i + 1) % 30 == 0:
                progress = ((i + 1) / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({i+1}/{total_frames} frames)", end='\r')
            
            frame = render_frame(features, i, WIDTH, HEIGHT, state)
            frames.append(frame)
        
        print(f"\nProgress: 100.0% ({total_frames}/{total_frames} frames)")
        
        # Create video clip
        print("\nEncoding video...")
        clip = ImageSequenceClip(frames, fps=FPS)
        
        # Add audio (compatible with both MoviePy 1.x and 2.0+)
        print("Adding audio track...")
        try:
            # Try MoviePy 2.0+ API first
            from moviepy import AudioFileClip
            audio_clip = AudioFileClip(str(audio_path))
            clip = clip.with_audio(audio_clip)
        except (ImportError, AttributeError):
            # Fallback to MoviePy 1.x API
            clip = clip.set_audio(str(audio_path))
        
        # Write video file
        print(f"Writing video to: {output_path}")
        clip.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            bitrate="8000k",
            fps=FPS,
            logger=None  # Suppress verbose moviepy output
        )
        
        # Clean up audio clip if we created it
        try:
            if 'audio_clip' in locals():
                audio_clip.close()
        except:
            pass
        
        print(f"\n✅ Complete! Video saved to: {output_path}")
        print(f"   Duration: {features['duration']:.2f}s")
        tempo_value = features['tempo']
        if isinstance(tempo_value, np.ndarray):
            tempo_value = tempo_value.item() if tempo_value.size > 0 else float(tempo_value[0])
        elif isinstance(tempo_value, list):
            tempo_value = float(tempo_value[0]) if tempo_value else 0.0
        else:
            tempo_value = float(tempo_value)
        print(f"   BPM: {tempo_value:.1f}")
        
        return str(output_path)
        
    finally:
        # Clean up temporary audio files only (not the final video)
        # Only delete files in temp_dir, not the directory itself if output is there
        if is_temp and temp_dir and os.path.exists(temp_dir):
            try:
                # List all files in temp directory
                temp_files = [f for f in os.listdir(temp_dir) if os.path.isfile(os.path.join(temp_dir, f))]
                # Delete only audio files, not video files
                for file in temp_files:
                    file_path = os.path.join(temp_dir, file)
                    # Only delete audio files, keep video files
                    if file_path.endswith(('.wav', '.mp3', '.m4a', '.webm', '.opus', '.ogg')):
                        os.remove(file_path)
                print(f"Cleaned up temporary audio files")
            except Exception as e:
                print(f"Warning: Could not clean up temporary files: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Cinematic Audio Visualizer - Generate cinematic videos from audio",
        epilog="""
Examples:
  # Local audio file
  python cinematic_visualizer.py track.mp3
  
  # YouTube URL
  python cinematic_visualizer.py "https://www.youtube.com/watch?v=VIDEO_ID"
  
  # Direct audio URL
  python cinematic_visualizer.py "https://example.com/audio.mp3"
  
  # Custom output path
  python cinematic_visualizer.py track.wav -o output.mp4
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "audio",
        type=str,
        help="Path to input audio file (MP3/WAV), YouTube URL, or direct audio URL"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to output video file (default: <audio_name>_cinematic_visualizer.mp4)"
    )
    
    args = parser.parse_args()
    
    try:
        create_visualization(args.audio, args.output)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
