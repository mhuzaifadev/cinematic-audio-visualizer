#!/usr/bin/env python3
"""
F1 Cinematic Audio Visualizer 🏎️🔥

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
import tempfile
import shutil
from pathlib import Path
from urllib.parse import urlparse
import yt_dlp
import requests


# Constants
WIDTH, HEIGHT = 1920, 1080
FPS = 30


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
        output_dir = tempfile.mkdtemp()
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
        output_dir = tempfile.mkdtemp()
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
                temp_dir = tempfile.mkdtemp(prefix="f1_visualizer_")
            audio_path = download_youtube_audio(input_path_str, temp_dir)
            return Path(audio_path), True, temp_dir
        else:
            print("Detected direct audio URL")
            if temp_dir is None:
                temp_dir = tempfile.mkdtemp(prefix="f1_visualizer_")
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
        dict with keys: y, sr, duration, rms, tempo, centroid, onset_env, bass_energy
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
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    print(f"Detected BPM: {tempo:.1f}")
    
    # Spectral Centroid (brightness / chaos)
    print("Extracting spectral centroid...")
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    centroid = librosa.util.normalize(centroid)
    
    # Onset Strength (drops / hits)
    print("Extracting onset strength...")
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_env = librosa.util.normalize(onset_env)
    
    # Bass energy (low frequency content for shake)
    print("Extracting bass energy...")
    stft = librosa.stft(y, hop_length=hop_length)
    # Focus on low frequencies (0-200 Hz)
    bass_freqs = librosa.fft_frequencies(sr=sr, n_fft=stft.shape[0] * 2 - 1)
    bass_mask = bass_freqs <= 200
    bass_energy = np.abs(stft[bass_mask, :]).mean(axis=0)
    bass_energy = librosa.util.normalize(bass_energy)
    
    return {
        'y': y,
        'sr': sr,
        'duration': duration,
        'rms': rms,
        'tempo': tempo,
        'centroid': centroid,
        'onset_env': onset_env,
        'bass_energy': bass_energy,
        'hop_length': hop_length
    }


def map_features_to_visuals(features, frame_idx):
    """
    Map audio features to visual parameters for a given frame.
    
    Returns:
        dict with visual parameters: speed, shake, zoom, flash, horizon_intensity
    """
    if frame_idx >= len(features['rms']):
        frame_idx = len(features['rms']) - 1
    
    rms = features['rms'][frame_idx]
    centroid = features['centroid'][frame_idx]
    onset = features['onset_env'][frame_idx]
    bass = features['bass_energy'][frame_idx]
    
    # Speed lines velocity (BPM → speed)
    # Higher RMS = faster motion
    speed = 5 + rms * 40
    
    # Screen shake (Bass → shake)
    # Stronger bass = more shake
    shake = int(bass * 25)
    
    # Zoom (Centroid → zoom, Orchestra swell → slow zoom)
    # Higher centroid = more zoom, but also consider RMS for orchestral swells
    zoom = 1.0 + (centroid * 0.1) + (rms * 0.05)
    
    # Flash on drops (Onset → flash)
    flash = onset > 0.85
    
    # Horizon intensity (for cinematic feel)
    # Lower during intense moments, higher during calm
    horizon_intensity = 1.0 - (rms * 0.3)
    
    return {
        'speed': speed,
        'shake': shake,
        'zoom': zoom,
        'flash': flash,
        'horizon_intensity': horizon_intensity
    }


def draw_speed_lines(frame, visuals, frame_idx, width, height):
    """Draw speed lines that move based on BPM/RMS."""
    speed = visuals['speed']
    
    # Draw multiple sets of speed lines
    num_lines = 15
    line_spacing = width // num_lines
    
    for i in range(num_lines):
        x_base = (i * line_spacing) % width
        
        # Calculate position based on speed and frame
        x1 = int((x_base + frame_idx * speed) % width)
        x2 = int((x_base + frame_idx * speed * 0.5) % width)
        
        # Vary line lengths for depth
        y1 = height
        y2 = height - 300 - int(visuals['rms'] * 200) if 'rms' in visuals else height - 300
        
        # Draw line with slight transparency effect
        color_intensity = int(200 + visuals.get('rms', 0.5) * 55)
        cv2.line(frame, (x1, y1), (x2, y2), (color_intensity, color_intensity, color_intensity), 2)
        
        # Add secondary lines for motion blur effect
        if i % 2 == 0:
            x1_blur = int((x_base + frame_idx * speed * 0.8) % width)
            x2_blur = int((x_base + frame_idx * speed * 0.4) % width)
            cv2.line(frame, (x1_blur, y1), (x2_blur, y2), 
                    (color_intensity // 2, color_intensity // 2, color_intensity // 2), 1)


def draw_horizon(frame, visuals, width, height):
    """Draw horizon gradient for cinematic depth."""
    intensity = visuals['horizon_intensity']
    
    # Create gradient from horizon line upward
    horizon_y = int(height * 0.7)
    
    for y in range(horizon_y, height):
        # Gradient from dark at horizon to lighter at bottom
        gradient_factor = (y - horizon_y) / (height - horizon_y)
        color_value = int(20 + gradient_factor * 30 * intensity)
        
        cv2.line(frame, (0, y), (width, y), 
                (color_value, color_value, color_value), 1)


def apply_screen_shake(frame, shake_amount, width, height):
    """Apply screen shake effect."""
    if shake_amount == 0:
        return frame
    
    dx = np.random.randint(-shake_amount, shake_amount + 1)
    dy = np.random.randint(-shake_amount, shake_amount + 1)
    
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    frame_shaken = cv2.warpAffine(frame, M, (width, height), 
                                  borderMode=cv2.BORDER_REPLICATE)
    return frame_shaken


def apply_zoom(frame, zoom_factor, width, height):
    """Apply zoom effect (centered)."""
    if zoom_factor == 1.0:
        return frame
    
    # Calculate new dimensions
    new_width = int(width / zoom_factor)
    new_height = int(height / zoom_factor)
    
    # Center crop
    x_offset = (width - new_width) // 2
    y_offset = (height - new_height) // 2
    
    # Resize and place
    frame_zoomed = cv2.resize(frame[y_offset:y_offset+new_height, 
                                    x_offset:x_offset+new_width], 
                             (width, height))
    return frame_zoomed


def apply_flash(frame, flash_active):
    """Apply white flash effect on drops."""
    if flash_active:
        # Blend white flash (80% white, 20% original)
        flash_frame = frame.copy()
        flash_frame[:] = (255, 255, 255)
        frame = cv2.addWeighted(frame, 0.2, flash_frame, 0.8, 0)
    return frame


def render_frame(features, frame_idx, width, height):
    """Render a single frame with all visual effects."""
    # Start with black frame
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Get visual parameters for this frame
    visuals = map_features_to_visuals(features, frame_idx)
    visuals['rms'] = features['rms'][min(frame_idx, len(features['rms']) - 1)]
    
    # Draw horizon first (background)
    draw_horizon(frame, visuals, width, height)
    
    # Draw speed lines
    draw_speed_lines(frame, visuals, frame_idx, width, height)
    
    # Apply zoom (before shake for better effect)
    frame = apply_zoom(frame, visuals['zoom'], width, height)
    
    # Apply flash on drops
    frame = apply_flash(frame, visuals['flash'])
    
    # Apply screen shake (last, so it affects everything)
    frame = apply_screen_shake(frame, visuals['shake'], width, height)
    
    return frame


def create_visualization(audio_input, output_path=None):
    """
    Main function to create the F1 cinematic visualization.
    
    Args:
        audio_input: Path to local audio file, YouTube URL, or direct audio URL
        output_path: Optional output path for the video
    """
    temp_dir = None
    is_temp = False
    
    try:
        # Prepare audio input (handles URLs and local files)
        audio_path, is_temp, temp_dir = prepare_audio_input(audio_input)
        
        # Set output path
        if output_path is None:
            output_path = audio_path.parent / f"{audio_path.stem}_f1_visualizer.mp4"
        else:
            output_path = Path(output_path)
        
        print(f"\n{'='*60}")
        print("F1 CINEMATIC AUDIO VISUALIZER 🏎️🔥")
        print(f"{'='*60}\n")
        
        # Extract audio features
        features = extract_audio_features(str(audio_path))
        
        # Calculate total frames
        total_frames = int(features['duration'] * FPS)
        print(f"\nRendering {total_frames} frames at {FPS} FPS...")
        print(f"Resolution: {WIDTH}×{HEIGHT}\n")
        
        # Render all frames
        frames = []
        for i in range(total_frames):
            if (i + 1) % 30 == 0:
                progress = ((i + 1) / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({i+1}/{total_frames} frames)", end='\r')
            
            frame = render_frame(features, i, WIDTH, HEIGHT)
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
        print(f"   BPM: {features['tempo']:.1f}")
        
        return str(output_path)
        
    finally:
        # Clean up temporary files if we downloaded from URL
        if is_temp and temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temporary files")
            except Exception as e:
                print(f"Warning: Could not clean up temporary files: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="F1 Cinematic Audio Visualizer - Generate cinematic videos from audio",
        epilog="""
Examples:
  # Local audio file
  python f1_visualizer.py track.mp3
  
  # YouTube URL
  python f1_visualizer.py "https://www.youtube.com/watch?v=VIDEO_ID"
  
  # Direct audio URL
  python f1_visualizer.py "https://example.com/audio.mp3"
  
  # Custom output path
  python f1_visualizer.py track.wav -o output.mp4
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
        help="Path to output video file (default: <audio_name>_f1_visualizer.mp4)"
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
