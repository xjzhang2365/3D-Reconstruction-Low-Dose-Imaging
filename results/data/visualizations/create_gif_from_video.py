"""
Create GIF preview from video for GitHub display.

Run this once to generate a GIF from your video.
Requires: pip install moviepy
"""

from moviepy.editor import VideoFileClip
from pathlib import Path

print("Creating GIF from video...")

# Input
video_file = 'dynamics_45frames.mp4'

if not Path(video_file).exists():
    print(f"❌ Video not found: {video_file}")
    print("Please copy your video to this directory first!")
    exit()

# Output
gif_file = 'dynamics_preview.gif'

print(f"\nProcessing: {video_file}")

# Load video
clip = VideoFileClip(video_file)

print(f"  Duration: {clip.duration:.1f}s")
print(f"  Size: {clip.size}")
print(f"  FPS: {clip.fps}")

# Resize for web (max 600px width, keeps aspect ratio)
clip_resized = clip.resize(width=600)

# Convert to GIF
print("\nCreating GIF (this may take a minute)...")
clip_resized.write_gif(
    gif_file,
    fps=10,  # Lower FPS = smaller file
    program='ffmpeg'
)

file_size_mb = Path(gif_file).stat().st_size / (1024 * 1024)

print(f"\n✓ Created: {gif_file}")
print(f"  Size: {file_size_mb:.2f} MB")

# If too large for GitHub (>10MB), create smaller version
if file_size_mb > 10:
    print(f"\n⚠ File is large ({file_size_mb:.1f}MB). Creating smaller version...")
    
    clip_small = clip.resize(width=400)
    clip_small.write_gif(
        'dynamics_preview_small.gif',
        fps=8
    )
    
    small_size_mb = Path('dynamics_preview_small.gif').stat().st_size / (1024 * 1024)
    print(f"✓ Created smaller version: {small_size_mb:.2f} MB")

clip.close()
print("\n✓ Done!")