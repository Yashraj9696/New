import cv2
import numpy as np
import moviepy.editor as mp
import os

def apply_sketch_effect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    inverted_blurred = cv2.bitwise_not(blurred)
    sketch = cv2.divide(gray, inverted_blurred, scale=256.0)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

def process_video(input_path, output_path):
    video = mp.VideoFileClip(input_path)
    fps = video.fps
    frames = []

    for frame in video.iter_frames(fps=fps, dtype="uint8"):
        sketch_frame = apply_sketch_effect(frame)
        frames.append(sketch_frame)

    output_clip = mp.ImageSequenceClip(frames, fps=fps)
    output_clip.write_videofile(output_path, codec="libx264", fps=fps)

if __name__ == "__main__":
    input_video = "input.mp4"
    output_video = "output_sketch.mp4"

    if not os.path.exists(input_video):
        print("ERROR: Input video not found!")
        exit(1)

    process_video(input_video, output_video)

    if os.path.exists(output_video):
        print("SUCCESS: Sketch video created!")
    else:
        print("ERROR: Sketch video was not generated.")
