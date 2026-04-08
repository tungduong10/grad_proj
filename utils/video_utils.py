import cv2
import os

def get_video_frame_count(video_path):
    """Get total number of frames in video without loading them all."""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

def read_video(video_path):
    """Generator that yields frames one at a time to save memory."""
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()

def save_video(output_video_frames, output_video_path, fps=24.0):
    frame_iter = iter(output_video_frames)
    try:
        first_frame = next(frame_iter)
    except StopIteration:
        return
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (first_frame.shape[1], first_frame.shape[0]))
    out.write(first_frame)
    for frame in frame_iter:
        out.write(frame)
    out.release()