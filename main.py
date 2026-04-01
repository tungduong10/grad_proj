import cv2
import numpy as np
import argparse
from utils.tracker5 import MultiSportTracker
from  utils.view_transformer import ViewTransformer 
from utils.configs5 import SPORT_CONFIGS
import os
import time

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


def validate_sport_selection(tracks, sport_config, sport_name):
    """
    FAILSAFE LOGIC: Checks if the visual evidence matches the selected sport.
    """
    print("\n--- Running Validation Failsafe ---")
    unique_players = set()
    
    # Check the first 100 frames (or less if video is short) to see how many players exist
    check_frames = min(100, len(tracks["players"]))
    for i in range(check_frames):
        for player_id in tracks["players"][i].keys():
            unique_players.add(player_id)
            
    total_detected_players = len(unique_players)
    max_allowed = sport_config['max_players']
    
    print(f"Selected Sport: {sport_name.upper()}")
    print(f"Detected unique players in early frames: {total_detected_players}")
    print(f"Max allowed for this sport config: {max_allowed}")
    
    if total_detected_players > max_allowed:
        print(f"\n[WARNING] You selected {sport_name}, but detected {total_detected_players} players!")
        print("This exceeds the logical maximum for this sport. Are you sure you selected the right profile?")
        response = input("Do you want to continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborting. Please restart with the correct sport configuration.")
            exit()
    print("Validation Passed! Proceeding to analytics...\n")

def draw_and_save_annotations_streaming(frame_generator, tracks, output_path, fps, video_dims, total_frames):
    """
    Draws annotations and saves directly to video to avoid memory issues with large videos.
    Accepts a frame generator and dimensions instead of loading all frames.
    """
    frame_width, frame_height = video_dims
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    for frame_num, frame in enumerate(frame_generator):
        annotated_frame = frame.copy()
        
        # 1. Draw Players (Class 0)
        player_dict = tracks["players"][frame_num]
        for track_id, player in player_dict.items():
            x1, y1, x2, y2 = map(int, player["bbox"])
            # Draw a green bounding box for players
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"P_{track_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 2. Draw the Ball (Class 1)
        ball_dict = tracks["ball"][frame_num]
        for _, ball in ball_dict.items():
            if "bbox" in ball and len(ball["bbox"]) > 0:
                x1, y1, x2, y2 = map(int, ball["bbox"])
                # Draw a distinct red/orange box for the ball
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 165, 255), 3)
                cv2.putText(annotated_frame, "BALL", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        out.write(annotated_frame)
        # Show progress every 30 frames or at the end
        if (frame_num + 1) % 30 == 0 or (frame_num + 1) == total_frames:
            progress_pct = (frame_num + 1) / total_frames * 100
            print(f"  [Annotation] Frame {frame_num + 1:5d} / {total_frames} ({progress_pct:5.1f}%)")
    
    out.release()

def process_video(video_path, model_path, sport):
    print(f"Loading {sport} configuration...")
    config = SPORT_CONFIGS[sport]

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    tracker = MultiSportTracker(model_path=model_path, sport=sport)

    tracks = tracker.get_object_tracks_streaming(
        read_video(video_path), total_frames
    )

    validate_sport_selection(tracks, config, sport)

    output_path = f"outputs/analyzed_{sport}_{os.path.basename(video_path)}"

    draw_and_save_annotations_streaming(
        read_video(video_path),
        tracks,
        output_path,
        fps,
        (frame_width, frame_height),
        total_frames
    )

    return output_path

def main():
    time_start = time.time()
    
    # Setup Argument Parser so you can easily run it from your terminal
    parser = argparse.ArgumentParser(description="Multi-Sport Tracking & Analytics")
    parser.add_argument('--video', type=str, required=True, help="Path to input video")
    parser.add_argument('--model', type=str, required=True, help="Path to YOLO best.onnx or best.pt")
    parser.add_argument('--sport', type=str, choices=['football', 'basketball', 'tennis'], required=True, help="Sport profile to load")
    args = parser.parse_args()

    print(f"Loading {args.sport} configuration...")
    config = SPORT_CONFIGS[args.sport]

    # Grab original FPS and frame count for saving the video later
    cap = cv2.VideoCapture(args.video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"Video Info: {total_frames} frames, {frame_width}x{frame_height} @ {fps} FPS")

    print("Initializing Multi-Sport Tracker...")
    # This automatically applies the ByteTrack for players and Pandas smoothing for the ball
    tracker = MultiSportTracker(model_path=args.model, sport=args.sport)
    
    # Time Tracking
    
    # Use streaming tracking to avoid loading all frames into memory
    tracks = tracker.get_object_tracks_streaming(read_video(args.video), total_frames)

    # Trigger the Failsafe to ensure the user didn't make a mistake
    validate_sport_selection(tracks, config, args.sport)

    # --- OPTIONAL: View Transformer for Speed/Distance ---
    # To use this, you'd define the 4 corners of the court in the video.
    # pixel_vertices = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    # transformer = UniversalViewTransformer(args.sport, pixel_vertices, SPORT_CONFIGS)
    # tracks = transformer.add_transformed_position_to_tracks(tracks) 

    print("Drawing annotations and saving video...")
    output_path = f"analyzed_{args.sport}_output1.mp4"
    draw_and_save_annotations_streaming(read_video(args.video), tracks, output_path, fps, (frame_width, frame_height), total_frames)
    print(f"Done! Video saved to {output_path}")
    
    time_finish = time.time()
    elapsed_time = time_finish - time_start
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()