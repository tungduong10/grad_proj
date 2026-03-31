import cv2
from ultralytics import YOLO

def run_sports_tracker(video_path, model_path, output_path):
    print(f"Loading custom weights from: {model_path}")
    model = YOLO(model_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing {total_frames} frames in the background...")
    print("Press Ctrl+C in your terminal to stop early.")
    
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        frame_count += 1
        
        # Print progress every 50 frames
        if frame_count % 50 == 0:
            print(f"Processing frame {frame_count}/{total_frames}...")
        
        results = model.track(
            frame, 
            persist=True, 
            tracker="bytetrack.yaml", 
            conf=0.25,     
            iou=0.5,       
            imgsz=1280,    
            verbose=False
        )
        
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

    cap.release()
    out.release()
    print(f"Tracking complete! Saved to {output_path}")

# --- Execute the Tracker ---
if __name__ == "__main__":
    run_sports_tracker(
        video_path="input_folder/test (1).mp4", 
        model_path="best.pt", 
        output_path="tracked_output.mp4"
    )