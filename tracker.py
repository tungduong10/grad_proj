import cv2
from ultralytics import YOLO

def fast_sports_tracker(video_path, model_path, output_path):
    print(f"Loading optimized model from: {model_path}")
    
    # Load the compiled model (best.engine or best.onnx) instead of best.pt
    model = YOLO(model_path) 
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Original video dimensions
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # --- APPROACH 4: Output Downscaling ---
    # Even if the source is 4K, write the output file at 1080p. 
    # This prevents your hard drive's write-speed from bottlenecking the AI.
    out_width, out_height = 1920, 1080
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
    
    # --- APPROACH 3: Frame Skipping Setup ---
    # Process every 3rd frame with YOLO. For the 2 frames in between, 
    # we will just redraw the last known bounding boxes to save massive compute time.
    frame_skip = 3 
    last_results = None 
    
    print(f"Processing {total_frames} frames at blazing speed...")
    
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        frame_count += 1
        
        # Only run the heavy YOLO engine every 3rd frame
        if frame_count % frame_skip == 1 or last_results is None:
            # --- APPROACH 1: Hardware Acceleration ---
            # half=True uses FP16 memory. device=0 forces the local GPU.
            results = model.track(
                frame, 
                persist=True, 
                tracker="bytetrack.yaml", 
                conf=0.1,     
                iou=0.5,       
                imgsz=1280,    
                verbose=False
            )
            # Cache the results for the next skipped frames
            last_results = results[0]
            annotated_frame = last_results.plot()
        else:
            # FRAME SKIP LOGIC: We don't run YOLO here. 
            # We simply take the boxes/IDs from the last processed frame 
            # and draw them directly onto this new, current frame.
            annotated_frame = last_results.plot(img=frame)

        # --- APPROACH 4: Output Downscaling (Execution) ---
        # Shrink the frame just before saving it to the MP4 file
        resized_frame = cv2.resize(annotated_frame, (out_width, out_height))
        out.write(resized_frame)
        
        if frame_count % 100 == 0:
            print(f"Fast-processed frame {frame_count}/{total_frames}...")

    cap.release()
    out.release()
    print(f"High-speed tracking complete! Saved to {output_path}")


# --- Execute the Tracker ---
if __name__ == "__main__":
    fast_sports_tracker(
        video_path="../input_folder/test (1).mp4", 
        model_path="../best.onnx", 
        output_path="tracked_output.mp4"
    )