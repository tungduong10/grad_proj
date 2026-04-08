import os
import gc

import modal

app = modal.App("tracker-only-test")

# Define an image with the required dependencies and system libraries for OpenCV/YOLO.
# We use an NVIDIA CUDA runtime base image because ONNX Runtime requires libcublas and cuDNN to connect to the GPU.
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.11")
    .pip_install(
        "ultralytics>=8.0.0",
        "opencv-python-headless>=4.8.0",
        "pandas",
        "supervision",
        "python-dotenv",
        "onnxruntime-gpu"
    )
    .apt_install("libgl1", "libglib2.0-0")
)

# We need to include the entire project directory into the image so the remote worker 
# can access your code (trackers, utils) and the video/model files.
# We explicitly ignore massive folders to make syncing extremely fast (from ~12GB down to ~400MB)
image = image.add_local_dir(
    "/home/moonscar_lap/Codes/graduation_project",
    remote_path="/root/project",
    ignore=[
        "grad_venv",        # 7.8GB Virtual environment
        ".git",             # Frequently changing index invalidates cache
        "outputs",          # Old video outputs
        "data",             # 500MB Data folder
        "raw_models",       # 300MB Raw models folder
        "player_referee_dataset.zip",
        "grad_proj/*.mp4",  # Stray output files in the code dir
        "__pycache__"
    ]
)

@app.function(image=image, gpu="any", timeout=3600)
def process_tracker_remote():
    """
    This function runs entirely on the remote Modal worker using a GPU.
    """
    import sys
    # Add project directory to python path so imports work correctly
    sys.path.insert(0, "/root/project/grad_proj")
    
    from utils import read_video, save_video, get_video_frame_count, Drawer
    from trackers import Tracker

    # Define team colors (dummy colors for tracker only test)
    TEAM_COLORS = {
        0: (0, 191, 255),  # Deep Sky Blue
        1: (147, 20, 255), # Deep Pink / Purple
    }

    print("--- Testing Tracker Module On Modal ---")
    
    # Make sure we use the paths inside the remote container
    video_path = '/root/project/input_folder/football_test.mp4'
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return None
        
    print("Loading video info...")
    total_frames = get_video_frame_count(video_path)
    print(f"Total {total_frames} frames.")

    # Initialize Tracker
    model_path = '/root/project/grad_proj/model/yolo11x_v2_best.onnx'
    print(f"Initializing Tracker with model: {os.path.basename(model_path)}...")
    tracker = Tracker(model_path)

    # Note: Setting read_from_stub=False to FORCE tracking for the test
    print("Running tracking (ignoring stubs)...")
    tracks = tracker.get_object_tracks(
        frame_generator=read_video(video_path),
        total_frames=total_frames,
        read_from_stub=False, 
        video_path=video_path,
        # Write stub to the container's directory (though it won't sync back to host right now)
        stub_dir='/root/project/grad_proj/stubs' 
    )
    
    # Get object positions 
    print("Adding positions to tracks...")
    tracker.add_position_to_tracks(tracks)

    # Free tracker
    del tracker
    gc.collect()

    # Assign all players to team 0 for visualization purposes
    print("Assigning dummy teams for drawing...")
    for frame_num in range(len(tracks['players'])):
        for player_id in tracks['players'][frame_num]:
            tracks['players'][frame_num][player_id]['team'] = 0
            tracks['players'][frame_num][player_id]['team_color'] = TEAM_COLORS[0]

    # Draw output 
    print("Drawing output frames...")
    drawer = Drawer()
    drawn_frames_generator = drawer.draw_annotations(read_video(video_path), tracks)

    # Save video to the container's /tmp directory temporarily
    output_path = '/tmp/tracker_only_test.avi'
    print(f"Saving video to remote {output_path}...")
    save_video(drawn_frames_generator, output_path)
    
    print("Reading output video bytes to send back to local machine...")
    with open(output_path, "rb") as f:
        video_bytes = f.read()
        
    print("Finished processing on Modal remote worker successfully!")
    return video_bytes

@app.local_entrypoint()
def main():
    print("Starting Modal remote tracking job...")
    
    # Call the remote function. This will push the container start and run execution.
    video_bytes = process_tracker_remote.remote()
    
    if video_bytes:
        # Save the returned video bytes locally
        output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'tracker_only_test.avi')
        
        with open(output_path, "wb") as f:
            f.write(video_bytes)
            
        print(f"Success! Video fetched from Modal and saved locally to {output_path}")
    else:
        print("Remote tracking failed or video bytes were empty.")

