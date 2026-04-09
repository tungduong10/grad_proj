import os
import modal

ignore_patterns = ["**/__pycache__", "**/*.pyc", "**/.DS_Store"]
# We use a CUDA 12.1 + cuDNN 8 devel base image to ensure all symlinks for ONNX Runtime GPU are present
image = (
    modal.Image.from_registry("nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04", add_python="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch", 
        "torchvision", 
        "transformers",         

        # Your explicit core ML & Data packages
        "ultralytics==8.4.32",  
        "onnx==1.17.0",
        "opencv-python-headless==4.11.0.86",
        "onnxruntime-gpu==1.18.1",
        "numpy==1.26.4",
        "pandas==2.2.3",
        "scikit-learn",
        "umap-learn",
        "setuptools"
    )
    # Point it to exactly where the file lives in your folder structure
    .pip_install_from_requirements("/home/moonscar_lap/Codes/graduation_project/requirements.txt")
    .add_local_dir("/home/moonscar_lap/Codes/graduation_project/grad_proj/trackers", remote_path="/root/trackers",ignore=ignore_patterns)
    .add_local_dir("/home/moonscar_lap/Codes/graduation_project/grad_proj/team", remote_path="/root/team",ignore=ignore_patterns)
    .add_local_dir("/home/moonscar_lap/Codes/graduation_project/grad_proj/utils", remote_path="/root/utils",ignore=ignore_patterns)
    .add_local_dir("/home/moonscar_lap/Codes/graduation_project/grad_proj/camera_movement_estimator", remote_path="/root/camera_movement_estimator",ignore=ignore_patterns)
    .add_local_dir("/home/moonscar_lap/Codes/graduation_project/grad_proj/view_transformer", remote_path="/root/view_transformer",ignore=ignore_patterns)
)

app = modal.App("thesis-tracker-pro", image=image)
volume=modal.Volume.from_name("grad_proj_vol")    

@app.function(
    cpu=2,
    gpu="L4",
    memory=6144,
    volumes={"/volume": volume},
    timeout=3600
)
def process_tracker_remote():
    import sys
    import gc
    import torch
    # Add root so mounted packages resolve correctly
    sys.path.insert(0, "/root")
    
    from utils import read_video, save_video, get_video_frame_count, Drawer
    from trackers import Tracker
    from team import TeamClassifier2

    # Define team colors (dummy colors for tracker only test)
    TEAM_COLORS = {
        0: (0, 191, 255),  # Deep Sky Blue
        1: (147, 20, 255), # Deep Pink / Purple
    }

    print("--- Running Tracker on Modal Volume ---")
    
    video_path = '/volume/input_folder/football_test.mp4'
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        print("Please ensure input_folder/football_test.mp4 is uploaded to the volume.")
        return None
        
    print("Loading video info...")
    total_frames = get_video_frame_count(video_path)
    print(f"Total {total_frames} frames.")

    model_path = '/volume/model/yolo11x_v2_best.pt'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please ensure model/yolo11x_v2_best.onnx is uploaded to the volume.")
        return None

    print(f"Initializing Tracker with model: {os.path.basename(model_path)}...")
    tracker = Tracker(model_path)

    print("Running tracking...")
    # Use the volume for stubs too to speed up subsequent runs
    stub_dir = '/volume/stubs'
    os.makedirs(stub_dir, exist_ok=True)
    
    tracks = tracker.get_object_tracks(
        frame_generator=read_video(video_path),
        total_frames=total_frames,
        read_from_stub=False, 
        video_path=video_path,
        stub_dir=stub_dir 
    )
    
    print("Adding positions to tracks...")
    tracker.add_position_to_tracks(tracks)

    del tracker
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Team Assignment Phase (SigLIP2)
    siglip2_path = '/volume/model/siglip2-base-patch16-224'
    if not os.path.exists(siglip2_path):
        print(f"Warning: SigLIP2 model not found at {siglip2_path}.")
        print("Falling back to dummy teams. Please ensure siglip2-base-patch16-224 is on the volume.")
        # Minimal dummy assignment if model is missing
        for frame_num in range(len(tracks['players'])):
            for player_id in tracks['players'][frame_num]:
                tracks['players'][frame_num][player_id]['team'] = 0
                tracks['players'][frame_num][player_id]['team_color'] = TEAM_COLORS[0]
    else:
        print(f"Initializing Team Classifier (SigLIP2) from {siglip2_path}...")
        team_classifier = TeamClassifier2(
            device="cuda" if torch.cuda.is_available() else "cpu",
            batch_size=64,
            model_path=siglip2_path,
            use_fp16=True,
            model_type="siglip2-base"
        )

        print("Fitting Team Classifier (Pass 1)...")
        team_classifier.fit_from_video(tracks['players'], read_video(video_path), sample_stride=10)

        print("Classifying Teams (Pass 2)...")
        team_classifier.classify_from_video(tracks, read_video(video_path), TEAM_COLORS, sample_stride=5)

        team_classifier.release_model()
        del team_classifier
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("Drawing output frames...")
    drawer = Drawer()
    drawn_frames_generator = drawer.draw_annotations(read_video(video_path), tracks)

    # Save video directly to the volume
    output_dir = '/volume/outputs'
    os.makedirs(output_dir, exist_ok=True)
    output_path = f'{output_dir}/tracker_only_test.avi'
    
    print(f"Saving video to volume at {output_path}...")
    save_video(drawn_frames_generator, output_path)
    
    print("Reading output video bytes to return to local machine...")
    with open(output_path, "rb") as f:
        video_bytes = f.read()
        
    print("Finished processing on Modal remote worker successfully!")
    return video_bytes

@app.local_entrypoint()
def main():
    print("Starting Modal remote tracking job using Volume...")
    
    video_bytes = process_tracker_remote.remote()
    
    if video_bytes:
        output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'tracker_only_test.avi')
        
        with open(output_path, "wb") as f:
            f.write(video_bytes)
            
        print(f"Success! Video saved to Volume AND downloaded locally to {output_path}")
    else:
        print("Remote tracking failed. Check Modal logs.")