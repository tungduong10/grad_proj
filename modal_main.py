import os
import modal

ignore_patterns = ["**/__pycache__", "**/*.pyc", "**/.DS_Store"]
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
    .pip_install_from_requirements("/home/moonscar_lap/Codes/graduation_project/grad_proj/requirements.txt")
    .add_local_dir("/home/moonscar_lap/Codes/graduation_project/grad_proj/trackers", remote_path="/root/trackers",ignore=ignore_patterns)
    .add_local_dir("/home/moonscar_lap/Codes/graduation_project/grad_proj/team", remote_path="/root/team",ignore=ignore_patterns)
    .add_local_dir("/home/moonscar_lap/Codes/graduation_project/grad_proj/utils", remote_path="/root/utils",ignore=ignore_patterns)
    .add_local_dir("/home/moonscar_lap/Codes/graduation_project/grad_proj/camera_movement_estimator", remote_path="/root/camera_movement_estimator",ignore=ignore_patterns)
    .add_local_dir("/home/moonscar_lap/Codes/graduation_project/grad_proj/view_transformer", remote_path="/root/view_transformer",ignore=ignore_patterns)
    .add_local_dir("/home/moonscar_lap/Codes/graduation_project/grad_proj/analysis", remote_path="/root/analysis",ignore=ignore_patterns)
    .add_local_dir("/home/moonscar_lap/Codes/graduation_project/grad_proj/keypoints_detectors", remote_path="/root/keypoints_detectors",ignore=ignore_patterns)
    .add_local_dir("/home/moonscar_lap/Codes/graduation_project/grad_proj/config", remote_path="/root/config",ignore=ignore_patterns)
    .add_local_dir("/home/moonscar_lap/Codes/graduation_project/grad_proj/images", remote_path="/root/images",ignore=ignore_patterns)
)

app = modal.App("thesis-tracker-pro", image=image)
volume=modal.Volume.from_name("grad_proj_vol")    



@app.function(
    gpu="A10G",
    memory=8192,
    volumes={"/volume": volume},
    timeout=3600
)
def process_tracker_remote(enable_team_assignment: bool = True):
    import sys
    import gc
    import torch
    # Skip Cudagraph
    torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
    # Add root so mounted packages resolve correctly
    sys.path.insert(0, "/root")
    
    SPORT = "football" # Change to "basketball" when you want to switch!

    from utils import read_video, save_video, get_video_frame_count, Drawer
    from keypoints_detectors.pitch_keypoint_detector import PitchKeypointDetector
    from view_transformer.tactical_view_converter import TacticalViewConverter
    from view_transformer.view_transformer import ViewTransformer
    from camera_movement_estimator.camera_movement_estimator import CameraMovementEstimator
    from analysis.football_acquisition import FootballBallAssigner
    
    if SPORT == "football":
        from trackers import FootballTracker as Tracker
        from team import FootballTeamClassifier as TeamClassifier
        video_path = '/volume/input_folder/football_test.mp4'
        model_path = '/volume/model/yolo11x_v2_best.pt'
        kp_model_path = '/volume/model/football_field_yolo11lv2.pt'
        from config.soccer_config import SoccerPitchConfiguration
        pitch_config = SoccerPitchConfiguration()
        court_image_path = '/root/images/football_pitch.png'
        # Soccer pitch is 10500x6800 cm — spread gates can be generous
        view_transformer = ViewTransformer(min_spread_x=2500.0, min_spread_y=1500.0)
        TEAM_COLORS = {
            0: (255, 191, 0),    # BGR for '#00BFFF' (Light Blue - Team 0)
            1: (147, 20, 255),   # BGR for '#FF1493' (Deep Pink - Team 1)
        }
    elif SPORT == "basketball":
        from trackers.basketball_tracker import BasketballTracker as Tracker
        from team import BasketballTeamClassifier as TeamClassifier
        video_path = '/volume/input_folder/basketball_test2.mp4'
        model_path = '/volume/model/basketball_yolo11l_v2.pt'
        kp_model_path = '/volume/model/basketball_court_yolo11lv2.pt'
        from config.basketball_config_v2 import BasketballPitchConfigurationV2
        pitch_config = BasketballPitchConfigurationV2()
        court_image_path = '/root/images/basketball_court.png'
        # Basketball court is only 2800x1500 cm — much smaller than soccer.
        # Default spread gates (2500x1500) reject almost everything.
        # Relax to ~500x300 so partial-court views still produce a valid homography.
        view_transformer = ViewTransformer(min_spread_x=500.0, min_spread_y=300.0)
        TEAM_COLORS = {
            0: (255, 191, 0),    # BGR for '#00BFFF' (Light Blue - Team 0)
            1: (147, 20, 255)    # BGR for '#FF1493' (Deep Pink - Team 1)
        }
    elif SPORT == "tennis":
        from trackers.tennis_tracker import TennisTracker as Tracker
        video_path = '/volume/input_folder/tennis_test.mp4'
        model_path = '/volume/model/tennis_yolo11l_v1.pt'
        kp_model_path = '/volume/model/tennis_court_yolo11lv1.pt'
        from config.tennis_config import TennisCourtConfiguration
        pitch_config = TennisCourtConfiguration()
        court_image_path = None  # Tennis skips mini-map PIP
        # Tennis court is 2377x1097 cm — relax spread gates accordingly
        view_transformer = ViewTransformer(min_spread_x=400.0, min_spread_y=200.0)
        TEAM_COLORS = {
            0: (0, 191, 255),   # Deep Sky Blue
            1: (147, 20, 255),  # Deep Pink / Purple
        }
    else:
        raise ValueError(f"Unknown sport: {SPORT}")

    print(f"--- Running Tracker ({SPORT.upper()}) on Modal Volume ---")
    
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        print(f"Please ensure input_folder/{os.path.basename(video_path)} is uploaded to the volume.")
        return None
        
    print("Loading video info...")
    total_frames = get_video_frame_count(video_path)
    print(f"Total {total_frames} frames.")

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print(f"Please ensure model/{os.path.basename(model_path)} is uploaded to the volume.")
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

    print("Detecting Court Keypoints...")
    kp_detector = PitchKeypointDetector(kp_model_path)
    court_keypoints = kp_detector.get_court_keypoints(
        frame_generator=read_video(video_path),
        total_frames=total_frames,
        read_from_stub=False, 
        video_path=video_path,
        stub_dir=stub_dir 
    )

    print("Estimating Camera Movement...")
    camera_movement_estimator = CameraMovementEstimator(next(read_video(video_path)))
    camera_movement = camera_movement_estimator.get_camera_movement(
        frame_generator=read_video(video_path),
        total_frames=total_frames,
        video_path=video_path,
        read_from_stub=True,
        stub_dir=stub_dir
    )

    print("Projecting players to tactical view...")
    tactical_converter = TacticalViewConverter(config=pitch_config, transformer=view_transformer)
    court_keypoints = tactical_converter.validate_keypoints(court_keypoints)
    tactical_player_positions = tactical_converter.transform_players_to_tactical_view(
        court_keypoints, tracks['players'], camera_movement_per_frame=camera_movement
    )

    # Debug: count how many frames actually got projections
    frames_with_projections = sum(1 for fp in tactical_player_positions if len(fp) > 0)
    total_projected_players = sum(len(fp) for fp in tactical_player_positions)
    print(f"  → Projected {total_projected_players} player-positions across {frames_with_projections}/{len(tactical_player_positions)} frames")

    # Convert cm (tactical pixels) to meters and store in position_transformed
    for frame_num, frame_positions in enumerate(tactical_player_positions):
        for player_id, pos in frame_positions.items():
            if frame_num < len(tracks['players']) and player_id in tracks['players'][frame_num]:
                x_cm, y_cm = pos
                tracks['players'][frame_num][player_id]['position_transformed'] = [x_cm / 100.0, y_cm / 100.0]

    from analysis import SpeedAndDistanceEstimator
    print("Estimating Speed and Distance...")
    speed_estimator = SpeedAndDistanceEstimator(frame_rate=30, window_size=5)
    speed_estimator.add_speed_and_distance_to_tracks(tracks)

    del tracker
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if SPORT == "tennis":
        # Tennis: assign teams by court position (near side vs far side)
        # Use tactical view if available, else fall back to foot Y-position
        print("Tennis mode — assigning teams by court position...")
        half_court_cm = pitch_config.length / 2  # net position in cm
        
        for frame_num in range(len(tracks['players'])):
            for player_id in tracks['players'][frame_num]:
                team = 0  # default
                
                # Try tactical coordinates first (most accurate)
                if frame_num < len(tactical_player_positions):
                    pos = tactical_player_positions[frame_num].get(player_id)
                    if pos is not None:
                        x_cm, _ = pos
                        team = 0 if x_cm < half_court_cm else 1
                    else:
                        # Fallback: use raw foot Y-position in video frame
                        foot_y = tracks['players'][frame_num][player_id]['bbox'][3]
                        # In broadcast tennis, lower Y = far side, higher Y = near side
                        # Use 40% of frame height as approximate net line (net appears
                        # above center due to perspective)
                        team = 0 if foot_y < 1080 * 0.4 else 1
                else:
                    foot_y = tracks['players'][frame_num][player_id]['bbox'][3]
                    team = 0 if foot_y < 1080 * 0.4 else 1
                
                tracks['players'][frame_num][player_id]['team'] = team
                tracks['players'][frame_num][player_id]['team_color'] = TEAM_COLORS[team]
    elif enable_team_assignment:
        # Team Assignment Phase (SigLIP2)
        siglip2_path = '/volume/model/siglip2-base-patch16-224'
        if not os.path.exists(siglip2_path):
            print(f"Warning: SigLIP2 model not found at {siglip2_path}.")
            print("Falling back to dummy teams. Please ensure siglip2-base-patch16-224 is on the volume.")
            for frame_num in range(len(tracks['players'])):
                for player_id in tracks['players'][frame_num]:
                    tracks['players'][frame_num][player_id]['team'] = 0
                    tracks['players'][frame_num][player_id]['team_color'] = TEAM_COLORS[0]
        else:
            print(f"Initializing Team Classifier (SigLIP2) from {siglip2_path}...")
            team_classifier = TeamClassifier(
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
    else:
        print("Team assignment DISABLED — assigning dummy teams for faster processing...")
        for frame_num in range(len(tracks['players'])):
            for player_id in tracks['players'][frame_num]:
                tracks['players'][frame_num][player_id]['team'] = 0
                tracks['players'][frame_num][player_id]['team_color'] = TEAM_COLORS[0]

    print("Assigning Ball to Players...")
    team_ball_control = []
    import numpy as np

    if SPORT == "basketball":
        from analysis.basketball_acquisition import BasketballBallAssigner
        player_assigner = BasketballBallAssigner()
        possession_list = player_assigner.detect_ball_possession(tracks['players'], tracks.get('ball', []))
        
        for frame_num in range(len(tracks['players'])):
            assigned_player = possession_list[frame_num] if frame_num < len(possession_list) else -1
            if assigned_player != -1:
                tracks['players'][frame_num][assigned_player]['has_ball'] = True
                team_ball_control.append(tracks['players'][frame_num][assigned_player].get('team', 0))
            else:
                if team_ball_control:
                    team_ball_control.append(team_ball_control[-1])
                else:
                    team_ball_control.append(0)
    else:
        player_assigner = FootballBallAssigner()
        for frame_num in range(len(tracks['players'])):
            assigned_player = -1
            if 'ball' in tracks and frame_num < len(tracks['ball']) and 1 in tracks['ball'][frame_num]:
                ball_bbox = tracks['ball'][frame_num][1]['bbox']
                assigned_player = player_assigner.assign_ball_to_player(tracks['players'][frame_num], ball_bbox)

            if assigned_player != -1:
                tracks['players'][frame_num][assigned_player]['has_ball'] = True
                team_ball_control.append(tracks['players'][frame_num][assigned_player].get('team', 0))
            else:
                if team_ball_control:
                    team_ball_control.append(team_ball_control[-1])
                else:
                    team_ball_control.append(0)

    team_ball_control = np.array(team_ball_control)

    print("Drawing output frames...")
    drawer = Drawer()
    drawn_frames_generator = drawer.draw_annotations(read_video(video_path), tracks, team_ball_control)
    drawn_frames_generator = drawer.draw_speed_and_distance(drawn_frames_generator, tracks)
    drawn_frames_generator = drawer.draw_keypoints(drawn_frames_generator, court_keypoints, pitch_config)
    
    # Only draw mini-map PIP for sports where it adds value (not tennis)
    if court_image_path is not None:
        drawn_frames_generator = drawer.draw_mini_map_pip(
            drawn_frames_generator, 
            tracks, 
            tactical_player_positions, 
            court_image_path, 
            pitch_config
        )

    # Save video directly to the volume
    output_dir = '/volume/outputs'
    os.makedirs(output_dir, exist_ok=True)
    output_path = f'{output_dir}/tracker_with_team.mp4'
    
    print(f"Saving video to volume at {output_path}...")
    save_video(drawn_frames_generator, output_path)
    
    print("Reading output video bytes to return to local machine...")
    with open(output_path, "rb") as f:
        video_bytes = f.read()
        
    print("Finished processing on Modal remote worker successfully!")
    return video_bytes

@app.local_entrypoint()
def main(no_teams: bool = False):
    enable_teams = not no_teams
    status = "ENABLED" if enable_teams else "DISABLED"
    print(f"Starting Modal remote tracking job (team assignment: {status})...")
    
    video_bytes = process_tracker_remote.remote(enable_team_assignment=enable_teams)
    
    if video_bytes:
        output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        suffix = 'tracker_with_team' if enable_teams else 'tracker_no_team1'
        output_path = os.path.join(output_dir, f'{suffix}.mp4')
        
        with open(output_path, "wb") as f:
            f.write(video_bytes)
            
        print(f"Success! Video saved to Volume AND downloaded locally to {output_path}")
    else:
        print("Remote tracking failed. Check Modal logs.")