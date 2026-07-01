import os
import sys
import gc
import torch
import argparse
import numpy as np

# Ensure the local modules can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import read_video, save_video, get_video_frame_count, Drawer
from keypoints_detectors.pitch_keypoint_detector import PitchKeypointDetector
from view_transformer.tactical_view_converter import TacticalViewConverter
from view_transformer.view_transformer import ViewTransformer
from camera_movement_estimator.camera_movement_estimator import CameraMovementEstimator
from analysis.football_acquisition import FootballBallAssigner

def process_tracker_local(sport: str, input_filename: str, enable_team_assignment: bool = True):
    # Skip Cudagraph
    torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
    
    SPORT = sport.lower()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    video_path = os.path.join(project_root, 'input_folder', input_filename)
    
    models_dir = os.path.join(base_dir, 'models')
    images_dir = os.path.join(base_dir, 'images')

    if SPORT == "football":
        from trackers import FootballTracker as Tracker
        from team import FootballTeamClassifier as TeamClassifier
        model_path = os.path.join(models_dir, 'yolo11x_v2_best.pt')
        kp_model_path = os.path.join(models_dir, 'football_field_yolo11lv2.pt')
        from config.soccer_config import SoccerPitchConfiguration
        pitch_config = SoccerPitchConfiguration()
        court_image_path = os.path.join(images_dir, 'football_pitch.png')
        view_transformer = ViewTransformer(min_spread_x=2500.0, min_spread_y=1500.0)
        TEAM_COLORS = {
            0: (255, 191, 0),    # BGR for '#00BFFF' (Light Blue - Team 0)
            1: (147, 20, 255),   # BGR for '#FF1493' (Deep Pink - Team 1)
        }
    elif SPORT == "basketball":
        from trackers.basketball_tracker import BasketballTracker as Tracker
        from team import BasketballTeamClassifier as TeamClassifier
        model_path = os.path.join(models_dir, 'basketball_yolo11l_v2.pt')
        kp_model_path = os.path.join(models_dir, 'basketball_court_yolo11lv2.pt')
        from config.basketball_config_v2 import BasketballPitchConfigurationV2
        pitch_config = BasketballPitchConfigurationV2()
        court_image_path = os.path.join(images_dir, 'basketball_court.png')
        view_transformer = ViewTransformer(min_spread_x=500.0, min_spread_y=300.0)
        TEAM_COLORS = {
            0: (255, 191, 0),
            1: (147, 20, 255)
        }
    elif SPORT == "tennis":
        from trackers.tennis_tracker import TennisTracker as Tracker
        model_path = os.path.join(models_dir, 'tennis_yolo11l_v1.pt')
        kp_model_path = os.path.join(models_dir, 'tennis_court_yolo11lv1.pt')
        from config.tennis_config import TennisCourtConfiguration
        pitch_config = TennisCourtConfiguration()
        court_image_path = None
        view_transformer = ViewTransformer(min_spread_x=400.0, min_spread_y=200.0)
        TEAM_COLORS = {
            0: (0, 191, 255),
            1: (147, 20, 255),
        }
    else:
        raise ValueError(f"Unknown sport: {SPORT}")

    print(f"--- Running Tracker ({SPORT.upper()}) Locally ---")
    
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return None
        
    print("Loading video info...")
    total_frames = get_video_frame_count(video_path)
    print(f"Total {total_frames} frames.")

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return None

    print(f"Initializing Tracker with model: {os.path.basename(model_path)}...")
    tracker = Tracker(model_path)

    print("Running tracking...")
    stub_dir = os.path.join(base_dir, 'stubs')
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

    if SPORT == "tennis" and "ball" in tracks:
        tactical_ball_positions = tactical_converter.transform_players_to_tactical_view(
            court_keypoints, tracks["ball"], camera_movement_per_frame=camera_movement
        )
        for frame_num, frame_positions in enumerate(tactical_ball_positions):
            for ball_id, pos in frame_positions.items():
                if frame_num < len(tracks["ball"]) and ball_id in tracks["ball"][frame_num]:
                    x_cm, y_cm = pos
                    tracks["ball"][frame_num][ball_id]["position_transformed"] = [x_cm / 100.0, y_cm / 100.0]

    frames_with_projections = sum(1 for fp in tactical_player_positions if len(fp) > 0)
    total_projected_players = sum(len(fp) for fp in tactical_player_positions)
    print(f"  → Projected {total_projected_players} player-positions across {frames_with_projections}/{len(tactical_player_positions)} frames")

    for frame_num, frame_positions in enumerate(tactical_player_positions):
        for player_id, pos in frame_positions.items():
            if frame_num < len(tracks['players']) and player_id in tracks['players'][frame_num]:
                x_cm, y_cm = pos
                tracks['players'][frame_num][player_id]['position_transformed'] = [x_cm / 100.0, y_cm / 100.0]

    from analysis import SpeedAndDistanceEstimator
    print("Estimating Speed and Distance...")
    calculate_ball_speed = (SPORT == "tennis")
    speed_estimator = SpeedAndDistanceEstimator(frame_rate=30, window_size=5, calculate_ball_speed=calculate_ball_speed)
    speed_estimator.add_speed_and_distance_to_tracks(tracks)

    del tracker
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if SPORT == "tennis":
        print("Tennis mode — assigning teams by court position...")
        half_court_cm = pitch_config.length / 2 
        
        for frame_num in range(len(tracks['players'])):
            for player_id in tracks['players'][frame_num]:
                team = 0 
                if frame_num < len(tactical_player_positions):
                    pos = tactical_player_positions[frame_num].get(player_id)
                    if pos is not None:
                        x_cm, _ = pos
                        team = 0 if x_cm < half_court_cm else 1
                    else:
                        foot_y = tracks['players'][frame_num][player_id]['bbox'][3]
                        team = 0 if foot_y < 1080 * 0.4 else 1
                else:
                    foot_y = tracks['players'][frame_num][player_id]['bbox'][3]
                    team = 0 if foot_y < 1080 * 0.4 else 1
                
                tracks['players'][frame_num][player_id]['team'] = team
                tracks['players'][frame_num][player_id]['team_color'] = TEAM_COLORS[team]
    elif enable_team_assignment:
        siglip2_path = os.path.join(models_dir, 'siglip2-base-patch16-224')
        if not os.path.exists(siglip2_path):
            print(f"Warning: SigLIP2 model not found at {siglip2_path}.")
            print("Falling back to dummy teams.")
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

    team_ball_control = None
    passes = None
    interceptions = None

    if SPORT in ["football", "basketball"]:
        print("Assigning Ball to Players...")
        team_ball_control_list = []

        if SPORT == "basketball":
            from analysis.basketball_acquisition import BasketballBallAssigner
            player_assigner = BasketballBallAssigner()
            possession_list = player_assigner.detect_ball_possession(tracks['players'], tracks.get('ball', []))
            
            for frame_num in range(len(tracks['players'])):
                assigned_player = possession_list[frame_num] if frame_num < len(possession_list) else -1
                if assigned_player != -1:
                    tracks['players'][frame_num][assigned_player]['has_ball'] = True
                    team_ball_control_list.append(tracks['players'][frame_num][assigned_player].get('team', 0))
                else:
                    if team_ball_control_list:
                        team_ball_control_list.append(team_ball_control_list[-1])
                    else:
                        team_ball_control_list.append(0)
        else:
            player_assigner = FootballBallAssigner()
            possession_list = [-1] * len(tracks['players'])
            for frame_num in range(len(tracks['players'])):
                assigned_player = -1
                if 'ball' in tracks and frame_num < len(tracks['ball']) and 1 in tracks['ball'][frame_num]:
                    ball_bbox = tracks['ball'][frame_num][1]['bbox']
                    assigned_player = player_assigner.assign_ball_to_player(tracks['players'][frame_num], ball_bbox)
                
                possession_list[frame_num] = assigned_player

                if assigned_player != -1:
                    tracks['players'][frame_num][assigned_player]['has_ball'] = True
                    team_ball_control_list.append(tracks['players'][frame_num][assigned_player].get('team', 0))
                else:
                    if team_ball_control_list:
                        team_ball_control_list.append(team_ball_control_list[-1])
                    else:
                        team_ball_control_list.append(0)

        team_ball_control = np.array(team_ball_control_list)

        print("Detecting Passes and Interceptions...")
        from analysis.pass_and_interception_detector import PassAndInterceptionDetector
        pass_detector = PassAndInterceptionDetector()
        passes = pass_detector.detect_passes(possession_list, tracks['players'])
        interceptions = pass_detector.detect_interceptions(possession_list, tracks['players'])
        
        total_passes = sum(1 for p in passes if p != -1)
        total_interceptions = sum(1 for i in interceptions if i != -1)
        print(f"  → Detected {total_passes} passes and {total_interceptions} interceptions!")

    print("Drawing output frames...")
    drawer = Drawer()
    drawn_frames_generator = drawer.draw_annotations(read_video(video_path), tracks, team_ball_control, passes, interceptions)
    drawn_frames_generator = drawer.draw_speed_and_distance(drawn_frames_generator, tracks, is_tennis=SPORT=="tennis")

    player_stats_df = None
    if SPORT == "tennis":
        print("Detecting tennis shots and calculating stats...")
        from analysis.tennis_shot_detector import TennisShotDetector
        shot_detector = TennisShotDetector()
        ball_shot_frames = shot_detector.get_ball_shot_frames(tracks.get('ball', []))
        player_stats_df = shot_detector.calculate_stats(tracks, ball_shot_frames, fps=30)
        
        if player_stats_df is not None and not player_stats_df.empty:
            drawn_frames_generator = drawer.draw_tennis_stats(drawn_frames_generator, player_stats_df, tracks=tracks)
    
    if court_image_path is not None:
        drawn_frames_generator = drawer.draw_mini_map_pip(
            drawn_frames_generator, 
            tracks, 
            tactical_player_positions, 
            court_image_path, 
            pitch_config
        )

    output_dir = os.path.join(base_dir, 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{input_filename}_processed.mp4')
    
    print(f"Saving video locally at {output_path}...")
    save_video(drawn_frames_generator, output_path)
    print(f"Success! Video saved to {output_path}")

    if SPORT in ["football", "basketball"] and court_image_path is not None:
        print("Generating per-team heatmaps...")
        from analysis.heatmap_generator import HeatmapGenerator
        heatmap_gen = HeatmapGenerator(config=pitch_config, court_image_path=court_image_path)

        for team_idx in (0, 1):
            png_bytes = heatmap_gen.generate(tactical_player_positions, tracks, team=team_idx)
            heatmap_path = os.path.join(output_dir, f'{input_filename}_team_{team_idx}_heatmap.png')
            with open(heatmap_path, "wb") as f:
                f.write(png_bytes)
            print(f"  → Team {team_idx + 1} heatmap saved to: {heatmap_path}")

    print("Finished processing successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sports tracker locally")
    parser.add_argument("--sport", type=str, default="basketball", help="Sport to analyze (football, basketball, tennis)")
    parser.add_argument("--input", type=str, default="basketball_test2.mp4", help="Input filename inside the input_folder")
    parser.add_argument("--no-teams", action="store_true", help="Disable team assignment")
    args = parser.parse_args()

    enable_teams = not args.no_teams
    status = "ENABLED" if enable_teams else "DISABLED"
    print(f"Starting local tracking job (sport={args.sport}, file={args.input}, teams={status})...")
    
    process_tracker_local(
        sport=args.sport,
        input_filename=args.input,
        enable_team_assignment=enable_teams
    )
