from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd
from utils.sport_logic import SportLogicFactory
from utils.configs5 import SPORT_CONFIGS

class MultiSportTracker:
    def __init__(self, model_path, sport):
        self.model = YOLO(model_path, task='detect')
        self.tracker = sv.ByteTrack()
        self.sport = sport
        self.config = SPORT_CONFIGS[sport]
        self.sport_logic = SportLogicFactory.get_logic(sport, self.config)

    def get_object_tracks_streaming(self, frame_generator, total_frames, fps=30):
        """
        Streaming tracker with sport-specific ball filtering
        **Better than tracker4 because it uses sport logic for ball selection**
        """
        tracks = {"players": [], "ball": []}
        
        max_ball_dist = self.config['max_ball_travel_per_frame']
        last_good_ball_box = None
        last_good_ball_frame = -1

        print(f"Running streaming inference & tracking for {self.sport}...")
        print(f"  - Using sport-specific ball filtering")
        
        for frame_num, frame in enumerate(frame_generator):            
            # Inference
            result = self.model.predict([frame], conf=0.1, imgsz=1280, verbose=False)[0]
            det_sv = sv.Detections.from_ultralytics(result)
            
            # Phase A: Player Tracking (ByteTrack)
            player_detections = det_sv[det_sv.class_id == 0]
            tracked_players = self.tracker.update_with_detections(player_detections)
            
            player_dict = {}
            for bbox, _, conf, cls_id, track_id, _ in tracked_players:
                player_dict[track_id] = {"bbox": bbox.tolist()}
            tracks["players"].append(player_dict)
            
            # Phase B: Ball Tracking with SPORT-SPECIFIC LOGIC (main advantage over tracker4)
            ball_detections = det_sv[det_sv.class_id == 1]
            ball_dict = {}
            
            if len(ball_detections) > 0:
                # **ADVANTAGE:** Sport-specific ball filtering
                # Football: picks highest confidence
                # Basketball: picks closest to previous position (more stable)
                # Tennis: picks closest to previous position (more stable)
                curr_box = self.sport_logic.filter_ball(
                    ball_detections,
                    last_good_ball_box,
                    frame_num,
                    last_good_ball_frame
                )
                
                # Physics-based validation
                if last_good_ball_box is None:
                    ball_dict[1] = {"bbox": curr_box}
                    last_good_ball_box = curr_box
                    last_good_ball_frame = frame_num
                else:
                    gap = frame_num - last_good_ball_frame
                    adjusted_max_dist = max_ball_dist * gap
                    
                    prev_center = np.array([(last_good_ball_box[0] + last_good_ball_box[2]) / 2,
                                           (last_good_ball_box[1] + last_good_ball_box[3]) / 2])
                    curr_center = np.array([(curr_box[0] + curr_box[2]) / 2,
                                           (curr_box[1] + curr_box[3]) / 2])
                    dist = np.linalg.norm(curr_center - prev_center)
                    
                    # Accept if within physics limits
                    if dist <= adjusted_max_dist:
                        ball_dict[1] = {"bbox": curr_box}
                        last_good_ball_box = curr_box
                        last_good_ball_frame = frame_num

            tracks["ball"].append(ball_dict)

            # Show progress every 30 frames
            if (frame_num + 1) % 30 == 0 or (frame_num + 1) == total_frames:
                progress_pct = (frame_num + 1) / total_frames * 100
                print(f"  [Pipeline] Frame {frame_num + 1:5d} / {total_frames} ({progress_pct:5.1f}%)")

        # Minimal post-processing: Just Pandas interpolation for gaps
        print("Filling ball trajectory gaps...")
        tracks["ball"] = self.interpolate_ball_positions(tracks["ball"])
        
        return tracks

    def interpolate_ball_positions(self, ball_positions):
        """Fill missing ball detections with linear interpolation"""
        bboxes = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df = pd.DataFrame(bboxes, columns=['x1', 'y1', 'x2', 'y2'])

        # Mathematically fill in the missing frames 
        df = df.interpolate(method='linear')
        df = df.bfill()

        return [{1: {"bbox": x}} for x in df.to_numpy().tolist()]