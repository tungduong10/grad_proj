from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd
from utils.configs import SPORT_CONFIGS

class MultiSportTracker:
    def __init__(self, model_path, sport):
        self.model = YOLO(model_path, task='detect')
        self.tracker = sv.ByteTrack()
        self.sport = sport
        self.config = SPORT_CONFIGS[sport]

    def get_object_tracks_streaming(self, frame_generator, total_frames):
        """
        High-Accuracy Streaming: Implements Loop Fusion and 
        Lower Inference Resolution (imgsz=960) for maximum speed but have to change the model first, 
        but avoids Frame Skipping to maintain maximum ByteTrack ID stability.
        """
        tracks = {"players": [], "ball": []}
        
        max_ball_dist = self.config['max_ball_travel_per_frame']
        last_good_ball_box = None
        last_good_ball_frame = -1

        print(f"Running high-accuracy streaming inference & tracking for {self.sport}...")
        
        for frame_num, frame in enumerate(frame_generator):
            
            # --- OPTION 2: Lower imgsz to 960 for math acceleration ---
            result = self.model.predict([frame], conf=0.1, imgsz=960, verbose=False)[0] #for now 1280
            det_sv = sv.Detections.from_ultralytics(result)
            
            # --- OPTION 3: Loop Fusion ---
            # Phase A: Player Tracking (ByteTrack)
            player_detections = det_sv[det_sv.class_id == 0]
            tracked_players = self.tracker.update_with_detections(player_detections)
            
            player_dict = {}
            for bbox, _, conf, cls_id, track_id, _ in tracked_players:
                player_dict[track_id] = {"bbox": bbox.tolist()}
            tracks["players"].append(player_dict)
            
            # Phase B: Ball Tracking (Physics Filter)
            ball_detections = det_sv[det_sv.class_id == 1]
            ball_dict = {}
            
            if len(ball_detections) > 0:
                best_ball_idx = np.argmax(ball_detections.confidence)
                curr_box = ball_detections.xyxy[best_ball_idx].tolist()
                
                # Apply the Physics Guardrail
                if last_good_ball_box is None:
                    ball_dict[1] = {"bbox": curr_box}
                    last_good_ball_box = curr_box
                    last_good_ball_frame = frame_num
                else:
                    gap = frame_num - last_good_ball_frame
                    adjusted_max_dist = max_ball_dist * gap
                    
                    prev_center = ((last_good_ball_box[0]+last_good_ball_box[2])/2, (last_good_ball_box[1]+last_good_ball_box[3])/2)
                    curr_center = ((curr_box[0]+curr_box[2])/2, (curr_box[1]+curr_box[3])/2)
                    dist = np.linalg.norm(np.array(prev_center) - np.array(curr_center))
                    
                    if dist <= adjusted_max_dist:
                        ball_dict[1] = {"bbox": curr_box}
                        last_good_ball_box = curr_box
                        last_good_ball_frame = frame_num

            tracks["ball"].append(ball_dict)

            # Show progress every 30 frames
            if (frame_num + 1) % 30 == 0 or (frame_num + 1) == total_frames:
                progress_pct = (frame_num + 1) / total_frames * 100
                print(f"  [Pipeline] Frame {frame_num + 1:5d} / {total_frames} ({progress_pct:5.1f}%)")

        # --- PHASE C: Pandas Interpolation ---
        print("Smoothing ball trajectory for natural occlusions...")
        tracks["ball"] = self.interpolate_ball_positions(tracks["ball"])
        
        return tracks

    def interpolate_ball_positions(self, ball_positions):
        # Extract boxes into a Pandas DataFrame
        bboxes = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df = pd.DataFrame(bboxes, columns=['x1', 'y1', 'x2', 'y2'])

        # Mathematically fill in the missing frames 
        df = df.interpolate()
        df = df.bfill() 

        return [{1: {"bbox": x}} for x in df.to_numpy().tolist()]