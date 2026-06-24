import cv2
import pickle
import pandas as pd
import sys
import os
import numpy as np
from ultralytics import YOLO 
import supervision as sv
from utils import measure_distance, get_center_of_bbox, get_foot_position, read_stub, save_stub

class TennisTracker:
    def __init__(self, model_path):
        # We assume a single YOLO model that detects both players and ball
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            minimum_matching_threshold=0.8,
            lost_track_buffer=100
        )

    def add_position_to_tracks(self, tracks):
        for object_type, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object_type == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object_type][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self, ball_positions):
        # ball_positions is a list of dicts: {1: {"bbox": [x1, y1, x2, y2]}} or {}
        ball_bboxes = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_bboxes, columns=['x1', 'y1', 'x2', 'y2'])

        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        interpolated = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]
        return interpolated

    def get_ball_shot_frames(self, ball_positions):
        ball_bboxes = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_bboxes, columns=['x1', 'y1', 'x2', 'y2'])

        df_ball_positions['ball_hit'] = 0
        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()
        
        minimum_change_frames_for_hit = 25
        for i in range(1, len(df_ball_positions) - int(minimum_change_frames_for_hit * 1.2)):
            negative_position_change = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[i+1] < 0
            positive_position_change = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[i+1] > 0

            if negative_position_change or positive_position_change:
                change_count = 0 
                for change_frame in range(i+1, i+int(minimum_change_frames_for_hit * 1.2) + 1):
                    negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[change_frame] < 0
                    positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[change_frame] > 0

                    if negative_position_change and negative_position_change_following_frame:
                        change_count += 1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count += 1
            
                if change_count > minimum_change_frames_for_hit - 1:
                    df_ball_positions.loc[i, 'ball_hit'] = 1

        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit'] == 1].index.tolist()
        return frame_nums_with_ball_hits

    def choose_and_filter_players(self, court_keypoints, player_detections):
        player_detections_first_frame = player_detections[0]
        chosen_players = self.choose_players(court_keypoints, player_detections_first_frame)
        
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {track_id: track_info for track_id, track_info in player_dict.items() if track_id in chosen_players}
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    def choose_players(self, court_keypoints, player_dict):
        distances = []
        for track_id, track_info in player_dict.items():
            bbox = track_info["bbox"]
            player_center = get_center_of_bbox(bbox)

            min_distance = float('inf')
            for i in range(0, len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i+1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))
        
        distances.sort(key = lambda x: x[1])
        # Choose the first 2 tracks
        chosen_players = [distances[0][0], distances[1][0]] if len(distances) >= 2 else [d[0] for d in distances]
        return chosen_players

    def detect_frames(self, frames):
        for frame in frames:
            results = self.model.predict(frame, conf=0.15, verbose=False)
            yield results[0]

    def _get_stub_path(self, video_path, stub_dir):
        if not video_path:
            return None
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        return os.path.join(stub_dir, f"{video_basename}_tennis.pkl")

    def get_object_tracks(self, frame_generator, total_frames, video_path=None, read_from_stub=False, stub_dir="/home/moonscar_lap/Codes/graduation_project/grad_proj/stubs"):
        stub_path = self._get_stub_path(video_path, stub_dir)
        
        if stub_path and read_from_stub:
            if os.path.isfile(stub_path):
                tracks = read_stub(True, stub_path)
                if tracks:
                    return tracks

        detections = self.detect_frames(frame_generator)
        
        tracks = {
            "players": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}
            
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            ball_id = cls_names_inv.get("ball", cls_names_inv.get("Ball", None))
            player_id = cls_names_inv.get("person", cls_names_inv.get("player", cls_names_inv.get("Player", None)))
            
            if player_id is not None:
                player_detections = detection_supervision[detection_supervision.class_id == player_id]
                player_detections = player_detections.with_nms(threshold=0.5, class_agnostic=True)
                detection_with_tracks = self.tracker.update_with_detections(player_detections)
            else:
                detection_with_tracks = []
                
            tracks["players"].append({})
            tracks["ball"].append({})
            
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                track_id = frame_detection[4]
                tracks["players"][frame_num][track_id] = {"bbox": bbox}

            # replace this block in get_object_tracks
            if ball_id is not None:
                ball_detections = detection_supervision[detection_supervision.class_id == ball_id]
                chosen_bbox = None
                
                if len(ball_detections) > 0:
                    # get all ball candidates this frame
                    candidates = [
                        (frame_det[0].tolist(), frame_det[2])   # (bbox, confidence)
                        for frame_det in ball_detections
                    ]
                    
                    if len(candidates) == 1:
                        # no ambiguity — use it directly
                        chosen_bbox = candidates[0][0]
                    else:
                        # multiple detections — use positional continuity
                        last_ball = None
                        for past_frame in range(frame_num - 1, max(frame_num - 10, -1), -1):
                            if tracks["ball"][past_frame].get(1):
                                last_ball = tracks["ball"][past_frame][1]["bbox"]
                                break

                        if last_ball is None:
                            # no prior position yet — fall back to max confidence
                            chosen_bbox = max(candidates, key=lambda x: x[1])[0]
                        else:
                            last_center = (
                                (last_ball[0] + last_ball[2]) / 2,
                                (last_ball[1] + last_ball[3]) / 2,
                            )
                            def score(candidate):
                                bbox, conf = candidate
                                cx = (bbox[0] + bbox[2]) / 2
                                cy = (bbox[1] + bbox[3]) / 2
                                dist = ((cx - last_center[0])**2 + (cy - last_center[1])**2) ** 0.5
                                return dist   # lower is better

                            chosen_bbox = min(candidates, key=score)[0]

                if chosen_bbox is not None:
                    tracks["ball"][frame_num][1] = {"bbox": chosen_bbox}
            
            if (frame_num + 1) % 30 == 0 or (frame_num + 1) == total_frames:
                print(f"  [Tennis Tracking] Frame {frame_num + 1:5d} / {total_frames}")

        tracks["ball"] = self.interpolate_ball_positions(tracks["ball"])    

        if stub_path:
            os.makedirs(stub_dir, exist_ok=True)
            save_stub(stub_path, tracks)

        return tracks

    def draw_bboxes(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            for track_id, track_info in player_dict.items():
                bbox = track_info["bbox"]
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

            for track_id, track_info in ball_dict.items():
                bbox = track_info["bbox"]
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Ball", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                
            output_video_frames.append(frame)
        
        return output_video_frames
