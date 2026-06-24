from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd
import os
from utils import save_stub, read_stub, get_foot_position, get_center_of_bbox

class BasketballTracker:
    def __init__(self, model_path):
        self.model = YOLO(model=model_path, task='detect')
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.55,  # High threshold to START a track (ignores false positive fans)
            minimum_matching_threshold=0.9,
            lost_track_buffer=150
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
        """Interpolate missing ball positions using pandas linear interpolation"""
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions

    def detect_frames(self, frames):
        """Streaming detection - yields one result per frame (memory efficient)"""
        for frame in frames:
            results = self.model.predict(
                frame,
                conf=0.15,  # Feed low-confidence boxes so ByteTrack can recover striding/blurred players
                verbose=False
            )
            yield results[0]

    def _get_stub_path(self, video_path, stub_dir):
        if not video_path:
            return None
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        return os.path.join(stub_dir, f"{video_basename}_basketball.pkl")

    def _load_tracks_from_stub(self, stub_path, total_frames):
        if not os.path.isfile(stub_path):
            return None
        try:
            tracks = read_stub(True, stub_path)
            if len(tracks.get("players", [])) == total_frames:
                print(f"✓ Loaded basketball tracks from stub: {stub_path}")
                return tracks
            else:
                print(f"⚠ Stub frame count mismatch ({len(tracks.get('players', []))} vs {total_frames}). Re-tracking...")
        except Exception as e:
            print(f"⚠ Failed to load stub: {e}. Re-tracking...")
        return None

    def _save_tracks_to_stub(self, stub_path, stub_dir, tracks):
        os.makedirs(stub_dir, exist_ok=True)
        print(f"Saving stub to: {os.path.abspath(stub_path)}")
        try:
            save_stub(stub_path, tracks)
            if os.path.isfile(stub_path):
                file_size = os.path.getsize(stub_path)
                print(f"✓ Saved basketball tracks to stub: {stub_path} ({file_size} bytes)")
            else:
                print(f"⚠ Stub file was not created at: {stub_path}")
        except Exception as e:
            print(f"⚠ Failed to save stub: {e}")
            import traceback
            traceback.print_exc()

    def _track_detections(self, detections, total_frames):
        # Classes to ignore (not used for tracking)
        IGNORED_CLASSES = {
            "number", "ball-in-basket", "player-in-possession",
            "player-jump-shot", "player-layup-dunk", "player-shot-block",
        }

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        # Physics filter state for ball (basketball-specific strength)
        max_allowed_distance = 25
        last_good_box = None
        last_good_frame_index = -1

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Convert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Resolve class IDs
            ball_id = cls_names_inv.get("Ball", cls_names_inv.get("ball", None))
            player_id = cls_names_inv.get("Player", cls_names_inv.get("player", None))
            ref_id = cls_names_inv.get("Ref", cls_names_inv.get("referee", None))
            ignored_ids = {cls_names_inv[c] for c in IGNORED_CLASSES if c in cls_names_inv}

            # --- Separate detections by category ---
            # Ball detections (handled separately with physics filter)
            if ball_id is not None:
                ball_detections = detection_supervision[detection_supervision.class_id == ball_id]
                if len(ball_detections) > 0:
                    ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)
            else:
                ball_detections = sv.Detections.empty()

            # Filter out ball + ignored classes, keep only trackable entities (Player, Ref)
            trackable_ids = {player_id, ref_id} - {None}
            mask = np.isin(detection_supervision.class_id, list(trackable_ids))
            trackable_detections = detection_supervision[mask]

            # Apply class-agnostic NMS to prevent the exact same person from being tracked 
            # twice (e.g., if the model is uncertain and outputs both a Player and Referee 
            # box, or a blurry split box). A threshold of 0.5 aggressively removes duplicates.
            # Temporary occlusions between different players are safely handled by ByteTrack's 
            # Kalman filter and lost_track_buffer, so aggressive NMS is preferred here.
            trackable_detections = trackable_detections.with_nms(threshold=0.5, class_agnostic=True)

            # Track players/refs with ByteTrack for persistent IDs
            detection_with_tracks = self.tracker.update_with_detections(trackable_detections)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == ref_id:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}
                else:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

            # --- Ball: highest-confidence pick + physics filter (fused) ---
            chosen_bbox = None
            max_confidence = 0

            for frame_detection in ball_detections:
                bbox = frame_detection[0].tolist()
                confidence = frame_detection[2]
                if max_confidence < confidence:
                    chosen_bbox = bbox
                    max_confidence = confidence

            if chosen_bbox is not None:
                if last_good_box is None:
                    # First detection, accept it
                    tracks["ball"][frame_num][1] = {"bbox": chosen_bbox}
                    last_good_box = chosen_bbox
                    last_good_frame_index = frame_num
                else:
                    # Physics validation: reject implausible jumps
                    frame_gap = frame_num - last_good_frame_index
                    adjusted_max_distance = max_allowed_distance * frame_gap
                    dist = np.linalg.norm(
                        np.array(last_good_box[:2]) - np.array(chosen_bbox[:2])
                    )
                    if dist <= adjusted_max_distance:
                        tracks["ball"][frame_num][1] = {"bbox": chosen_bbox}
                        last_good_box = chosen_bbox
                        last_good_frame_index = frame_num
                    # else: skip this detection as physically implausible

            # Progress monitoring (every 30 frames for consistency)
            if (frame_num + 1) % 30 == 0 or (frame_num + 1) == total_frames:
                progress_pct = (frame_num + 1) / total_frames * 100
                print(f"  [Basketball Tracking] Frame {frame_num + 1:5d} / {total_frames} ({progress_pct:5.1f}%)")

        return tracks

    def get_object_tracks(self, frame_generator, total_frames, video_path=None, read_from_stub=False, stub_dir="/home/moonscar_lap/Codes/graduation_project/grad_proj/stubs"):
        """
        Get tracking results for all objects (players + ball) with optional caching.

        Args:
            frame_generator: Generator yielding frames
            total_frames: Total number of frames (integer)
            video_path: Path to input video file (used to generate stub filename)
            read_from_stub: Whether to attempt reading cached results
            stub_dir: Directory to save/load stub files

        Returns:
            dict: Dictionary with 'players' and 'ball' keys, each containing per-frame tracking data.
        """
        stub_path = self._get_stub_path(video_path, stub_dir)

        if stub_path and read_from_stub:
            tracks = self._load_tracks_from_stub(stub_path, total_frames)
            if tracks:
                return tracks

        detections = self.detect_frames(frame_generator)
        tracks = self._track_detections(detections, total_frames)

        # Interpolate ball positions (post-processing)
        tracks["ball"] = self.interpolate_ball_positions(tracks["ball"])

        if stub_path:
            self._save_tracks_to_stub(stub_path, stub_dir, tracks)

        # --- Post-process: Make referee assignment permanent via majority vote ---
        # We do this here so it applies even if tracks were loaded from an old stub.
        track_counts = {}
        
        for frame_num in range(total_frames):
            if frame_num < len(tracks.get("players", [])):
                for pid in tracks["players"][frame_num].keys():
                    if pid not in track_counts:
                        track_counts[pid] = {'players': 0, 'referees': 0}
                    track_counts[pid]['players'] += 1
            
            if frame_num < len(tracks.get("referees", [])):
                for rid in tracks["referees"][frame_num].keys():
                    if rid not in track_counts:
                        track_counts[rid] = {'players': 0, 'referees': 0}
                    track_counts[rid]['referees'] += 1

        referee_track_ids = set()
        player_track_ids = set()
        for tid, counts in track_counts.items():
            if counts['referees'] > counts['players']:
                referee_track_ids.add(tid)
            else:
                player_track_ids.add(tid)
                
        for frame_num in range(total_frames):
            if frame_num < len(tracks.get("players", [])):
                for ref_id in list(referee_track_ids):
                    if ref_id in tracks["players"][frame_num]:
                        tracks["referees"][frame_num][ref_id] = tracks["players"][frame_num].pop(ref_id)
            
            if frame_num < len(tracks.get("referees", [])):
                for pid in list(player_track_ids):
                    if pid in tracks["referees"][frame_num]:
                        tracks["players"][frame_num][pid] = tracks["referees"][frame_num].pop(pid)

        return tracks