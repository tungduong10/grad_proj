from ultralytics import YOLO
import supervision as sv
import os
import pandas as pd
from utils import save_stub, read_stub, get_foot_position, get_center_of_bbox

class Tracker:
    def __init__(self, model_path, sport="placeholder"):
        self.model = YOLO(model_path,task='detect')
        self.tracker = sv.ByteTrack()
        self.sport = sport # Defined so the print statement works
    
    def add_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self,ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
        for frame in frames:
            results = self.model.predict(frame, conf=0.3, imgsz=1280, verbose=False)
            yield results[0]

    def _get_stub_path(self, video_path, stub_dir):
        if not video_path:
            return None
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        return os.path.join(stub_dir, f"{video_basename}_player.pkl")

    def _load_tracks_from_stub(self, stub_path, total_frames):
        if not os.path.isfile(stub_path):
            return None
        try:
            tracks = read_stub(True, stub_path)
            # Verify stub has correct number of frames
            if len(tracks.get("players", [])) == total_frames:
                print(f"✓ Loaded player tracks from stub: {stub_path}")
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
            # Verify file was created
            if os.path.isfile(stub_path):
                file_size = os.path.getsize(stub_path)
                print(f"✓ Saved player tracks to stub: {stub_path} ({file_size} bytes)")
            else:
                print(f"⚠ Stub file was not created at: {stub_path}")
        except Exception as e:
            print(f"⚠ Failed to save stub: {e}")
            import traceback
            traceback.print_exc()

    def _track_detections(self, detections, total_frames):
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert GoalKeeper to player object
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            ball_id = cls_names_inv.get("ball", 0)

            # Separate Ball from other detections
            ball_detections = detection_supervision[detection_supervision.class_id == ball_id]
            if len(ball_detections) > 0:
                ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

            all_detections = detection_supervision[detection_supervision.class_id != ball_id]
            all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(all_detections)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
            
            for frame_detection in ball_detections:
                bbox = frame_detection[0].tolist()
                # Ball doesn't pass through tracker so we just write its bounding box directly
                tracks["ball"][frame_num][1] = {"bbox":bbox}
            
            # Progress monitoring (every 30 frames for consistency)
            if (frame_num + 1) % 30 == 0 or (frame_num + 1) == total_frames:
                progress_pct = (frame_num + 1) / total_frames * 100
                print(f"  [Object Tracking] Frame {frame_num + 1:5d} / {total_frames} ({progress_pct:5.1f}%)")
        
        return tracks

    def get_object_tracks(self, frame_generator, total_frames, video_path=None, read_from_stub=False, stub_dir="/home/moonscar_lap/Codes/graduation_project/grad_proj/stubs"):
        """        
        Args:
            frame_generator: Generator yielding frames
            total_frames: Total number of frames (integer)
            video_path: Path to input video file (used to generate stub filename)
            stub_dir: Directory to save/load stub files
        """
        stub_path = self._get_stub_path(video_path, stub_dir)
        
        if stub_path and read_from_stub:
            tracks = self._load_tracks_from_stub(stub_path, total_frames)
            if tracks:
                return tracks

        detections = self.detect_frames(frame_generator)
        tracks = self._track_detections(detections, total_frames)

        if stub_path:
            self._save_tracks_to_stub(stub_path, stub_dir, tracks)

        return tracks