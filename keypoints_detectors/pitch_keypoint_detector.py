import os
import itertools
import numpy as np
from ultralytics import YOLO
import supervision as sv
from utils import read_stub, save_stub

class PitchKeypointDetector:
    """
    The PitchKeypointDetector class uses a YOLO model to detect court keypoints in image frames. 
    It integrates smoothly with the streaming/generator-based tracking pipeline.
    """
    def __init__(self, model_path):
        self.model = YOLO(model_path, task='pose')
        
    def _get_stub_path(self, video_path, stub_dir):
        if not video_path:
            return None
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        return os.path.join(stub_dir, f"{video_basename}_court_kp.pkl")

    def _load_tracks_from_stub(self, stub_path, total_frames):
        if not os.path.isfile(stub_path):
            return None
        try:
            keypoints_list = read_stub(True, stub_path)
            # Verify stub has correct number of frames
            if len(keypoints_list) == total_frames:
                print(f"✓ Loaded court keypoints from stub: {stub_path}")
                return keypoints_list
            else:
                print(f"⚠ Stub frame count mismatch ({len(keypoints_list)} vs {total_frames}). Re-tracking...")
        except Exception as e:
            print(f"⚠ Failed to load stub: {e}. Re-tracking...")
        return None

    def _save_tracks_to_stub(self, stub_path, stub_dir, keypoints_list):
        os.makedirs(stub_dir, exist_ok=True)
        print(f"Saving stub to: {os.path.abspath(stub_path)}")
        try:
            save_stub(stub_path, keypoints_list)
            # Verify file was created
            if os.path.isfile(stub_path):
                file_size = os.path.getsize(stub_path)
                print(f"✓ Saved court keypoints to stub: {stub_path} ({file_size} bytes)")
            else:
                print(f"⚠ Stub file was not created at: {stub_path}")
        except Exception as e:
            print(f"⚠ Failed to save stub: {e}")

    def get_court_keypoints(self, frame_generator, total_frames, video_path=None, read_from_stub=False, stub_dir="/home/moonscar_lap/Codes/graduation_project/grad_proj/stubs"):
        """
        Detect court keypoints for frames from a generator using the YOLO model. 
        Implements memory-efficient batching and stub saving to fit the streaming pipeline.

        Args:
            frame_generator (generator): A generator yielding frames (images).
            total_frames (int): Total number of frames in the video.
            video_path (str, optional): The file path for the input video.
            read_from_stub (bool, optional): Indicates whether to read keypoints from a stub file.
            stub_dir (str, optional): Directory to save/read the stub file.

        Returns:
            list: A list of detected supervision KeyPoints objects for each input frame.
        """
        stub_path = self._get_stub_path(video_path, stub_dir)
        
        if stub_path and read_from_stub:
            court_keypoints = self._load_tracks_from_stub(stub_path, total_frames)
            if court_keypoints:
                return court_keypoints
        
        batch_size = 12  # Moderate batch size for GPU efficiency without memory exhaustion
        court_keypoints = []
        frames_processed = 0

        while True:
            # Consume next `batch_size` frames from generator
            batch_frames = list(itertools.islice(frame_generator, batch_size))
            if not batch_frames:
                break
            
            # Predict in batches
            detections_batch = self.model.predict(batch_frames, conf=0.5, verbose=False)
            
            for detection in detections_batch:
                if detection.keypoints is None or len(detection.keypoints) == 0:
                    # Create an empty KeyPoints object or append a placeholder
                    empty_kps = sv.KeyPoints(xy=np.zeros((1, 0, 2)), confidence=np.zeros((1, 0)))
                    court_keypoints.append(empty_kps)
                else:
                    # Store keypoints in Supervision's format explicitly 
                    keypoints_sv = sv.KeyPoints.from_ultralytics(detection)
                    court_keypoints.append(keypoints_sv)
                frames_processed += 1
                
                # Progress logging
                if frames_processed % 30 == 0 or frames_processed == total_frames:
                    progress_pct = frames_processed / total_frames * 100
                    print(f"  [Court Keypoints] Frame {frames_processed:5d} / {total_frames} ({progress_pct:5.1f}%)")

        if stub_path:
            self._save_tracks_to_stub(stub_path, stub_dir, court_keypoints)
        
        return court_keypoints