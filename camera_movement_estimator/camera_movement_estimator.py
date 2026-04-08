from utils import measure_distance, measure_xy_distance, read_stub, save_stub
import cv2
import os
import numpy as np

class CameraMovementEstimator():
    def __init__(self, first_frame):

        self.lk_params = dict(
            winSize = (15,15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        first_frame_grayscale= cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:,0:20]=1
        mask_features[:,900:1050]=1

        self.features = dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance = 3,
            blockSize = 7,
            mask = mask_features
        )

    def get_camera_movement(self, frame_generator, total_frames, video_path=None, read_from_stub=False, stub_dir="/home/moonscar_lap/Codes/graduation_project/grad_proj/stubs"):
        """        
        Args:
            frame_generator: List of video frames
            total_frames: Total number of frames (integer)
            video_path: Path to input video file (used to generate stub filename)
            stub_dir: Directory to save/load stub files
        """
        stub_path = None
        if video_path:
            video_basename = os.path.splitext(os.path.basename(video_path))[0]
            stub_path = os.path.join(stub_dir, f"{video_basename}_camera_movement.pkl")            
            # Check if stub exists and is valid
            if read_from_stub and os.path.isfile(stub_path):
                try:
                    tracks = read_stub(True, stub_path)
                    # Verify stub has correct number of frames
                    if len(tracks) == total_frames:
                        print(f"✓ Loaded camera movement from stub: {stub_path}")
                        return tracks
                    else:
                        print(f"⚠ Stub frame count mismatch ({len(tracks)} vs {total_frames}). Re-tracking...")
                except Exception as e:
                    print(f"⚠ Failed to load stub: {e}. Re-tracking...")

        camera_movement = [[0, 0] for _ in range(total_frames)]

        try:
            first_frame = next(frame_generator)
        except StopIteration:
            return camera_movement
            
        old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray,**self.features)

        for frame_num, frame in enumerate(frame_generator, start=1):
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(old_gray,frame_gray,old_features,None,**self.lk_params)
            
            max_distance = 0
            camera_movement_x, camera_movement_y = 0,0
            
            for i, (new, old) in enumerate(zip(new_features, old_features)):
                new_features_point=new.ravel()
                old_features_point=old.ravel()

                distance = measure_distance(new_features_point,old_features_point)
                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(old_features_point, new_features_point)

            # accumulate
            if frame_num > 0:
                camera_movement[frame_num] = [camera_movement[frame_num-1][0] + camera_movement_x, camera_movement[frame_num-1][1] + camera_movement_y]

            old_gray=frame_gray.copy()
            old_features = cv2.goodFeaturesToTrack(old_gray,**self.features)
        
        if stub_path:
            os.makedirs(stub_dir, exist_ok=True)
            print(f"Saving stub to: {os.path.abspath(stub_path)}")
            try:
                save_stub(stub_path, camera_movement)
                if os.path.isfile(stub_path):
                    file_size = os.path.getsize(stub_path)
                    print(f"✓ Saved camera movement to stub: {stub_path} ({file_size} bytes)")
                else:
                    print(f"⚠ Stub file was not created at: {stub_path}")
            except Exception as e:
                print(f"⚠ Failed to save stub: {e}")
                import traceback
                traceback.print_exc()
        
        return camera_movement

    def draw_camera_movement(self, frames, camera_movement_per_frame):
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            overlay = frame.copy()
            cv2.rectangle(overlay, (0,0), (500,100), (255,255,255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame, f"Camera Movement X: {x_movement:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            frame = cv2.putText(frame, f"Camera Movement Y: {y_movement:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            yield frame

    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (position[0]-camera_movement[0], position[1]-camera_movement[1])
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted