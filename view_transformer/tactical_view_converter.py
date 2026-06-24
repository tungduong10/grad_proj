import numpy as np
import cv2
from copy import deepcopy

from utils import get_foot_position, measure_distance
from view_transformer import ViewTransformer

class TacticalViewConverter:
    def __init__(self, config, transformer: ViewTransformer = None):
        """
        Initializes the converter dynamically based on the provided configuration (Soccer or Basketball).

        Args:
            config: An object (e.g. SoccerPitchConfiguration or BasketballPitchConfiguration) providing 
                    dimensions and `.vertices` corresponding to tactical keypoints.
            transformer: An optional pre-initialized ViewTransformer to perform temporally smoothed 
                         homography. If None, it creates a new one with filter_window=1 
                         (no sliding window memory).
        """
        self.config = config
        
        # Ensure we have standardized generic canvas dimensions for OOB filtering.
        # X-axis is mapped to length, Y-axis mapped to width depending on the sport.
        self.width = getattr(config, 'length', getattr(config, 'width', 2800))
        self.height = getattr(config, 'width', getattr(config, 'height', 1500))
        self.key_points = config.vertices

        if transformer is None:
            # Default to an un-smoothed ViewTransformer for isolated batch processing
            self.transformer = ViewTransformer(filter_window=1)
        else:
            self.transformer = transformer

    def validate_keypoints(self, keypoints_list):
        """
        Pass-through — validation is now handled entirely by RANSAC pre-filtering
        inside transform_players_to_tactical_view. Kept for API compatibility.
        """
        return keypoints_list

    def transform_players_to_tactical_view(self, keypoints_list, player_tracks, camera_movement_per_frame=None):
        """
        Transform player positions from video frame coordinates to tactical view coordinates.
        Uses RANSAC pre-filtering to remove outlier keypoints before computing the homography.
        
        Args:
            keypoints_list (list): List of detected court keypoints for each frame.
            player_tracks (list): List of dictionaries containing player tracking info for each frame.
            camera_movement_per_frame (list, optional): List of cumulative (dx, dy) camera movements.
        
        Returns:
            list: Tactical player positions matching the frames order.
        """
        tactical_player_positions = []
        last_good_camera_movement = [0.0, 0.0]

        
        # Create a list of only the non-None config points.
        # The YOLO model outputs a contiguous array of N points (e.g. 33), 
        # which map exactly to the N non-None points in the config.
        valid_config_points = [kp for kp in self.key_points if kp is not None]
        
        for frame_ix, (frame_keypoints, frame_tracks) in enumerate(zip(keypoints_list, player_tracks)):
            tactical_positions = {}
            frame_xy = np.array(frame_keypoints.xy[0])

            if frame_xy.size == 0:
                tactical_player_positions.append(tactical_positions)
                continue
            
            # Map valid config points dynamically
            bound_limit = min(len(frame_xy), len(valid_config_points))
            conf = np.array(frame_keypoints.confidence[0]) if getattr(frame_keypoints, 'confidence', None) is not None else np.ones(len(frame_xy))
            
            valid_indices = [
                i for i in range(bound_limit) 
                if frame_xy[i][0] > 0 and frame_xy[i][1] > 0 
                and conf[i] > 0.3
            ]
            
            # --- YOLO Symmetric Error Correction for Basketball ---
            # Broadcast cameras always view the court from one side.
            # "Top" keypoints (far side) MUST have smaller Y coordinates than "Bottom" keypoints.
            # If the model confuses them, we forcefully swap them to prevent a twisted homography.
            if "Basketball" in type(self.config).__name__:
                symmetric_pairs = [
                    (1, 4),   # Left 3pt line
                    (2, 3),   # Left paint
                    (6, 7),   # Left FT corner
                    (11, 14), # Right 3pt line
                    (12, 13), # Right paint
                    (16, 17), # Right FT corner
                    (0, 5),   # Left corner
                    (10, 15), # Right corner
                    (8, 9),   # Middle line
                ]
                for top_idx, bot_idx in symmetric_pairs:
                    if top_idx in valid_indices and bot_idx in valid_indices:
                        if frame_xy[top_idx][1] > frame_xy[bot_idx][1]:
                            # Swap coordinates
                            frame_xy[[top_idx, bot_idx]] = frame_xy[[bot_idx, top_idx]]
                            # Swap confidences
                            conf[top_idx], conf[bot_idx] = conf[bot_idx], conf[top_idx]
            
            # RANSAC pre-filter: remove gross outliers before homography update
            inlier_indices = valid_indices
            if len(valid_indices) >= 4:
                source_points = np.array([frame_xy[i] for i in valid_indices], dtype=np.float32)
                target_points = np.array([valid_config_points[i] for i in valid_indices], dtype=np.float32)

                # 500cm threshold — generous enough to keep imprecise-but-useful detections,
                # strict enough to reject completely wrong ones (e.g. wrong court feature)
                _, mask = cv2.findHomography(source_points, target_points, cv2.RANSAC, 500.0)

                if mask is not None:
                    inlier_indices = [valid_indices[j] for j in range(len(valid_indices)) if mask[j][0] == 1]
                else:
                    inlier_indices = []

            # Update homography using ONLY geometric inliers
            if len(inlier_indices) >= 4:
                source_inliers = np.array([frame_xy[i] for i in inlier_indices], dtype=np.float32)
                target_inliers = np.array([valid_config_points[i] for i in inlier_indices], dtype=np.float32)

                # Diagnostic logging
                if frame_ix % 10 == 0:
                    sx = float(np.max(target_inliers[:, 0]) - np.min(target_inliers[:, 0]))
                    sy = float(np.max(target_inliers[:, 1]) - np.min(target_inliers[:, 1]))
                    print(f"  [Tactical] Frame {frame_ix} | valid={len(valid_indices)} → "
                          f"inliers={len(inlier_indices)} | "
                          f"spread=({int(sx)}cm, {int(sy)}cm) | "
                          f"has_homography={'YES' if self.transformer.m is not None else 'NO'}")

                if self.transformer:
                    try:
                        self.transformer.update(source_inliers, target_inliers)
                        if camera_movement_per_frame:
                            last_good_camera_movement = camera_movement_per_frame[frame_ix]
                    except ValueError:
                        pass
            elif frame_ix % 10 == 0:
                print(f"  [Tactical] Frame {frame_ix} | valid={len(valid_indices)} → "
                      f"inliers={len(inlier_indices)} | SKIPPED (< 4 inliers)")

            # Map the positions locally using current transformation state
            if self.transformer.m is not None:
                delta = np.array([0.0, 0.0])
                if camera_movement_per_frame:
                    delta = np.array(camera_movement_per_frame[frame_ix]) - np.array(last_good_camera_movement)
                    
                for player_id, player_data in frame_tracks.items():
                    bbox = player_data["bbox"]
                    player_position = np.array([get_foot_position(bbox)]) - delta
                    
                    tactical_position = self.transformer.transform_points(player_position)

                    if tactical_position.size > 0:
                        x, y = tactical_position[0]
                        # Filter out-of-bounds maps 
                        # Apply a generous 1000cm margin exclusively for tennis where players stand far behind the baseline
                        margin = 1000.0 if "Tennis" in type(self.config).__name__ else 0.0
                        if -margin <= x <= self.width + margin and -margin <= y <= self.height + margin:
                            tactical_positions[player_id] = [x, y]
            
            tactical_player_positions.append(tactical_positions)
            
        return tactical_player_positions