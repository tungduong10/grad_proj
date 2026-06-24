import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils import measure_distance

class SpeedAndDistanceEstimator():
    def __init__(self, frame_rate=30, window_size=5):
        self.frame_rate = frame_rate
        self.window_size = window_size
        
    def add_speed_and_distance_to_tracks(self, tracks):
        total_distance = {}
        
        for object_type, object_tracks in tracks.items():
            if object_type in ["ball", "referees"]:
                continue 
                
            number_of_frames = len(object_tracks)
            total_distance[object_type] = {}
            frame_distances = [{} for _ in range(number_of_frames)]
            previous_players_position = {}
            
            # 1. Calculate frame-by-frame distance and accumulate total distance
            for frame_num in range(number_of_frames):
                for track_id, track_info in object_tracks[frame_num].items():
                    if 'position_transformed' not in track_info:
                        continue
                        
                    current_position = track_info['position_transformed']
                    if current_position is None:
                        continue
                        
                    if track_id not in total_distance[object_type]:
                        total_distance[object_type][track_id] = 0
                        
                    if track_id in previous_players_position:
                        previous_position = previous_players_position[track_id]
                        distance_covered = measure_distance(previous_position, current_position)
                        
                        # Jitter filter for total distance: ignore tiny frame-to-frame movements (< 20cm) 
                        # which are likely just homography/bbox noise, not actual player movement.
                        if distance_covered > 0.2:
                            frame_distances[frame_num][track_id] = distance_covered
                            total_distance[object_type][track_id] += distance_covered
                        else:
                            frame_distances[frame_num][track_id] = 0
                            
                    previous_players_position[track_id] = current_position
                    
                    # Store accumulated distance
                    tracks[object_type][frame_num][track_id]['distance'] = total_distance[object_type][track_id]
            
            # 2. Calculate rolling speed
            for frame_num in range(number_of_frames):
                for track_id, track_info in object_tracks[frame_num].items():
                    if 'position_transformed' not in track_info:
                        continue
                        
                    start_frame = max(0, frame_num - (self.window_size * 3) + 1)
                    
                    intervals_present = 0
                    last_frame_present = None
                    first_pos_in_window = None
                    last_pos_in_window = None
                    
                    for i in range(start_frame, frame_num + 1):
                        if track_id in object_tracks[i] and 'position_transformed' in object_tracks[i][track_id]:
                            pos = object_tracks[i][track_id]['position_transformed']
                            if first_pos_in_window is None:
                                first_pos_in_window = pos
                            last_pos_in_window = pos
                            
                            if last_frame_present is not None:
                                intervals_present += 1
                            last_frame_present = i
                            
                    # Need at least (window_size - 1) intervals to calculate speed
                    if intervals_present >= self.window_size - 1 and intervals_present > 0 and first_pos_in_window and last_pos_in_window:
                        # Use DISPLACEMENT over the time window instead of accumulating noisy frame-to-frame path length
                        displacement_in_window = measure_distance(first_pos_in_window, last_pos_in_window)
                        
                        time_in_seconds = intervals_present / self.frame_rate
                        time_in_hours = time_in_seconds / 3600
                        
                        speed_kmh = (displacement_in_window / 1000) / time_in_hours
                        tracks[object_type][frame_num][track_id]['speed'] = speed_kmh
                    else:
                        # Fallback to previous speed or 0
                        if frame_num > 0 and track_id in object_tracks[frame_num-1] and 'speed' in object_tracks[frame_num-1][track_id]:
                            tracks[object_type][frame_num][track_id]['speed'] = object_tracks[frame_num-1][track_id]['speed']
                        else:
                            tracks[object_type][frame_num][track_id]['speed'] = 0
