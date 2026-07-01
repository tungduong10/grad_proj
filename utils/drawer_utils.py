import cv2
import numpy as np
import supervision as sv
from typing import Optional, List

class Drawer:
    def __init__(self):
        # Annotators from football_ai.ipynb sample
        self.ellipse_annotator = sv.EllipseAnnotator(
            color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
            thickness=2
        )
        self.label_annotator = sv.LabelAnnotator(
            color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
            text_color=sv.Color.from_hex('#000000'),
            text_position=sv.Position.BOTTOM_CENTER
        )
        self.triangle_annotator = sv.TriangleAnnotator(
            color=sv.Color.from_hex('#FFD700'),
            base=25,
            height=21,
            outline_thickness=1
        )

    def draw_team_stats(self, frame, frame_num, team_ball_control, passes=None, interceptions=None):
        # Draw a semi-transparent rectangle
        overlay = frame.copy()
        
        width = 350
        height = 230
        start_x = frame.shape[1] - 400
        start_y = frame.shape[0] - 500
        end_x = start_x + width
        end_y = start_y + height
        
        cv2.rectangle(overlay, (start_x, start_y), (end_x, end_y), (0, 0, 0), -1)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        
        # Calculate ball control percentages
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 0].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        total_frames = team_1_num_frames + team_2_num_frames
        team_1_control = (team_1_num_frames / total_frames * 100) if total_frames > 0 else 0
        team_2_control = (team_2_num_frames / total_frames * 100) if total_frames > 0 else 0
        
        # Calculate passes
        team_1_passes = 0
        team_2_passes = 0
        if passes is not None:
            passes_till_frame = passes[:frame_num + 1]
            team_1_passes = passes_till_frame.count(0)
            team_2_passes = passes_till_frame.count(1)
            
        # Calculate interceptions
        team_1_interceptions = 0
        team_2_interceptions = 0
        if interceptions is not None:
            interceptions_till_frame = interceptions[:frame_num + 1]
            team_1_interceptions = interceptions_till_frame.count(0)
            team_2_interceptions = interceptions_till_frame.count(1)

        text = "     Team 1       Team 2"
        cv2.putText(frame, text, (start_x+80, start_y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        text = "Ball Control"
        cv2.putText(frame, text, (start_x+10, start_y+80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{team_1_control:.1f}%       {team_2_control:.1f}%"
        cv2.putText(frame, text, (start_x+130, start_y+80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        text = "Passes"
        cv2.putText(frame, text, (start_x+10, start_y+120), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{team_1_passes}           {team_2_passes}"
        cv2.putText(frame, text, (start_x+130, start_y+120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        text = "Interceptions"
        cv2.putText(frame, text, (start_x+10, start_y+160), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{team_1_interceptions}           {team_2_interceptions}"
        cv2.putText(frame, text, (start_x+130, start_y+160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control=None, passes=None, interceptions=None):
        for frame_num, frame in enumerate(video_frames):
            annotated_frame = frame.copy()

            # --- Construct detections for Players and Referees ---
            xyxy = []
            class_ids = []
            tracker_ids = []

            # 1. Players (teams 0 and 1)
            if frame_num < len(tracks.get("players", [])):
                for track_id, track_info in tracks["players"][frame_num].items():
                    xyxy.append(track_info["bbox"])
                    class_ids.append(track_info.get("team", 0))
                    tracker_ids.append(track_id)

            # 2. Referees (assigned class 2 for #FFD700 color mapping)
            if frame_num < len(tracks.get("referees", [])):
                for track_id, track_info in tracks["referees"][frame_num].items():
                    xyxy.append(track_info["bbox"])
                    class_ids.append(2) 
                    tracker_ids.append(track_id)

            if len(xyxy) > 0:
                detections = sv.Detections(
                    xyxy=np.array(xyxy),
                    class_id=np.array(class_ids),
                    tracker_id=np.array(tracker_ids)
                )

                annotated_frame = self.ellipse_annotator.annotate(
                    scene=annotated_frame,
                    detections=detections
                )

            # --- Construct detections for Ball ---
            if frame_num < len(tracks.get("ball", [])):
                ball_dict = tracks["ball"][frame_num]
                if 1 in ball_dict:
                    ball_bbox = ball_dict[1]["bbox"]
                    
                    ball_xyxy = np.array([ball_bbox])
                    # Pad boxes for the ball
                    ball_xyxy = sv.pad_boxes(xyxy=ball_xyxy, px=10)
                    
                    ball_detections = sv.Detections(
                        xyxy=ball_xyxy,
                        class_id=np.array([0]) 
                    )

                    annotated_frame = self.triangle_annotator.annotate(
                        scene=annotated_frame,
                        detections=ball_detections
                    )


            # Draw team ball control statistics if provided
            if team_ball_control is not None:
                annotated_frame = self.draw_team_stats(annotated_frame, frame_num, team_ball_control, passes, interceptions)

            yield annotated_frame

    def draw_speed_and_distance(self, video_frames, tracks, is_tennis=False):
        for frame_num, frame in enumerate(video_frames):
            annotated_frame = frame.copy() if hasattr(frame, 'copy') else frame
            for object_type, object_tracks in tracks.items():
                if object_type in ["ball", "referees"]:
                    continue 
                if frame_num < len(object_tracks):
                    for track_id, track_info in object_tracks[frame_num].items():
                        if "speed" in track_info and "distance" in track_info:
                            speed = track_info['speed']
                            distance = track_info['distance']
                            if speed is None or distance is None:
                                continue
                            
                            bbox = track_info['bbox']
                            x1, y1, x2, y2 = bbox
                            position = (int((x1 + x2) / 2) - 30, int(y2) + 20) # Slightly offset for better UI
                            
                            if is_tennis:
                                team = track_info.get('team', 0)
                                player_text = f"Player {team + 1}"
                                cv2.putText(annotated_frame, player_text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 3)
                                cv2.putText(annotated_frame, player_text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
                            else:
                                # Draw shadow/outline first for readability
                                cv2.putText(annotated_frame, f"{speed:.1f} km/h", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 3)
                                cv2.putText(annotated_frame, f"{distance:.1f} m", (position[0], position[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 3)
                                # Draw actual text
                                cv2.putText(annotated_frame, f"{speed:.1f} km/h", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
                                cv2.putText(annotated_frame, f"{distance:.1f} m", (position[0], position[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
            
            yield annotated_frame

    def draw_tennis_stats(self, video_frames, player_stats_df, tracks=None):
        for index, frame in enumerate(video_frames):
            annotated_frame = frame.copy() if hasattr(frame, 'copy') else frame
            if player_stats_df is not None and not player_stats_df.empty and index < len(player_stats_df):
                row = player_stats_df.iloc[index]
                
                player_1_shot_speed = row['player_1_last_shot_speed']
                player_2_shot_speed = row['player_2_last_shot_speed']
                player_1_speed = row['player_1_last_player_speed']
                player_2_speed = row['player_2_last_player_speed']

                if tracks is not None and index < len(tracks.get('players', [])):
                    for pid, t_info in tracks['players'][index].items():
                        team = t_info.get('team', 0)
                        if team == 0:
                            player_1_speed = t_info.get('speed', player_1_speed)
                        elif team == 1:
                            player_2_speed = t_info.get('speed', player_2_speed)

                avg_player_1_shot_speed = row['player_1_average_shot_speed']
                avg_player_2_shot_speed = row['player_2_average_shot_speed']
                avg_player_1_speed = row['player_1_average_player_speed']
                avg_player_2_speed = row['player_2_average_player_speed']

                width = 350
                height = 230

                start_x = annotated_frame.shape[1] - 400
                start_y = annotated_frame.shape[0] - 500
                end_x = start_x + width
                end_y = start_y + height

                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (start_x, start_y), (end_x, end_y), (0, 0, 0), -1)
                alpha = 0.5 
                cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)

                text = "     Player 1     Player 2"
                cv2.putText(annotated_frame, text, (start_x+80, start_y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                text = "Shot Speed"
                cv2.putText(annotated_frame, text, (start_x+10, start_y+80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                text = f"{player_1_shot_speed:.1f} km/h    {player_2_shot_speed:.1f} km/h"
                cv2.putText(annotated_frame, text, (start_x+130, start_y+80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                text = "Player Speed"
                cv2.putText(annotated_frame, text, (start_x+10, start_y+120), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                text = f"{player_1_speed:.1f} km/h    {player_2_speed:.1f} km/h"
                cv2.putText(annotated_frame, text, (start_x+130, start_y+120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                text = "avg. S. Speed"
                cv2.putText(annotated_frame, text, (start_x+10, start_y+160), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                text = f"{avg_player_1_shot_speed:.1f} km/h    {avg_player_2_shot_speed:.1f} km/h"
                cv2.putText(annotated_frame, text, (start_x+130, start_y+160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                text = "avg. P. Speed"
                cv2.putText(annotated_frame, text, (start_x+10, start_y+200), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                text = f"{avg_player_1_speed:.1f} km/h    {avg_player_2_speed:.1f} km/h"
                cv2.putText(annotated_frame, text, (start_x+130, start_y+200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            yield annotated_frame

    def draw_keypoints(self, video_frames, court_keypoints, config):
        """
        Draw detected court keypoints on each frame.

        Args:
            video_frames: Generator yielding video frames.
            court_keypoints: List of supervision KeyPoints objects per frame.
            config: Pitch configuration with .vertices for bound checking.
        """
        for frame_num, frame in enumerate(video_frames):
            annotated_frame = frame.copy() if hasattr(frame, 'copy') else frame

            if frame_num < len(court_keypoints):
                frame_kps = court_keypoints[frame_num]
                frame_xy = np.array(frame_kps.xy[0])
                conf = (
                    np.array(frame_kps.confidence[0])
                    if getattr(frame_kps, 'confidence', None) is not None
                    else np.ones(len(frame_xy))
                )

                bound_limit = min(len(frame_xy), len(config.vertices))

                # Map model output contiguous indices to the original config indices (which have None gaps)
                valid_config_indices = [idx for idx, kp in enumerate(config.vertices) if kp is not None]
                bound_limit = min(len(frame_xy), len(valid_config_indices))

                for i in range(bound_limit):
                    if (frame_xy[i][0] > 0
                            and frame_xy[i][1] > 0
                            and conf[i] > 0.3):
                        x, y = int(frame_xy[i][0]), int(frame_xy[i][1])
                        color = (0, 255, 0)  # Green
                        cv2.circle(annotated_frame, (x, y), 6, color, -1)
                        cv2.circle(annotated_frame, (x, y), 6, (255, 255, 255), 1)
                        
                        # Label with the original 1-based template index (e.g. 01 to 41)
                        label_idx = valid_config_indices[i] + 1
                        cv2.putText(annotated_frame, str(label_idx), (x + 8, y - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
                        cv2.putText(annotated_frame, str(label_idx), (x + 8, y - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            yield annotated_frame

    def draw_mini_map_pip(self, video_frames, tracks, tactical_player_positions, court_image_path, config):
        court_image = cv2.imread(court_image_path)
        if court_image is None:
            # Fallback if image not found
            for frame in video_frames:
                yield frame
            return

        img_h, img_w = court_image.shape[:2]
        # PIP size (larger for better visibility)
        pip_h = 350
        pip_w = int(img_w * (pip_h / img_h))
        
        # Scaling factors from tactical config space (cm) to original image pixels
        config_length = getattr(config, 'length', getattr(config, 'width', 2800))
        config_width = getattr(config, 'width', getattr(config, 'height', 1500))
        scale_x = img_w / config_length
        scale_y = img_h / config_width

        for frame_num, frame in enumerate(video_frames):
            annotated_frame = frame.copy() if hasattr(frame, 'copy') else frame
            pip_frame = court_image.copy()

            if frame_num < len(tactical_player_positions):
                positions = tactical_player_positions[frame_num]
                
                # We need to draw the points with correct team colors. We look up tracks.
                player_colors = {}
                if frame_num < len(tracks.get("players", [])):
                    for tid, tinfo in tracks["players"][frame_num].items():
                        # Default fallback color is RED #FF0000 (0, 0, 255)
                        player_colors[tid] = tinfo.get("team_color", (0, 0, 255))
                
                for player_id, pos in positions.items():
                    x_cm, y_cm = pos
                    
                    pt_x = int(x_cm * scale_x)
                    pt_y = int(y_cm * scale_y)
                    
                    color = player_colors.get(player_id, (0, 0, 255))
                    
                    # Draw filled circle
                    cv2.circle(pip_frame, (pt_x, pt_y), 8, color, -1)
                    # Draw outline
                    cv2.circle(pip_frame, (pt_x, pt_y), 8, (0, 0, 0), 2)
                    
            # Resize pip to fit corner
            pip_resized = cv2.resize(pip_frame, (pip_w, pip_h))
            
            # Add a white border around pip
            pip_resized = cv2.copyMakeBorder(pip_resized, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            new_pip_h, new_pip_w = pip_resized.shape[:2]
            
            main_h, main_w = annotated_frame.shape[:2]
            # Overlay PIP in top-left corner
            y_offset = 20
            x_offset = 20
            
            # Ensure boundaries are respected
            if y_offset >= 0 and x_offset >= 0 and y_offset + new_pip_h <= main_h and x_offset + new_pip_w <= main_w:
                annotated_frame[y_offset:y_offset+new_pip_h, x_offset:x_offset+new_pip_w] = pip_resized
            
            yield annotated_frame
    
    def draw_pitch(
        self,
        config,
        background_color: sv.Color = sv.Color(34, 139, 34),
        line_color: sv.Color = sv.Color.WHITE,
        padding: int = 50,
        line_thickness: int = 4,
        point_radius: int = 8,
        scale: float = 0.1
    ) -> np.ndarray:
        """
        Draws a soccer pitch with specified dimensions, colors, and scale.

        Args:
            config: Configuration object containing the
                dimensions and layout of the pitch.
            background_color (sv.Color, optional): Color of the pitch background.
                Defaults to sv.Color(34, 139, 34).
            line_color (sv.Color, optional): Color of the pitch lines.
                Defaults to sv.Color.WHITE.
            padding (int, optional): Padding around the pitch in pixels.
                Defaults to 50.
            line_thickness (int, optional): Thickness of the pitch lines in pixels.
                Defaults to 4.
            point_radius (int, optional): Radius of the penalty spot points in pixels.
                Defaults to 8.
            scale (float, optional): Scaling factor for the pitch dimensions.
                Defaults to 0.1.

        Returns:
            np.ndarray: Image of the soccer pitch.
        """
        scaled_width = int(config.width * scale)
        scaled_length = int(config.length * scale)
        scaled_circle_radius = int(config.centre_circle_radius * scale)
        scaled_penalty_spot_distance = int(config.penalty_spot_distance * scale)

        pitch_image = np.ones(
            (scaled_width + 2 * padding,
            scaled_length + 2 * padding, 3),
            dtype=np.uint8
        ) * np.array(background_color.as_bgr(), dtype=np.uint8)

        for start, end in config.edges:
            point1 = (int(config.vertices[start - 1][0] * scale) + padding,
                    int(config.vertices[start - 1][1] * scale) + padding)
            point2 = (int(config.vertices[end - 1][0] * scale) + padding,
                    int(config.vertices[end - 1][1] * scale) + padding)
            cv2.line(
                img=pitch_image,
                pt1=point1,
                pt2=point2,
                color=line_color.as_bgr(),
                thickness=line_thickness
            )

        centre_circle_center = (
            scaled_length // 2 + padding,
            scaled_width // 2 + padding
        )
        cv2.circle(
            img=pitch_image,
            center=centre_circle_center,
            radius=scaled_circle_radius,
            color=line_color.as_bgr(),
            thickness=line_thickness
        )

        penalty_spots = [
            (
                scaled_penalty_spot_distance + padding,
                scaled_width // 2 + padding
            ),
            (
                scaled_length - scaled_penalty_spot_distance + padding,
                scaled_width // 2 + padding
            )
        ]
        for spot in penalty_spots:
            cv2.circle(
                img=pitch_image,
                center=spot,
                radius=point_radius,
                color=line_color.as_bgr(),
                thickness=-1
            )

        return pitch_image

    def draw_points_on_pitch(
        self,
        config,
        xy: np.ndarray,
        face_color: sv.Color = sv.Color.RED,
        edge_color: sv.Color = sv.Color.BLACK,
        radius: int = 10,
        thickness: int = 2,
        padding: int = 50,
        scale: float = 0.1,
        pitch: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Draws points on a soccer pitch.

        Args:
            config: Configuration object containing the
                dimensions and layout of the pitch.
            xy (np.ndarray): Array of points to be drawn, with each point represented by
                its (x, y) coordinates.
            face_color (sv.Color, optional): Color of the point faces.
                Defaults to sv.Color.RED.
            edge_color (sv.Color, optional): Color of the point edges.
                Defaults to sv.Color.BLACK.
            radius (int, optional): Radius of the points in pixels.
                Defaults to 10.
            thickness (int, optional): Thickness of the point edges in pixels.
                Defaults to 2.
            padding (int, optional): Padding around the pitch in pixels.
                Defaults to 50.
            scale (float, optional): Scaling factor for the pitch dimensions.
                Defaults to 0.1.
            pitch (Optional[np.ndarray], optional): Existing pitch image to draw points on.
                If None, a new pitch will be created. Defaults to None.

        Returns:
            np.ndarray: Image of the soccer pitch with points drawn on it.
        """
        if pitch is None:
            pitch = self.draw_pitch(
                config=config,
                padding=padding,
                scale=scale
            )

        for point in xy:
            scaled_point = (
                int(point[0] * scale) + padding,
                int(point[1] * scale) + padding
            )
            cv2.circle(
                img=pitch,
                center=scaled_point,
                radius=radius,
                color=face_color.as_bgr(),
                thickness=-1
            )
            cv2.circle(
                img=pitch,
                center=scaled_point,
                radius=radius,
                color=edge_color.as_bgr(),
                thickness=thickness
            )

        return pitch