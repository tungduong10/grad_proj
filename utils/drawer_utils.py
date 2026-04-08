import cv2
import numpy as np
import supervision as sv

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

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Draw a semi-transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        
        # Calculate ball control percentages
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 0].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        
        # Avoid division by zero
        total_frames = team_1_num_frames + team_2_num_frames
        team_1_control = (team_1_num_frames / total_frames * 100) if total_frames > 0 else 0
        team_2_control = (team_2_num_frames / total_frames * 100) if total_frames > 0 else 0

        cv2.putText(frame, f"Team 1 Ball Control: {team_1_control:.2f}%", (1400, 900), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2_control:.2f}%", (1400, 950), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control=None):
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

                labels = [f"#{t_id}" for t_id in detections.tracker_id]

                annotated_frame = self.ellipse_annotator.annotate(
                    scene=annotated_frame,
                    detections=detections
                )

                annotated_frame = self.label_annotator.annotate(
                    scene=annotated_frame,
                    detections=detections,
                    labels=labels
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
                annotated_frame = self.draw_team_ball_control(annotated_frame, frame_num, team_ball_control)

            yield annotated_frame