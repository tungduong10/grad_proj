import pandas as pd
from utils import measure_distance
from copy import deepcopy

class TennisShotDetector:
    def __init__(self):
        pass

    def get_ball_shot_frames(self, ball_tracks):
        """
        Detects frames where the ball is hit by a player based on trajectory changes.
        """
        # Ensure we have enough frames
        if not ball_tracks or len(ball_tracks) == 0:
            return []

        # Convert tracks to pandas DataFrame format expected by the logic
        ball_positions = []
        for frame_num in range(len(ball_tracks)):
            if 1 in ball_tracks[frame_num]:
                bbox = ball_tracks[frame_num][1]['bbox']
                ball_positions.append({'x1': bbox[0], 'y1': bbox[1], 'x2': bbox[2], 'y2': bbox[3]})
            else:
                ball_positions.append({'x1': None, 'y1': None, 'x2': None, 'y2': None})

        df_ball_positions = pd.DataFrame(ball_positions)
        
        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        
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
                for change_frame in range(i + 1, i + int(minimum_change_frames_for_hit * 1.2) + 1):
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

    def calculate_stats(self, tracks, ball_shot_frames, fps=30):
        """
        Attributes shots to players and calculates speeds, returning a DataFrame of stats.
        """
        player_stats_data = [{
            'frame_num': 0,
            'player_1_number_of_shots': 0,
            'player_1_total_shot_speed': 0,
            'player_1_last_shot_speed': 0,
            'player_1_total_player_speed': 0,
            'player_1_last_player_speed': 0,

            'player_2_number_of_shots': 0,
            'player_2_total_shot_speed': 0,
            'player_2_last_shot_speed': 0,
            'player_2_total_player_speed': 0,
            'player_2_last_player_speed': 0,
        }]

        if len(ball_shot_frames) < 2:
            return pd.DataFrame()

        for ball_shot_ind in range(len(ball_shot_frames) - 1):
            start_frame = ball_shot_frames[ball_shot_ind]
            end_frame = ball_shot_frames[ball_shot_ind + 1]
            ball_shot_time_in_seconds = (end_frame - start_frame) / fps

            # Get distance covered by the ball in meters
            if 1 in tracks['ball'][start_frame] and 1 in tracks['ball'][end_frame]:
                ball_pos_start = tracks['ball'][start_frame][1].get('position_transformed')
                ball_pos_end = tracks['ball'][end_frame][1].get('position_transformed')
                if ball_pos_start and ball_pos_end:
                    distance_covered_by_ball_meters = measure_distance(ball_pos_start, ball_pos_end)
                    speed_of_ball_shot = (distance_covered_by_ball_meters / ball_shot_time_in_seconds) * 3.6
                else:
                    speed_of_ball_shot = 0
            else:
                speed_of_ball_shot = 0

            # Find which player hit the ball (the one closest to it at start_frame)
            player_positions = {}
            for player_id, track_info in tracks['players'][start_frame].items():
                if 'position_transformed' in track_info:
                    player_positions[player_id] = track_info['position_transformed']

            if not player_positions or 1 not in tracks['ball'][start_frame] or 'position_transformed' not in tracks['ball'][start_frame][1]:
                continue
                
            ball_start_pos = tracks['ball'][start_frame][1]['position_transformed']
            
            player_shot_ball = min(
                player_positions.keys(),
                key=lambda pid: measure_distance(player_positions[pid], ball_start_pos)
            )

            # In our pipeline, team is 0 or 1. Let's map team 0 -> player 1, team 1 -> player 2.
            # To be safe, we determine the team of the player who hit the ball
            hitting_team = tracks['players'][start_frame][player_shot_ball].get('team', 0)
            player_num = hitting_team + 1
            opponent_num = 2 if player_num == 1 else 1

            # Opponent player speed
            # Find the track ID of the opponent
            opponent_track_id = None
            for pid, t_info in tracks['players'][start_frame].items():
                if t_info.get('team') != hitting_team:
                    opponent_track_id = pid
                    break

            speed_of_opponent = 0
            if opponent_track_id and opponent_track_id in tracks['players'][start_frame] and opponent_track_id in tracks['players'][end_frame]:
                opp_start_pos = tracks['players'][start_frame][opponent_track_id].get('position_transformed')
                opp_end_pos = tracks['players'][end_frame][opponent_track_id].get('position_transformed')
                if opp_start_pos and opp_end_pos:
                    distance_covered_by_opponent_meters = measure_distance(opp_start_pos, opp_end_pos)
                    speed_of_opponent = (distance_covered_by_opponent_meters / ball_shot_time_in_seconds) * 3.6

            current_player_stats = deepcopy(player_stats_data[-1])
            current_player_stats['frame_num'] = start_frame
            current_player_stats[f'player_{player_num}_number_of_shots'] += 1
            current_player_stats[f'player_{player_num}_total_shot_speed'] += speed_of_ball_shot
            current_player_stats[f'player_{player_num}_last_shot_speed'] = speed_of_ball_shot

            current_player_stats[f'player_{opponent_num}_total_player_speed'] += speed_of_opponent
            current_player_stats[f'player_{opponent_num}_last_player_speed'] = speed_of_opponent

            player_stats_data.append(current_player_stats)

        player_stats_data_df = pd.DataFrame(player_stats_data)
        frames_df = pd.DataFrame({'frame_num': list(range(len(tracks['players'])))})
        player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
        player_stats_data_df = player_stats_data_df.ffill()

        player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed'] / player_stats_data_df['player_1_number_of_shots'].replace(0, 1)
        player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed'] / player_stats_data_df['player_2_number_of_shots'].replace(0, 1)
        player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed'] / player_stats_data_df['player_2_number_of_shots'].replace(0, 1)
        player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed'] / player_stats_data_df['player_1_number_of_shots'].replace(0, 1)

        return player_stats_data_df
