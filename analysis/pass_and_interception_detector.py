from copy import deepcopy

class PassAndInterceptionDetector:
    """
    A class that detects passes between teammates and interceptions by opposing teams.
    """
    def __init__(self):
        pass 

    def detect_passes(self, possession_list, player_tracks):
        """
        Detects successful passes between players of the same team.

        Args:
            possession_list (list): A list indicating which player has possession of the ball in each frame.
            player_tracks (list): The list of player tracks for each frame (from tracks['players']).

        Returns:
            list: A list where each element indicates if a pass occurred in that frame
                (-1: no pass, 0: Team 0 pass, 1: Team 1 pass).
        """
        passes = [-1] * len(possession_list)
        prev_holder = -1
        previous_frame = -1

        for frame in range(1, len(possession_list)):
            if possession_list[frame - 1] != -1:
                prev_holder = possession_list[frame - 1]
                previous_frame = frame - 1
            
            current_holder = possession_list[frame]
            
            if prev_holder != -1 and current_holder != -1 and prev_holder != current_holder:
                prev_team = player_tracks[previous_frame].get(prev_holder, {}).get('team', -1)
                current_team = player_tracks[frame].get(current_holder, {}).get('team', -1)

                if prev_team == current_team and prev_team != -1:
                    passes[frame] = prev_team

        return passes

    def detect_interceptions(self, possession_list, player_tracks):
        """
        Detects interceptions where the ball possession changes between opposing teams.

        Args:
            possession_list (list): A list indicating which player has possession of the ball in each frame.
            player_tracks (list): The list of player tracks for each frame (from tracks['players']).

        Returns:
            list: A list where each element indicates if an interception occurred in that frame
                (-1: no interception, 0: Team 0 interception, 1: Team 1 interception).
        """
        interceptions = [-1] * len(possession_list)
        prev_holder = -1
        previous_frame = -1
        
        for frame in range(1, len(possession_list)):
            if possession_list[frame - 1] != -1:
                prev_holder = possession_list[frame - 1]
                previous_frame = frame - 1

            current_holder = possession_list[frame]
            
            if prev_holder != -1 and current_holder != -1 and prev_holder != current_holder:
                prev_team = player_tracks[previous_frame].get(prev_holder, {}).get('team', -1)
                current_team = player_tracks[frame].get(current_holder, {}).get('team', -1)
                
                if prev_team != current_team and prev_team != -1 and current_team != -1:
                    interceptions[frame] = current_team
        
        return interceptions