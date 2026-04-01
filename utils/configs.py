SPORT_CONFIGS = {
    'football': {
        'court_dim': (68, 105), # Width, Length in real-world meters
        'max_players': 25,      # Includes referees and subs on the sideline
        'max_ball_travel_per_frame': 35, # Max pixels the ball can move between frames
    },
    'basketball': {
        'court_dim': (15, 28),
        'max_players': 12,
        'max_ball_travel_per_frame': 50, # Basketballs move faster relative to camera
    },
    'tennis': {
        'court_dim': (10.97, 23.77),
        'max_players': 4,
        'max_ball_travel_per_frame': 80, # Highest velocity
    }
}