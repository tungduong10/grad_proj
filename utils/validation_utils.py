def validate_sport_selection(tracks, sport_config, sport_name):
    """
    FAILSAFE LOGIC: Checks if the visual evidence matches the selected sport.
    """
    print("\n--- Running Validation Failsafe ---")
    unique_players = set()
    
    # Check the first 100 frames (or less if video is short) to see how many players exist
    check_frames = min(100, len(tracks["players"]))
    for i in range(check_frames):
        for player_id in tracks["players"][i].keys():
            unique_players.add(player_id)
            
    total_detected_players = len(unique_players)
    max_allowed = sport_config['max_players']
    
    print(f"Selected Sport: {sport_name.upper()}")
    print(f"Detected unique players in early frames: {total_detected_players}")
    print(f"Max allowed for this sport config: {max_allowed}")
    
    if total_detected_players > max_allowed:
        print(f"\n[WARNING] You selected {sport_name}, but detected {total_detected_players} players!")
        print("This exceeds the logical maximum for this sport. Are you sure you selected the right profile?")
        response = input("Do you want to continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborting. Please restart with the correct sport configuration.")
            exit()
    print("Validation Passed! Proceeding to analytics...\n")
