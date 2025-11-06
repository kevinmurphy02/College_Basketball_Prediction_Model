# upset_factors.py

# Historical upset win percentages for NCAA tournament matchups
# Format: (underdog_seed, favorite_seed): underdog_win_rate
HISTORICAL_UPSETS = {
    (12, 5): 0.35,
    (11, 6): 0.40,
    (10, 7): 0.39,
    # You can add more seed matchups as needed.
}

def adjust_for_upset_trends(teamA, teamB, win_probs, round_name):
    """
    Adjust win probabilities based on historical upset frequencies in tournaments.
    Only applies in tournament rounds.
    """
    seedA = teamA.get("Seed", 0)
    seedB = teamB.get("Seed", 0)
    if seedA and seedB and round_name.lower() in ["round1", "round2", "sweet16", "elite8", "final4", "championship"]:
        if seedA > seedB:
            underdog_seed, favorite_seed = seedA, seedB
            underdog_name, favorite_name = teamA["name"], teamB["name"]
        else:
            underdog_seed, favorite_seed = seedB, seedA
            underdog_name, favorite_name = teamB["name"], teamA["name"]
        matchup = (underdog_seed, favorite_seed)
        if matchup in HISTORICAL_UPSETS:
            upset_rate = HISTORICAL_UPSETS[matchup]
            win_probs[underdog_name] = max(win_probs.get(underdog_name, 0), upset_rate)
            win_probs[favorite_name] = 1 - win_probs[underdog_name]
    return win_probs
