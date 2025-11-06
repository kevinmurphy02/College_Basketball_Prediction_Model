#adjustments.py

HOME_COURT_ADV = 0.014  #~1.4% boost; note: in tournament mode, this is effectively disabled.

def apply_home_court(teamA_stats, teamB_stats, location):
    """
    Adjust team efficiencies based on game location.
    For tournament mode (neutral sites), location should be "neutral".
    """
    loc = location.lower()
    if loc == "home":
        teamA_stats["AdjO"] *= (1 + HOME_COURT_ADV)
        teamA_stats["AdjD"] *= (1 - HOME_COURT_ADV)
        teamB_stats["AdjO"] *= (1 - HOME_COURT_ADV)
        teamB_stats["AdjD"] *= (1 + HOME_COURT_ADV)
    elif loc == "away":
        teamA_stats["AdjO"] *= (1 - HOME_COURT_ADV)
        teamA_stats["AdjD"] *= (1 + HOME_COURT_ADV)
        teamB_stats["AdjO"] *= (1 + HOME_COURT_ADV)
        teamB_stats["AdjD"] *= (1 - HOME_COURT_ADV)
    #For "neutral", no adjustment.
