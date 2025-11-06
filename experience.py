# experience.py

EXPERIENCE_THRESHOLD = 1.0  # years difference threshold
EXPERIENCE_BONUS = 0.02     # win probability bonus

def apply_experience_bonus(teamA, teamB, win_prob_A, win_prob_B):
    """
    Applies a bonus to the team with significantly higher experience.
    """
    expA = teamA.get("Experience", 0)
    expB = teamB.get("Experience", 0)
    if abs(expA - expB) >= EXPERIENCE_THRESHOLD:
        if expA > expB:
            win_prob_A += EXPERIENCE_BONUS
        else:
            win_prob_B += EXPERIENCE_BONUS
        total = win_prob_A + win_prob_B
        win_prob_A /= total
        win_prob_B /= total
    return win_prob_A, win_prob_B
