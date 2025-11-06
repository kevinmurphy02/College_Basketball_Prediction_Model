
# main_model.py
# Uses notebook-exported logistic artifacts to reproduce probabilities from the Jupyter model.

from predictor import predict_logistic_from_artifacts
from data_loader import load_team_stats

def main():
    print("Loading live team stats...")
    teams = load_team_stats()
    print(f"Loaded {len(teams)} teams.")

    team1 = input("Team 1: ").strip()
    team2 = input("Team 2: ").strip()
    loc   = input("Location [home/away/neutral]: ").strip().lower()

    A = teams.get(team1)
    B = teams.get(team2)
    if A is None or B is None:
        print("Team not found in live stats. Check spelling.")
        return

    pA, pB, used = predict_logistic_from_artifacts(A, B, loc)
    print(f"\nNotebook-aligned Logistic probability: {team1} {pA:.3f}  |  {team2} {pB:.3f}")
    print("Key diffs:")
    for k in ["OffRating_diff","DefRating_diff","home_advantage","Off_OR%_diff","Off_eFG_diff"]:
        if k in used:
            print(f"  {k}: {used[k]:.4f}")

if __name__ == "__main__":
    main()
