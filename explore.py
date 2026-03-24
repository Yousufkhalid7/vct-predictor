import pandas as pd

# Load the key files
score = pd.read_csv('data/score.csv')
stats = pd.read_csv('data/stats.csv')
match = pd.read_csv('data/match_id.csv')
players = pd.read_csv('data/player_id.csv')
teams = pd.read_csv('data/team_id.csv')
ban = pd.read_csv('data/pick_ban.csv')

# See the first few rows of each
print("=== SCORE.CSV ===")
print(score.head())
print("\nColumns:", score.columns.tolist())

print("\n=== STATS.CSV ===")
print(stats.head())
print("\nColumns:", stats.columns.tolist())

print("\n=== MATCH_ID.CSV ===")
print(match.head())
print("\nColumns:", match.columns.tolist())

print("\n=== TEAM_ID.CSV ===")
print(teams.head())
print("\nColumns:", teams.columns.tolist())

print("\n=== PICK_BAN.CSV ===")
print(ban.head())
print("\nColumns:", ban.columns.tolist())

print("\n=== PLAYER_ID.CSV ===")
print(players.head())
print("\nColumns:", players.columns.tolist())