import pandas as pd

score = pd.read_csv('data/score.csv')
stats = pd.read_csv('data/stats.csv')
teams = pd.read_csv('data/team_id.csv')
ban = pd.read_csv('data/pick_ban.csv')

score['team_1_wl'] = score['team_1_wl'].map({"win": 1, "lose": 0})
print("Score shape: ",score.shape)
print('Unique values in team_1_wl: ', score['team_1_wl'].unique())

stats['kast'] = stats['kast'].str.replace("%", "").astype(float)
stats['hs'] = stats['hs'].str.replace("%", "").astype(float)

team_stats = stats.groupby(['match_id', 'game_id', 'team_id'])[['rating', 'acs', 'kill', 'death', 'assist', 'kast', 'adr', 'hs', 'fk', 'fd']].mean().reset_index()

print("\nTeam Stats Shape: ", team_stats.shape)
print(team_stats.head())

score.to_csv('data/score_clean.csv', index="False")
team_stats.to_csv('data/team_stats_clean.csv', index="False")

print("\nClean files saved")

