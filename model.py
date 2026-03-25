import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

score = pd.read_csv('data/score_clean.csv')
team_stats = pd.read_csv('data/team_stats_clean.csv')

merged = score.merge(
    team_stats, left_on=['match_id', 'game_id', 'team_1_id'], right_on=['match_id', 'game_id', 'team_id']
).merge(
    team_stats, left_on=['match_id', 'game_id', 'team_2_id'], right_on=['match_id', 'game_id', 'team_id'],
    suffixes=('_t1', '_t2')
)

#
print("Merged shape: ", merged.shape)
print("Columns: ", merged.columns.tolist())

featured_col = ['rating_t1', 'acs_t1', 'kast_t1', 'adr_t1', 'hs_t1', 'kill_t1', 'death_t1', 
            'rating_t2', 'acs_t2', 'kast_t2', 'adr_t2', 'hs_t2', 'kill_t2', 'death_t2']
x = merged[featured_col]
y = merged['team_1_wl']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

model = RandomForestClassifier(random_state = 42, n_estimators = 100)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))

with open('model/features.pkl', 'wb') as f:
    pickle.dump(featured_col, f)

with open('model/vct_model.pkl', 'wb') as f:
    pickle.dump(model, f)

    