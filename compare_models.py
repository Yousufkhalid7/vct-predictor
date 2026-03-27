import pandas as pd
import pickle

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

score = pd.read_csv('data/score_clean.csv')
team_stats = pd.read_csv('data/team_stats_clean.csv')

merged = score.merge(
    team_stats, left_on = [ 'match_id', 'game_id', 'team_1_id' ], right_on = [ 'match_id', 'game_id', 'team_id']
).merge(team_stats, left_on = [ 'match_id', 'game_id', 'team_2_id'], right_on = ['match_id', 'game_id', 'team_id'],
        suffixes=('_t1','_t2')
)

featured_col = ['rating_t1', 'acs_t1', 'kast_t1', 'adr_t1', 'hs_t1', 'kill_t1', 'death_t1', 
            'rating_t2', 'acs_t2', 'kast_t2', 'adr_t2', 'hs_t2', 'kill_t2', 'death_t2']

x = merged[featured_col].dropna()
y = merged['team_1_wl'][x.index]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 42, test_size = 0.2)

models = {
    'Random Forest': RandomForestClassifier(n_estimators = 100, random_state = 42),
    'Gradient Boost': GradientBoostingClassifier(n_estimators = 100, random_state = 42),
    'Logistic Regression': LogisticRegression(max_iter = 100)
}

results = {}
for name, model in models.items():
    model.fit(x_train, y_train)
    acc = accuracy_score(y_test, model.predict(x_test))
    results[name] = round( acc*100, 2)
    print(f"{name}: {acc*100:.2f}%")

plt.figure(figsize = (8,5))
plt.bar(results.keys(), results.values(), color=["#4687FF", "#4000FF", "#A600FF"])
plt.title('Model Comparison', fontsize = 14, fontweight = 'bold' )
plt.ylabel('Accuracy(%)')
plt.ylim(50, 100)
for i, (name, acc) in enumerate(results.items()):
    plt.text(i, acc+0.5, f'{acc}%', ha = 'center', fontweight = 'bold')
plt.tight_layout()
plt.savefig('model/model_comparison.png')
plt.show()

rf_model = models['Random Forest']
importance_df = pd.DataFrame({
    'feature': featured_col,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=True)

plt.figure(figsize=(8, 6)) 
plt.barh(importance_df['feature'], importance_df['importance'], color="#4677FF")
plt.title('Feature Importance', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('model/feature_importance.png')
plt.show()

plt.barh(importance_df['feature'], importance_df['importance'], color = "#4677FF")


