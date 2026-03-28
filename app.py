import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title = "VCT PREDICTOR 2026",
    page_icon = "🎮",
    layout = "wide"
)

@st.cache_resource
def load_models():
    score = pd.read_csv('data/score_clean.csv')
    team_stats = pd.read_csv('data/team_stats_clean.csv')

    merged = score.merge(team_stats, left_on = ['match_id', 'game_id', 'team_1_id'],
                          right_on = ['match_id', 'game_id', 'team_id']).merge(team_stats, left_on = ['match_id', 'game_id', 'team_2_id'],
                                                                               right_on = ['match_id', 'game_id', 'team_id'], suffixes = ('_t1','_t2')
                          )
    featured_col = ['rating_t1', 'acs_t1', 'kast_t1', 'adr_t1', 'hs_t1', 'kill_t1', 'death_t1', 
            'rating_t2', 'acs_t2', 'kast_t2', 'adr_t2', 'hs_t2', 'kill_t2', 'death_t2']
    
    x = merged[featured_col].dropna()
    y = merged['team_1_wl'][x.index]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42 )

    model = RandomForestClassifier(n_estimators = 100, random_state = 42)
    model.fit(x_train, y_train)
    return model, featured_col

model, featured_col = load_models()

st.title("VCT Match Predictor")
st.markdown("Predict who wins based on team performance stats")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("🔵 Team 1")
    t1_name   = st.text_input("Team 1 Name", value="Team A")
    t1_rating = st.slider("Rating",  0.0, 2.0, 1.2, step=0.01, key="r1")
    t1_acs    = st.slider("ACS",     0,   400,  220, key="a1")
    t1_kast   = st.slider("KAST %",  0,   100,  75,  key="k1")
    t1_adr    = st.slider("ADR",     0,   250,  140, key="d1")
    t1_hs     = st.slider("HS %",    0,   60,   25,  key="h1")
    t1_kill   = st.slider("Kills",   0,   30,   15,  key="kl1")
    t1_death  = st.slider("Deaths",  0,   30,   14,  key="dt1")

with col2:
    st.subheader("🔴 Team 2")
    t2_name   = st.text_input("Team 2 Name", value="Team B")
    t2_rating = st.slider("Rating",  0.0, 2.0, 1.1, step=0.01, key="r2")
    t2_acs    = st.slider("ACS",     0,   400,  200, key="a2")
    t2_kast   = st.slider("KAST %",  0,   100,  70,  key="k2")
    t2_adr    = st.slider("ADR",     0,   250,  130, key="d2")
    t2_hs     = st.slider("HS %",    0,   60,   22,  key="h2")
    t2_kill   = st.slider("Kills",   0,   30,   14,  key="kl2")
    t2_death  = st.slider("Deaths",  0,   30,   15,  key="dt2")

st.divider()

if st.button("Predict the winner", use_container_width = True):
    input_data = pd.DataFrame([{
        'rating_t1': t1_rating, 'acs_t1': t1_acs, 'kast_t1': t1_kast,
        'adr_t1':    t1_adr,    'hs_t1':  t1_hs,  'kill_t1': t1_kill,
        'death_t1':  t1_death,
        'rating_t2': t2_rating, 'acs_t2': t2_acs, 'kast_t2': t2_kast,
        'adr_t2':    t2_adr,    'hs_t2':  t2_hs,  'kill_t2': t2_kill,
        'death_t2':  t2_death,
    }])

    predict = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    winner = t1_name if predict == 1 else t2_name
    win_prob = round(probability[predict] * 100, 1)
    t1_prob = round(probability[1] * 100, 1)
    t2_prob = round(probability[0] * 100, 1)

    st.divider()

    st.markdown(f"Winner: {winner}")
    st.markdown(f"### Model Confidence: {win_prob}")
    st.divider()

    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.metric(label=f"🔵 {t1_name}", value=f"{t1_prob}%")
        st.progress(t1_prob / 100)

    with res_col2:
        st.metric(label=f"🔵 {t2_name}", value=f"{t2_prob}%")
        st.progress(t2_prob / 100)

    st.divider()

    st.markdown("Stat Comparison")
    comparison = pd.DataFrame({
        'Stat':    ['Rating', 'ACS', 'KAST%', 'ADR', 'HS%', 'Kills', 'Deaths'],
        t1_name:   [t1_rating, t1_acs, t1_kast, t1_adr, t1_hs, t1_kill, t1_death],
        t2_name:   [t2_rating, t2_acs, t2_kast, t2_adr, t2_hs, t2_kill, t2_death],
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    