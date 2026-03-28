# vct-predictor
# 🎮 VCT Match Predictor

A machine learning web application that predicts **Valorant Champions Tour (VCT) 2025** match outcomes based on real tournament data. Built with Python, scikit-learn, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![Accuracy](https://img.shields.io/badge/Accuracy-88.89%25-brightgreen)

---

## 📌 Overview

This project builds an end-to-end ML pipeline that:
- Downloads and processes real VCT 2025 match data from Kaggle
- Engineers team-level features from player statistics
- Trains and compares multiple ML models to predict match winners
- Serves predictions through an interactive web UI

---

## 🚀 Demo

> Enter team stats using the sliders, click **Predict Winner**, and get an instant prediction with win probability for both teams.

---

## 📊 Model Performance

| Model | Accuracy |
|-------|----------|
| ✅ Random Forest | **88.89%** |
| ✅ Gradient Boosting | **88.89%** |
| Logistic Regression | 83.33% |

Random Forest was selected as the final model based on its performance and interpretability.

---

## 🗂️ Dataset

**Source:** [Valorant VCT Champions 2025 Dataset](https://www.kaggle.com/datasets/kierru/valorant-vct-champions-2025-dataset) — Kaggle

The dataset includes 9 CSV files:

| File | Contents |
|------|----------|
| `score.csv` | Match results — who won each game |
| `stats.csv` | Player stats: ACS, rating, K/D, KAST, ADR, HS% |
| `economy.csv` | Buy patterns and economy rounds |
| `pick_ban.csv` | Map pick and ban phase data |
| `match_id.csv` | Match ID reference table |
| `player_id.csv` | Player ID reference table |
| `team_id.csv` | Team names, IDs, and logo URLs |
| `agent_id.csv` | Agent reference table |
| `1v1.csv` | Duel outcome data |

---

## 🧠 How It Works

```
Raw CSVs (9 files)
      ↓
Data Cleaning (prepare.py)
  - Fix win/loss column encoding
  - Remove % signs from KAST and HS columns
  - Average 5 players into 1 team row per game
      ↓
Feature Engineering
  - Merge score + team stats into single match rows
  - 14 features: rating, ACS, kills, deaths, KAST, ADR, HS% for each team
      ↓
Model Training (model.py)
  - 80/20 train/test split
  - Random Forest Classifier (100 trees)
      ↓
Model Comparison (compare_models.py)
  - Random Forest vs Gradient Boosting vs Logistic Regression
  - Feature importance analysis
      ↓
Web UI (app.py)
  - Streamlit sliders for team stats
  - Live win probability prediction
```

---

## 📁 Project Structure

```
vct-predictor/
├── app.py                  # Streamlit web UI
├── prepare.py              # Data cleaning pipeline
├── model.py                # Model training and evaluation
├── compare_models.py       # Multi-model comparison + charts
├── requirements.txt        # Python dependencies
├── data/
│   ├── score.csv           # Raw match results
│   ├── stats.csv           # Raw player stats
│   ├── score_clean.csv     # Cleaned match results
│   └── team_stats_clean.csv # Cleaned team-level stats
└── model/
    ├── model_comparison.png    # Accuracy bar chart
    └── feature_importance.png  # Feature importance chart
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/yousufkhalid/vct-predictor.git
cd vct-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Set up your Kaggle API key at `~/.kaggle/kaggle.json`, then run:
```bash
kaggle datasets download -d kierru/valorant-vct-champions-2025-dataset --path ./data --unzip
```

### 4. Clean the data
```bash
python prepare.py
```

### 5. Train the model
```bash
python model.py
```

### 6. Run the web app
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 🖥️ Features

- **Two-panel input UI** — enter stats for Team 1 and Team 2 side by side
- **Live prediction** — instantly shows predicted winner on button click
- **Win probability bars** — visual confidence display for both teams
- **Stat comparison table** — side-by-side breakdown of all input stats
- **Model trained on real data** — uses actual VCT 2025 tournament matches

---

## 📦 Dependencies

```
pandas
scikit-learn
matplotlib
streamlit
kaggle
numpy
```

---

## 🔮 Future Improvements

- Add map pick/ban win rate as a feature
- Historical team form (rolling last 5 games win rate)
- Agent composition analysis
- Full analytics dashboard with tournament insights
- Automated model retraining as new data becomes available

---

## 👤 Author

**Yousuf Khalid**
- GitHub: [@yousufkhalid](https://github.com/yousufkhalid7)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).