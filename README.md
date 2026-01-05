Advanced Streamlit dashboard for IPL cricket analytics with interactive charts, player comparisons, advanced metrics & ML predictions.

[

✨ Features
📊 Interactive Dashboard - 4 tabs: Overview, Players, Metrics, ML Predict

🏏 Player Analysis - Runs, wickets, strike rate, economy comparisons

📈 Visualizations - Plotly charts, heatmaps, subplots

🤖 ML Prediction - RandomForest match winner predictor (~70% accuracy)

🔍 Filters - Season, team, venue multi-select

⚡ Performance - Cached data loading, responsive design

📋 Quick Setup
bash
# 1. Clone/Download
git clone <your-repo>
cd ipl-analysis

# 2. Virtual Environment
python -m venv venv
venv\Scripts\activate  # Windows

# 3. Install Dependencies
pip install -r requirements.txt

# 4. Download Datasets
# matches.csv, deliveries.csv → Kaggle IPL dataset [web:15]

# 5. Run Dashboard
streamlit run app.py
📦 Requirements
text
streamlit==1.38.0
pandas==2.2.2
numpy==2.1.1
plotly==5.24.1
scikit-learn==1.5.2
🗄️ Dataset
Source: Kaggle IPL Complete Ball-by-Ball 2008-2025

text
matches.csv     → Match details (winner, toss, venue)
deliveries.csv  → Ball-by-ball data (runs, wickets)
📁 Project Structure
text
ipl-analysis/
├── app.py               # Main Streamlit app
├── matches.csv         # Match data
├── deliveries.csv      # Ball data
├── requirements.txt    # Dependencies
├── README.md          # This file!
└── venv/              # Python virtual env
🎯 Key Analytics
Feature	Description
Team Wins	Season-wise win distribution + pie charts
Player Stats	Runs, SR, wickets, economy leaderboards
Comparisons	Batsman vs Batsman strike rate showdown
Venue Heatmap	Win % by ground visualization
ML Model	Predict match winner (teams + toss features)
🚀 Deployment
GitHub + Streamlit Cloud (Free):

text
git add .
git commit -m "Initial dashboard"
git push origin main
Deploy: https://share.streamlit.io

Heroku / Render - Production ready
ML Accuracy ~70-75%

🔧 Customization
python
# Add new metric
def batting_average(deliveries, player):
    innings = deliveries[deliveries['batsman'] == player].groupby('match_id')['batsman_runs'].sum()
    return innings.mean()

# New chart
fig_new = px.scatter(df, x='balls', y='runs', size='sr')
📚 Tech Stack
text
Frontend: Streamlit + Plotly
Backend: Pandas + NumPy
ML: Scikit-learn (RandomForest)
Data: CSV (2008-2025 IPL)
🤝 Contributing
Fork repository

Add new charts/metrics

PR with description

📄 License
MIT License - Free for commercial use.

👨‍💻 Author
Your Name - Final Year CS Student
Portfolio: https://github.com/sajitha-shanmugam/ipl-analysis


Made with ❤️ for IPL Analytics 🏏 | Deployed 2026

Run now: streamlit run app.py → Live dashboard ready! 🎉
