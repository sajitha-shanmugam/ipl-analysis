import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Pro IPL Analytics", layout="wide", initial_sidebar_state="expanded")

@st.cache_data
def load_data():
    try:
        matches = pd.read_csv('matches.csv')
        deliveries = pd.read_csv('deliveries.csv')
        return matches, deliveries
    except FileNotFoundError:
        st.error("Download matches.csv & deliveries.csv from Kaggle IPL dataset [web:15]")
        st.stop()
        return pd.DataFrame(), pd.DataFrame()

matches, deliveries = load_data()

# Sidebar Filters
st.sidebar.title("🔍 Filters")
season_options = sorted(matches['season'].unique())
season = st.sidebar.selectbox("Season:", season_options)
venue_options = matches['venue'].unique()
venue = st.sidebar.multiselect("Venue:", venue_options)
team_options = sorted(set(matches['team1'].unique()) | set(matches['team2'].unique()))
team = st.sidebar.selectbox("Team:", team_options)

if st.sidebar.button("🔄 Refresh"):
    st.cache_data.clear()
    st.rerun()

# Advanced Calculations
def player_strike_rate(deliveries, batsman):
    player_data = deliveries[deliveries['batsman'] == batsman]
    balls = len(player_data)
    runs = player_data['batsman_runs'].sum()
    return (runs / balls * 100) if balls > 0 else 0

def bowler_economy(deliveries, bowler):
    bowler_data = deliveries[deliveries['bowler'] == bowler]
    overs = len(bowler_data) / 6
    runs = bowler_data['total_runs'].sum()
    return runs / overs if overs > 0 else 0

# Multi-tab Dashboard
tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🏏 Players", "📊 Metrics", "🤖 ML Predict"])

with tab1:
    st.header(f"IPL {season} Overview")
    season_filter = matches['season'] == season
    team_filter = (matches['team1'] == team) | (matches['team2'] == team)
    venue_filter = ~matches['venue'].isin(venue) if venue else matches.index
    filtered_matches = matches[season_filter & team_filter & venue_filter].copy()
    
    col1, col2, col3, col4 = st.columns(4)
    total_matches = len(filtered_matches)
    wins = len(filtered_matches[filtered_matches['winner'] == team])
    win_pct = (wins / total_matches * 100) if total_matches > 0 else 0
    avg_runs = filtered_matches['dl_applied'].sum()  # Example metric
    col1.metric("Matches", total_matches)
    col2.metric("Wins", wins)
    col3.metric("Win %", f"{win_pct:.1f}%")
    col4.metric("High Scoring", avg_runs)
    
    # Team wins bar + pie
    team_wins = matches[matches['season'] == season]['winner'].value_counts().head(8)
    fig1 = make_subplots(rows=1, cols=2, 
                        subplot_titles=('Top Teams Wins', 'Win Distribution'),
                        specs=[[{"type": "bar"}, {"type": "pie"}]])
    fig1.add_trace(go.Bar(x=team_wins.index, y=team_wins.values, name="Wins"), row=1, col=1)
    fig1.add_trace(go.Pie(labels=team_wins.index, values=team_wins.values), row=1, col=2)
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    st.header("Player Performance")
    col_a, col_b = st.columns(2)
    
    with col_a:
        # Top Batsmen with SR
        deliveries['batsman_runs'] = pd.to_numeric(deliveries['batsman_runs'], errors='coerce')
        bat_stats = deliveries.groupby('batsman').agg({
            'batsman_runs': ['sum', 'count']
        }).round(2)
        bat_stats.columns = ['total_runs', 'balls_faced']
        bat_stats['strike_rate'] = (bat_stats['total_runs'] / (bat_stats['balls_faced']/100)) * 100
        top_batsmen = bat_stats.nlargest(10, 'total_runs')
        fig_bat = px.bar(top_batsmen.reset_index(), 
                        x='total_runs', y='batsman', 
                        color='strike_rate',
                        title="Top Batsmen: Runs + Strike Rate")
        st.plotly_chart(fig_bat, use_container_width=True)
    
    with col_b:
        # Top Wicket Takers
        wickets = deliveries[deliveries['dismissal_kind'].notna()]['bowler'].value_counts().head(10)
        fig_wkt = px.bar(x=wickets.values, y=wickets.index, orientation='h', title="Top Wicket Takers")
        st.plotly_chart(fig_wkt, use_container_width=True)
    
    # Compare 2 Players
    st.subheader("👥 Player Comparison")
    p1 = st.selectbox("Player 1:", deliveries['batsman'].value_counts().index[:50])
    p2 = st.selectbox("Player 2:", deliveries['batsman'].value_counts().index[:50])
    if p1 != p2:
        sr1 = player_strike_rate(deliveries, p1)
        sr2 = player_strike_rate(deliveries, p2)
        comp_data = pd.DataFrame({
            'Player': [p1, p2],
            'Strike Rate': [sr1, sr2],
            'Total Runs': [deliveries[deliveries['batsman']==p1]['batsman_runs'].sum(),
                          deliveries[deliveries['batsman']==p2]['batsman_runs'].sum()]
        })
        st.dataframe(comp_data.style.highlight_max(axis=0))
        fig_comp = px.bar(comp_data, x='Player', y=['Strike Rate', 'Total Runs'], barmode='group')
        st.plotly_chart(fig_comp)

with tab3:
    st.header("Advanced Analytics")
    col1, col2 = st.columns(2)
    with col1:
        # Economy Rates
        deliveries['total_runs'] = pd.to_numeric(deliveries['total_runs'], errors='coerce')
        econ_stats = deliveries.groupby('bowler')['total_runs'].apply(
            lambda x: bowler_economy(deliveries, x.name)
        ).nsmallest(10).round(2)
        st.subheader("Best Bowling Economies")
        st.dataframe(pd.DataFrame({'Economy': econ_stats}).reset_index(), use_container_width=True)
    
    with col2:
        # Venue Stats
        venue_win_pct = matches.groupby('venue')['winner'].value_counts(normalize=True).unstack().fillna(0) * 100
        st.subheader("Venue Win % Heatmap")
        fig_heat = px.imshow(venue_win_pct.head(), title="Venue Performance", aspect="auto", color_continuous_scale="RdYlGn")
        st.plotly_chart(fig_heat)

with tab4:
    st.header("🔮 Match Outcome Predictor")
    if st.button("🚀 Train ML Model"):
        with st.spinner("Training RandomForest..."):
            matches_ml = matches.dropna(subset=['winner', 'toss_winner'])
            matches_ml['team1_win'] = (matches_ml['winner'] == matches_ml['team1']).astype(int)
            features = pd.get_dummies(matches_ml[['team1', 'team2', 'toss_winner', 'toss_decision', 'venue']])
            X_train, X_test, y_train, y_test = train_test_split(features, matches_ml['team1_win'], test_size=0.2, random_state=42)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            st.success(f"✅ Model Accuracy: **{acc:.2%}**")
            st.info("Features: Teams, Toss, Venue | Improve with ball data [web:55]")
    
    st.caption("Production ready | Deploy: Streamlit Cloud [web:21]")

if matches.empty:
    st.info("📥 Datasets ready ah? Kaggle IPL full data download pannu [web:1]")
