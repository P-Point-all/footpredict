import streamlit as st
import requests
import time
import pandas as pd
import plotly.express as px
from src.utils import highlight_standing_rows

API_URL = "http://localhost:8000/api/v1"

st.set_page_config(page_title="FootPredict", page_icon="⚽", layout="wide")

# --- SESSION Management ---
if "session_id" not in st.session_state:
    st.session_state.session_id = "sess_" + str(int(time.time()))

st.title("⚽ FootPredict - Modelisation & Simulation")

tab_data, tab_match, tab_saison = st.tabs(["📁 Data & Models", "⚔️ Match Simulation", "🏆 Season Simulation"])

# ==========================================
# ONGLET 1 :  Data & Models
# ==========================================
with tab_data:
    st.header("1. Data Importation")
    st.info("⚠️ Dont forget to include the file of the season you want to simulate (current or fictive season).")
    
    uploaded_files = st.file_uploader("Load your CSV files (Historical + Current season)", type="csv", accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Upload files"):
            with st.spinner("Upload in progress..."):
                files = [("files", (file.name, file.getvalue(), "text/csv")) for file in uploaded_files]
                res = requests.post(f"{API_URL}/data/upload", data={"session_id": st.session_state.session_id}, files=files)
                if res.status_code == 200:
                    st.session_state.columns = res.json()["columns"]
                    st.success(f"{res.json()['files_saved']} fichier(s) uploadé(s) avec succès !")
    
    if "columns" in st.session_state:
        st.header("2. Model configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Columns mapping")
            cols = st.session_state.columns
            map_date = st.selectbox("Date", cols)
            map_home = st.selectbox("Home Team", cols)
            map_away = st.selectbox("Away Team", cols)
            map_hg = st.selectbox("Home Goals", cols)
            map_ag = st.selectbox("Away goals", cols)
            
        with col2:
            st.subheader("Algorithm parameters")
            model_type = st.radio("Model type :", ["Static Double Poisson (Maher)", "Bivariate Dynamic Poisson (Koopman & Lit)"])
            
            if st.button("🚀 Launch Training", type="primary"):
                
                selected_cols = [map_date,map_home,  map_away, map_hg, map_ag]
                payload = {"session_id": st.session_state.session_id, "selected_cols": selected_cols, "model_type": model_type}
                
                with st.spinner("Training in progress..."):
                    # Request training route
                    res = requests.post(f"{API_URL}/model/train", json=payload)
                    if res.status_code == 200:
                        st.success(res.json()["message"])
                        st.session_state.model_ready = True
                        st.balloons()

# ==========================================
# ONGLET 2 : MATCH SIMULATION
# ==========================================
with tab_match:
    st.header("One-Shot Prediction")
    if not st.session_state.get("model_ready"):
        st.warning("Please first train a model in the Data corner.")
    else:
        # Get team names
        res_teams = requests.get(f"{API_URL}/model/{st.session_state.session_id}/teams")
        teams = res_teams.json().get("teams", []) if res_teams.status_code == 200 else []
        
        col1, col2 = st.columns(2)
        home_team = col1.selectbox("Domicile", teams, key="home")
        away_team = col2.selectbox("Extérieur", teams, key="away")
        
        if home_team == away_team:
            st.error("Players can't play against themselves")
        elif st.button("⚽ Simulate match"):
            with st.spinner("Retrieving coefficients..."):
                # 1. Coefficients
                coef_h = requests.get(f"{API_URL}/model/{st.session_state.session_id}/coefficients?team={home_team}").json()
                coef_a = requests.get(f"{API_URL}/model/{st.session_state.session_id}/coefficients?team={away_team}").json()
                
                c1, c2 = st.columns(2)
                c1.metric(f"Attaque {home_team} (α)", coef_h["alpha"])
                c1.metric(f"Défense {home_team} (β)", coef_h["beta"])
                c2.metric(f"Attaque {away_team} (α)", coef_a["alpha"])
                c2.metric(f"Défense {away_team} (β)", coef_a["beta"])
            
            with st.spinner("Simulating match..."):
                # 2. Prediction
                res = requests.get(f"{API_URL}/predict/match?session_id={st.session_state.session_id}&home_team={home_team}&away_team={away_team}").json()
            
            st.markdown(f"<h1 style='text-align: center; color: #1f77b4;'>Simulated score : {res['simulated_score']}</h1>", unsafe_allow_html=True)
            
            st.divider()
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.subheader("Probabilities (Skellam Law)")
                df_probs = pd.DataFrame(list(res["probabilities"].items()), columns=["Issue", "Probability"])
                fig1 = px.pie(df_probs, values='Probability', names='Issue', color='Issue',
                              color_discrete_map={home_team:"green", "Draw":"gray", away_team:"red"})
                st.plotly_chart(fig1, use_container_width=True)
                
            with col_chart2:
                st.subheader("Top 5 Exact Scores")
                df_scores = pd.DataFrame(list(res["top_scores"].items()), columns=["Score", "Prob (%)"])
                # Keep only 5 most probable scores
                df_scores = df_scores.sort_values(by="Prob (%)", ascending=False).head(5)
                fig2 = px.bar(df_scores.sort_values(by="Prob (%)", ascending=True), x="Prob (%)", y="Score", orientation='h', text = "Prob (%)")
                fig2.update_traces(texttemplate='%{text:.2%}%', textposition='outside')
                fig2.update_layout(xaxis_tickformat=".1%",yaxis={'categoryorder': 'total ascending'}, margin=dict(l=10, r=10, t=20, b=20))
                st.plotly_chart(fig2, use_container_width=True)

# ==========================================
# ONGLET 3 : SEASON SIMULATION
# ==========================================
with tab_saison:
    st.header("Season simulator")
    if not st.session_state.get("model_ready"):
        st.warning("Please first train a model in the Data corner.")
    else:
        sim_mode = st.radio("Simulation mode", ["Unique Season", "Multi-Simulations (Monte Carlo)"], horizontal=True)
        
        if sim_mode == "Unique Season":
            if st.button("Launch simulation"):
                with st.spinner("Simulation of left matches..."):
                    res = requests.post(f"{API_URL}/simulate/season/single?session_id={st.session_state.session_id}").json()
                    df_standings = pd.DataFrame(res["standings"])
                    n_teams = len(df_standings)
                    styled_table = (
                        df_standings.style
                        .apply(highlight_standing_rows, n_teams=n_teams, axis=1)
                        .format({"points": "{:.0f}", "gd": "{:+d}", "gf": "{:.0f}", "ga": "{:.0f}"}, na_rep="-")
                    )
                    st.dataframe(styled_table, use_container_width=True, hide_index=True)
                    
        else:
            # For later, to store results of simulation in the session data
            if "multi_sim_results" not in st.session_state:
                st.session_state.multi_sim_results = None
            n_sims = st.slider("Number of seasons to simulate", min_value=10, max_value=1000, value=100, step=10)
            st.warning(f"⏳ Estimation : {n_sims} simulations approximately takes {int(n_sims * 3)} seconds.") 
            
            if st.button("Launch Monte Carlo"):
                # 1. Asynchrone request
                res = requests.post(f"{API_URL}/simulate/season/multi", json={"session_id": st.session_state.session_id, "n_simulations": n_sims}).json()
                task_id = res["task_id"]
                
                # 2. Polling : Waiting interface with progression bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status = "processing"

                while status != "completed":
                    time.sleep(1) # Vérify API every second
                    poll_res = requests.get(f"{API_URL}/simulate/status/{task_id}").json()
                    status = poll_res["status"]
                    progress = poll_res.get("progress", 0)
                    
                    progress_bar.progress(progress)
                    status_text.text(f"Simulations in progress... {progress}%")
                    
                    if status == "completed":
                        #Persistent save in session data
                        st.session_state.multi_sim_results = poll_res["result"]
                        status_text.success("Simulations terminées !")
                
            # KPIs display
            if st.session_state.multi_sim_results is not None:
                st.divider()
                results = st.session_state.multi_sim_results
                kpis = results["kpis"]
                st.subheader("Key Performance Indicators (KPIs)")
                teams = list(kpis.keys())
                selected_team = st.selectbox("Show stats for :", teams,key="kpi_team_selector")
                
                team_stats = kpis[selected_team]
                c1, c2, c3 = st.columns(3)
                c1.metric("🏆 Champion", f"{team_stats['Title']} %")
                c2.metric("⭐ Top 4 (LDC)", f"{team_stats['Top 4']} %")
                c3.metric("🔻 Relegation", f"{team_stats['Relegation']} %")