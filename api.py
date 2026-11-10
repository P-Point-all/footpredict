import os
import uuid
import time
import json
import pandas as pd
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form
from pydantic import BaseModel
from typing import List, Dict
from src.data import preprocess_data
from src.model import StaticDoublePoisson, DynamicBivariatePoisson

app = FastAPI(title="FootPredict API")

# --- MEMORY STORAGE (Demo mode) ---
SESSIONS: Dict[str, dict] = {}
TASKS: Dict[str, dict] = {}
UPLOAD_DIR = "temp_data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- PYDANTIC SCHEMES ---
class MappingRequest(BaseModel):
    session_id: str
    selected_cols: list[str]
    model_type: str

class MultiSimRequest(BaseModel):
    session_id: str
    n_simulations: int

# --- ROUTES DATA & TRAIN ---
@app.post("/api/v1/data/upload")
async def upload_data(session_id: str = Form(...), files: List[UploadFile] = File(...)):
    session_dir = os.path.join(UPLOAD_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    
    saved_files = []
    columns = []
    for file in files:
        file_path = os.path.join(session_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        saved_files.append(file_path)
    
    # Reading columns of first file for front mapping
    if saved_files:
        df_temp = pd.read_csv(saved_files[0], nrows=0)
        columns = df_temp.columns.tolist()
        
    SESSIONS[session_id] = {"files": saved_files}
    return {"columns": columns, "files_saved": len(saved_files)}

@app.post("/api/v1/model/train")
def train_model(req: MappingRequest):
    session_data = SESSIONS.get(req.session_id)
    if not session_data:
        return {"error": "Session introuvable"}
    
    df_train, df_test = preprocess_data(session_data["files"],req.selected_cols)
    if req.model_type == "Static Double Poisson (Maher)":
        SPoisson = StaticDoublePoisson()
        SPoisson.fit(df_train)
        session_data["teams"] = SPoisson.teams
        session_data["model"] = SPoisson
    elif req.model_type == "Bivariate Dynamic Poisson (Koopman & Lit)":
        DPoisson = DynamicBivariatePoisson()
        DPoisson.fit(df_train)
        session_data["teams"] = DPoisson.teams
        session_data["model"] = DPoisson
    else:
        {"status": "error", "message": "No model has been chosen", "n_teams": 0}

    session_data["df_train"] = df_train
    session_data["df_test"] = df_test

    return {"status": "success", "message": "Model trained with sucess", "n_teams": 6}

@app.get("/api/v1/model/{session_id}/teams")
def get_teams(session_id: str):
    return {"teams": SESSIONS.get(session_id, {}).get("teams", [])}

@app.get("/api/v1/model/{session_id}/coefficients")
def get_coefficients(session_id: str, team: str):
    # Get model
    session_data = SESSIONS.get(session_id)
    model = session_data["model"] 
    #Get coefficients
    alpha = model.alpha.get(team)
    beta = model.beta.get(team)
    return {"alpha": alpha, "beta": beta} 

# --- ROUTES PREDICTIONS ---
@app.get("/api/v1/predict/match")
def predict_match(session_id: str, home_team: str, away_team: str):
    # Get model
    session_data = SESSIONS.get(session_id)
    model = session_data["model"]
    # Simple simulation
    res_sim = model.predict(home_team,away_team)
    sim_string = f"{res_sim[home_team]}-{res_sim[away_team]}"
    # Proba simulation
    res_prob = model.predict_result(home_team,away_team)
    # Multi-simulation
    res_score = model.predict_proba(home_team,away_team)
    return {
        "simulated_score": sim_string,
        "probabilities": res_prob,
        "top_scores": res_score
    }

@app.post("/api/v1/simulate/season/single")
def simulate_single_season(session_id: str):
    # Get model and test DataFrame
    session_data = SESSIONS.get(session_id)
    model = session_data["model"]
    df_test = session_data["df_test"]
    # Simulate season
    standings, season_sim = model.predict_season(df_test)
    standings_clean = json.loads(standings.to_json(orient="records"))
    # season_sim_clean = json.loads(season_sim.to_json(orient="records", date_format="iso"))
    return {"standings": standings_clean}

# --- MULTI-SIMULATION LOGIC (ASYNCHRONE) ---
def run_monte_carlo(task_id: str, session_id: str, n_sims: int):
    """Background task to simulate N seasons without API blockage """
    TASKS[task_id] = {"status": "processing", "progress": 0, "result": None}
    # Get model and test DataFrame
    session_data = SESSIONS.get(session_id)
    model = session_data["model"]
    df_test = session_data["df_test"]

    # List to score ranks
    records = []
    for sim_id in range(n_sims):
        standings,_ = model.predict_season(df_test)
        # Store ranks
        sim_data = standings[['team', 'rank']].copy()
        sim_data['sim_id'] = sim_id
        records.append(sim_data)
        TASKS[task_id]["progress"] = int(((sim_id + 1) / n_sims) * 100)
        
    #Concatenate all simulation results
    all_sims_df = pd.concat(records, ignore_index=True)
    #Rank occurence matrix
    rank_counts = pd.crosstab(
        index=all_sims_df['team'], 
        columns=all_sims_df['rank']
    )
    
    # Rank probability matrix
    rank_probs = (rank_counts / n_sims * 100).round(1)
    # Title probability
    title_prob = rank_probs[1]

    # UCL qualification probability
    top_4_cols = [c for c in [1, 2, 3, 4] if c in rank_probs.columns]
    top_4_prob = rank_probs[top_4_cols].sum(axis=1)

    # Direct relegation probability
    relegation_cols = [c for c in [17, 18] if c in rank_probs.columns]
    relegation_prob = rank_probs[relegation_cols].sum(axis=1)

    # Recap
    season_recap = pd.DataFrame({
        "Title": title_prob,
        "Top 4": top_4_prob,
        "Relegation": relegation_prob
    }).sort_values(by="Title", ascending=False)
    # Conversion to dict
    kpis_dict = season_recap.to_dict(orient="index")
    # Store result
    TASKS[task_id]["result"] = {
        "kpis": kpis_dict
    }
    TASKS[task_id]["status"] = "completed"

@app.post("/api/v1/simulate/season/multi")
def start_multi_simulation(req: MultiSimRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    background_tasks.add_task(run_monte_carlo, task_id, req.session_id, req.n_simulations)
    return {"task_id": task_id}

@app.get("/api/v1/simulate/status/{task_id}")
def get_simulation_status(task_id: str):
    task = TASKS.get(task_id)
    if not task:
        return {"status": "error", "message": "Task introuvable"}
    return task