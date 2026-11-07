import pandas as pd
import numpy as np
from scipy.optimize import minimize
import scipy.stats as stats
from collections import Counter

# Base football model class
class BaseFootballModel:
    """
    Base class for Football models
    Need to specify the training DataFrame at instantiation to be able to use the dynamic prediction of a season
    """
    
    def __init__(self, df_train: pd.DataFrame = None):
        self.teams = []
        self.team_to_idx = {}
        self.idx_to_team = {}
        self.n_teams = 0
        self.df_train = df_train

    def _fit_team_indices(self, df: pd.DataFrame):
        """Identify unique team names and generate index mapping. Usefull to manipulate matrices instead of DataFrames
        """
        # Get all team names
        unique_teams = sorted(
            list(set(df["home_team"].unique()) | set(df["away_team"].unique()))
        )
        self.teams = unique_teams
        self.n_teams = len(unique_teams)
        # Create mapping team to index and reverse mapping index to team
        self.team_to_idx = {team: idx for idx, team in enumerate(unique_teams)}
        self.idx_to_team = {idx: team for idx, team in enumerate(unique_teams)}
        
    def _split_matches(self, season: pd.DataFrame):
        """
        Split the matches of the season between Played and Left
        """
        played = season[season["played"] == True].copy()
        left = season[season["played"] == False].copy()
        
        return played, left

    def _compute_standings(self, season: pd.DataFrame):
        """
        Compute standings for a given season
        """
        # Home games
        home = season[
            ["home_team", "home_goals", "away_goals"]
        ].copy().rename(
            columns={
                "home_team": "team",
                "home_goals": "gf",
                "away_goals": "ga",
            }
        )
        home["points"] = (home["gf"] > home["ga"]) * 3 + (
            home["gf"] == home["ga"]
        ) * 1

        # Away games
        away = season[
            ["away_team", "away_goals", "home_goals"]
        ].copy().rename(
            columns={
                "away_team": "team",
                "away_goals": "gf",
                "home_goals": "ga",
            }
        )
        away["points"] = (away["gf"] > away["ga"]) * 3 + (
            away["gf"] == away["ga"]
        ) * 1

        # Aggregation of home and away games
        table = pd.concat([home, away], ignore_index=True)
        standings = (
            table.groupby("team")
            .agg(
                played=("points", "count"),
                points=("points", "sum"),
                gf=("gf", "sum"),
                ga=("ga", "sum"),
            )
            .reset_index()
        )

        # Compute statistics to break a tie
        standings["gd"] = standings["gf"] - standings["ga"]
        standings = standings.sort_values(
            by=["points", "gd", "gf"], ascending=[False, False, False]
        ).reset_index(drop=True)

        # Add rank
        standings["rank"] = standings.index + 1

        return standings
        


# Classic Double Poisson model

class StaticDoublePoisson(BaseFootballModel):
    """
    Independant Poisson model for score prediction (one for home team and one for away team)
    """
    
    def __init__(self):
        super().__init__()
        self.params = None
        
    def prepare_design_matrix(self, df: pd.DataFrame):
        """
        Build target Y and design matrix X
        
        Each match produces 2 lines of observations:
        - Home scored goals
        - Away scored goals

        """
        if not self.teams:
            self._fit_team_indices(df)
        
        n_matches = len(df)
        n_obs = 2 * n_matches

        # Column structure of X :
        # [0] : intercept / home_advantage
        # [1 : 1 + n_teams] : offensive (alpha)
        # [1 + n_teams : 1 + 2*n_teams] : defensive (beta)
        n_features = 1 + 2 * self.n_teams
        X = np.zeros((n_obs, n_features), dtype=np.float64)
        Y = np.zeros(n_obs, dtype=np.float64)

        # Map teams to index 
        home_indices = df["home_team"].map(self.team_to_idx).to_numpy()
        away_indices = df["away_team"].map(self.team_to_idx).to_numpy()
        home_goals = df["home_goals"].to_numpy()
        away_goals = df["away_goals"].to_numpy()

        for k in range(n_matches):
            i = home_indices[k]
            j = away_indices[k]

            # Rows index in the matrix for k-th match (one for home team and one for away team)
            row_home = 2 * k
            row_away = 2 * k + 1

            # Home 
            Y[row_home] = home_goals[k]
            X[row_home, 0] = 1.0  # Home advantage
            X[row_home, 1 + i] = 1.0
            X[row_home, 1 + self.n_teams + j] = -1.0  # Away team defensive 
            # Away
            Y[row_away] = away_goals[k]
            X[row_away, 0] = 0.0  # No home advantage
            X[row_away, 1 + j] = 1.0  # Away team offensive
            X[row_away, 1 + self.n_teams + i] = -1.0  # Home team defensive


        return X, Y
    
    def _negative_log_likelihood(self, params, X, Y):
        """
        Compute Negative Log-Likelihood for a Poisson distribution
        """
        
        log_lambdas = X @ params
        #Avoid overflow problems
        log_lambdas = np.clip(log_lambdas, a_min=-20, a_max=20)
        lambdas = np.exp(log_lambdas)
        nll = np.sum(lambdas - Y * log_lambdas)
        
        return nll

    def fit(self, df: pd.DataFrame):
        """
        Parameter Estimation of the model by minimizing the NLL
        """
        # If training DataFrame is not provided, use the one specified at instantiation
        if df is None:
            if self.df_train is None:
                raise ValueError("No training DataFrame provided.")
            df = self.df_train
        elif self.df_train is None:
            self.df_train = df
            
        # Prepare matrices
        X, Y = self.prepare_design_matrix(self.df_train)

        # Initialisation : 0 for all coefficients => lambda = 1 for all teams and all matches
        initial_guess = np.zeros(X.shape[1])
        
        # Add constraint on coefficients to avoid the problem of ranking
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x[1:1 + self.n_teams])}]
        
        # Optimizing with SLSQP (ideal for eq constraints)
        result = minimize(
            self._negative_log_likelihood,
            x0=initial_guess,
            args=(X, Y),
            method='SLSQP',
            constraints=constraints,
            options={'disp': False, 'maxiter': 1000}
        )
        
        if result.success:
            # Store optimal params
            self.params = result.x
            
            # Storing home advantage
            self.home_advantage = self.params[0]
            
            # Storing offensive and defensive coefficients in dictionnary with team names as key using mapping
            self.alpha = {team: self.params[1 + i] for i, team in self.idx_to_team.items()}
            self.beta = {team: self.params[1 + self.n_teams + i] for i, team in self.idx_to_team.items()}
        else:
            raise ValueError(f"Optimization failed : {result.message}")

    def predict(self, home_team, away_team):
        """
        Predict a score for a match between home_team and away_team
        """
        # Get the coefficients for home and away teams
        alpha_h = self.alpha.get(home_team, 0)
        beta_h = self.beta.get(home_team, 0)
        alpha_a = self.alpha.get(away_team, 0)
        beta_a = self.beta.get(away_team, 0)
        
        # Compute lambdas for both teams
        lambda_home = np.exp(self.home_advantage + alpha_h - beta_a)
        lambda_away = np.exp(alpha_a - beta_h)
        
        #Simulate the scored goals
        home_goals = np.random.poisson(lambda_home,1)[0]
        away_goals = np.random.poisson(lambda_away,1)[0]
        return {
        home_team: home_goals,
        away_team: away_goals
    }
    
    def predict_proba(self, home_team, away_team , n_simu:int=1000):
            """
            Predict the probability of the exact scores for a match between home_team and away_team
            """
            # Get the coefficients for home and away teams
            alpha_h = self.alpha.get(home_team, 0)
            beta_h = self.beta.get(home_team, 0)
            alpha_a = self.alpha.get(away_team, 0)
            beta_a = self.beta.get(away_team, 0)
            
            # Compute lambdas for both teams
            lambda_home = np.exp(self.home_advantage + alpha_h - beta_a)
            lambda_away = np.exp(alpha_a - beta_h)
            
            #Simulate the scored goals for n matches
            home_goals = np.random.poisson(lambda_home,n_simu)
            away_goals = np.random.poisson(lambda_away,n_simu)
            #List with all the results
            simulated_scores = [f"{h}-{a}" for h, a in zip(home_goals, away_goals)]
            #Count the occurences of each score
            counts = Counter(simulated_scores)
            # Normalization and sort of the dictionnary
            empirical_results = {score: occurrences / n_simu for score, occurrences in counts.items()}
            empirical_results = dict(sorted(empirical_results.items(), key=lambda x: x[1], reverse=True))
            
            return empirical_results
 
    def predict_result(self, home_team, away_team):
            """
            Predict the probability of each outcome for a match between home_team and away_team
            """
            # Get the coefficients for home and away teams
            alpha_h = self.alpha.get(home_team, 0)
            beta_h = self.beta.get(home_team, 0)
            alpha_a = self.alpha.get(away_team, 0)
            beta_a = self.beta.get(away_team, 0)
            
            # Compute lambdas for both teams
            lambda_home = np.exp(self.home_advantage + alpha_h - beta_a)
            lambda_away = np.exp(alpha_a - beta_h)
            
            #Simulate the probability of the outcomes using Skellam distribution
            prob_draw = stats.skellam.pmf(0, lambda_home, lambda_away)
            prob_away = stats.skellam.cdf(-1, lambda_home, lambda_away)
            prob_home = 1 - stats.skellam.cdf(0, lambda_home, lambda_away)
            return {
        home_team: round(prob_home, 3),
        "Draw": round(prob_draw, 3),
        away_team: round(prob_away, 3)
    }

    def predict_season(self, season : pd.DataFrame, dynamic: bool = True):
        """
        Predict the ranking for the given season dynamically.
        If dynamic=True, after each week, we update the coefficients by adding the simulated results in the training DataFrame.
        """
        # Get the training DataFrame to be able to update it without forgetting the initial training (based on truth)
        df_train_bis = self.df_train.copy()
        #Split the season
        played, left = self._split_matches(season)
        left = left.copy()
        
        # Group the left matches by week
        weeks_groups = list(left.groupby("t", sort=True))
        total_weeks = len(weeks_groups)
        simulated_weeks = []
        
        for i, (t_week, df_week) in enumerate(weeks_groups):
            df_week = df_week.copy()
                
            # Get coefficients
            alpha_h = df_week["home_team"].map(self.alpha).fillna(0).to_numpy()
            beta_h = df_week["home_team"].map(self.beta).fillna(0).to_numpy()
            alpha_a = df_week["away_team"].map(self.alpha).fillna(0).to_numpy()
            beta_a = df_week["away_team"].map(self.beta).fillna(0).to_numpy()   
                
            # Lambda computation for each match
            l_home = np.exp(self.home_advantage + alpha_h - beta_a)
            l_away = np.exp(alpha_a - beta_h)
            
            # Simulation
            df_week["home_goals"] = np.random.poisson(l_home)
            df_week["away_goals"] = np.random.poisson(l_away)
            df_week["played"] = True
            
            # Add the simulated week in a list to merge later with the played matches
            simulated_weeks.append(df_week)
            
            # Retraining the model with the new simulated week to update the coefficients for the next week
            if dynamic and i < total_weeks - 1:
                df_train_bis = pd.concat([df_train_bis, df_week], ignore_index=True)
                self.fit(df_train_bis)
        
        # Rebuild the DataFrame with all the simulated weeks
        left_sim = pd.concat(simulated_weeks, ignore_index=True)
        # Merge back both played and simulated matches
        season_sim = pd.concat([played,left_sim],ignore_index=True)
        
        # Compute standings
        standings = self._compute_standings(season_sim)
        
        return standings, season_sim
    
    def predict_season_proba(self, season : pd.DataFrame, n_simu = 100,dynamic: bool= True, stats: bool= True):
        """
        Predict the ranking probabilities over multiple simulated seasons by simulating them dynamically
        """
        # Get team names
        unique_teams = sorted(
                    list(set(season["home_team"].unique()) | set(season["away_team"].unique()))
                )
        # List to score ranks
        records = []
        
        for sim_id in range (n_simu):
            #Simulate season
            standings, _ = self.predict_season(season)
            # Store ranks
            sim_data = standings[['team', 'rank']].copy()
            sim_data['sim_id'] = sim_id
            records.append(sim_data)
        
        #Concatenate all simulation results
        all_sims_df = pd.concat(records, ignore_index=True)
        #Rank occurence matrix
        rank_counts = pd.crosstab(
            index=all_sims_df['team'], 
            columns=all_sims_df['rank']
        )
        
        # Compute general statistics
        if stats:
            # Rank probability matrix
            rank_probs = (rank_counts / n_simu * 100).round(1)
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
                "Title (%)": title_prob,
                "Top 4 (%)": top_4_prob,
                "Relegation (%)": relegation_prob
            }).sort_values(by="Title (%)", ascending=False)
            print(season_recap)
            
        return rank_counts

# Dynamic Bivariate Poisson model (cref S.Koopman & R. Lit)
class DynamicBivariatePoisson(BaseFootballModel):
    def __init__(self, home_goals, away_goals):
        self.home_goals = home_goals
        self.away_goals = away_goals
        self.likelihood = None
        self.params = None

    def fit(self):
        # Fit the model to the data and estimate parameters
        # This is a placeholder for the actual fitting logic
        pass

    def predict(self, home_team, away_team):
        # Predict the outcome of a match between home_team and away_team
        # This is a placeholder for the actual prediction logic
        pass

    def evaluate(self, test_data):
        # Evaluate the model's performance on test_data
        # This is a placeholder for the actual evaluation logic
        pass