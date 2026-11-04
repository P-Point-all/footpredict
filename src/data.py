# Functions for data preprocessing and loading
import pandas as pd
import numpy as np
from thefuzz import fuzz
import os

def load_and_clean_data(file_path: str, cols_to_keep: list[str] = ["Date","HomeTeam","AwayTeam","FTHG","FTAG"]) -> pd.DataFrame:
    """
    Load data from a CSV file and perform cleaning: general column names, date conversion and calendar week number

    Args:
        file_path (str): Path to the CSV file.
        cols_to_keep (list[str]): List of columns to keep in the dataframe.
    """
    df = pd.read_csv(file_path)
    df = df[cols_to_keep]
    
    # Clean column names
    clean_columns = ["date", "home", "away", "home_goals", "away_goals"]
    dict_rename = dict(zip(cols_to_keep, clean_columns))
    df = df.rename(columns=dict_rename)
    
    # Add week number based on the date
    datetime = pd.to_datetime(df["date"], dayfirst=True)
    weeks = datetime.dt.isocalendar().week
    df["week"] = weeks
    
    return df

def clean_team_names(df: pd.DataFrame, threshold: int = 85) -> pd.DataFrame:
    """
    Create a mapping of team names to unify the names in the final DataFrame
    This function is not 100% performant. In case where the full name differ with the city name or when short abreviations then the 
    model could possibly not detect similarity. Also, when two teams are called by their city and the two city names are close (Marseille/Lille)
    the model could detect a false similarity. We encourage you to check the exported pre-processed data and change using Ctrl + F 
    if necessary to finish the job.

    Args:
        df (pd.DataFrame): DataFrame containing all the matches results
        threshold (int): Threshold to use for similarity
    """
    # Get all different team names and counts
    teams = list(pd.concat([df["home"], df["away"]]).unique())
    counts = pd.concat([df["home"], df["away"]]).value_counts()
    
    #Computing similarities across team names
    matrix = pd.DataFrame(index=teams, columns=teams, dtype=float)

    for home in teams:
        for away in teams:
            matrix.loc[home, away] = fuzz.partial_token_sort_ratio(home, away)
    # Create a dynamic mapping with a 80% threshold, if no name above the threshold, then no better name, if multiple names,
    # take the one with more occurence, if the current name as more occurence than the replacement name, dont change
    mapping = dict()
    for team in teams:
        #Get similarities above the threshold
        similarities = matrix[team]
        best_names = similarities[similarities>threshold]
        # Dataframe with the best names and their occurences
        df_candidates = best_names.to_frame(name="score").join(counts.rename("count"))
        #Get the candidate with more occurences (could be team himself)
        canonical_name = df_candidates["count"].idxmax() 
        # Update mapping
        mapping[team] = canonical_name
    
    return mapping

def preprocess_data(folder: str, output_file: str,cols_to_keep: list[str] = ["Date","HomeTeam","AwayTeam","FTHG","FTAG"],clean_names: bool = False) -> pd.DataFrame:
    """
    Preprocess data from multiple CSV files: load and clean, add season, concatenate and add week temporality index.
    Clean team names if demanded.
    Write the final dataframe to a CSV file.

    Args:
        folder (str): Path to the folder containing CSV files.
        output_file (str): Name of the output file in the data folder
        cols_to_keep (list[str]): List of columns to keep in the dataframe.
        clean_names (Bool): Indicate if team names should be unified or not using a similarity algorithm

    Returns:
        pd.DataFrame: Preprocessed dataframe.
    """
    # Get data files
    files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.csv')]
    # Load and clean all data files
    dfs = [load_and_clean_data(file, cols_to_keep) for file in files]
    # Concatenate data files in a single Dataframe with season information
    seasons = range(len(dfs))
    df_concatenated = pd.concat(dfs, keys=seasons, names = ["season", "match_index"])
    # Add a week temporality index
    df_concatenated['t'] = df_concatenated.groupby(['season', 'week'], sort=False).ngroup() + 1
    # Clean team names
    if clean_names:
        mapping = clean_team_names(df_concatenated)
        df_concatenated["home"]= df_concatenated["home"].replace(mapping)
        df_concatenated["away"]= df_concatenated["away"].replace(mapping)
    df_concatenated["date"] = pd.to_datetime(df_concatenated["date"], dayfirst=True)
    # Write outuput file
    output_path = f"data/{output_file}"
    df_concatenated.to_csv(output_path)
    return df_concatenated
