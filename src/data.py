# Functions for data preprocessing and loading
import pandas as pd
import numpy as np
from thefuzz import fuzz
import os

def load_and_clean_data(file_path: str, cols_to_keep: list[str] = ["Date","HomeTeam","AwayTeam","FTHG","FTAG"]) -> pd.DataFrame:
    """
    Load data from a CSV file and perform cleaning: general column names, date conversion and calendar week number
    and status of the match (played or not).
    
    Args:
        file_path (str): Path to the CSV file.
        cols_to_keep (list[str]): List of columns to keep in the dataframe.
    """
    df = pd.read_csv(file_path)
    df = df[cols_to_keep]
    
    # Clean column names
    clean_columns = ["date", "home_team", "away_team", "home_goals", "away_goals"]
    dict_rename = dict(zip(cols_to_keep, clean_columns))
    df = df.rename(columns=dict_rename)
    
    # Add week number based on the date
    datetime = pd.to_datetime(df["date"], dayfirst=True)
    df["week"] = datetime.dt.isocalendar().week
    
    df = df.astype({"home_goals": "Int64", "away_goals": "Int64"})
    
    # Add a flag to tell if the game is played or not
    df["played"] = df["home_goals"].notna() & df["away_goals"].notna()
    
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
    all_seasons_list = []
    
    # Sort to conserve season order, imply that the file names contain the years of the season
    files = sorted([os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.csv')])
    
    # Load and clean all data files
    for season_idx, file_path in enumerate(files):
        df = load_and_clean_data(file_path, cols_to_keep)
        
        # Add season index
        df['season'] = season_idx 
                
        all_seasons_list.append(df)
            
    # Concatenate all seasons
    df_all = pd.concat(all_seasons_list, ignore_index=True)
    df_all["date"] = pd.to_datetime(df_all["date"], dayfirst=True)
    
    # Sort by date to assure the Dataframe is chronologically ordered
    df_all = df_all.sort_values(by=["date"]).reset_index(drop=True)
    
    # Cleaning names if necessary
    if clean_names:
        mapping = clean_team_names(df)
        df_all["home_team"] = df_all["home_team"].replace(mapping)
        df_all["away_team"] = df_all["away_team"].replace(mapping)
    
    #Creation of time index for weeks
    df_all['week'] = df_all['date'].dt.to_period('W')
    df_all['t'] = df_all.groupby('week', sort=False).ngroup() + 1
    df_all = df_all.drop(columns=['week'])
    
    # Train/Test separation 
    unplayed_matches = df_all[~df_all["played"]]
    if not unplayed_matches.empty:
        current_season = unplayed_matches["season"].max()
    else:
        current_season = df_all["season"].max()

    # Test : Current season (with played and unplayed matches)
    df_test = df_all[df_all["season"] == current_season].copy()
    
    # Train : All the played matches
    df_train = df_all[df_all["played"]].copy()
    
    # Upload data
    df_train.to_csv(f"data/train_{output_file}.csv", index=False)
    df_test.to_csv(f"data/test_{output_file}.csv", index=False)
    
    return df_train, df_test
    
    
    
    # Load and clean all data files
    for file_path in files:
        
        df = load_and_clean_data(file_path)
        #Clean team names
        if clean_names:
                mapping = clean_team_names(df)
                df["home"]= df["home"].replace(mapping)
                df["away"]= df["away"].replace(mapping)
                
        # If at list one match is not played than its the current season
        if not df["played"].all():  
            df_test = df.copy()
            # We add the played match of the current season in the train set
            train_list.append(df[df["played"]].copy())
        else:
            #If the season is over, we had all matches
            train_list.append(df)
            
    # Concatenate data files in a single Dataframe with season information
    seasons = range(len(df_train))
    df_train = pd.concat(train_list, keys=seasons, names = ["season", "match_index"],ignore_index=True)
    # Add a week temporality index
    df_train['t'] = df_concatenated.groupby(['season', 'week'], sort=False).ngroup() + 1
    #
        
    df_concatenated["date"] = pd.to_datetime(df_concatenated["date"], dayfirst=True)
    # Separate train data and test data 
    # Write outuput file
    output_path = f"data/{output_file}"
    df_concatenated.to_csv(output_path)
    return df_concatenated
