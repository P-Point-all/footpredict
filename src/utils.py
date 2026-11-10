import pandas as pd
import numpy as np

"""
Utils functions for the App.
"""

def highlight_standing_rows(row, n_teams):
    """
    Highlight standing rows depending on the standing (Title, LDC, Relegation)
    """
    
    rank = row["rank"]
    
    # 1er : Jaune doré
    if rank == 1:
        return ["background-color: #ffe066; color: black; font-weight: bold;"] * len(row)
    # 2e à 4e : Vert clair (Ligue des Champions)
    elif 2 <= rank <= 4:
        return ["background-color: #d3f9d8; color: black;"] * len(row)
    # 2 derniers : Rouge clair (Relégation directe)
    elif rank >= n_teams - 1:
        return ["background-color: #ffc9c9; color: black;"] * len(row)
    
    return [""] * len(row)