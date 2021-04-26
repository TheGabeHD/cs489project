# -*- coding: utf-8 -*-
import pandas as pd
import helperFunctions as hf
import numpy as np

# Read in the Dataset
gameDataCSV_2 = pd.read_csv('FinalDatasetComplete.csv', index_col = 0)
gameDataCSV_2 = gameDataCSV_2.iloc[::-1] # Reverse Dataframe

# Grab team names
all_NBA_teams = set(gameDataCSV_2['H_TEAM'])

# Create dataframe
gameScore = pd.DataFrame(columns=['TEAM', 'LAST_15_FG3_PCT', 'LAST_15_FT_PCT', 'LAST_15_REB', 'LAST_15_AST', 'LAST_15_STL', 
                                  'LAST_15_BLK', 'TEAM_ELO'])

index = 0
seenTeams = 0
for team in all_NBA_teams:
    gameScore.loc[len(gameScore.index)] = [team, 0, 0, 0, 0, 0, 0, 0]
    home_teams = gameDataCSV_2['H_TEAM']
    away_teams = gameDataCSV_2['A_TEAM']
    