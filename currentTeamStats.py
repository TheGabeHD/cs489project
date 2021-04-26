# -*- coding: utf-8 -*-
import pandas as pd
import helperFunctions as hf
import numpy as np

# Read in the Dataset
gameDataCSV_2 = pd.read_csv('FinalDatasetComplete.csv', index_col=0)
gameDataCSV_2 = gameDataCSV_2.iloc[::-1]  # Reverse Dataframe

# Grab team names
all_NBA_teams = set(gameDataCSV_2['H_TEAM'])

# Create dataframe
gameScore = pd.DataFrame(
    columns=['TEAM', 'LAST_15_FG3_PCT', 'LAST_15_FT_PCT', 'LAST_15_REB', 'LAST_15_AST', 'LAST_15_STL',
             'LAST_15_BLK', 'TEAM_ELO'])

for index, row in gameDataCSV_2.iterrows():
    teams_list = gameScore['TEAM'].to_list()
    home_team = row['H_TEAM']
    away_team = row['A_TEAM']

    if home_team not in teams_list:
        gameScore.loc[len(gameScore.index)] = [home_team, row['H_LAST_15_FG3_PCT'], row['H_LAST_15_FT_PCT'],
                                               row['H_LAST_15_REB'], row['H_LAST_15_AST'], row['H_LAST_15_STL'],
                                               row['H_LAST_15_BLK'], row['H_TEAM_ELO']]

    if away_team not in teams_list:
        gameScore.loc[len(gameScore.index)] = [away_team, row['A_LAST_15_FG3_PCT'], row['A_LAST_15_FT_PCT'],
                                               row['A_LAST_15_REB'], row['A_LAST_15_AST'], row['A_LAST_15_STL'],
                                               row['A_LAST_15_BLK'], row['A_TEAM_ELO']]

    n, p = gameScore.shape
    if n >= 30:
        break

gameScore.to_csv("currentTeamStats.csv")