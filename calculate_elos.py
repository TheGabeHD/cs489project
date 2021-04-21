# -*- coding: utf-8 -*-

import pandas as pd
import helperFunctions as hf
import numpy as np

# Setting Up Costant Values for ELO Score
K_VALUE = 20
HOME_ADVANTAGE = 100
AVERAGE = 1500

# Elo Calculation Functions
def expectedWinProb(h_elo, a_elo):
    bottom_half = 1 + (10 ** ((a_elo - h_elo) / 400))
    return (1 / bottom_half)

def adjustmentRating(w_elo, l_elo, marginOfVic):
    diff_elo = w_elo - l_elo
    top_half = ((marginOfVic + 3) ** 0.8)
    bottom_half = 7.5 + (0.006 * diff_elo)
    return (K_VALUE * (top_half / bottom_half))

def eloUpdate(w_elo, l_elo, marginOfVic):
    e_team = expectedWinProb(w_elo, l_elo)
    new_k = adjustmentRating(w_elo, l_elo, marginOfVic)
    return (new_k * (1-e_team))

def eloYearUpdate(elo_dict):
    for key, value in elo_dict.items():
        elo_dict[key] = (.75 * value) + (.25 * AVERAGE)
    
# Read CSV
# 'GAME_ID', 'H_TEAM', 'A_TEAM', 'H_POINTS', 'A_POINTS', 'TOTAL', 'M_OF_VIC', 'WINNER'
gameDataCSV = pd.read_csv('ScoresData.csv')
additionalgameDataCSV = pd.read_csv('TeamGamesSorted.csv')
teamBaseElos = pd.read_csv('EosELO2013.csv')
helperObj = hf.HelperFunctions()

def evalTeamName(game_id, location):
    return str(gameDataCSV.loc[gameDataCSV.GAME_ID == game_id, location].values[0])



# Add 2 New Columns for ELO's
gameDataCSV['H_TEAM_ELO'] = np.zeros(gameDataCSV.shape[0])
gameDataCSV['A_TEAM_ELO'] = np.zeros(gameDataCSV.shape[0])

all_NBA_teams = set(gameDataCSV['H_TEAM'])

# Grabbing the Elos for each team at the end of 2013-14 season (before we start our data)
eloArray = []
for team in list(all_NBA_teams):
    id = int(helperObj.getTeamId(team))
    eloArray += [float(teamBaseElos.loc[teamBaseElos['TEAM_ID'] == id, 'EOS_ELO'].values[0])]

elo_dict = dict(zip(list(all_NBA_teams),  eloArray)) # Change 1500 to the 2013-14 Year Elo's

currentYear = ""
for index, row in gameDataCSV.iterrows():    
    game_id = row['GAME_ID']
    
    # Update On New Year
    theYear = additionalgameDataCSV.loc[additionalgameDataCSV.GAME_ID == game_id, 'SEASON_YEAR'].values[0]
    if (currentYear != theYear):
        eloYearUpdate(elo_dict)
        currentYear = theYear
    
    win_Team = row['WINNER']
    lose_Team = (row['H_TEAM']) if (win_Team != row['H_TEAM']) else (row['A_TEAM'])
    marginOfVic = row['M_OF_VIC']
    
    isWinTeamHome = helperObj.isHomeTeam(game_id, str(win_Team))
    
    win_Adj, loser_Adj = 0, 0
    if isWinTeamHome:
        win_Adj += HOME_ADVANTAGE
    else:
        loser_Adj += HOME_ADVANTAGE
    
    eloUpdateWeight = eloUpdate(elo_dict[win_Team] + win_Adj, elo_dict[lose_Team] + loser_Adj, marginOfVic)
    elo_dict[win_Team] += eloUpdateWeight
    elo_dict[lose_Team] -= eloUpdateWeight
    
    if isWinTeamHome:
        gameDataCSV.loc[gameDataCSV.GAME_ID == game_id, 'H_TEAM_ELO'] = elo_dict[win_Team]
        gameDataCSV.loc[gameDataCSV.GAME_ID == game_id, 'A_TEAM_ELO'] = elo_dict[lose_Team]
    else:
        gameDataCSV.loc[gameDataCSV.GAME_ID == game_id, 'A_TEAM_ELO'] = elo_dict[win_Team]
        gameDataCSV.loc[gameDataCSV.GAME_ID == game_id, 'H_TEAM_ELO'] = elo_dict[lose_Team]
        
    
    
