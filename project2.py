# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
#from prettytable import PrettyTable
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from scipy import stats  # stats.zscore(pandaDataFrame)

"""Grabs the Team ID of the given team"""
def getTeamId(df, teamName):
    teamIDIndex = df.loc[df['TEAM_NAME'] == teamName].index[0]
    return df['TEAM_ID'].loc[teamIDIndex]

def getGameTeams(df, gameID):
    team_1_index = df.loc[df['GAME_ID'] == gameID].index[0]
    team_2_index = df.loc[df['GAME_ID'] == gameID].index[1]
    return [df['TEAM_NAME'].loc[team_1_index], df['TEAM_NAME'].loc[team_2_index]]

def getGameWinner(df, gameID):
    index = df.loc[df['GAME_ID'] == gameID].index[0]
    teams = getGameTeams(df, gameID)
    if df['WL'].loc[index] == 'W' and df['TEAM_NAME'].loc[index] == teams[0]:
        return teams[0]
    else:
        return teams[1]

def getHomeTeam(df, gameID):
    index = df.loc[df['GAME_ID'] == gameID].index[0]
    matchup = df['MATCHUP'].loc[index]
    teams = getGameTeams(df, gameID)
    if '@' in matchup and df['TEAM_NAME'].loc[index] == teams[0]:
        return teams[0]
    else:
        return teams[1]


teamGamesCSV = pd.read_csv('TeamGames.csv')

wl = teamGamesCSV['WL']
n = wl.shape[0]
labels = np.zeros(n)
labels[wl == 'W'] = 1  # ground_truth

print(getTeamId(teamGamesCSV, 'Atlanta Hawks'))
print(getGameTeams(teamGamesCSV, 22000069))
print(getHomeTeam(teamGamesCSV, 22000069))
print(getGameWinner(teamGamesCSV, 22000069))


"""
cv = KFold(n_splits=10, random_state=225, shuffle=True)

for train_index, test_index in cv.split(teamGamesCSV):
    X_train, X_test = teamGamesCSV.iloc[train_index, :], teamGamesCSV.iloc[test_index, :]
    y_train, y_test = labels[train_index], labels[test_index]

"""
