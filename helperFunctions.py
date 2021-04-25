# -*- coding: utf-8 -*-
import pandas as pd
from scipy import stats  # stats.zscore(pandaDataFrame)


class HelperFunctions:
    _teamGames = pd.DataFrame()
    _totalScores = pd.DataFrame()

    def __init__(self):
        self._teamGames = pd.read_csv('TeamGamesSorted.csv')
        self._totalScores = pd.read_csv('TotalScores.csv')
        self._fullData = pd.read_csv('ScoresData.csv')

    """Grabs the Team ID of the given team"""

    def getTeamId(self, team_name):
        print(team_name)
        print(type(team_name))
        df = self._teamGames
        team_id_index = df.loc[df['TEAM_NAME'] == team_name].index[0]
        return df['TEAM_ID'].loc[team_id_index]

    """Grabs the Team Name of the given team ID"""

    def getTeamName(self, team_id):
        df = self._teamGames
        team_name_index = df.loc[df['TEAM_ID'] == team_id].index[0]
        return df['TEAM_NAME'].loc[team_name_index]

    """Returns an array of Strings containing the Teams playing in the game id given"""

    def getGameTeams(self, game_id):
        df = self._teamGames
        team_1_index = df.loc[df['GAME_ID'] == game_id].index[0]
        team_2_index = df.loc[df['GAME_ID'] == game_id].index[1]
        return [df['TEAM_NAME'].loc[team_1_index], df['TEAM_NAME'].loc[team_2_index]]

    """Return the Team Name (string) of the winner of game id"""

    def getYear(self, game_id):
        df = self._teamGames
        year_index = df.loc[df['GAME_ID'] == game_id].index[0]
        return df['SEASON_YEAR'].loc[year_index]

    """Return the Year (string) that the game was played in """

    def getGameWinner(self, game_id):
        df = self._teamGames
        index = df.loc[df['GAME_ID'] == game_id].index[0]
        teams = self.getGameTeams(game_id)
        if df['WL'].loc[index] == 'W' and df['TEAM_NAME'].loc[index] == teams[0]:
            return teams[0]
        else:
            return teams[1]

    """Returns True if home team won, false otherwise"""

    def isHomeWinner(self, game_id):
        df = self._fullData
        index = df.loc[df['GAME_ID'] == game_id].index[0]
        if df['H_TEAM'].loc[index] == df['WINNER'].loc[index]:
            return True
        else:
            return False

    """Return the Team (string) that was the Home team in game id"""

    def isHomeTeam(self, game_id, teamName):
        df = self._teamGames
        index = df.loc[df['GAME_ID'] == game_id].index[0]
        matchup = df['MATCHUP'].loc[index]
        if '@' in matchup and df['TEAM_NAME'].loc[index] == teamName:
            return True
        else:
            return False

    """col_name is string, col num is the index of the column"""

    def normalizeData(self, col_name, col_num):
        df = self._teamGames
        a = df[col_name]
        n = df.columns[col_num]
        df[n] = stats.zscore(a)
        # print(df)

    """Returns inputted value as a percentage"""

    def ReturnPercent(self, not_percent):
        return '{0:.2f}%'.format(not_percent * 100)

    """Finds the average performance of teams for the past n games"""

    def get_performance_of_n_last_games(self, game_id, team_name, game_stats_df, last_n_games):
        prevGames = game_stats_df[game_stats_df['GAME_ID'] < game_id][
            (game_stats_df['H_TEAM'] == team_name) | (game_stats_df['A_TEAM'] == team_name)].tail(last_n_games)

        h_df = prevGames.iloc[:, 10:16]
        h_df.insert(0, 'H_TEAM', prevGames.iloc[:, 2], True)
        h_df.columns = [x[2:] for x in h_df.columns]

        a_df = prevGames.iloc[:, 16:]
        a_df.insert(0, 'A_TEAM', prevGames.iloc[:, 3], True)
        a_df.columns = [x[2:] for x in a_df.columns]

        df = pd.concat([h_df, a_df])
        df = df[df['TEAM'] == team_name]
        df.drop(columns='TEAM', inplace=True)

        return df.mean()


"""
import numpy as np
from prettytable import PrettyTable
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

wl = teamGamesCSV['WL']
n = wl.shape[0]
labels = np.zeros(n)
labels[wl == 'W'] = 1  # ground_truth

cv = KFold(n_splits=10, random_state=225, shuffle=True)

for train_index, test_index in cv.split(teamGamesCSV):
    X_train, X_test = teamGamesCSV.iloc[train_index, :], teamGamesCSV.iloc[test_index, :]
    y_train, y_test = labels[train_index], labels[test_index]

"""
