# -*- coding: utf-8 -*-
import pandas as pd
import time
import helperFunctions as hf
import numpy as np

game_stats = pd.read_csv('ScoresData1.csv', index_col=0)
helperObj = hf.HelperFunctions()


def get_last_15(game_id, team):
    last_15 = game_stats[game_stats['GAME_ID'] < game_id][(game_stats['H_TEAM'] == team) |
                                                          (game_stats['A_TEAM'] == team)].tail(15)

    # Grab the stats relevant to home and away teams separately
    home_games = last_15.iloc[:, 9:15]
    away_games = last_15.iloc[:, 15:21]

    # Prefix removal - "H_", "A_" ...
    home_games.columns = [('LAST_15_' + col[2:]) for col in home_games.columns]
    away_games.columns = [('LAST_15_' + col[2:]) for col in away_games.columns]

    # Insert the associated team names to stats, this way we are able to find the correct stats for team_name later
    home_games.insert(0, 'TEAM', last_15.iloc[:, 1], True)  # True doesn't allow duplicates, none in our data,
    away_games.insert(0, 'TEAM', last_15.iloc[:, 2], True)  # but just for good practice

    # Combine all the stats and filter out for specified team_name, then return mean of performance
    team_performance = pd.concat([home_games, away_games])
    team_performance = team_performance[team_performance['TEAM'] == team]
    team_performance.drop(columns='TEAM', inplace=True)  # remove team name for mean

    return pd.DataFrame(team_performance.mean())


stat_array = ['LAST_15_FG3_PCT', 'LAST_15_FT_PCT', 'LAST_15_REB', 'LAST_15_AST', 'LAST_15_STL', 'LAST_15_BLK']
for prefix in ['H_', 'A_']:
    for post in stat_array:
        game_stats[prefix + post] = np.zeros(game_stats.shape[0])

for index, row in game_stats.iterrows():
    game_id = row['GAME_ID']
    home_team = row['H_TEAM']
    away_team = row['A_TEAM']

    home_team_performance = get_last_15(game_id, home_team)
    home_team_performance.index = ['H_' + col for col in home_team_performance.index]
    home_team_performance = home_team_performance.transpose()
    for col in home_team_performance.columns:
        game_stats.loc[game_stats.GAME_ID == game_id, col] = home_team_performance.loc[0, col]

    away_team_performance = get_last_15(game_id, away_team)
    away_team_performance.index = ['A_' + col for col in away_team_performance.index]
    away_team_performance = away_team_performance.transpose()
    for col in away_team_performance.columns:
        game_stats.loc[game_stats.GAME_ID == game_id, col] = away_team_performance.loc[0, col]


game_stats.to_csv('AttemptDataSet.csv')
