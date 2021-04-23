# -*- coding: utf-8 -*-

import pandas as pd

# Read data
data = pd.read_csv('TeamGamesSorted.csv')
gameScore = pd.DataFrame(columns=['GAME_ID', 'H_TEAM', 'A_TEAM', 'H_POINTS', 'A_POINTS', 'TOTAL', 'M_OF_VIC', 'WINNER',
                                  'PLUS_MINUS', 'H_FG3_PCT', 'H_FT_PCT', 'H_REB', 'H_AST', 'H_STL', 'H_BLK',
                                  'A_FG3_PCT', 'A_FT_PCT', 'A_REB', 'A_AST', 'A_STL', 'A_BLK'])

# Grab all the data on a per-game basis
for index, row in data.iterrows():
    game_id = row['GAME_ID']
    matchup = row['MATCHUP']
    pts = row['PTS']
    team = row['TEAM_NAME']
    fg3 = row['FG3_PCT']
    ft = row['FT_PCT']
    reb = row['REB']
    ast = row['AST']
    stl = row['STL']
    blk = row['BLK']

    if (index % 2 == 0):
        gameScore.loc[len(gameScore.index)] = [game_id, None, None, 0, 0, 0, 0, None, None, None, None, None, None,
                                               None, None, None, None, None, None, None, None]

    gameScore.loc[gameScore.GAME_ID == game_id, 'PLUS_MINUS'] = abs(row['PLUS_MINUS'])

    if '@' in matchup:
        gameScore.loc[gameScore.GAME_ID == game_id, 'H_POINTS'] = pts
        gameScore.loc[gameScore.GAME_ID == game_id, 'H_TEAM'] = team
        gameScore.loc[gameScore.GAME_ID == game_id, 'H_FG3_PCT'] = fg3
        gameScore.loc[gameScore.GAME_ID == game_id, 'H_FT_PCT'] = ft
        gameScore.loc[gameScore.GAME_ID == game_id, 'H_REB'] = reb
        gameScore.loc[gameScore.GAME_ID == game_id, 'H_AST'] = ast
        gameScore.loc[gameScore.GAME_ID == game_id, 'H_STL'] = stl
        gameScore.loc[gameScore.GAME_ID == game_id, 'H_BLK'] = blk
        gameScore.loc[gameScore.GAME_ID == game_id, 'TOTAL'] += pts
    else:
        gameScore.loc[gameScore.GAME_ID == game_id, 'A_POINTS'] = pts
        gameScore.loc[gameScore.GAME_ID == game_id, 'A_TEAM'] = team
        gameScore.loc[gameScore.GAME_ID == game_id, 'A_FG3_PCT'] = fg3
        gameScore.loc[gameScore.GAME_ID == game_id, 'A_FT_PCT'] = ft
        gameScore.loc[gameScore.GAME_ID == game_id, 'A_REB'] = reb
        gameScore.loc[gameScore.GAME_ID == game_id, 'A_AST'] = ast
        gameScore.loc[gameScore.GAME_ID == game_id, 'A_STL'] = stl
        gameScore.loc[gameScore.GAME_ID == game_id, 'A_BLK'] = blk
        gameScore.loc[gameScore.GAME_ID == game_id, 'TOTAL'] += pts

print("Done Main For Loop")

# Calculate margin of victory
for index, row in gameScore.iterrows():
    game_id = row['GAME_ID']
    gameScore.loc[gameScore.GAME_ID == game_id, 'M_OF_VIC'] = abs(row['H_POINTS'] - row['A_POINTS'])
    gameScore.loc[gameScore.GAME_ID == game_id, 'WINNER'] = row['H_TEAM'] if (row['H_POINTS'] > row['A_POINTS']) else \
        row['A_TEAM']

for index, row in gameScore.iterrows():
    game_id = row['GAME_ID']
    if (row['H_TEAM'] == 'LA Clippers'):
        gameScore.loc[gameScore.GAME_ID == game_id, 'H_TEAM'] = 'Los Angeles Clippers'
    if (row['A_TEAM'] == 'LA Clippers'):
        gameScore.loc[gameScore.GAME_ID == game_id, 'A_TEAM'] = 'Los Angeles Clippers'
    if (row['WINNER'] == 'LA Clippers'):
        gameScore.loc[gameScore.GAME_ID == game_id, 'WINNER'] = 'Los Angeles Clippers'

gameScore.to_csv('ScoresData1.csv')

'''
# Sort by game id
data = data.sort_values(by=['GAME_ID'])

# Separate odd/even rows
gameScore = data.iloc[::2][['GAME_ID','PTS']]
dataOdd = data.iloc[1:].iloc[::2]['PTS']

# Add them together
# Using a for loop because the dataframes won't add correctly...
for i in range(gameScore.shape[0]):
    gameScore.iloc[i]['PTS'] += dataOdd.iloc[i]
    
# Create csv
# gameScore.to_csv('TotalScores.csv')
'''
