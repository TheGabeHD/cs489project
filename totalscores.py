# -*- coding: utf-8 -*-

import pandas as pd

# Read data
data = pd.read_csv('TeamGamesSorted.csv')
gameScore = pd.DataFrame(columns=['GAME_ID', 'H_TEAM', 'A_TEAM', 'H_POINTS', 'A_POINTS', 'TOTAL', 'M_OF_VIC', 'WINNER'])

# Grab all the data on a per-game basis
for index, row in data.iterrows():
    game_id = row['GAME_ID']
    matchup = row['MATCHUP']
    pts = row['PTS']
    team = row['TEAM_NAME']
        
    if (index % 2 == 0):
        gameScore.loc[len(gameScore.index)] = [game_id, None, None, 0, 0, 0, 0, None]
    
    if '@' in matchup:
        gameScore.loc[gameScore.GAME_ID == game_id, 'H_POINTS'] = pts
        gameScore.loc[gameScore.GAME_ID == game_id, 'H_TEAM'] = team
        gameScore.loc[gameScore.GAME_ID == game_id, 'TOTAL'] += pts
    else:
        gameScore.loc[gameScore.GAME_ID == game_id, 'A_POINTS'] = pts
        gameScore.loc[gameScore.GAME_ID == game_id, 'A_TEAM'] = team
        gameScore.loc[gameScore.GAME_ID == game_id, 'TOTAL'] += pts

print("Done Main For Loop")
            
# Calculate margin of victory
for index, row in gameScore.iterrows():
    game_id = row['GAME_ID']
    gameScore.loc[gameScore.GAME_ID == game_id, 'M_OF_VIC'] = abs(row['H_POINTS'] - row['A_POINTS'])
    gameScore.loc[gameScore.GAME_ID == game_id, 'WINNER'] = row['H_TEAM'] if (row['H_POINTS'] > row['A_POINTS']) else row['A_TEAM']
    
for index, row in gameScore.iterrows():
    game_id = row['GAME_ID']
    if (row['H_TEAM'] == 'LA Clippers'):
         gameScore.loc[gameScore.GAME_ID == game_id, 'H_TEAM'] = 'Los Angeles Clippers'
    if (row['A_TEAM'] == 'LA Clippers'):
         gameScore.loc[gameScore.GAME_ID == game_id, 'A_TEAM'] = 'Los Angeles Clippers'
    if (row['WINNER'] == 'LA Clippers'):
        gameScore.loc[gameScore.GAME_ID == game_id, 'WINNER'] = 'Los Angeles Clippers'
    
gameScore.to_csv('ScoresData.csv')

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