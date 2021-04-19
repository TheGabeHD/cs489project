# -*- coding: utf-8 -*-

import pandas as pd

# Read data
data = pd.read_csv('TeamGamesSorted.csv')
gameScore = pd.DataFrame(columns=['GAME_ID', 'H_TEAM', 'A_TEAM', 'H_POINTS', 'A_POINTS', 'TOTAL', 'M_OF_VIC'])

for index, row in data.iterrows():
    game_id = row['GAME_ID']
    print(game_id)
    if (index % 2 == 0):
        gameScore.loc[len(gameScore.index)] = [game_id, None, None, 0, 0, 0, 0]


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