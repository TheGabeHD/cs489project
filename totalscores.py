# -*- coding: utf-8 -*-

import pandas as pd

# Read data
data = pd.read_csv('TeamGames.csv')

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
gameScore.to_csv('TotalScores.csv')