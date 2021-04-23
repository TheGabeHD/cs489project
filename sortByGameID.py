# -*- coding: utf-8 -*-

import pandas as pd

# Read data
data = pd.read_csv('TeamGamesSorted.csv')

# Sort by game id
data = data.sort_values(by=['GAME_ID'])

# Print out new CSV
data.to_csv('TeamGamesSorted.csv')