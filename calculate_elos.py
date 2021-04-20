# -*- coding: utf-8 -*-

import pandas as pd
import helperFunctions as hf

# Read CSV
# 'GAME_ID', 'H_TEAM', 'A_TEAM', 'H_POINTS', 'A_POINTS', 'TOTAL', 'M_OF_VIC', 'WINNER'
gameDataCSV = pd.read_csv('ScoresData.csv')
additionalgameDataCSV = pd.read_csv('TeamGamesSorted.csv')

# Setting Up Costant Values for ELO Score
K_VALUE = 20
HOME_ADVANTAGE = 100

test = hf.getTeamId(gameDataCSV[0]['H_TEAM'])