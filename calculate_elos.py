# -*- coding: utf-8 -*-

import pandas as pd
import project2 as helper

# Read CSV
# 'GAME_ID', 'H_TEAM', 'A_TEAM', 'H_POINTS', 'A_POINTS', 'TOTAL', 'M_OF_VIC', 'WINNER'
gameDataCSV = pd.read_csv('ScoresData.csv')
additionalgameDataCSV = pd.read_csv('TeamGameSorted.csv')

# Setting Up Costant Values for ELO Score
K_VALUE = 20
HOME_ADVANTAGE = 100

test = helper.getTeamID(gameDataCSV[0]['H_TEAM'])