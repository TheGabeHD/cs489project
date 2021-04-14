# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 15:50:01 2021

@author: Gabriel
"""

import pandas as pd
from nba_api.stats.static import teams 
from nba_api.stats.endpoints import teamgamelogs, leaguegamefinder


teams = teams.get_teams()

team = teams[0]
# Get all the games for 'team'
games = teamgamelogs.TeamGameLogs(team_id_nullable=team['id'],date_from_nullable='10/31/2014').get_data_frames()[0]
games = games[['GAME_ID','GAME_DATE','TEAM_ID','TEAM_NAME','WL','W_RANK','L_RANK','W_PCT_RANK','FG_PCT_RANK','PLUS_MINUS','PLUS_MINUS_RANK']]
games.to_csv("teamgamelogs.csv")


