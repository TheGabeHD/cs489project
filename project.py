# -*- coding: utf-8 -*-
# Retrieves data for every NBA game since 2014

import pandas as pd
import time
from nba_api.stats.static import teams
from nba_api.stats.endpoints import teamgamelogs

# Get list of all NBA teams
teams = teams.get_teams()

# Get the data for each team and season
teamGames = pd.DataFrame()
for team in teams:
    for date in ['2014-15','2015-16', '2016-17', '2017-18', '2018-19', '2019-20', '2020-21']:
        
        # Prevent timeout
        time.sleep(1)
        
        # Using a get_request workaround to get all seasons
        raw_data = teamgamelogs.TeamGameLogs(team_id_nullable=team['id'], get_request=False)
        raw_data.parameters['Season'] = date
        raw_data.get_request()
        
        # Concatenate
        teamGames = pd.concat([teamGames, raw_data.get_data_frames()[0]])
        print('Done', team['full_name'], ':', date)

# Keep relevant columns and save
teamGamesRel = teamGames[['SEASON_YEAR','GAME_ID','GAME_DATE','TEAM_ID','TEAM_NAME','MATCHUP','WL','W_RANK','L_RANK','W_PCT_RANK','PLUS_MINUS','PLUS_MINUS_RANK','PTS_RANK']]
teamGamesRel.to_csv('TeamGames.csv')