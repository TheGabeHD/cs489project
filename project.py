# -*- coding: utf-8 -*-
import pandas as pd
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder

# Get list of all NBA teams
teams = teams.get_teams()

# For each NBA team, get all of the games
leagueGames = pd.DataFrame()
for team in teams:
    leagueGames = pd.concat([leagueGames, leaguegamefinder.LeagueGameFinder(team_id_nullable=team['id'], date_from_nullable='10/31/2014', timeout=100).get_data_frames()[0]])
    print('Done', team['full_name'])
    
leagueGames.tocsv('leagueGames.csv')
    
# Only keep the necessary columns
# leagueGames = leagueGames[['GAME_ID','TEAM_ID','WL','FG_PCT','PLUS_MINUS']]