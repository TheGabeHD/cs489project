# -*- coding: utf-8 -*-
import pandas as pd
from nba_api.stats.endpoints import teamgamelogs, leaguegamefinder

# 30000 games since 2014 (includes duplicates)
leagueGames = leaguegamefinder.LeagueGameFinder(date_from_nullable='10/31/2014').get_data_frames()[0]

# 2460 games since 2014 (includes duplicates)
teamGames = teamgamelogs.TeamGameLogs(date_from_nullable='10/31/2014').get_data_frames()[0]