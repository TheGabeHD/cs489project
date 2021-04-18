# NBA Game Predictor
## Collected Data
The data collected to make NBA game predictons used the nba_api to scrape the appropriate basketball data from NBA.com starting from the 2014 NBA season until now.
To use the most recent data, run **project.py** which will re-create the **leagueGames.csv** including the most recent games.

project.py uses the LeagueGameFinder endpoint to collect the following data:
- SEASON_ID
- TEAM_ID
- TEAM_ABBREVIATION
- TEAM_NAME
- GAME_ID
- GAME_DATE
- MATCHUP
- WL
- MIN
- PTS
- FGM
- FGA
- FG_PCT
- FG3M
- FG3A
- FG3_PCT
- FTM
- FTA
- FT_PCT
- OREB
- DREB
- REB
- AST
- STL
- BLK
- TOV
- PF
- PLUS_MINUS
