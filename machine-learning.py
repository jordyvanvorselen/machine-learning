#%% Loading libraries and imports
import pandas as pd
print('Finished loading libraries and imports')

#%% Loading and merging the data
df_matches = pd.read_csv('data/matchResults.csv').merge(pd.read_csv('data/matchLineups.csv'), on='Match ID')
df_statistics = pd.read_csv('data/playerStats.csv')
df_teams = pd.read_csv('data/teams.csv')
print('Finished loading and merging the data')

#%%
def get_player_statistics_for_map(match_id, map_id):
    df_match_stats = df_statistics.loc[df_statistics['MatchID'] == match_id]
    df_match_stats = df_match_stats.drop(df_match_stats[df_match_stats.Map != map_id].index)
    return df_match_stats

#%%
def get_team_by_id(team_id):
    return df_teams.loc[df_teams['ID'] == team_id]