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

#%%
def clean_matches_dataset():
    df = pd.DataFrame()
    df['match_id'] = df_matches['Match ID'].head(10)
    df['map'] = df_matches['Map'].head(10)
    df['team_1_id'] = df_matches['Team 1 ID'].head(10)
    df['team_2_id'] = df_matches['Team 2 ID'].head(10)
    df['team_1_score'] = df_matches['Team 1 Half 1 Score'] + df_matches['Team 1 Half 2 Score'] + df_matches['Team 1 Overtime Score'].head(10)
    df['team_2_score'] = df_matches['Team 2 Half 1 Score'] + df_matches['Team 2 Half 2 Score'] + df_matches['Team 2 Overtime Score'].head(10)
    return df

#%%
clean_matches_dataset()