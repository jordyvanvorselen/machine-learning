#%% Loading libraries and imports
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import metrics
print('Finished loading libraries and imports')

#%% Loading and merging the data
df_matches = pd.read_csv(
    'data/matchResults.csv').merge(pd.read_csv('data/matchLineups.csv'), on='Match ID')
df_statistics = pd.read_csv('data/playerStats.csv')
df_teams = pd.read_csv('data/teams.csv')
print('Finished loading and merging the data')

#%% Loading functions


def get_player_statistics_for_map(match_id, map_id):
    df_match_stats = df_statistics.loc[df_statistics['MatchID'] == match_id]
    df_match_stats = df_match_stats.drop(
        df_match_stats[df_match_stats.Map != map_id].index)
    return df_match_stats


def get_team_by_id(team_id):
    return df_teams.loc[df_teams['ID'] == team_id]


def clean_matches_dataset():
    df = pd.DataFrame()
    df['match_id'] = df_matches['Match ID']
    df['map'] = df_matches['Map']
    df['team_1_id'] = df_matches['Team 1 ID']
    df['team_2_id'] = df_matches['Team 2 ID']
    df['team_1_score'] = df_matches['Team 1 Half 1 Score'] + \
        df_matches['Team 1 Half 2 Score'] + df_matches['Team 1 Overtime Score']
    df['team_2_score'] = df_matches['Team 2 Half 1 Score'] + \
        df_matches['Team 2 Half 2 Score'] + df_matches['Team 2 Overtime Score']
    return df


def decide_winners(dataframe):
    for i, row in dataframe.iterrows():
        dataframe.set_value(
            i, 'winning_team_nr', 1 if row['team_1_score'] > row['team_2_score'] else 2)
        dataframe.set_value(
            i, 'winning_team_id', row['team_1_id'] if row['team_1_score'] > row['team_2_score'] else row['team_2_id'])
    return dataframe


def calculate_winrate_for_team_on_map(df_total, team_id, map_id):
    matches = pd.DataFrame()
    won_matches = pd.DataFrame()

    df_filtered = df_total.loc[(df_total['team_1_id'] == team_id) & (
        df_total['map'] == map_id)]
    df_filtered = df_filtered.append(
        df_total.loc[(df_total['team_2_id'] == team_id) & (df_total['map'] == map_id)])
    matches = matches.append(df_filtered)
    won_matches = won_matches.append(df_total.loc[(
        df_total['winning_team_id'] == team_id) & (df_total['map'] == map_id)])

    return len(won_matches) / len(matches) * 100


def calculate_winrate_for_matches(df_total):
    already_calculated_keys = {}
    for i, row in df_total.iterrows():
        if str(row['map']) + str(row['team_1_id']) not in already_calculated_keys:
            winrate = calculate_winrate_for_team_on_map(
                df_total, row['team_1_id'], row['map'])
            already_calculated_keys[str(
                row['map']) + str(row['team_1_id'])] = winrate
            df_total.set_value(i, 'team_1_winrate', winrate)
        else:
            df_total.set_value(i, 'team_1_winrate', already_calculated_keys[str(
                row['map']) + str(row['team_1_id'])])
        if str(row['map']) + str(row['team_2_id']) not in already_calculated_keys:
            winrate = calculate_winrate_for_team_on_map(
                df_total, row['team_2_id'], row['map'])
            already_calculated_keys[str(
                row['map']) + str(row['team_2_id'])] = winrate
            df_total.set_value(i, 'team_2_winrate', winrate)
        else:
            df_total.set_value(i, 'team_2_winrate', already_calculated_keys[str(
                row['map']) + str(row['team_2_id'])])
        print('Calculated winrate for row ' + str(i))
    df_total.to_csv('winrates.csv', index=False)
    return df_total


def split_data_train_test(df):
    x = df['team_1_winrate', 'team_2_winrate'].values
    y = df['winning_team_nr'].values
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=1337)
    return {'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test}


print('Finished loading functions')

#%%
df_cleaned_data = decide_winners(clean_matches_dataset())
df_final_data = calculate_winrate_for_matches(df_cleaned_data)

#%%
train_test_sets = split_data_train_test()
