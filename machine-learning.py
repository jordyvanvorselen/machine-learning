#%% Loading libraries and imports
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics
from sklearn import tree, svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import GaussianNB
print('Finished loading libraries and imports')

#%% Loading and merging the data
df_matches = pd.read_csv(
    'data/matchResults.csv').merge(pd.read_csv('data/matchLineups.csv'), on='Match ID')
df_statistics = pd.read_csv('data/playerStats.csv')
df_teams = pd.read_csv('data/teams.csv')
print('Finished loading and merging the data')

#%% Loading functions


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
    print('Finished cleaning matches dataset')
    return df


def decide_winners(dataframe):
    for i, row in dataframe.iterrows():
        dataframe.set_value(
            i, 'winning_team_nr', 1 if row['team_1_score'] > row['team_2_score'] else 2)
        dataframe.set_value(
            i, 'winning_team_id', row['team_1_id'] if row['team_1_score'] > row['team_2_score'] else row['team_2_id'])
    print('Finished deciding winners')
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
        team_1_key = str(row['map']) + str(row['team_1_id'])
        team_2_key = str(row['map']) + str(row['team_2_id'])
        if team_1_key not in already_calculated_keys:
            winrate = calculate_winrate_for_team_on_map(
                df_total, row['team_1_id'], row['map'])
            already_calculated_keys[team_1_key] = winrate
            df_total.set_value(i, 'team_1_winrate', winrate)
        else:
            df_total.set_value(i, 'team_1_winrate',
                               already_calculated_keys[team_1_key])
        if team_2_key not in already_calculated_keys:
            winrate = calculate_winrate_for_team_on_map(
                df_total, row['team_2_id'], row['map'])
            already_calculated_keys[team_2_key] = winrate
            df_total.set_value(i, 'team_2_winrate', winrate)
        else:
            df_total.set_value(i, 'team_2_winrate',
                               already_calculated_keys[team_2_key])
    print('Finished calculating winrates')
    df_total.to_csv('winrates.csv', index=False)
    return df_total


def split_data_train_test(df):
    x = df[['team_1_winrate', 'team_2_winrate']].values
    y = df['winning_team_nr'].values
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=1337)
    return {'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test}


def decision_trees(sets):
    clf = DecisionTreeClassifier(criterion="gini", random_state=100,
                                 max_depth=3, min_samples_leaf=5)
    clf.fit(sets['x_train'], sets['y_train'])
    y_pred = clf.predict(sets['x_test'])
    print(
        f"Decision trees cross validation score is {cross_val_score(clf, df_final_data[['team_1_winrate', 'team_2_winrate']].values, df_final_data['winning_team_nr'].values, cv=5)}")


def support_vector_machines(sets):
    clf = svm.SVC()
    clf.fit(sets['x_train'], sets['y_train'])
    y_pred = clf.predict(sets['x_test'])
    print(
        f"Support Vector Machines cross validation score is {cross_val_score(clf, df_final_data[['team_1_winrate', 'team_2_winrate']].values, df_final_data['winning_team_nr'].values, cv=5)}")


def neural_network_mlp(sets):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(sets['x_train'], sets['y_train'])
    y_pred = clf.predict(sets['x_test'])
    print(
        f"Neural Network MLP cross validation score is {cross_val_score(clf, df_final_data[['team_1_winrate', 'team_2_winrate']].values, df_final_data['winning_team_nr'].values, cv=5)}")


def random_forest(sets):
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(sets['x_train'], sets['y_train'])
    y_pred = clf.predict(sets['x_test'])
    print(
        f"Random Forest cross validation score is {cross_val_score(clf, df_final_data[['team_1_winrate', 'team_2_winrate']].values, df_final_data['winning_team_nr'].values, cv=5)}")


def linear_regression(sets):
    clf = LinearRegression()
    clf.fit(sets['x_train'], sets['y_train'])
    y_pred = clf.predict(sets['x_test'])
    print(
        f"Linear Regression cross validation score is {cross_val_score(clf, df_final_data[['team_1_winrate', 'team_2_winrate']].values, df_final_data['winning_team_nr'].values, cv=5)}")


def naive_bayes(sets):
    clf = GaussianNB()
    clf.fit(sets['x_train'], sets['y_train'])
    y_pred = clf.predict(sets['x_test'])
    print(
        f"Naive Bayes cross validation score is {cross_val_score(clf, df_final_data[['team_1_winrate', 'team_2_winrate']].values, df_final_data['winning_team_nr'].values, cv=5)}")


print('Finished loading functions')

#%%
df_cleaned_data = decide_winners(clean_matches_dataset())
df_final_data = calculate_winrate_for_matches(df_cleaned_data)

#%%
train_test_sets = split_data_train_test(pd.read_csv('winrates.csv'))
print('Finished splitting test and train data')

#%%
decision_trees(train_test_sets)
support_vector_machines(train_test_sets)
neural_network_mlp(train_test_sets)
random_forest(train_test_sets)
linear_regression(train_test_sets)
naive_bayes(train_test_sets)
print('Finished getting accuracy of all alghoritms')
