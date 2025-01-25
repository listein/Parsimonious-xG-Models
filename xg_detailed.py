from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from getdata import *
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 24,  # Base font size
    'axes.titlesize': 28,  # Title font size
    'axes.labelsize': 24,  # x and y label font size
    'xtick.labelsize': 24,  # x-axis tick font size
    'ytick.labelsize': 24,  # y-axis tick font size
})

def xG_log(X, y, param):
    split = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

    test_log = []  # log losses by testing data
    number_of_shots = []

    for train_index, test_index in split.split(X, y):
        # split data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # create model and train it
        model = LogisticRegression(solver='lbfgs', C=1, max_iter=100)
        model.fit(X_train, y_train)

        # get predictions
        y_test_pred = model.predict_proba(X_test)[:, 1]

        # only take the entries that are within the interval
        valid_indices = (y_test_pred >= param[0] - param[1]) & (y_test_pred < param[0])
        y_test = y_test[valid_indices]
        y_test_pred = y_test_pred[valid_indices]

        #store number of shots within testing dataset
        no_shots = len(y_test)
        number_of_shots.append(no_shots)

        if no_shots >= 2 and len(set(y_test)) > 1:
            # calculate log loss on testing dataset
            test_log_loss = log_loss(y_test, y_test_pred)
            test_log.append(test_log_loss)

    if len(number_of_shots) > 0:
        print(f"avg number of shots: {sum(number_of_shots) / len(number_of_shots):.1f}")
    else:
        print('avg number of shots: 0')
    if len(test_log) > 0:
        print(f"avg test log loss: {(sum(test_log) / len(test_log)):.4f}")
    else:
        print('avg test log loss: 0')

def xG_Boost(X, y, param):
    split = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

    test_log = []  # log losses by testing data
    number_of_shots = []

    for train_index, test_index in split.split(X, y):
        # split data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # create model and train it
        model = XGBClassifier(max_depth=4, learning_rate=0.1, gamma=0.3, random_state=42)
        model.fit(X_train, y_train)

        # get predictions
        y_test_pred = model.predict_proba(X_test)[:, 1]

        # only take the entries that are within the interval
        valid_indices = (y_test_pred >= param[0] - param[1]) & (y_test_pred < param[0])
        y_test = y_test[valid_indices]
        y_test_pred = y_test_pred[valid_indices]

        #store number of shots within testing dataset
        no_shots = len(y_test)
        number_of_shots.append(no_shots)

        if no_shots >= 2 and len(set(y_test)) > 1:
            # calculate log loss on testing dataset
            test_log_loss = log_loss(y_test, y_test_pred)
            test_log.append(test_log_loss)

    if len(number_of_shots) > 0:
        print(f"avg number of shots: {sum(number_of_shots) / len(number_of_shots):.1f}")
    else:
        print('avg number of shots: 0')
    if len(test_log) > 0:
        print(f"avg test log loss: {(sum(test_log) / len(test_log)):.4f}")
    else:
        print('avg test log loss: 0')

def xG_statsbomb(X, y, xG, param):
    split = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

    test_log = []  # log losses by testing data
    number_of_shots = []

    for train_index, test_index in split.split(X, y):
        # split data, take only data from testing set, as we don't train the StatsBomb model
        _, y_test = y.iloc[train_index], y.iloc[test_index]
        _, xG_test = xG.iloc[train_index], xG.iloc[test_index]

        #only take the entries that are within the interval
        valid_indices = (xG_test >= param[0] - param[1]) & (xG_test < param[0])
        y_test = y_test[valid_indices]
        xG_test = xG_test[valid_indices]

        #store number of shots within testing dataset
        no_shots = len(y_test)
        number_of_shots.append(no_shots)

        if no_shots >= 2 and len(set(y_test)) > 1:
            #calculate the log loss on the testing dataset
            test_log_loss = log_loss(y_test, xG_test)
            test_log.append(test_log_loss)

    if len(number_of_shots) > 0:
        print(f"avg number of shots: {sum(number_of_shots) / len(number_of_shots):.1f}")
    else:
        print('avg number of shots: NaN')

    if len(test_log) > 0:
        print(f"avg test log loss: {(sum(test_log) / len(test_log)):.4f}")
    else:
        print('avg test log loss: NaN')

def xG_models(shot_df):
    X = shot_df[['distance', 'angle', 'vis_angle']]
    y = shot_df['goal']
    statsbomb_xG = shot_df['xG']

    # loop through various intervals [a, b]: a is the upper value and b is the range ([a-b, a])
    for j in [[0.1, 0.1], [0.2, 0.1], [0.3, 0.1], [0.5, 0.2], [0.75, 0.25], [1, 0.25]]:
        print("StatsBomb")
        xG_statsbomb(X, y, statsbomb_xG, j)

    #loop through every relevant combination of features
    for i in [['vis_angle'], ['distance', 'vis_angle'], ['distance', 'vis_angle', 'angle'], ['distance'], ['angle'], ['distance','angle']]:
        print(i)

        for j in [[0.1, 0.1], [0.2, 0.1], [0.3, 0.1], [0.5, 0.2], [0.75, 0.25], [1, 0.25]]:
            print("Logistic Regression")
            xG_log(X[i], y, j)

        for j in [[0.1, 0.1], [0.2, 0.1], [0.3, 0.1], [0.5, 0.2], [0.75, 0.25], [1, 0.25]]:
            print("XGBoost")
            xG_Boost(X[i], y, j)

#This code is used for a more detailed analysis of the predictive performance of the xG models.
def main():
    m_shot, w_shot = get_data()

    print(m_shot[m_shot['angle'] < m_shot['vis_angle']])

    print("men")
    xG_models(m_shot)

    print("women")
    xG_models(w_shot)

if __name__ == "__main__":
    main()