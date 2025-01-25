from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from getdata import *
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 24,           # Base font size
    'axes.titlesize': 28,      # Title font size
    'axes.labelsize': 24,      # x and y label font size
    'xtick.labelsize': 24,     # x-axis tick font size
    'ytick.labelsize': 24,     # y-axis tick font size
})

#creates predictive performance graph
def prediction_analysis(predictions_df, gender, model, feat):
    output_folder = '/Users/livio/Documents/img_BT/test/predictive_analysis'
    os.makedirs(output_folder, exist_ok=True)
    bin_edges = np.arange(0, 1.1, 0.1) #steps of 0.1 from 0 to 1.0

    # concat all ten dataframes together in one large and assign each entry to a bin based on their prediction
    predictions_df = pd.concat(predictions_df, ignore_index=True)
    predictions_df['xG_bin'] = pd.cut(predictions_df['y_pred'], bins=bin_edges, include_lowest=True)
    total = len(predictions_df)

    bin_stats = predictions_df.groupby('xG_bin').agg(
        avg_xG=('y_pred', 'mean'),  #average xG for each bin (x-axis value)
        actl_goals=('y_actual', 'sum'), #total goals
        total_shots=('y_actual', 'size'), #total shots
        bin_size=('y_pred', 'size') #number of shots in this group
    ).reset_index()

    plt.figure(figsize=(12, 8))

    # Scatter plot: x-axis is avg of predicted xG / y-axis is actual xG based on all shots in this bin
    plt.scatter(bin_stats['avg_xG'], bin_stats['actl_goals'] / bin_stats['total_shots'],
                s= bin_stats['bin_size']/total*5000,color='#00796b' if gender=='m' else '#ff7043' ,alpha=0.7, label="Shots in Bin")

    # Add perfect prediction line
    plt.plot([0, 1], [0, 1], color="black", linestyle="-")

    #ratio of x-axis and y-axis is the same
    plt.gca().set_aspect('equal', adjustable='box')

    #create title for the plot
    #assign abbreviations
    feature_mapping = {
        'distance': 'D',
        'vis_angle': r'$A_{V}$',
        'angle': r'$A_{S}$'
    }
    shortened_features = [feature_mapping.get(feature, 'unknown') for feature in feat]
    feature_str = ', '.join(shortened_features)

    #model, gender and features extraction
    model_name = 'LR' if model == 'LR' else ('XGB' if model == 'xgb' else 'StatsBomb')
    gender_name = 'Men: ' if gender == 'm' else 'Women: '
    trained_str = ' '
    if model != 'statsbomb':
        trained_str = f" trained on {feature_str}"

    title = f"Predictive Performance\n{gender_name} {model_name}{trained_str}"

    plt.xlabel('Predicted xG')
    plt.ylabel('Actual xG')
    plt.title(title)

    filename = os.path.join(output_folder, f"{gender}{model}{feat}.pdf")
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

# this functions creates a naive model, that assigns to each shot the same probability
def xG_naive(X, y):
    split = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

    train_log = []  # log losses by training data
    test_log = []  # log losses by testing data
    train_pred = []  # goal predicted on training set
    test_pred = []  # goal predicted on testing set

    for train_index, test_index in split.split(X, y):
        # split data
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # calculate the average probability a goal is scored
        total_shots = len(y_train)
        total_goals = sum(y_train)
        ratio = total_goals / total_shots

        # assign to each shot the same value
        y_train_pred = [ratio] * len(y_train)
        y_test_pred = [ratio] * len(y_test)

        # calculate log loss on training and testing dataset
        train_log_loss = log_loss(y_train, y_train_pred)
        train_log.append(train_log_loss)

        test_log_loss = log_loss(y_test, y_test_pred)
        test_log.append(test_log_loss)

        # predict number of goals on training and testing dataset
        pred_train = sum(y_train_pred)
        train_pred.append(pred_train)

        pred_test = sum(y_test_pred)
        test_pred.append(pred_test)

    print(f"avg train log loss: {(sum(train_log) / len(train_log)):.4f}")
    print(f"avg test log loss: {(sum(test_log) / len(test_log)):.4f}")
    print(f"avg train goal prediction: {(sum(train_pred) / len(train_pred)):.1f}")
    print(f"avg test goal prediction: {(sum(test_pred) / len(test_pred)):.1f}")

def xG_log(X, y, gender, feat):

    split = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

    train_log = [] #log losses by training data
    test_log = [] #log losses by testing data
    train_pred = [] #goal predicted on training set
    test_pred = [] #goal predicted on testing set

    predictions = [] #store each prediction and its actual value (0 or 1)

    for train_index, test_index in split.split(X, y):
        #split data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        #create LR model and train it
        model = LogisticRegression(solver='lbfgs', C=1, max_iter=100)
        model.fit(X_train, y_train)

        #get predictions
        y_train_pred = model.predict_proba(X_train)[:, 1]
        y_test_pred = model.predict_proba(X_test)[:, 1]

        # create dataframe which is later used for predictive performance analysis
        preds = pd.DataFrame({
            "y_pred": y_test_pred,
            "y_actual": y_test.values  # convert y to array
        })
        predictions.append(preds)

        # calculate log loss on training and testing dataset
        train_log_loss = log_loss(y_train, y_train_pred)
        train_log.append(train_log_loss)

        test_log_loss = log_loss(y_test, y_test_pred)
        test_log.append(test_log_loss)

        # predict number of goals on training and testing dataset
        pred_train = sum(y_train_pred)
        train_pred.append(pred_train)

        pred_test = sum(y_test_pred)
        test_pred.append(pred_test)

    print(f"avg train log loss: {(sum(train_log) / len(train_log)):.4f}")
    print(f"avg test log loss: {(sum(test_log) / len(test_log)):.4f}")
    print(f"avg train goal prediction: {(sum(train_pred) / len(train_pred)):.1f}")
    print(f"avg test goal prediction: {(sum(test_pred) / len(test_pred)):.1f}")

    prediction_analysis(predictions, gender, 'LR', feat)

def xG_Boost(X, y, gender, feat):

    split = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

    train_log = []  # log losses by training data
    test_log = []  # log losses by testing data
    train_pred = []  # goal predicted on training set
    test_pred = []  # goal predicted on testing set

    predictions = [] #store each prediction and its actual value (0 or 1)

    for train_index, test_index in split.split(X, y):
        # split data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        #create XGB model and train it
        model = XGBClassifier(max_depth=4, learning_rate=0.1, gamma=0.3, random_state=42)
        model.fit(X_train, y_train)

        # get predictions
        y_train_pred = model.predict_proba(X_train)[:, 1]
        y_test_pred = model.predict_proba(X_test)[:, 1]

        # create dataframe which is later used for predictive performance analysis
        preds = pd.DataFrame({
            "y_pred": y_test_pred,
            "y_actual": y_test.values #convert y to array
        })
        predictions.append(preds)

        # calculate log loss on training and testing dataset
        train_log_loss = log_loss(y_train, y_train_pred)
        train_log.append(train_log_loss)

        test_log_loss = log_loss(y_test, y_test_pred)
        test_log.append(test_log_loss)

        # predict number of goals on training and testing dataset
        pred_train = sum(y_train_pred)
        train_pred.append(pred_train)

        pred_test = sum(y_test_pred)
        test_pred.append(pred_test)

    print(f"avg train log loss: {(sum(train_log) / len(train_log)):.4f}")
    print(f"avg test log loss: {(sum(test_log) / len(test_log)):.4f}")
    print(f"avg train goal prediction: {(sum(train_pred) / len(train_pred)):.1f}")
    print(f"avg test goal prediction: {(sum(test_pred) / len(test_pred)):.1f}")

    prediction_analysis(predictions, gender, 'xgb', feat)

# check xG from statsbomb
def xG_statsbomb(X, y, xG, gender):
    split = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

    train_log = []  # log losses by training data
    test_log = []  # log losses by testing data
    train_pred = []  # goal predicted on training set
    test_pred = []  # goal predicted on testing set

    predictions = [] #store each prediction and its actual value (0 or 1)

    for train_index, test_index in split.split(X, y):
        # split data
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        xG_train, xG_test = xG.iloc[train_index], xG.iloc[test_index]

        # create dataframe which is later used for predictive performance analysis
        preds = pd.DataFrame({
            "y_pred": xG_test.values, #convert xG to array
            "y_actual": y_test.values  # convert y to array
        })
        predictions.append(preds)

        # calculate log loss on training and testing dataset
        train_log_loss = log_loss(y_train, xG_train)
        train_log.append(train_log_loss)

        test_log_loss = log_loss(y_test, xG_test)
        test_log.append(test_log_loss)

        # predict number of goals on training and testing dataset
        pred_train = sum(xG_train)
        train_pred.append(pred_train)

        pred_test = sum(xG_test)
        test_pred.append(pred_test)

    print(f"avg train log loss: {(sum(train_log) / len(train_log)):.4f}")
    print(f"avg test log loss: {(sum(test_log) / len(test_log)):.4f}")
    print(f"avg train goal prediction: {(sum(train_pred) / len(train_pred)):.1f}")
    print(f"avg test goal prediction: {(sum(test_pred) / len(test_pred)):.1f}")

    prediction_analysis(predictions, gender, 'statsbomb', '')

def xG_models(shot_df, gender):
    X = shot_df[['distance', 'angle', 'vis_angle']]
    y = shot_df['goal']
    statsbomb_xG = shot_df['xG']

    print("Naive Model")
    xG_naive(X, y)

    print("StatsBomb")
    xG_statsbomb(X, y, statsbomb_xG, gender)

    for i in [['vis_angle'], ['distance', 'vis_angle'], ['distance', 'vis_angle', 'angle'], ['distance'], ['angle'], ['distance','angle']]:
        print(i)

        print("Logistic Regression")
        xG_log(X[i], y, gender, i)

        print("XGBoost")
        xG_Boost(X[i], y, gender, i)

#this code is used to analyse the performance of the models and also creating a predictive performance graph
def main():
    m_shot, w_shot = get_data()

    print("men")
    xG_models(m_shot, 'm')

    print("women")
    xG_models(w_shot, 'w')

if __name__ == "__main__":
    main()