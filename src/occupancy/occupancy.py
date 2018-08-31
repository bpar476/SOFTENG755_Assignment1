import numpy as np
import pandas as pd
import math
import re
import argparse

from sklearn import linear_model, svm, tree, neighbors, naive_bayes
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

parser = argparse.ArgumentParser(description='Machine learning algorithm for 2018 world cup data.')
parser.add_argument('-t', '--test-data', help='path to additional features to test against. Path must be relative to the current directory. If supplied, results of predictions against this test data will be the last thing printed by this script (optional)')

parsed_args = parser.parse_args()

TEST_FILE_PATH = parsed_args.test_data

ROOT_DIR='../..'
FILE_PATH_FROM_ROOT='/Occupancy_sensor/occupancy_sensor_data.csv'

'''
    Used to transform the date into a number. We ignore the date, and only
    take the time. We can't figure out the day of the week so the actual
    date probably doesn't matter.
'''
def date_to_minutes(time_string):
    match = re.match('.*([0-9]{1,2}):([0-9]{2})', time_string)
    hours = int(match.group(1))
    minutes = int(match.group(2))
    return 60 * hours + minutes

def preprocess_features(features):
    # Transform the date into a number
    features = features.loc[:, :'HumidityRatio']
    features['date'] = features['date'].apply(date_to_minutes)

    # Process the features
    pipeline = Pipeline([
            ('imputer', Imputer(strategy='median')),
            ('std_scaler', StandardScaler())
        ])

    processed_features = pd.DataFrame(pipeline.fit_transform(features))

    return processed_features

occupancy_df = pd.read_csv(filepath_or_buffer=ROOT_DIR + FILE_PATH_FROM_ROOT)

# Extract Features
features = occupancy_df.loc[:, :'Occupancy']
targets = occupancy_df.loc[:, 'Occupancy']

processed_features = preprocess_features(features)

# Partition data into training and test sets
data_len = processed_features.shape[0]

class_freq = min(data_len - math.ceil(data_len/10)/2, math.ceil(min(occupancy_df.groupby('Occupancy').size()) * 0.9))
training_size = data_len - math.ceil(data_len/10)

# Get an even distribution of occupied and unoccupied data samples
num_occupied = 0
num_unoccupied = 0

train_features_list = []
train_targets_list = []
rows_to_drop = []

total_training_samples = 0;

for i in range(data_len):
    if total_training_samples == training_size:
        break

    if targets.iloc[i] == 0 and num_unoccupied < class_freq:
        train_features_list.append(processed_features.iloc[i].tolist())
        train_targets_list.append(targets.iloc[i])
        rows_to_drop.append(i)
        num_unoccupied += 1
        total_training_samples += 1
    elif targets.iloc[i] == 1 and num_occupied < class_freq:
        train_features_list.append(processed_features.iloc[i].tolist())
        train_targets_list.append(targets.iloc[i])
        rows_to_drop.append(i)
        num_occupied += 1
        total_training_samples += 1

for x in rows_to_drop:
    processed_features.drop(x, inplace=True)
    targets.drop(x, inplace=True)

training_features = pd.DataFrame(train_features_list, columns=processed_features.columns)
training_targets = pd.Series(train_targets_list)

print('-------------------------------------------')
print('------------CLASSIFICATION TASK------------')
print('-------------------------------------------')

print('F1 scores given in numeric order: 0 (unoccupied), 1 (occupied)')
# Perceptron model
perceptron_parameters = {'max_iter': [5, 10, 20], 'penalty': ['elasticnet', None], 'alpha': [1e-3, 1e-4, 1e-5]}
perceptron = GridSearchCV(linear_model.Perceptron(), perceptron_parameters, cv=5, scoring='f1_weighted')

# Train the model
perceptron.fit(training_features, training_targets)

perceptron_prediction = perceptron.predict(processed_features)

# Evaluate the model
print('-----------PERFORMANCE OF PERCEPTRON----------')
print('Tuned perceptron parameters: {}'.format(perceptron.best_params_))
print('Mean accuracy of predictions: {:.2f}'.format(perceptron.score(processed_features, targets)))
print('f1 score of perceptron: {}'.format(f1_score(targets, perceptron_prediction, average=None)))
print('Area under ROC curve score: {}'.format(roc_auc_score(targets, perceptron.decision_function(processed_features))))

# SVM model
svm_parameters = [{'C': [1, 10, 100, 1000], 'gamma': [1e-3, 1e-4, 1e-5], 'kernel': ['rbf'], 'class_weight':['balanced', None]},
    {'C': [1, 10, 100, 1000], 'kernel': ['linear'], 'class_weight': ['balanced', None]}]

svm_clf = GridSearchCV(svm.SVC(), svm_parameters, cv=5, scoring='f1_weighted')
svm_clf.fit(training_features, training_targets)

svm_prediction = svm_clf.predict(processed_features)

# Evaluate the model
print('--------------PERFORMANCE OF SVM--------------')
print('Tuned SVM parameters: {}'.format(svm_clf.best_params_))
print('Mean accuracy of predictions: {:.2f}'.format(svm_clf.score(processed_features, targets)))
print('f1 score of svm: {}'.format(f1_score(targets, svm_prediction, average=None)))
print('Area under ROC curve score: {}'.format(roc_auc_score(targets, svm_clf.decision_function(processed_features))))

# Decision Tree Model
tree_parameters = {'criterion': ['entropy', 'gini'], 'max_depth': [None, 3, 8, 12], 'min_samples_leaf': list(range(1,9))}
tree_clf = RandomizedSearchCV(tree.DecisionTreeClassifier(), tree_parameters, cv=10, scoring='f1_weighted')
tree_clf.fit(training_features, training_targets)

tree_prediction = tree_clf.predict(processed_features)

print('--------PERFORMANCE OF DECISION TREES---------')
print('Tuned decision tree parameters: {}'.format(tree_clf.best_params_))
print('Mean accuracy of predictions: {:.2f}'.format(tree_clf.score(processed_features, targets)))
print('f1 score of decision trees: {}'.format(f1_score(targets, tree_prediction, average=None)))

# K-Nearest Model
knear_params = {'n_neighbors': [3,5,10, 15], 'weights': ['uniform', 'distance']}
knear_clf = GridSearchCV(neighbors.KNeighborsClassifier(), knear_params, cv=10, scoring='f1_weighted')
knear_clf.fit(training_features, training_targets)

knear_prediction = knear_clf.predict(processed_features)

print('-----PERFORMANCE OF K-NEAREST NEIGHBOURS------')
print('Tuned Nearest Neighbours parameters: {}'.format(knear_clf.best_params_))
print('Mean accuracy of predictions: {:.2f}'.format(knear_clf.score(processed_features, targets)))
print('f1 score of nearest neighbours: {}'.format(f1_score(targets, knear_prediction, average=None)))

# Naive Bayes Model
bayes_clf = naive_bayes.GaussianNB()
bayes_clf.fit(training_features, training_targets)

bayes_prediction = bayes_clf.predict(processed_features)

print('-------PERFORMANCE OF NAIVE BAYES--------')
print('Mean accuracy of predictions: {:.2f}'.format(bayes_clf.score(processed_features, targets)))
print('f1 score of naive bayes: {}'.format(f1_score(targets, bayes_prediction, average=None)))

if TEST_FILE_PATH is not None:
    print('Running predictions against supplied test data')

    test_data_df = pd.read_csv(filepath_or_buffer=TEST_FILE_PATH)
    processed_test_data = preprocess_features(test_data_df)

    perceptron_prediction = perceptron.predict(processed_test_data)
    svm_prediction = svm_clf.predict(processed_test_data)
    tree_prediction = tree_clf.predict(processed_test_data)
    knear_prediction = knear_clf.predict(processed_test_data)
    bayes_prediction = bayes_clf.predict(processed_test_data)

    with open('perceptron_prediction.txt', 'w') as perceptron_out:
        for pred in perceptron_prediction:
            perceptron_out.write('{}\n'.format(pred))

    with open('svm_prediction.txt', 'w') as svm_out:
        for pred in svm_prediction:
            svm_out.write('{}\n'.format(pred))

    with open('tree_prediction.txt', 'w') as tree_out:
        for pred in tree_prediction:
            tree_out.write('{}\n'.format(pred))

    with open('knear_prediction.txt', 'w') as knear_out:
        for pred in knear_prediction:
            knear_out.write('{}\n'.format(pred))

    with open('bayes_prediction.txt', 'w') as bayes_out:
        for pred in bayes_prediction:
            bayes_out.write('{}\n'.format(pred))

