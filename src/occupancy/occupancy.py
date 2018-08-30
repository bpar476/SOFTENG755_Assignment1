import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import re

from sklearn import linear_model, svm, tree, neighbors, naive_bayes
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, roc_auc_score

ROOT_DIR='../..'

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
    features['date'] = features['date'].apply(date_to_minutes)

    # Process the features
    pipeline = Pipeline([
            ('imputer', Imputer(strategy='median')),
            ('std_scaler', StandardScaler())
        ])

    processed_features = pd.DataFrame(pipeline.fit_transform(features))

    return processed_features

occupancy_df = pd.read_csv(filepath_or_buffer=ROOT_DIR + '/Occupancy_sensor/occupancy_sensor_data.csv')

# Extract Features
features = occupancy_df.loc[:, :'HumidityRatio']
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
perceptron = linear_model.Perceptron()

# Train the model
perceptron.fit(training_features, training_targets)

perceptron_prediction = perceptron.predict(processed_features)

# Evaluate the model
print('-----------PERFORMANCE OF PERCEPTRON----------')
print('Mean accuracy of predictions: {:.2f}'.format(perceptron.score(processed_features, targets)))
print('f1 score of perceptron: {}'.format(f1_score(targets, perceptron_prediction, average=None)))
print('Area under ROC curve score: {}'.format(roc_auc_score(targets, perceptron.decision_function(processed_features))))

# SVM model
svm_clf = svm.SVC(kernel='linear')
svm_clf.fit(training_features, training_targets)

svm_prediction = svm_clf.predict(processed_features)

# Evaluate the model
print('--------------PERFORMANCE OF SVM--------------')
print('Mean accuracy of predictions: {:.2f}'.format(svm_clf.score(processed_features, targets)))
print('f1 score of svm: {}'.format(f1_score(targets, svm_prediction, average=None)))
print('Area under ROC curve score: {}'.format(roc_auc_score(targets, svm_clf.decision_function(processed_features))))

# Decision Tree Model
tree_clf = tree.DecisionTreeClassifier()
tree_clf.fit(training_features, training_targets)

tree_prediction = tree_clf.predict(processed_features)

print('--------PERFORMANCE OF DECISION TREES---------')
print('Mean accuracy of predictions: {:.2f}'.format(tree_clf.score(processed_features, targets)))
print('f1 score of decision trees: {}'.format(f1_score(targets, tree_prediction, average=None)))

# K-Nearest Model
knear_clf = neighbors.KNeighborsClassifier(n_neighbors=3)
knear_clf.fit(training_features, training_targets)

knear_prediction = knear_clf.predict(processed_features)

print('-----PERFORMANCE OF K-NEAREST NEIGHBOURS------')
print('Mean accuracy of predictions: {:.2f}'.format(knear_clf.score(processed_features, targets)))
print('f1 score of nearest neighbours: {}'.format(f1_score(targets, knear_prediction, average=None)))

# Naive Bayes Model
bayes_clf = naive_bayes.GaussianNB()
bayes_clf.fit(training_features, training_targets)

bayes_prediction = bayes_clf.predict(processed_features)

print('-------PERFORMANCE OF NAIVE BAYES--------')
print('Mean accuracy of predictions: {:.2f}'.format(bayes_clf.score(processed_features, targets)))
print('f1 score of naive bayes: {}'.format(f1_score(targets, bayes_prediction, average=None)))
