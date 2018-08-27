import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import re

from sklearn import linear_model, svm, tree, neighbors, naive_bayes
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.pipeline import Pipeline

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


ROOT_DIR='../..'

occupancy_df = pd.read_csv(filepath_or_buffer=ROOT_DIR + '/Occupancy_sensor/occupancy_sensor_data.csv')

# Extract Features
features = occupancy_df.loc[:, :'HumidityRatio']
targets = occupancy_df.loc[:, 'Occupancy']

# Transform the date into a number
features['date'] = features['date'].apply(date_to_minutes)

# Process the features
pipeline = Pipeline([
        ('imputer', Imputer(strategy='median')),
        ('std_scaler', StandardScaler())
    ])

processed_features = pd.DataFrame(pipeline.fit_transform(features))

# Partition data into training and test sets
data_len = processed_features.shape[0]
test_size = math.floor(data_len/10)

# Get an even distribution of occupied and unoccupied data samples
num_occupied = 0
num_unoccupied = 0;
max_unoccupied = math.floor(test_size/2)
max_occupied = test_size - max_unoccupied

test_features_list = []
test_targets_list = []
rows_to_drop = []

for i in range(data_len):
    if targets.iloc[i] == 0 and num_unoccupied < max_unoccupied:
        test_features_list.append(processed_features.iloc[i].tolist())
        test_targets_list.append(targets.iloc[i])
        rows_to_drop.append(i)
        num_unoccupied += 1
    elif targets.iloc[i] == 1 and num_occupied < max_occupied:
        test_features_list.append(processed_features.iloc[i].tolist())
        test_targets_list.append(targets.iloc[i])
        rows_to_drop.append(i)
        num_occupied += 1

for x in rows_to_drop:
    processed_features.drop(processed_features.index[x], inplace=True)
    targets.drop(targets.index[x], inplace=True)



test_features = pd.DataFrame(test_features_list, columns=processed_features.columns)
test_targets = pd.Series(test_targets_list)

print('-------------------------------------------')
print('------------CLASSIFICATION TASK------------')
print('-------------------------------------------')
# Perceptron model
perceptron = linear_model.Perceptron()

# Train the model
perceptron.fit(processed_features, targets)

# Evaluate the model
print('-----------PERFORMANCE OF PERCEPTRON----------')
print('Coefficients: ', perceptron.coef_)
print('Mean accuracy of predictions: {:.2f}'.format(perceptron.score(test_features, test_targets)))

# SVM model
svm_clf = svm.SVC(kernel='linear')
svm_clf.fit(processed_features, targets)

# Evaluate the model
print('--------------PERFORMANCE OF SVM--------------')
# print('Coefficients: ', svm_clf.coef_)
print('Mean accuracy of predictions: {:.2f}'.format(svm_clf.score(test_features, test_targets)))

# Decision Tree Model
tree_clf = tree.DecisionTreeClassifier()
tree_clf.fit(processed_features, targets)

print('--------PERFORMANCE OF DECISION TREES---------')
print('Mean accuracy of predictions: {:.2f}'.format(tree_clf.score(test_features, test_targets)))

# K-Nearest Model
knear_clf = neighbors.KNeighborsClassifier(n_neighbors=3)
knear_clf.fit(processed_features, targets)

print('-----PERFORMANCE OF K-NEAREST NEIGHBOURS------')
print('Mean accuracy of predictions: {:.2f}'.format(knear_clf.score(test_features, test_targets)))

# Naive Bayes Model
bayes_clf = naive_bayes.GaussianNB()
bayes_clf.fit(processed_features, targets)

print('-------PERFORMANCE OF NAIVE BAYES--------')
print('Mean accuracy of predictions: {:.2f}'.format(bayes_clf.score(test_features, test_targets)))
