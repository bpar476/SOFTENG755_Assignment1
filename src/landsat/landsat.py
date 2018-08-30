import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from sklearn import linear_model, svm, tree, neighbors, naive_bayes
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, log_loss

ROOT_DIR='../..'

ls_df = pd.read_csv(filepath_or_buffer=ROOT_DIR + '/Landsat/lantsat.csv', header=None)

rows, cols = ls_df.shape

features = ls_df.iloc[:, :cols-1]
targets = ls_df.iloc[:, cols-1]

# Pre-process the features
pipeline = Pipeline([
        ('imputer', Imputer(strategy='median')),
        ('std_scaler', StandardScaler())
    ])

processed_features = pd.DataFrame(pipeline.fit_transform(features))

# Partition data into training and test sets
class_freq = min(math.floor((rows - rows/10)/6), math.floor(min(ls_df.groupby(cols-1).size()) * 0.9))
training_set_limit = rows - math.floor(rows/10)

# Get an even distribution of each class
class_counts = {1:0,2:0,3:0,4:0,5:0,7:0}

train_features_list = []
train_targets_list = []
rows_to_drop = []

total_training_samples = 0

for row in range(rows):
    row_class = targets.iloc[row]
    if class_counts[row_class] < class_freq and total_training_samples < training_set_limit:
        class_counts[row_class] += 1
        train_features_list.append(processed_features.iloc[row].tolist())
        train_targets_list.append(row_class)
        rows_to_drop.append(row)
        total_training_samples += 1

for x in rows_to_drop:
    processed_features.drop(x, inplace=True)
    targets.drop(x, inplace=True)

train_features = pd.DataFrame(train_features_list)
train_targets = pd.Series(train_targets_list)

print('-------------------------------------------')
print('------------CLASSIFICATION TASK------------')
print('-------------------------------------------')
# Perceptron model
perceptron = linear_model.Perceptron()

# Train the model
perceptron.fit(train_features, train_targets)

perceptron_prediction_results = perceptron.predict(processed_features)

# Evaluate the model
print('-----------PERFORMANCE OF PERCEPTRON----------')
print('Mean accuracy of predictions (perceptron): {:.2f}'.format(perceptron.score(processed_features, targets)))
print('f1 score of perceptron: {}'.format(f1_score(perceptron_prediction_results, targets, average=None)))

# SVM model
svm_clf = svm.SVC(kernel='linear')
svm_clf.fit(train_features, train_targets)

svm_prediction = svm_clf.predict(processed_features)

# Evaluate the model
print('--------------PERFORMANCE OF SVM--------------')
print('Mean accuracy of predictions (svm): {:.2f}'.format(svm_clf.score(processed_features, targets)))
print('f1 score of svm: {}'.format(f1_score(svm_prediction, targets, average=None)))

# Decision Tree Model
tree_clf = tree.DecisionTreeClassifier()
tree_clf.fit(train_features, train_targets)

tree_prediction = tree_clf.predict(processed_features)

print('--------PERFORMANCE OF DECISION TREES---------')
print('Mean accuracy of predictions (decision trees): {:.2f}'.format(tree_clf.score(processed_features, targets)))
print('f1 score of decision trees: {}'.format(f1_score(tree_prediction, targets, average=None)))

# K-Nearest Model
knear_clf = neighbors.KNeighborsClassifier(n_neighbors=3)
knear_clf.fit(train_features, train_targets)

knear_prediction = knear_clf.predict(processed_features)

print('-----PERFORMANCE OF K-NEAREST NEIGHBOURS------')
print('Mean accuracy of predictions (nearest neighbours): {:.2f}'.format(knear_clf.score(processed_features, targets)))
print('f1 score of nearest neighbours: {}'.format(f1_score(knear_prediction, targets, average=None)))

# Naive Bayes Model
bayes_clf = naive_bayes.GaussianNB()
bayes_clf.fit(train_features, train_targets)

bayes_prediction = bayes_clf.predict(processed_features)

print('-------PERFORMANCE OF NAIVE BAYES--------')
print('Mean accuracy of predictions (naive bayes): {:.2f}'.format(bayes_clf.score(processed_features, targets)))
print('f1 score of naive bayes: {}'.format(f1_score(bayes_prediction, targets, average=None)))
