import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from sklearn import linear_model, svm, tree, neighbors, naive_bayes
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.pipeline import Pipeline

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
test_size = math.floor(rows/10)
class_freq = math.floor(test_size/6)

# Get an even distribution of each class
class_counts = {1:0,2:0,3:0,4:0,5:0,7:0}

test_features_list = []
test_targets_list = []
rows_to_drop = []

for row in range(rows):
    row_class = targets.iloc[row]
    if class_counts[row_class] < class_freq:
        class_counts[row_class] += 1
        test_features_list.append(processed_features.iloc[row].tolist())
        test_targets_list.append(row_class)
        rows_to_drop.append(row)

for x in rows_to_drop:
    processed_features.drop(processed_features.index[x], inplace=True)
    targets.drop(targets.index[x], inplace=True)

test_features = pd.DataFrame(test_features_list)
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
print('Mean accuracy of predictions (perceptron): {:.2f}'.format(perceptron.score(test_features, test_targets)))

# SVM model
svm_clf = svm.SVC(kernel='linear')
svm_clf.fit(processed_features, targets)

# Evaluate the model
print('--------------PERFORMANCE OF SVM--------------')
# print('Coefficients: ', svm_clf.coef_)
print('Mean accuracy of predictions (svm): {:.2f}'.format(svm_clf.score(test_features, test_targets)))

# Decision Tree Model
tree_clf = tree.DecisionTreeClassifier()
tree_clf.fit(processed_features, targets)

print('--------PERFORMANCE OF DECISION TREES---------')
print('Mean accuracy of predictions (decision trees): {:.2f}'.format(tree_clf.score(test_features, test_targets)))

# K-Nearest Model
knear_clf = neighbors.KNeighborsClassifier(n_neighbors=3)
knear_clf.fit(processed_features, targets)

print('-----PERFORMANCE OF K-NEAREST NEIGHBOURS------')
print('Mean accuracy of predictions (nearest neighbours): {:.2f}'.format(knear_clf.score(test_features, test_targets)))

# Naive Bayes Model
bayes_clf = naive_bayes.GaussianNB()
bayes_clf.fit(processed_features, targets)

print('-------PERFORMANCE OF NAIVE BAYES--------')
print('Mean accuracy of predictions (naive bayes): {:.2f}'.format(bayes_clf.score(test_features, test_targets)))
