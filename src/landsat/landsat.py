import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import argparse

from sklearn import linear_model, svm, tree, neighbors, naive_bayes
from sklearn.preprocessing import Imputer, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

parser = argparse.ArgumentParser(description='Machine learning algorithm for 2018 world cup data.')
parser.add_argument('-t', '--test-data', help='path to additional features to test against. Path must be relative to the current directory. If supplied, results of predictions against this test data will be the last thing printed by this script (optional)')

parsed_args = parser.parse_args()

TEST_FILE_PATH = parsed_args.test_data

ROOT_DIR='../..'
FILE_PATH_FROM_ROOT='/Landsat/lantsat.csv'

def preprocess_features(features):
    # Pre-process the features
    pipeline = Pipeline([
            ('imputer', Imputer(strategy='median')),
            ('std_scaler', StandardScaler())
        ])

    processed_features = pd.DataFrame(pipeline.fit_transform(features))

    return processed_features

ls_df = pd.read_csv(filepath_or_buffer=ROOT_DIR + FILE_PATH_FROM_ROOT, header=None)

rows, cols = ls_df.shape

features = ls_df.iloc[:, :cols-2]
targets = ls_df.iloc[:, cols-1]

processed_features = preprocess_features(features)

le = LabelEncoder()
le.fit(targets)
targets = le.transform(targets)

# Partition data into training and test sets
class_freq = min(math.floor((rows - rows/10)/6), math.floor(min(ls_df.groupby(cols-1).size()) * 0.9))
training_set_limit = rows - math.floor(rows/10)

# Get an even distribution of each class
class_counts = {0:0,1:0,2:0,3:0,4:0,5:0}

train_features_list = []
train_targets_list = []
rows_to_drop = []

total_training_samples = 0

for row in range(rows):
    row_class = targets[row]
    if class_counts[row_class] < class_freq and total_training_samples < training_set_limit:
        class_counts[row_class] += 1
        train_features_list.append(processed_features.iloc[row].tolist())
        train_targets_list.append(row_class)
        rows_to_drop.append(row)
        total_training_samples += 1

for x in rows_to_drop:
    processed_features.drop(x, inplace=True)

test_targets = []
row_to_drop_index = 0
for i in range(len(targets)):
    if row_to_drop_index < len(rows_to_drop) and i == rows_to_drop[row_to_drop_index]:
        row_to_drop_index += 1
    else:
        test_targets.append(targets[i])

train_features = pd.DataFrame(train_features_list)
train_targets = pd.Series(train_targets_list)
test_targets = pd.Series(test_targets)

print('-------------------------------------------')
print('------------CLASSIFICATION TASK------------')
print('-------------------------------------------')

print('F1 scores given for each class in numeric order: 1,2,3,4,5,7')
# Perceptron model
perceptron_parameters = {'max_iter': [5, 10, 20], 'penalty': ['elasticnet', None], 'alpha': [1e-3, 1e-4, 1e-5]}
perceptron = GridSearchCV(linear_model.Perceptron(), perceptron_parameters, cv=5, scoring='f1_weighted')

# Train the model
perceptron.fit(train_features, train_targets)

perceptron_prediction_results = perceptron.predict(processed_features)

# Evaluate the model
print('-----------PERFORMANCE OF PERCEPTRON----------')
print('Tuned perceptron parameters: {}'.format(perceptron.best_params_))
print('Mean accuracy of predictions (perceptron): {:.2f}'.format(perceptron.score(processed_features, test_targets)))
print('f1 score of perceptron: {}'.format(f1_score(test_targets, perceptron_prediction_results, average=None)))
# SVM model
svm_parameters = [{'C': [1, 10, 100, 1000], 'gamma': [1e-3, 1e-4, 1e-5], 'kernel': ['rbf'], 'class_weight':['balanced', None]},
    {'C': [1, 10, 100, 1000], 'kernel': ['linear'], 'class_weight': ['balanced', None]}]

svm_clf = GridSearchCV(svm.SVC(), svm_parameters, cv=5, scoring='f1_weighted')
svm_clf.fit(train_features, train_targets)

svm_prediction = svm_clf.predict(processed_features)

# Evaluate the model
print('--------------PERFORMANCE OF SVM--------------')
print('Tuned SVM parameters: {}'.format(svm_clf.best_params_))
print('Mean accuracy of predictions (svm): {:.2f}'.format(svm_clf.score(processed_features, test_targets)))
print('f1 score of svm: {}'.format(f1_score(test_targets, svm_prediction, average=None)))

# Decision Tree Model
tree_parameters = {'criterion': ['entropy', 'gini'], 'max_depth': [None, 3, 8, 12], 'min_samples_leaf': list(range(1,9))}
tree_clf = RandomizedSearchCV(tree.DecisionTreeClassifier(), tree_parameters, cv=10, scoring='f1_weighted')
tree_clf.fit(train_features, train_targets)

tree_prediction = tree_clf.predict(processed_features)

print('--------PERFORMANCE OF DECISION TREES---------')
print('Tuned decision tree parameters: {}'.format(tree_clf.best_params_))
print('Mean accuracy of predictions (decision trees): {:.2f}'.format(tree_clf.score(processed_features, test_targets)))
print('f1 score of decision trees: {}'.format(f1_score(test_targets, tree_prediction, average=None)))

# K-Nearest Model
knear_params = {'n_neighbors': [3,5,10, 15], 'weights': ['uniform', 'distance']}
knear_clf = GridSearchCV(neighbors.KNeighborsClassifier(), knear_params, cv=10, scoring='f1_weighted')
knear_clf.fit(train_features, train_targets)

knear_prediction = knear_clf.predict(processed_features)

print('-----PERFORMANCE OF K-NEAREST NEIGHBOURS------')
print('Tuned Nearest Neighbours parameters: {}'.format(knear_clf.best_params_))
print('Mean accuracy of predictions (nearest neighbours): {:.2f}'.format(knear_clf.score(processed_features, test_targets)))
print('f1 score of nearest neighbours: {}'.format(f1_score(test_targets, knear_prediction, average=None)))

# Naive Bayes Model
bayes_clf = naive_bayes.GaussianNB()
bayes_clf.fit(train_features, train_targets)

bayes_prediction = bayes_clf.predict(processed_features)

print('-------PERFORMANCE OF NAIVE BAYES--------')
print('Mean accuracy of predictions (naive bayes): {:.2f}'.format(bayes_clf.score(processed_features, test_targets)))
print('f1 score of naive bayes: {}'.format(f1_score(test_targets, bayes_prediction, average=None)))

if TEST_FILE_PATH is not None:
    print('Running predictions against supplied test data')

    test_data_df = pd.read_csv(filepath_or_buffer=TEST_FILE_PATH, header=None)
    test_data = test_data_df.iloc[:, :cols-2]

    processed_test_data = preprocess_features(test_data)

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

