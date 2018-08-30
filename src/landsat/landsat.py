import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from sklearn import linear_model, svm, tree, neighbors, naive_bayes
from sklearn.preprocessing import Imputer, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

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

print('Encoding classes 1,2,3,4,5,7 as {}'.format(le.transform([1,2,3,4,5,7])))

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
print(test_targets)

print('-------------------------------------------')
print('------------CLASSIFICATION TASK------------')
print('-------------------------------------------')

print('F1 scores given for each class in numeric order: 1,2,3,4,5,7')
# Perceptron model
perceptron = linear_model.Perceptron()

# Train the model
perceptron.fit(train_features, train_targets)

perceptron_prediction_results = perceptron.predict(processed_features)

# Evaluate the model
print('-----------PERFORMANCE OF PERCEPTRON----------')
print('Mean accuracy of predictions (perceptron): {:.2f}'.format(perceptron.score(processed_features, test_targets)))
print('f1 score of perceptron: {}'.format(f1_score(test_targets, perceptron_prediction_results, average=None)))
# SVM model
svm_clf = svm.SVC(kernel='linear')
svm_clf.fit(train_features, train_targets)

svm_prediction = svm_clf.predict(processed_features)

# Evaluate the model
print('--------------PERFORMANCE OF SVM--------------')
print('Mean accuracy of predictions (svm): {:.2f}'.format(svm_clf.score(processed_features, test_targets)))
print('f1 score of svm: {}'.format(f1_score(test_targets, svm_prediction, average=None)))

# Decision Tree Model
tree_clf = tree.DecisionTreeClassifier()
tree_clf.fit(train_features, train_targets)

tree_prediction = tree_clf.predict(processed_features)

print('--------PERFORMANCE OF DECISION TREES---------')
print('Mean accuracy of predictions (decision trees): {:.2f}'.format(tree_clf.score(processed_features, test_targets)))
print('f1 score of decision trees: {}'.format(f1_score(test_targets, tree_prediction, average=None)))

# K-Nearest Model
knear_clf = neighbors.KNeighborsClassifier(n_neighbors=3)
knear_clf.fit(train_features, train_targets)

knear_prediction = knear_clf.predict(processed_features)

print('-----PERFORMANCE OF K-NEAREST NEIGHBOURS------')
print('Mean accuracy of predictions (nearest neighbours): {:.2f}'.format(knear_clf.score(processed_features, test_targets)))
print('f1 score of nearest neighbours: {}'.format(f1_score(test_targets, knear_prediction, average=None)))

# Naive Bayes Model
bayes_clf = naive_bayes.GaussianNB()
bayes_clf.fit(train_features, train_targets)

bayes_prediction = bayes_clf.predict(processed_features)

print('-------PERFORMANCE OF NAIVE BAYES--------')
print('Mean accuracy of predictions (naive bayes): {:.2f}'.format(bayes_clf.score(processed_features, test_targets)))
print('f1 score of naive bayes: {}'.format(f1_score(test_targets, bayes_prediction, average=None)))
