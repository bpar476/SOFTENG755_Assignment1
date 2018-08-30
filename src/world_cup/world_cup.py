import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import argparse
import scipy

from sklearn import linear_model, svm, tree, neighbors, naive_bayes
from sklearn.metrics import mean_squared_error, r2_score, f1_score
from sklearn.preprocessing import Imputer, StandardScaler, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import category_encoders as cs

parser = argparse.ArgumentParser(description='Machine learning algorithm for 2018 world cup data.')
parser.add_argument('-t', '--test-data', help='path to additional features to test against. Path must be relative to the current directory. If supplied, results of predictions against this test data will be the last thing printed by this script (optional)')
parser.add_argument('-d', '--different-values', help='use this flag if the number values of categorical columns in the evaluation dataset is different to the number in the learning dataset', action='store_const', default=False, const=True)

parsed_args = parser.parse_args()

TEST_FILE_PATH = parsed_args.test_data

ROOT_DIR='../..'
FILE_PATH_FROM_ROOT='/World_Cup_2018/2018_worldcup.csv'

def preprocess_features(features_to_process, test_set=False):
    # Drop unimportant columns
    features.drop(['Date', 'Team1_Ball_Possession(%)'],axis=1,inplace=True)


    # Separate categorical columns from numerical columns
    categorical_features_list = ['Location', 'Phase', 'Team1', 'Team2', 'Team1_Continent', 'Team2_Continent', 'Normal_Time']
    numerical_features = features_to_process.drop(categorical_features_list, axis=1, inplace=False)
    categorical_features = features_to_process[categorical_features_list].copy()


    # Preprocess features
    numerical_pipeline = Pipeline([
            ('selector', DataFrameSelector(list(numerical_features))),
            ('imputer', Imputer(strategy='median')),
            ('std_scaler', StandardScaler())
        ])

    category_pipeline = Pipeline([
            ('selector', DataFrameSelector(list(categorical_features))),
            ('cat_encoder', cs.OneHotEncoder(drop_invariant=True))
        ])

    if parsed_args.different_values:
        full_pipeline = FeatureUnion(transformer_list=[
                ('num_pipeline', numerical_pipeline),
            ])
    else:
        full_pipeline = FeatureUnion(transformer_list=[
                ('num_pipeline', numerical_pipeline),
                ('cat_pipeline', category_pipeline)
            ])

    prepared_features = pd.DataFrame(data=full_pipeline.fit_transform(features_to_process),index=np.arange(1,features_to_process.shape[0]+1))

    return prepared_features

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
            return self
    def transform(self, X):
        return X[self.attribute_names].values



wc_df = pd.read_csv(filepath_or_buffer=ROOT_DIR + FILE_PATH_FROM_ROOT, index_col=0)

features = wc_df.loc[:, :'Normal_Time'].copy()
goals = wc_df.loc[:, 'Total_Scores']
results = wc_df.loc[:, 'Match_result']

prepared_features = preprocess_features(features)

# Partition data into training and test data
num_rows = wc_df.shape[0]
training_threshold = math.floor(num_rows/10)

# Prepare inputs for ML algorithms

# Features
training_features = prepared_features.iloc[:num_rows - training_threshold]
testing_features = prepared_features.iloc[num_rows - training_threshold:]

# Scores for regression task
training_scores = goals.to_frame().iloc[:num_rows - training_threshold]
testing_scores = goals.to_frame().iloc[num_rows - training_threshold:]

# Results for classification task
le = LabelEncoder()
le.fit(results)
results = le.transform(results)

training_results = results[:num_rows - training_threshold]
testing_results = results[num_rows - training_threshold:]

print('-------------------------------------------')
print('--------------REGRESSION TASK--------------')
print('-------------------------------------------')

# Ordinary regression
ord_regr = linear_model.LinearRegression()
ord_regr.fit(training_features, training_scores)
ord_prediction_scores = ord_regr.predict(testing_features)

# Ridge Regression
ridge_regr = linear_model.RidgeCV(alphas = [0.1, 1.0, 10.0, 100.0])
ridge_regr.fit(training_features, training_scores)
ridge_prediction_scores = ridge_regr.predict(testing_features)

# Evaluate the models
print('-----------PERFORMANCE OF ORDINARY REGRESSION----------')
print('Root mean Squared error: {:.2}'.format(math.sqrt(mean_squared_error(testing_scores, ord_prediction_scores))))
print('r^2 Variance score: {:.2}'.format(r2_score(testing_scores, ord_prediction_scores)))

print('------------PERFORMANCE  OF RIDGE REGRESSION-----------')
print('Root mean Squared error: {:.2}'.format(math.sqrt(mean_squared_error(testing_scores, ridge_prediction_scores))))
print('r^2 Variance score: {:.2}'.format(r2_score(testing_scores, ridge_prediction_scores)))

print('-------------------------------------------')
print('------------CLASSIFICATION TASK------------')
print('-------------------------------------------')

print('F1 scores given in alphabetic order: draw loss win')

# Perceptron model
perceptron_parameters = {'max_iter': [5, 10, 20], 'penalty': ['elasticnet', None], 'alpha': [1e-3, 1e-4, 1e-5]}
perceptron = GridSearchCV(linear_model.Perceptron(), perceptron_parameters, cv=5, scoring='f1_weighted')


# Train the model
perceptron.fit(training_features, training_results)


# Make some predictions
perceptron_prediction_results = perceptron.predict(testing_features)

# Evaluate the model
print('-----------PERCEPTRON----------')
print('Tuned perceptron parameters: {}'.format(perceptron.best_params_))
print('Mean accuracy of predictions: {:.2f}'.format(perceptron.score(testing_features, testing_results)))
print('f1 score of perceptron: {}'.format(f1_score(testing_results, perceptron_prediction_results, average=None)))

# SVM model
svm_parameters = [{'C': [1, 10, 100, 1000], 'gamma': [1e-3, 1e-4, 1e-5], 'kernel': ['rbf'], 'class_weight':['balanced', None]},
    {'C': [1, 10, 100, 1000], 'kernel': ['linear'], 'class_weight': ['balanced', None]}]

svm_clf = GridSearchCV(svm.SVC(), svm_parameters, cv=10, scoring='f1_weighted')
svm_clf.fit(training_features, training_results)


svm_prediction = svm_clf.predict(testing_features)

# Evaluate the model
print('--------------SVM--------------')
print('Tuned SVM parameters: {}'.format(svm_clf.best_params_))
print('Mean accuracy of predictions: {:.2f}'.format(svm_clf.score(testing_features, testing_results)))
print('f1 score of svm: {}'.format(f1_score(testing_results, svm_prediction, average=None)))

# Decision Tree Model
tree_parameters = {'criterion': ['entropy', 'gini'], 'max_depth': [None, 3, 8, 12], 'min_samples_leaf': list(range(1,9))}
tree_clf = RandomizedSearchCV(tree.DecisionTreeClassifier(), tree_parameters, cv=10, scoring='f1_weighted')
tree_clf.fit(training_features, training_results)


tree_prediction = tree_clf.predict(testing_features)

print('--------DECISION TREES---------')
print('Tuned decision tree parameters: {}'.format(tree_clf.best_params_))
print('Mean accuracy of predictions: {:.2f}'.format(tree_clf.score(testing_features, testing_results)))
print('f1 score of decision trees: {}'.format(f1_score(testing_results, tree_prediction, average=None)))

# K-Nearest Model
knear_params = {'n_neighbors': [3,5,10, 15], 'weights': ['uniform', 'distance']}
knear_clf = GridSearchCV(neighbors.KNeighborsClassifier(), knear_params, cv=10, scoring='f1_weighted')
knear_clf.fit(training_features, training_results)

knear_prediction = knear_clf.predict(testing_features)

print('-----K-NEAREST NEIGHBOURS------')
print('Tuned Nearest Neighbours parameters: {}'.format(knear_clf.best_params_))
print('Mean accuracy of predictions: {:.2f}'.format(knear_clf.score(testing_features, testing_results)))
print('f1 score of nearest neighbours: {}'.format(f1_score(testing_results, knear_prediction, average=None)))

# Naive Bayes Model
bayes_clf = naive_bayes.GaussianNB()
bayes_clf.fit(training_features, training_results)

bayes_prediction = bayes_clf.predict(testing_features)

print('-------NAIVE BAYES--------')
print('Mean accuracy of predictions: {:.2f}'.format(bayes_clf.score(testing_features, testing_results)))
print('f1 score of naive bayes: {}'.format(f1_score(testing_results, bayes_prediction, average=None)))

if TEST_FILE_PATH is not None:
    print('Running predictions against supplied test data')

    test_data_df = pd.read_csv(filepath_or_buffer=TEST_FILE_PATH, index_col=0)
    features = test_data_df.loc[:, :'Normal_Time'].copy()
    processed_test_data = preprocess_features(features)

    ord_prediction = ord_regr.predict(processed_test_data)
    ridge_prediction = ridge_regr.predict(processed_test_data)

    with open('ord_regression.txt', 'w') as ord_out:
        for pred in ord_prediction:
            ord_out.write('{}\n'.format(pred))

    with open('ridge_regression.txt', 'w') as ridge_out:
        for pred in ridge_prediction:
            ridge_out.write('{}\n'.format(pred))

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

