import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from sklearn import linear_model, svm, tree, neighbors, naive_bayes
from sklearn.metrics import mean_squared_error, r2_score, f1_score, log_loss
from sklearn.preprocessing import Imputer, StandardScaler, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
import category_encoders as cs

ROOT_DIR='../..'

wc_df = pd.read_csv(filepath_or_buffer=ROOT_DIR + '/World_Cup_2018/2018_worldcup.csv', index_col=0)
wc_df.drop(['Date', 'Team1_Ball_Possession(%)'],axis=1,inplace=True)

# Extract features

features = wc_df.loc[:, :'Normal_Time'].copy()
goals = wc_df.loc[:, 'Total_Scores']
results = wc_df.loc[:, 'Match_result']

# Transform the data to normalise it
# Create a selector class to select categorical or numerical columns
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
            return self
    def transform(self, X):
        return X[self.attribute_names].values

# Separate categorical columns from numerical columns
categorical_features_list = ['Location', 'Phase', 'Team1', 'Team2', 'Team1_Continent', 'Team2_Continent', 'Normal_Time']
numerical_features = features.drop(categorical_features_list, axis=1, inplace=False)
categorical_features = features[categorical_features_list].copy()

numerical_pipeline = Pipeline([
        ('selector', DataFrameSelector(list(numerical_features))),
        ('imputer', Imputer(strategy='median')),
        ('std_scaler', StandardScaler())
    ])

category_pipeline = Pipeline([
        ('selector', DataFrameSelector(list(categorical_features))),
        ('cat_encoder', cs.OneHotEncoder(drop_invariant=True))
    ])

full_pipeline = FeatureUnion(transformer_list=[
        ('num_pipeline', numerical_pipeline),
        ('cat_pipeline', category_pipeline)
    ])

prepared_features = pd.DataFrame(data=full_pipeline.fit_transform(features),index=np.arange(1,65))

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

# Perceptron model
perceptron = linear_model.Perceptron()

# Train the model
perceptron.fit(training_features, training_results)

# Make some predictions
perceptron_prediction_results = perceptron.predict(testing_features)

# Evaluate the model
print('-----------PERFORMANCE OF PERCEPTRON----------')
print('Mean accuracy of predictions: {:.2f}'.format(perceptron.score(testing_features, testing_results)))
print('f1 score of perceptron: {}'.format(f1_score(perceptron_prediction_results, testing_results, average=None)))

print('log loss of perceptron: {}'.format(log_loss(perceptron_prediction_results, testing_results)))

# SVM model
svm_clf = svm.SVC(kernel='linear')
svm_clf.fit(training_features, training_results)

svm_prediction = svm_clf.predict(testing_features)

# Evaluate the model
print('--------------PERFORMANCE OF SVM--------------')
print('Mean accuracy of predictions: {:.2f}'.format(svm_clf.score(testing_features, testing_results)))
print('f1 score of svm: {}'.format(f1_score(svm_prediction, testing_results, average=None)))
print('log loss of svm: {}'.format(log_loss(svm_prediction, testing_results)))

# Decision Tree Model
tree_clf = tree.DecisionTreeClassifier()
tree_clf.fit(training_features, training_results)

tree_prediction = tree_clf.predict(testing_features)

print('--------PERFORMANCE OF DECISION TREES---------')
print('Mean accuracy of predictions: {:.2f}'.format(tree_clf.score(testing_features, testing_results)))
print('f1 score of decision trees: {}'.format(f1_score(tree_prediction, testing_results, average=None)))
print('log loss of decision trees: {}'.format(log_loss(tree_prediction, testing_results)))

# K-Nearest Model
knear_clf = neighbors.KNeighborsClassifier(n_neighbors=3)
knear_clf.fit(training_features, training_results)

knear_prediction = knear_clf.predict(testing_features)

print('-----PERFORMANCE OF K-NEAREST NEIGHBOURS------')
print('Mean accuracy of predictions: {:.2f}'.format(knear_clf.score(testing_features, testing_results)))
print('f1 score of nearest neighbours: {}'.format(f1_score(knear_prediction, testing_results, average=None)))
print('log loss of nearest neighbours: {}'.format(log_loss(knear_prediction, testing_results)))

# Naive Bayes Model
bayes_clf = naive_bayes.GaussianNB()
bayes_clf.fit(training_features, training_results)

bayes_prediction = bayes_clf.predict(testing_features)

print('-------PERFORMANCE OF NAIVE BAYES--------')
print('Mean accuracy of predictions: {:.2f}'.format(bayes_clf.score(testing_features, testing_results)))
print('f1 score of naive bayes: {}'.format(f1_score(bayes_prediction, testing_results, average=None)))
print('log loss of naive bayes: {}'.format(log_loss(bayes_prediction, testing_results)))
