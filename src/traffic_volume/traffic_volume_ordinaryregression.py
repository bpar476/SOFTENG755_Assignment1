import numpy
import pandas as pd
import math

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

ROOT_DIR='../..'

tf_df = pd.read_csv(filepath_or_buffer=ROOT_DIR + '/Traffic_flow/traffic_flow_data.csv')

# Partition data into training and test data
num_rows = tf_df.shape[0]
training_threshold = math.floor(num_rows/10)

training_set = tf_df.iloc[:num_rows - training_threshold]
testing_set = tf_df.iloc[num_rows - training_threshold:]

# Extract features
last_feature = 'Segment_45(t)'
target = ['Segment23_(t+1)']

training_features = training_set.loc[:, :last_feature]
training_targets = training_set.loc[:, target]

testing_features = testing_set.loc[:, :last_feature]
testing_targets = testing_set.loc[:, target]

regr = linear_model.LinearRegression()

# Train the model
regr.fit(training_features, training_targets)

# Make some predictions
prediction_targets = regr.predict(testing_features)

# Evaluate the model
print('Coefficients: ', regr.coef_)
print('Mean Squared error: {:.2}'.format(mean_squared_error(testing_targets, prediction_targets)))
print('Variance score: {:.2}'.format(r2_score(testing_targets, prediction_targets)))
