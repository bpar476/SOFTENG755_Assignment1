import numpy
import pandas as pd
import math

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.pipeline import Pipeline

ROOT_DIR='../..'

tf_df = pd.read_csv(filepath_or_buffer=ROOT_DIR + '/Traffic_flow/traffic_flow_data.csv')

# Extract features
last_feature = 'Segment_45(t)'
target = ['Segment23_(t+1)']

features = tf_df.loc[:, :last_feature]
targets = tf_df.loc[:, target]

# Preprocess features
pipeline = Pipeline([
        ('imputer', Imputer(strategy='median')),
        ('std_scaler', StandardScaler())
    ])
processed_features = pd.DataFrame(pipeline.fit_transform(features))

# Partition data into training and test data
num_rows = processed_features.shape[0]
training_threshold = math.floor(num_rows/10)

training_features = processed_features.iloc[:num_rows - training_threshold]
testing_features = processed_features.iloc[num_rows - training_threshold:]


training_targets = targets.iloc[:num_rows - training_threshold]
testing_targets = targets.iloc[num_rows - training_threshold:]

print(training_features.shape)
print(training_targets.shape)

regr = linear_model.LinearRegression()

# Train the model
regr.fit(training_features, training_targets)

# Make some predictions
prediction_targets = regr.predict(testing_features)

# Evaluate the model
print('Coefficients: ', regr.coef_)
print('Mean Squared error: {:.2}'.format(mean_squared_error(testing_targets, prediction_targets)))
print('Variance score: {:.2}'.format(r2_score(testing_targets, prediction_targets)))
