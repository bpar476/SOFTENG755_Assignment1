import numpy
import pandas as pd
import math

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.pipeline import Pipeline

ROOT_DIR='../..'

def preprocess_features(features):
    # Preprocess features
    pipeline = Pipeline([
            ('imputer', Imputer(strategy='median')),
            ('std_scaler', StandardScaler())
        ])
    processed_features = pd.DataFrame(pipeline.fit_transform(features))

    return processed_features

tf_df = pd.read_csv(filepath_or_buffer=ROOT_DIR + '/Traffic_flow/traffic_flow_data.csv')

# Extract features
last_feature = 'Segment_45(t)'
target = ['Segment23_(t+1)']

features = tf_df.loc[:, :last_feature]
targets = tf_df.loc[:, target]

processed_features = preprocess_features(features)

# Partition data into training and test data
num_rows = processed_features.shape[0]
training_threshold = math.ceil(num_rows/10)

training_features = processed_features.iloc[:num_rows - training_threshold]
testing_features = processed_features.iloc[num_rows - training_threshold:]


training_targets = targets.iloc[:num_rows - training_threshold]
testing_targets = targets.iloc[num_rows - training_threshold:]

# Ordinary regression
ord_regr = linear_model.LinearRegression()
ord_regr.fit(training_features, training_targets)
ord_prediction_targets = ord_regr.predict(testing_features)

# Ridge regression
ridge_regr = linear_model.RidgeCV(alphas = [0.1, 1.0, 10.0, 100.0])
ridge_regr.fit(training_features, training_targets)
ridge_prediction_targets = ridge_regr.predict(testing_features)

# Evaluate the model
print('------------ORDINARY REGRESSION RESULTS------------')
print('Root mean Squared error: {:.2}'.format(math.sqrt(mean_squared_error(testing_targets, ord_prediction_targets))))
print('r^2 Variance score: {:.2}'.format(r2_score(testing_targets, ord_prediction_targets)))


print('-------------RIDGE  REGRESSION RESULTS------------')
print('Root mean Squared error: {:.2}'.format(math.sqrt(mean_squared_error(testing_targets, ridge_prediction_targets))))
print('r^2 Variance score: {:.2}'.format(r2_score(testing_targets, ridge_prediction_targets)))
