import numpy
import pandas as pd
import math
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

wc_df = pd.read_csv(filepath_or_buffer='../World_Cup_2018/2018_worldcup.csv')

# Partition data into training and test data
num_rows = wc_df.shape[0]
training_threshold = math.floor(num_rows/10)

training_set = wc_df.iloc[:num_rows - training_threshold]
testing_set = wc_df.iloc[num_rows - training_threshold:]

# Extract features
features = ['Team1_Attempts','Team1_Corners','Team1_Offsides','Team1_Ball_Possession(%)','Team1_Pass_Accuracy(%)','Team1_Distance_Covered','Team1_Ball_Recovered','Team1_Yellow_Card','Team1_Red_Card','Team1_Fouls','Team2_Attempts','Team2_Corners','Team2_Offsides','Team2_Ball_Possession(%)','Team2_Pass_Accuracy(%)','Team2_Distance_Covered','Team2_Ball_Recovered','Team2_Yellow_Card','Team2_Red_Card','Team2_Fouls']
target = ['Total_Scores']

training_features = training_set.loc[:, features]
training_targets = training_set.loc[:, target]

testing_features = testing_set.loc[:, features]
testing_targets = testing_set.loc[:, target]

regr = linear_model.RidgeCV(alphas = [0.1, 1.0, 10.0, 100.0])

# Train the model
regr.fit(training_features, training_targets)

# Make some predictions
prediction_targets = regr.predict(testing_features)

# Evaluate the model
print('Coefficients: ', regr.coef_)
print('Mean Squared error: {:.2}'.format(mean_squared_error(testing_targets, prediction_targets)))
print('Variance score: {:.2}'.format(r2_score(testing_targets, prediction_targets)))
