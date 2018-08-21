import numpy
import pandas as pd
import math
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

ROOT_DIR='../..'

wc_df = pd.read_csv(filepath_or_buffer=ROOT_DIR + '/World_Cup_2018/2018_worldcup.csv')

# Partition data into training and test data
num_rows = wc_df.shape[0]
training_threshold = math.floor(num_rows/10)

training_set = wc_df.iloc[:num_rows - training_threshold]
testing_set = wc_df.iloc[num_rows - training_threshold:]

# Extract features
features = ['Team1_Attempts','Team1_Corners','Team1_Offsides','Team1_Ball_Possession(%)','Team1_Pass_Accuracy(%)','Team1_Distance_Covered','Team1_Ball_Recovered','Team1_Yellow_Card','Team1_Red_Card','Team1_Fouls','Team2_Attempts','Team2_Corners','Team2_Offsides','Team2_Ball_Possession(%)','Team2_Pass_Accuracy(%)','Team2_Distance_Covered','Team2_Ball_Recovered','Team2_Yellow_Card','Team2_Red_Card','Team2_Fouls']
target = 'Match_result'

training_features = training_set.loc[:, features]
training_targets = training_set.loc[:, target]

testing_features = testing_set.loc[:, features]
testing_targets = testing_set.loc[:, target]

perceptron = linear_model.Perceptron()

# Train the model
perceptron.fit(training_features, training_targets)

# Make some predictions
prediction_targets = perceptron.predict(testing_features)

# Evaluate the model
print('Coefficients: ', perceptron.coef_)
print('Mean accuracy of predictions: {:.2f}'.format(perceptron.score(testing_features, testing_targets)))
