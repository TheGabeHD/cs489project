# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import helperFunctions as hf
from prettytable import PrettyTable
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# Initials -----------------------------------------------------------------------------------------------  

# Create a HelperFunctions GameObject
helperObj = hf.HelperFunctions()

# Create PrettyTables to hold prediction statistics
linear_table = PrettyTable()
linear_table.field_names = ["Experiment #", "Accuracy"]

# Preprocessing ------------------------------------------------------------------------------------------

# Read in the Dataset
gameDataCSV = pd.read_csv('ScoresData.csv', index_col = 0)

# Grab the labels (Game Winner) 
# Need to convert the winner data to numerical data:
# 1 -> Home Team Wins | 0 -> Away Team Wins  
n = gameDataCSV.shape[0]
ground_Truth = np.zeros(n)
for index, row in gameDataCSV.iterrows():
    if helperObj.isHomeWinner(row['GAME_ID']):
        ground_Truth[index] = True

# Drop non-Feature columns
columns = [0, 1, 2, 7]
gameDataCSV.drop(gameDataCSV.columns[columns], axis=1, inplace=True)

# Creating Test/Training Data & Training the Model ---------------------------------------------------------
experiment = 1      # Keep Track of Current Experiment
l_avg_accuracy = 0  # For Final Calculation of Average Accuracy at the end (Linear Regression)
# Use 10-Fold CV to split Dataset
cv = KFold(n_splits=5, random_state=225, shuffle=True)

for train_index, test_index in cv.split(gameDataCSV):
    X_train, X_test = gameDataCSV.iloc[train_index, :], gameDataCSV.iloc[test_index, :]
    y_train, y_test = ground_Truth[train_index], ground_Truth[test_index]

    # Create Logistic Regression Model
    logistic_model = LogisticRegression(solver='liblinear', random_state=1)
    logistic_model.fit(X_train, y_train)
    
    game_predictions = logistic_model.predict(X_test)
    print(game_predictions)
    linearReg_accuracy = accuracy_score(y_test, game_predictions)
    print(linearReg_accuracy)
    
    # Add the Accuracy Score to the table
    linear_table.add_row([experiment, helperObj.ReturnPercent(linearReg_accuracy)])
    l_avg_accuracy += linearReg_accuracy
    
    # Move to next experiment
    experiment = experiment + 1
    
# Print Out Results
# Print Out Results w/ Linear Regression
print("----------------------------------------------------------------")
print("Linear Regression Results:")
print(linear_table)
print("Average Accuracy: {0:.0f}%".format((l_avg_accuracy / (experiment-1)) * 100))