# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import helperFunctions as hf
from prettytable import PrettyTable
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import svm

# SVM Functions
def Linear_Kernel_Classification(X_TrainData, y_TrainData, X_TestData):
    # Creating the SVM Classifier
    clf = svm.SVC(kernel='linear') # Linear Kernel

    # Train our model using the given training data
    clf.fit(X_TrainData, y_TrainData)

    # Calculate and return predictions on the Test Data
    return clf.predict(X_TestData)

def Poly_Kernel_Classification(X_TrainData, y_TrainData, X_TestData):
    # Creating the SVM Classifier
    clf = svm.SVC(kernel='poly') # Poly Kernel

    # Train our model using the given training data
    clf.fit(X_TrainData, y_TrainData)

    # Calculate and return predictions on the Test Data
    return clf.predict(X_TestData)

def RBF_Kernel_Classification(X_TrainData, y_TrainData, X_TestData):
    # Creating the SVM Classifier
    clf = svm.SVC(kernel='rbf') # RBF Kernel

    # Train our model using the given training data
    clf.fit(X_TrainData, y_TrainData)

    # Calculate and return predictions on the Test Data
    return clf.predict(X_TestData)

# Initials -----------------------------------------------------------------------------------------------  

# Create a HelperFunctions GameObject
helperObj = hf.HelperFunctions()

# Create PrettyTables to hold prediction statistics
linear_table = PrettyTable()
svmLinear_table = PrettyTable()
svmPoly_table = PrettyTable()
svmRBF_table = PrettyTable()
linear_table.field_names = ["Experiment #", "Accuracy"]
svmLinear_table.field_names = ["Experiment #", "Accuracy"]
svmPoly_table.field_names = ["Experiment #", "Accuracy"]
svmRBF_table.field_names = ["Experiment #", "Accuracy"]

# Preprocessing ------------------------------------------------------------------------------------------

# Read in the Dataset
gameDataCSV = pd.read_csv('FinalDatasetComplete.csv', index_col = 0)

# Grab the labels (Game Winner) 
# Need to convert the winner data to numerical data:
# 1 -> Home Team Wins | 0 -> Away Team Wins  
n = gameDataCSV.shape[0]
ground_Truth = np.zeros(n)
for index, row in gameDataCSV.iterrows():
    if helperObj.isHomeWinner(row['GAME_ID']):
        ground_Truth[index] = True

# Drop non-Feature columns
columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
gameDataCSV.drop(gameDataCSV.columns[columns], axis=1, inplace=True)

# Creating Test/Training Data & Training the Model ---------------------------------------------------------
experiment = 1       # Keep Track of Current Experiment
l_avg_accuracy = 0   # For Final Calculation of Average Accuracy at the end (Linear Regression)
lin_avg_accuracy = 0 # For Final Calculation of Average Accuracy at the end (SVM Linear Kernel)
p_avg_accuracy = 0   # For Final Calculation of Average Accuracy at the end (SVM Poly Kernel)
r_avg_accuracy = 0   # For Final Calculation of Average Accuracy at the end (SVM RBF Kernel)

# Use 10-Fold CV to split Dataset
cv = KFold(n_splits=10, random_state=225, shuffle=True)

for train_index, test_index in cv.split(gameDataCSV):
    X_train, X_test = gameDataCSV.iloc[train_index, :], gameDataCSV.iloc[test_index, :]
    y_train, y_test = ground_Truth[train_index], ground_Truth[test_index]

    # Normalize the Data
    # Must use the parameters from the training set to normalize the test set
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # -------------------------------------------------------------------------------
    # Create Logistic Regression Model
    logistic_model = LogisticRegression(random_state=1)
    logistic_model.fit(X_train, y_train)
  
    game_predictions = logistic_model.predict(X_test)
    linearReg_accuracy = accuracy_score(y_test, game_predictions)

    # Add the Accuracy Score to the table
    linear_table.add_row([experiment, helperObj.ReturnPercent(linearReg_accuracy)])
    l_avg_accuracy += linearReg_accuracy
    
    '''
    # -------------------------------------------------------------------------------
    # Linear Kernel Calculation
    linear_y_predictions = Linear_Kernel_Classification(X_train, y_train, X_test)
    linear_accuracy = accuracy_score(y_test, linear_y_predictions)
    lin_avg_accuracy += linear_accuracy

    # Add Results of Experiment to Table (rounded to 2 decimal places)
    svmLinear_table.add_row([experiment, helperObj.ReturnPercent(linear_accuracy)])

    # -------------------------------------------------------------------------------
    # Poly Kernel Calculation
    poly_y_predictions = Poly_Kernel_Classification(X_train, y_train, X_test)
    poly_accuracy = accuracy_score(y_test, poly_y_predictions)
    p_avg_accuracy += poly_accuracy

    # Add Results of Experiment to Table (rounded to 2 decimal places)
    svmPoly_table.add_row([experiment, helperObj.ReturnPercent(poly_accuracy)])
    '''

    # -------------------------------------------------------------------------------
    # RBF Kernel Calculation
    rbf_y_predictions = RBF_Kernel_Classification(X_train, y_train, X_test)
    rbf_accuracy = accuracy_score(y_test, rbf_y_predictions)
    r_avg_accuracy += rbf_accuracy

    # Add Results of Experiment to Table (rounded to 2 decimal places)
    svmRBF_table.add_row([experiment, helperObj.ReturnPercent(rbf_accuracy)])

    # -------------------------------------------------------------------------------
    # Move to next experiment
    experiment = experiment + 1
    print("Round: {}".format(experiment))
    
    
# Print Out Results
# Print Out Results w/ Linear Regression
print("----------------------------------------------------------------")
print("Linear Regression Results:")
print(linear_table)
print("Average Accuracy: {0:.0f}%".format((l_avg_accuracy / (experiment-1)) * 100))

print("----------------------------------------------------------------")
print("SVM Results:")
print("1) Linear Kernel:")
print(svmLinear_table)
print("Average Accuracy: {0:.0f}%".format((lin_avg_accuracy / (experiment-1)) * 100))

print("2) Poly Kernel:")
print(svmPoly_table)
print("Average Accuracy: {0:.0f}%".format((p_avg_accuracy / (experiment-1)) * 100))

print("2) RBF Kernel:")
print(svmRBF_table)
print("Average Accuracy: {0:.0f}%".format((r_avg_accuracy / (experiment-1)) * 100))