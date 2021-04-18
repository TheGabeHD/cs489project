# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from prettytable import PrettyTable
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

data = pd.read_csv('TeamGames.csv')
