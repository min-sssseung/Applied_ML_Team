# 1) Make sure that all of part IV is commented out. Copy the following code and run it.
# import libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import data 
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()

# check the data 
print(housing.keys())

# What result do you get? Copy the result below the code in the comment. Think about what kind of data in the dataset. Comment out what you did from line 10
# dict_keys(['data', 'target', 'frame', 'target_names', 'feature_names', 'DESCR'])
# I imported the housing data
# Through checking the data, it turns out 
# # housing data is a dictionary which carries datasets like 'data', 'target', 'frame', 'target_names', 'feature_names' and 'DESCR'.

# 2) Add the following code and run it
print(housing.data.shape) #(20640, 8)
print(housing.target.shape) # (20640,)

# # What result do you get? Copy the result below the code in the comment. Guess what the results mean? Comment out what you did from line 10.
# # housing.data has 20640 rows with 8 features
# # housing.target has 20640 rows with 1 feature, targer feature

# 3) Add the following code and run it.
print(housing['DESCR'])
# Read the result carefully. Comment out what you did from line 10.
# .. _california_housing_dataset:

# California Housing dataset
# --------------------------

# **Data Set Characteristics:**

#     :Number of Instances: 20640

#     :Number of Attributes: 8 numeric, predictive attributes and the target

#     :Attribute Information:
#         - MedInc        median income in block group
#         - HouseAge      median house age in block group
#         - AveRooms      average number of rooms per household
#         - AveBedrms     average number of bedrooms per household
#         - Population    block group population
#         - AveOccup      average number of household members
#         - Latitude      block group latitude
#         - Longitude     block group longitude

#     :Missing Attribute Values: None

# This dataset was obtained from the StatLib repository.
# https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html

# The target variable is the median house value for California districts,
# expressed in hundreds of thousands of dollars ($100,000).

# This dataset was derived from the 1990 U.S. census, using one row per census
# block group. A block group is the smallest geographical unit for which the U.S.
# Census Bureau publishes sample data (a block group typically has a population
# of 600 to 3,000 people).

# An household is a group of people residing within a home. Since the average
# number of rooms and bedrooms in this dataset are provided per household, these
# columns may take surpinsingly large values for block groups with few households
# and many empty houses, such as vacation resorts.

# It can be downloaded/loaded using the
# :func:`sklearn.datasets.fetch_california_housing` function.

# .. topic:: References

#     - Pace, R. Kelley and Ronald Barry, Sparse Spatial Autoregressions,
#       Statistics and Probability Letters, 33 (1997) 291-297

# 4) Add the following code and run it.
print(housing['feature_names'])
print(housing['target_names'])

# What result do you get? Copy the result below the code in the comment. You will use this features later for learning. Comment out what you did from line 10.
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'] # this shows the names of features in housing.data
# ['MedHouseVal'] # this shows the name of feature in housing.target

# 5) Let’s convert data from nd-array to data frame and adding feature names to the data. Add the following code and run it.
cal = pd.DataFrame(housing.data, columns=housing.feature_names)
print(cal.head(10).to_string())

# What result do you get? Copy the result below the code in the comment. Comment out what you did from line 12. 
#    MedInc  HouseAge  AveRooms  AveBedrms  Population  AveOccup  Latitude  Longitude
# 0  8.3252      41.0  6.984127   1.023810       322.0  2.555556     37.88    -122.23
# 1  8.3014      21.0  6.238137   0.971880      2401.0  2.109842     37.86    -122.22
# 2  7.2574      52.0  8.288136   1.073446       496.0  2.802260     37.85    -122.24
# 3  5.6431      52.0  5.817352   1.073059       558.0  2.547945     37.85    -122.25
# 4  3.8462      52.0  6.281853   1.081081       565.0  2.181467     37.85    -122.25
# 5  4.0368      52.0  4.761658   1.103627       413.0  2.139896     37.85    -122.25
# 6  3.6591      52.0  4.931907   0.951362      1094.0  2.128405     37.84    -122.25
# 7  3.1200      52.0  4.797527   1.061824      1157.0  1.788253     37.84    -122.25
# 8  2.0804      42.0  4.294118   1.117647      1206.0  2.026891     37.84    -122.26
# 9  3.6912      52.0  4.970588   0.990196      1551.0  2.172269     37.84    -122.25
# this table represents the top 10 rows of housing data with feature names

# 6) Add ’Price’ (target – ’MedHouseVal’) column to the data frame. Write the following code and run it.
cal['Price'] = housing.target
print(cal.head(10).to_string())
# What result do you get? Copy the result below the code in the comment. Comment out what you did from line 14.
#    MedInc  HouseAge  AveRooms  AveBedrms  Population  AveOccup  Latitude  Longitude  Price
# 0  8.3252      41.0  6.984127   1.023810       322.0  2.555556     37.88    -122.23  4.526
# 1  8.3014      21.0  6.238137   0.971880      2401.0  2.109842     37.86    -122.22  3.585
# 2  7.2574      52.0  8.288136   1.073446       496.0  2.802260     37.85    -122.24  3.521
# 3  5.6431      52.0  5.817352   1.073059       558.0  2.547945     37.85    -122.25  3.413
# 4  3.8462      52.0  6.281853   1.081081       565.0  2.181467     37.85    -122.25  3.422
# 5  4.0368      52.0  4.761658   1.103627       413.0  2.139896     37.85    -122.25  2.697
# 6  3.6591      52.0  4.931907   0.951362      1094.0  2.128405     37.84    -122.25  2.992
# 7  3.1200      52.0  4.797527   1.061824      1157.0  1.788253     37.84    -122.25  2.414
# 8  2.0804      42.0  4.294118   1.117647      1206.0  2.026891     37.84    -122.26  2.267
# 9  3.6912      52.0  4.970588   0.990196      1551.0  2.172269     37.84    -122.25  2.611
# this table is a comnbination with housing.data and hosuing.target which shows top 10 data instances

# 7) Let’s check the prepared data. Add the following code and run it.
print(cal.describe())
print(cal.info())
# Check the description and information of the data. Comment out what you did from line 14.
#              MedInc      HouseAge      AveRooms     AveBedrms    Population      AveOccup      Latitude     Longitude
# count  20640.000000  20640.000000  20640.000000  20640.000000  20640.000000  20640.000000  20640.000000  20640.000000
# mean       3.870671     28.639486      5.429000      1.096675   1425.476744      3.070655     35.631861   -119.569704
# std        1.899822     12.585558      2.474173      0.473911   1132.462122     10.386050      2.135952      2.003532
# min        0.499900      1.000000      0.846154      0.333333      3.000000      0.692308     32.540000   -124.350000
# 25%        2.563400     18.000000      4.440716      1.006079    787.000000      2.429741     33.930000   -121.800000
# 50%        3.534800     29.000000      5.229129      1.048780   1166.000000      2.818116     34.260000   -118.490000
# 75%        4.743250     37.000000      6.052381      1.099526   1725.000000      3.282261     37.710000   -118.010000
# max       15.000100     52.000000    141.909091     34.066667  35682.000000   1243.333333     41.950000   -114.310000

# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 20640 entries, 0 to 20639
# Data columns (total 8 columns):
#  #   Column      Non-Null Count  Dtype
# ---  ------      --------------  -----
#  0   MedInc      20640 non-null  float64
#  1   HouseAge    20640 non-null  float64
#  2   AveRooms    20640 non-null  float64
#  3   AveBedrms   20640 non-null  float64
#  4   Population  20640 non-null  float64
#  5   AveOccup    20640 non-null  float64
#  6   Latitude    20640 non-null  float64
#  7   Longitude   20640 non-null  float64
# dtypes: float64(8)
# memory usage: 1.3 MB
# None

# 8) Let’s preprocess the data. Add the following code and run it.
print(cal.isnull().sum())
# What result do you get? Copy the result below the code in the comment. Is there any missing data? Comment out what you did from line 14.
# MedInc        0
# HouseAge      0
# AveRooms      0
# AveBedrms     0
# Population    0
# AveOccup      0
# Latitude      0
# Longitude     0
# dtype: int64
# No missing data

# 9) Let’s check the distribution of the target variable ’Price’. Add the following code and run it.
import seaborn as sns
# sns.set(rc={'figure.figsize':(9.7,6.27)})
# sns.distplot(cal['Price'], bins=50)
# plt.show()

# You may need to install the seaborn package using pip. What result do you get? What kind of graph do you observe? Comment out what you did from line 16.
# it shows the cal['Price']'s distribution. The graphs are a dist graph and line graph.

# 10) Continue with seaborn library using heatmap → correlation matrix that measures the linear relationships between the variables.
fig, ax = plt.subplots(figsize=(12, 9))
correlation_matrix = cal.corr().round(2)
ax = sns.heatmap(data=correlation_matrix , annot=True, fmt='.2f')
plt.show()

# What result do you get? What kind of graph do you observe? Which feature is most related to the Price? Comment out what you did from line 16.
# It shows the matrix with correlation coefficients.
# Except of Price feature itself, Medinc features is the most related feature with 0.69 correaltion value

# 11) Let’s plot the data: Price vs. Feature variable. Try the following code.
plt.figure(figsize=(20, 5))
features = ['MedInc', 'AveRooms', 'HouseAge']
target = cal['Price']
for i, col in enumerate(features):
    plt.subplot(1, len(features) , i+1)
    x = cal[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('Price')
plt.show()

# What result do you get? What kind of graph do you observe? Which feature is most related to the Price? Comment out what you did from line 16.
# I got 3 subplots where each feature shows scatter matrix with Price feature.
# MedInc feature is most related to the Price

# 12) It is a time to start Linear Regression Learning for the California Housing Dataset. First, single feature linear regression with ’MedInc’. Try the following code from line 16. 
# Prepare train/test data
X = pd.DataFrame(np.c_[cal['MedInc']], columns = ['MedInc'])
Y = cal['Price']

from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
print(X_train.shape) 
print(X_test.shape)
print(Y_train.shape) 
print(Y_test.shape)

# Training and testing the model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score , mean_absolute_error
lin_model = LinearRegression()
lin_model.fit(X_train , Y_train)

# model evaluation for training set
from sklearn.metrics import r2_score
y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train , y_train_predict)))
r2 = r2_score(Y_train , y_train_predict)
mae = mean_absolute_error(Y_train , y_train_predict)
print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print('MAE is {}'.format(mae))

y_test_predict = lin_model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test , y_test_predict)))
r2 = r2_score(Y_test , y_test_predict)
k = len(X.columns) # number of features
n = len(X_test) # number of observations
adj_r2 = 1- ((1-r2) * (n-1)/(n-k-1))
mae = mean_absolute_error(Y_test , y_test_predict)
print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print('Adjusted R2 score is {}'.format(adj_r2))
print('MAE is {}'.format(mae))

# What result do you get? Copy the result below the code in the comment below the code.
# I get the model performance summary of linear regression with 1 independent variable 
# (16512, 1)
# (4128, 1)
# (16512,)
# (4128,)
# The model performance for training set
# --------------------------------------
# RMSE is 0.8359223196455023
# R2 score is 0.4703942846392545
# MAE is 0.6247244339495553
# The model performance for testing set
# --------------------------------------
# RMSE is 0.8430087153316839
# R2 score is 0.48490837623606453
# Adjusted R2 score is 0.48478353580374167
# MAE is 0.6295893594192818

# 13) Let’s add more features one by one in the line 17. Replace the line 17 and run again. Check the performance and compare it with previous one.
# (a) ’MedInc’, ’AveRooms’
X = pd.DataFrame(np.c_[cal['MedInc'], cal['AveRooms']], columns = ['MedInc', 'AveRooms'])
Y = cal['Price']

from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape) 
print(Y_train.shape) 
print(Y_test.shape) 

# Training and testing the model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score , mean_absolute_error
lin_model = LinearRegression()
lin_model.fit(X_train , Y_train)

# model evaluation for training set
from sklearn.metrics import r2_score
y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train , y_train_predict)))
r2 = r2_score(Y_train , y_train_predict)
mae = mean_absolute_error(Y_train , y_train_predict)
print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print('MAE is {}'.format(mae))

y_test_predict = lin_model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test , y_test_predict)))
r2 = r2_score(Y_test , y_test_predict)
k = len(X.columns) # number of features
n = len(X_test) # number of observations
adj_r2 = 1- ((1-r2) * (n-1)/(n-k-1))
mae = mean_absolute_error(Y_test , y_test_predict)
print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print('Adjusted R2 score is {}'.format(adj_r2))
print('MAE is {}'.format(mae))

# (16512, 2)
# (4128, 2)
# (16512,)
# (4128,)
# The model performance for training set
# --------------------------------------
# RMSE is 0.8317685105583229
# R2 score is 0.4756445698174918
# MAE is 0.6182564587991637
# The model performance for testing set
# --------------------------------------
# RMSE is 0.8360370094765344
# R2 score is 0.49339279190200946
# Adjusted R2 score is 0.4931471641647499
# MAE is 0.6230198436241503

# About trainging set
# RMSE and R2 score decrease slightly, but MAE increases
# About test set
# RMSE and  MAE increase, but R2 and adjusted R2 score decrease.

# (b) ’MedInc’, ’AveRooms’,’HouseAge
X = pd.DataFrame(np.c_[cal['MedInc'], cal['AveRooms'], cal['HouseAge']], columns = ['MedInc','AveRooms' ,'HouseAge'])
Y = cal['Price']

from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
print(X_train.shape) 
print(X_test.shape) 
print(Y_train.shape) 
print(Y_test.shape) 

# Training and testing the model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score , mean_absolute_error
lin_model = LinearRegression()
lin_model.fit(X_train , Y_train)

# model evaluation for training set
from sklearn.metrics import r2_score
y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train , y_train_predict)))
r2 = r2_score(Y_train , y_train_predict)
mae = mean_absolute_error(Y_train , y_train_predict)
print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print('MAE is {}'.format(mae))

y_test_predict = lin_model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test , y_test_predict)))
r2 = r2_score(Y_test , y_test_predict)
k = len(X.columns) # number of features
n = len(X_test) # number of observations
adj_r2 = 1- ((1-r2) * (n-1)/(n-k-1))
mae = mean_absolute_error(Y_test , y_test_predict)
print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print('Adjusted R2 score is {}'.format(adj_r2))
print('MAE is {}'.format(mae))

# (16512, 3)
# (4128, 3)
# (16512,)
# (4128,)
# The model performance for training set
# --------------------------------------
# RMSE is 0.8059845768343475
# R2 score is 0.5076496134492752
# MAE is 0.5990553328517139
# The model performance for testing set
# --------------------------------------
# RMSE is 0.8062990087498704
# R2 score is 0.5287920486680441
# Adjusted R2 score is 0.5284492688780353
# MAE is 0.6021406536817527

# Comparing it with the linear model with (a) ’MedInc’, ’AveRooms’, 
# there is a decrease on RMSE and MAE score, but a increase on R2 score about training set
# there is a increase on RMSE and MAE score, but a decrease on R2 and adjusted score on test set

# (c) ’MedInc’, ’AveRooms’,’HouseAge’, ’AveOccup’
X = pd.DataFrame(np.c_[cal['MedInc'], cal['AveRooms'], cal['HouseAge'], cal['AveOccup']], columns = ['MedInc','AveRooms' ,'HouseAge', 'AveOccup'])
Y = cal['Price']

from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
print(X_train.shape) 
print(X_test.shape) 
print(Y_train.shape) 
print(Y_test.shape) 

# Training and testing the model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score , mean_absolute_error
lin_model = LinearRegression()
lin_model.fit(X_train , Y_train)

# model evaluation for training set
from sklearn.metrics import r2_score
y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train , y_train_predict)))
r2 = r2_score(Y_train , y_train_predict)
mae = mean_absolute_error(Y_train , y_train_predict)
print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print('MAE is {}'.format(mae))

y_test_predict = lin_model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test , y_test_predict)))
r2 = r2_score(Y_test , y_test_predict)
k = len(X.columns) # number of features
n = len(X_test) # number of observations
adj_r2 = 1- ((1-r2) * (n-1)/(n-k-1))
mae = mean_absolute_error(Y_test , y_test_predict)
print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print('Adjusted R2 score is {}'.format(adj_r2))
print('MAE is {}'.format(mae))

# (16512, 4)
# (4128, 4)
# (16512,)
# (4128,)
# The model performance for training set
# --------------------------------------
# RMSE is 0.8042017651680179
# R2 score is 0.5098253305563252
# MAE is 0.5977507201798694
# The model performance for testing set
# --------------------------------------
# RMSE is 0.8068247868669579
# R2 score is 0.5281773099559234
# Adjusted R2 score is 0.5277195629852282
# MAE is 0.6017939317232517

# Comparing it with the linear model with # (b) ’MedInc’, ’AveRooms’,’HouseAge,
# there is a slight increase on R2 and MAE score except for R2 score which decreases on training set
# there is a slight increase on RMSE score, and a decrease on other performance evaluation. 
