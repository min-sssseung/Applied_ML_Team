# I. Logistic Regression Implementation
# 1) Download ’car_data.csv’ from iCampus into your current working directory.
# 2) Import libraries in Lab5.py.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 3) Import dataset from car_data.csv as data preparation.
dataset = pd.read_csv('./lab/car_data.csv')

# 4) The followings are the code for checking the information of the dataset. Add each of statement and run it.
print(dataset.head())
print(dataset.describe())
print(dataset.info())
print(dataset.isnull().sum())
# What do you get? Copy the result below the code. Comment out the result and all print statements.
# dataset's first 5 rows, 
# description(count, mean, std...) of dataset's columns, 
# information(count, the number of null values) of dataset's columns
# counts of how many null values each column has

#    User ID Gender  Age  AnnualSalary  Purchased
# 0      385   Male   35         20000          0
# 1      681   Male   40         43500          0
# 2      353   Male   49         74000          0
# 3      895   Male   40        107500          1
# 4      661   Male   25         79000          0
#            User ID          Age   AnnualSalary    Purchased
# count  1000.000000  1000.000000    1000.000000  1000.000000
# mean    500.500000    40.106000   72689.000000     0.402000
# std     288.819436    10.707073   34488.341867     0.490547
# min       1.000000    18.000000   15000.000000     0.000000
# 25%     250.750000    32.000000   46375.000000     0.000000
# 50%     500.500000    40.000000   72000.000000     0.000000
# 75%     750.250000    48.000000   90000.000000     1.000000
# max    1000.000000    63.000000  152500.000000     1.000000
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 1000 entries, 0 to 999
# Data columns (total 5 columns):
#  #   Column        Non-Null Count  Dtype
# ---  ------        --------------  -----
#  0   User ID       1000 non-null   int64
#  1   Gender        1000 non-null   object
#  2   Age           1000 non-null   int64
#  3   AnnualSalary  1000 non-null   int64
#  4   Purchased     1000 non-null   int64
# dtypes: int64(4), object(1)
# memory usage: 39.2+ KB
# None
# User ID         0
# Gender          0
# Age             0
# AnnualSalary    0
# Purchased       0
# dtype: int64

# 5) Let’s extract feature variables (independent variables(x) and dependent variable(y)).
x = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

print(x.shape , y.shape)
print(x[:5])
print(y[:5])
# What do you get? Copy the result below the code. Comment out the result and all print statements (line 12 to line 14).
# dimension of shape in x and y
# first 5 rows of x and y

# (1000, 2) (1000,)
# [[    35  20000]
#  [    40  43500]
#  [    49  74000]
#  [    40 107500]
#  [    25  79000]]
# [0 0 0 1 0]

# 6) Split the dataset into training and test set.
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test= train_test_split(x, y, test_size=0.25, random_state=0)
print(x_train)
print(x_test)
# What do you get? Comment out your answer and all print statements (line 18 to line 19).
# I got x_train dataset and x_test dataset which are divded from x dataset, x_train having 75% of x data and x_test having 25% of x data

# [[    20  20500]
#  [    50  25500]
#  [    51 136500]
#  ...
#  [    61  84500]
#  [    30  62000]
#  [    19  45500]]

# [[    54 136500]
#  [    41  67500]
#  [    50  29500]
#  [    20  86000]
#  [    27  31500]
#  [    36  48500]
#  [    40  97500]
#  [    51  45500]
#  [    55  71500]
#  [    28  89500]
#  [    47  80500]
#  [    19  21000]
#  [    35  44500]
#  [    27  31000]
#  [    46  33500]
#  [    49  36000]
#  [    33  60000]
#  [    52  67500]
#  [    42 108000]
#  [    34  32500]
#  [    32  67500]
#  [    38  58500]
#  [    27  82500]
#  [    51  98500]
#  [    55 140500]
#  [    41  72500]
#  [    37  75000]
#  [    39  42000]
#  [    38 145500]
#  [    40  43500]
#  [    40  80500]
#  [    41  73500]
#  [    29  86500]
#  [    27  88000]
#  [    25  80000]
#  [    37  52000]
#  [    26  80500]
#  [    34  72000]
#  [    38  71000]
#  [    58 127500]
#  [    36  76500]
#  [    47  23500]
#  [    54  35500]
#  [    38  61000]
#  [    51  37500]
#  [    42  77500]
#  [    30  15000]
#  [    40  75000]
#  [    31  18500]
#  [    18  82000]
#  [    57 134500]
#  [    34  44500]
#  [    42  65000]
#  [    38  65000]
#  [    41  53500]
#  [    48  96000]
#  [    34 150500]
#  [    42  53000]
#  [    31  71000]
#  [    20  36000]
#  [    47  97500]
#  [    48  52500]
#  [    57  72500]
#  [    38  54500]
#  [    26  80000]
#  [    39  62500]
#  [    42 136500]
#  [    35  73500]
#  [    27  36500]
#  [    41  48500]
#  [    41  58500]
#  [    49 141000]
#  [    25  59500]
#  [    28  59500]
#  [    38  75500]
#  [    56  84500]
#  [    60  42000]
#  [    49  97500]
#  [    53  72000]
#  [    22  73500]
#  [    40  60000]
#  [    37 127500]
#  [    54 104000]
#  [    39  60500]
#  [    39  62500]
#  [    35  60000]
#  [    62  90500]
#  [    46  22000]
#  [    38  94500]
#  [    47  49000]
#  [    43  77500]
#  [    41 108500]
#  [    59 102500]
#  [    39  75000]
#  [    46 114500]
#  [    53  39500]
#  [    51  92500]
#  [    18  44000]
#  [    24  21500]
#  [    42 108500]
#  [    44  62500]
#  [    33 118500]
#  [    35  22000]
#  [    35  26500]
#  [    47  51000]
#  [    40  82500]
#  [    37  76500]
#  [    47  20000]
#  [    38  34500]
#  [    27  16500]
#  [    26  17000]
#  [    45  55500]
#  [    19  69500]
#  [    20  26500]
#  [    21  83500]
#  [    55  92500]
#  [    20  22500]
#  [    19  70000]
#  [    40 123500]
#  [    36  63000]
#  [    27  85500]
#  [    37  34500]
#  [    40  72500]
#  [    42  54000]
#  [    50 109500]
#  [    39 127500]
#  [    50  44000]
#  [    58  75500]
#  [    56 145500]
#  [    31  81500]
#  [    43  74500]
#  [    28  85000]
#  [    27  90000]
#  [    60 131500]
#  [    28  97500]
#  [    27  44500]
#  [    27 137000]
#  [    30  84500]
#  [    47  83500]
#  [    23  28500]
#  [    50  88000]
#  [    37  57000]
#  [    56  40500]
#  [    58  38000]
#  [    42  81500]
#  [    63  36500]
#  [    38  61000]
#  [    23  28000]
#  [    49  43500]
#  [    44 113500]
#  [    38  80000]
#  [    45  26000]
#  [    52 147500]
#  [    38  76500]
#  [    24  32000]
#  [    25  56500]
#  [    52  90000]
#  [    29  43000]
#  [    37  33000]
#  [    30  29500]
#  [    35 108000]
#  [    48  33000]
#  [    26  88500]
#  [    43 150500]
#  [    36  61500]
#  [    40 139500]
#  [    40  43500]
#  [    49  34500]
#  [    51  89500]
#  [    36  66500]
#  [    48 138000]
#  [    52  30500]
#  [    40  74500]
#  [    45  72500]
#  [    56 126500]
#  [    27  87500]
#  [    42 104000]
#  [    39  71000]
#  [    45 106500]
#  [    22  89500]
#  [    41  79000]
#  [    44  73500]
#  [    22  63000]
#  [    60  34000]
#  [    27  84000]
#  [    29  28000]
#  [    46 117000]
#  [    46  23000]
#  [    58 144000]
#  [    35  88000]
#  [    59 130000]
#  [    27  54000]
#  [    43 112000]
#  [    47  33500]
#  [    55 109500]
#  [    35  55000]
#  [    39 128500]
#  [    23  64500]
#  [    40  80500]
#  [    39  62500]
#  [    26  84000]
#  [    40  57000]
#  [    33  19500]
#  [    32  59500]
#  [    37  53500]
#  [    45  81500]
#  [    48 141000]
#  [    59  29000]
#  [    40  72500]
#  [    35  57000]
#  [    29  83000]
#  [    36 125000]
#  [    39 114500]
#  [    57 122000]
#  [    49  31500]
#  [    24  67500]
#  [    30  89000]
#  [    37  79000]
#  [    21  37500]
#  [    54 105500]
#  [    40  95500]
#  [    21  16000]
#  [    24  83500]
#  [    21  72000]
#  [    40  65000]
#  [    43  54500]
#  [    44  74500]
#  [    26  91500]
#  [    46  79000]
#  [    50  36000]
#  [    27  58000]
#  [    20  82000]
#  [    63  44500]
#  [    50  87500]
#  [    49  88000]
#  [    50  53500]
#  [    50  45500]
#  [    19  19000]
#  [    34 115000]
#  [    38  59500]
#  [    59  24500]
#  [    28  37000]
#  [    59  83000]
#  [    50  52500]
#  [    30  48500]
#  [    19  26000]
#  [    41  52500]
#  [    49  74000]
#  [    44 130500]
#  [    35  91000]]

# 7) Add the following code for feature scaling and run it.
from sklearn.preprocessing import StandardScaler
st_x= StandardScaler()
x_train= st_x.fit_transform(x_train)
x_test= st_x.transform(x_test)

print(x_train[0:10, :])
print(x_test[0:10, :])
# What do you get? Copy the result below the code. Comment out your answer, the result and all print statements (line 25 to line 26).
# x_train and x_test are normalized with standardScaler
# all data are distributed in a standard normal distribution where mean is 0 and standard deviation is 1

# [[-1.92084369 -1.50249061]
#  [ 0.91133082 -1.35965201]
#  [ 1.00573664  1.81136479]
#  [-1.07119134  2.13989355]
#  [ 0.91133082  1.21144269]
#  [-0.3159448   1.5256876 ]
#  [ 1.85538899  0.468682  ]
#  [-0.03272735  0.26870796]
#  [-0.59916225 -0.85971693]
#  [ 0.15608428 -0.00268537]]

# [[ 1.28895409  1.81136479]
#  [ 0.06167847 -0.15980782]
#  [ 0.91133082 -1.24538114]
#  [-1.92084369  0.36869498]
#  [-1.26000297 -1.1882457 ]
#  [-0.41035062 -0.70259448]
#  [-0.03272735  0.69722375]
#  [ 1.00573664 -0.78829764]
#  [ 1.3833599  -0.04553694]
#  [-1.16559715  0.468682  ]]

# 8) Add the following code for fitting Logistic Regression to the training set and run it.
from sklearn.linear_model import LogisticRegression
lrclassifier= LogisticRegression(random_state=0)
lrclassifier.fit(x_train , y_train)

# 9) Predict the test set.
y_pred= lrclassifier.predict(x_test)
print(y_pred[0:10])
print(y_test[0:10])
# What do you get? Copy the result below the code. Comment out the result and all print statements (line and line 32).
# Through logistic regression, I get predicted values of x_test by the model fitted with x_train and y_train, y_pred values 
# [1 0 0 0 0 0 1 1 1 0]
# [0 0 1 0 0 0 1 0 1 0]

# 10) Create confusion matrix.
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test , y_pred)
print ("Confusion Matrix : \n", cm)
# What do you get? Copy the result below the code. Comment out the result.
# I get a matrix 
# where first row and second row stand for the number of y_test == 1 and the number of y_test ==0
# and first column and second column are the number of y_pred == 1 and the number of y_pred ==0
#  [[138  14]
#  [ 24  74]]

# 11) Find the accuracy score.
from sklearn.metrics import accuracy_score
print ("Accuracy : ", accuracy_score(y_test , y_pred))
# What do you get? Copy the result below the code. Comment out the result.
# accuracy is the possibility where the number of cases where y_test == y_pred over the number of cases
# Accuracy :  0.848

# 12) Find the precision and recall.
from sklearn.metrics import precision_score , recall_score
print("Precision : ", precision_score(y_test , y_pred))
print("Recall : ", recall_score(y_test , y_pred))
# What do you get? Copy the result below the code. Comment out the result.
# precision is the possibility where the number of cases where y_test == y_pred == 1 over the number of cases where y_pred== 1
# recall is the possibility where the number of cases where y_test == y_pred == 1 over the number of cases where y_test== 1
# Precision :  0.8409090909090909
# Recall :  0.7551020408163265

# 13) Find the F1 score.
from sklearn.metrics import f1_score
print("F1 Score : ", f1_score(y_test , y_pred))
# What do you get? Copy the result below the code. Comment out the result.
# f1_score is the value of (2 * precision * recall) divided by (precision + recall)
# F1 Score :  0.7956989247311828

# 14) Let’s make a function evaluatingMetrics to include the above performance evaluation code (from 10) to 13
def evaluationMetrics(y_test , y_pred):
    # Creating the Confusion matrix
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score , recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import classification_report
    
    cm = confusion_matrix(y_test , y_pred)
    print ("Confusion Matrix : \n", cm)
    print ("Accuracy : ", accuracy_score(y_test , y_pred))

    # Finding precision and recall
    print("Precision : ", precision_score(y_test , y_pred))
    print("Recall : ", recall_score(y_test , y_pred))

    # Finding F1 score (harmonic mean of precision and recall)
    print("F1 Score : ", f1_score(y_test , y_pred))

    # Report Summary
    report = classification_report(y_test , y_pred)
    print(report)

# 15) Let’s make a function visualizingPerformance to visualize the performance of the model.
def visualizingPerformance(x_set , y_set , classifier , title , color1 ,color2):
    from matplotlib.colors import ListedColormap
    x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, 
                                   stop = x_set[:, 0].max() + 1, step=0.01),
                        np.arange(start = x_set[:, 1].min() - 1, 
                                    stop = x_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
                 alpha = 0.75, 
                 cmap = ListedColormap([color1 , color2]))
    plt.xlim(x1.min(), x1.max())
    plt.ylim(x2.min(), x2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
        c = ListedColormap([color1 , color2])(i), label = j)
    plt.title(title)
    plt.xlabel('Age')
    plt.ylabel('Annual Salary')
    plt.legend()
    plt.show()

# 16) Check the performance of the training set result. Run the following code.
visualizingPerformance(x_train , y_train , lrclassifier , 'Logistic Regression (Training set)', 'purple','green')
# What do you observe? Describe what you understand from the graph. Comment out your answer and the code of line 80.

# It shows that a scatter graph of x_train values and each x value has a different color according to predicted y_values (either 0 or 1) 
# The graph has also the classification line dividing x_values according to y_values 
# Here, lrclassifier are already trained throug x_train and y_train.

# 17) Check the performance of the test set result. Run the following code.
visualizingPerformance(x_test , y_test , lrclassifier , 'Logistic Regression (Test set)', 'red','green')
# What do you observe? Describe what you understand from the graph. Comment out your answer and the code of line 81.

# It shows that a scatter graph of x_test values and each x value has a different color according to predicted y_test_values (either 0 or 1) 
# The graph has also the classification line dividing x_test_values according to predicted y__test 
# Here, lrclassifier is already trained throug x_train and y_train.

# II. Support Vector Machine (SVM)
# 1) We will use the same dataset for SVM Classifier, so you need to keep using many parts of the Part I code.
# 2) Add the following code for fitting SVM to the training set and run it.
from sklearn.svm import SVC # "Support vector classifier"
svmclassifier = SVC(kernel='linear', random_state=0)
svmclassifier.fit(x_train , y_train)

# 3) Predict the test set.
y_pred= svmclassifier.predict(x_test)
print(y_pred[0:10])
print(y_test[0:10])
# What do you get? Copy the result below the code. Comment out the result and all print statements (line 89 and line 90).
# y_pred is the predicted y values using x_test

# I get the arrays of y_pred and y_test
# [1 0 0 0 0 0 0 1 1 0]
# [0 0 1 0 0 0 1 0 1 0]

# 4) Call the evaluatingMetrics function.
evaluationMetrics(y_test , y_pred)
# What do you get? Copy the result below the code. Comment out the result.

# I get accuracy, precision, recall, f1_score values 
# I also get the classification_report between y_test and y_pred which was made from svmclassifier
# Accuracy :  0.84
# Precision :  0.8372093023255814
# Recall :  0.7346938775510204
# F1 Score :  0.782608695652174
#               precision    recall  f1-score   support

#            0       0.84      0.91      0.87       152
#            1       0.84      0.73      0.78        98

#     accuracy                           0.84       250
#    macro avg       0.84      0.82      0.83       250
# weighted avg       0.84      0.84      0.84       250

# 5) Check the performance of the training set result. Run the following code.
visualizingPerformance(x_train , y_train , svmclassifier , 'SVM (Training set)', 'purple','green')
# What do you observe? Describe what you understand from the graph. Comment out your answer and the code of line 90.

# It shows that a scatter graph of x_train values and each x value has a different color according to predicted y_values (either 0 or 1) 
# The graph has also the classifying line - hyperplane- dividing x_values according to y_values. 
# Here, svmclassifier is already trained throug x_train and y_train.

# 6) Check the performance of the test set result. Run the following code.
visualizingPerformance(x_test , y_test , svmclassifier , 'SVM (Test set)', 'red','green')
# What do you observe? Describe what you understand from the graph. Comment out your answer and the code of line 91.

# It shows that a scatter graph of x_test values and each x value has a different color according to predicted y_values (either 0 or 1) 
# The graph has also the classification line dividing x_values according to y_values 
# Here, svmclassifier is already trained throug x_train and y_train.

# III. K-Nearest Neighbors (KNN)
# 1) We will use the same dataset for KNN Classifier, so you need to keep using many parts of the Part I code.
# 2) Add the following code for fitting KNN to the training set and run it.
from sklearn.neighbors import KNeighborsClassifier
knnclassifier= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )
knnclassifier.fit(x_train , y_train)

# 3) Predict the test set.
y_pred= knnclassifier.predict(x_test)
print(y_pred[0:10])
print(y_test[0:10])
# # What do you get? Copy the result below the code. Comment out the result and all print statements (line 99 and line 100).
# y_pred is the predicted y values using x_test

# I get arrays of y_pred and y_test
# [1 0 1 0 0 0 1 1 1 0]
# [0 0 1 0 0 0 1 0 1 0]

# # 4) Call the evaluatingMetrics function.
evaluationMetrics(y_test , y_pred)
# What do you get? Copy the result below the code. Comment out the result.
# I get accuracy, precision, recall, f1_score values 
# I also get the classification_report between y_test and y_pred which was made from knnclassifier
# Confusion Matrix :
#  [[142  10]
#  [ 10  88]]
# Accuracy :  0.92
# Precision :  0.8979591836734694
# Recall :  0.8979591836734694
# F1 Score :  0.8979591836734694
#               precision    recall  f1-score   support

#            0       0.93      0.93      0.93       152
#            1       0.90      0.90      0.90        98

#     accuracy                           0.92       250
#    macro avg       0.92      0.92      0.92       250
# weighted avg       0.92      0.92      0.92       250


# # 5) Check the performance of the training set result. Run the following code.
visualizingPerformance(x_train , y_train , knnclassifier , 'KNN (Training set)', 'purple','green')
# What do you observe? Describe what you understand from the graph. Comment out your answer and the code of line 102.
# It shows that a scatter graph of x_train values and each x value has a different color according to predicted y_values (either 0 or 1) 
# The graph has also the classification line dividing x_values according to y_values 
# Here, knnclassifier is already trained throug x_train and y_train.

# # 6) Check the performance of the test set result. Run the following code.
visualizingPerformance(x_test , y_test , knnclassifier , 'KNN (Test set)','red','green')
# What do you observe? Describe what you understand from the graph. Comment out your answer and the code of line 103.
# It shows that a scatter graph of x_train values and each x value has a different color according to predicted y_values (either 0 or 1) 
# The graph has also the classification line dividing x_values according to y_values 
# Here, knnclassifier is already trained throug x_train and y_train.

# IV. Discuss Classification Algorithms
# We try 3 representative classification algorithms today. 
# At the bottom of Lab5.py, compare the performance of three algorithms (Logistic Regression, SVM and KNN), 
# and discuss the results in your own words.

# According to evaluation values about 3 algorithms, KNN classifier shows the best performance among those
# KNN classifier has a higher accuracy with about 0.92 compared to logistic regression(0.85) and SVM classifier(0.84)
# Given f1_score which consider both precision and recall, KNN classifier is a highest-performing classifier

# 1) evaluations of Logistic Regression
# Accuracy :  0.848
# Precision :  0.8409090909090909
# Recall :  0.7551020408163265
# F1 Score :  0.7956989247311828

# 2) evaluations of SVM classifier
# Accuracy :  0.84
# Precision :  0.8372093023255814
# Recall :  0.7346938775510204
# F1 Score :  0.782608695652174

# 3) evaluations of SVM classifier
# Accuracy :  0.92
# Precision :  0.8979591836734694
# Recall :  0.8979591836734694
# F1 Score :  0.8979591836734694