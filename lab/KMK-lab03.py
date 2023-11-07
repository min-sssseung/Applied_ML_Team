# I. Plotting using Seaborn Library
# (B) Write the following code for this part without commenting it out, as it will still be needed.

# importing packages
import matplotlib.pyplot as plt
import seaborn as sns

# loading dataset
iris = sns.load_dataset("iris")

# (C) Run the following code to explore the iris data.
# print(type(iris))
# print(iris.describe)
# print(iris.columns)
# <class 'pandas.core.frame.DataFrame'>
# <bound method NDFrame.describe of      sepal_length  sepal_width  petal_length  petal_width    species
# 0             5.1          3.5           1.4          0.2     setosa
# 1             4.9          3.0           1.4          0.2     setosa
# 2             4.7          3.2           1.3          0.2     setosa
# 3             4.6          3.1           1.5          0.2     setosa
# 4             5.0          3.6           1.4          0.2     setosa
# ..            ...          ...           ...          ...        ...
# 145           6.7          3.0           5.2          2.3  virginica
# 146           6.3          2.5           5.0          1.9  virginica
# 147           6.5          3.0           5.2          2.0  virginica
# 148           6.2          3.4           5.4          2.3  virginica
# 149           5.9          3.0           5.1          1.8  virginica

# [150 rows x 5 columns]>
# Index(['sepal_length', 'sepal_width', 'petal_length', 'petal_width',
#        'species'],
#       dtype='object')

# (D) Simple plotting using the iris data.
# draw lineplot
# sns.lineplot(x="sepal_length", y="sepal_width", data=iris)
# _ = plt.show()

# (E) More Plotting with Iris dataset
# (a) Lineplot with a new theme and x limit

# changing the theme to dark grid
sns.set_style("darkgrid")

# draw lineplot
sns.lineplot(x="sepal_length", y="sepal_width", data=iris)

# setting the x limit of the plot
plt.xlim(5)
_ = plt.show()

# (b) Rug plot
import seaborn as sns
iris = sns.load_dataset("iris")
x = iris.petal_length.values
sns.rugplot(x)
plt.title("Petal Length of Iris - Rug Plot")
plt.grid(True)
_ = plt.show()

# (c) KDE (Kernel Density Plot) plot
import seaborn as sns
iris = sns.load_dataset("iris")
x = iris.petal_length.values
sns.kdeplot(x)
plt.title("Petal Length of Iris - Kernel Density Plot")
plt.grid(True)
_ = plt.show()

# (d) Dist Plot
import seaborn as sns
iris = sns.load_dataset("iris")
x = iris.petal_length.values
sns.displot(x = "species", data = iris)
plt.title("Iris Species - Dist Plot")
plt.grid(True)
_ = plt.show()

# (e) Count Plot
sns.countplot(x = "species", data = iris) #  sns.countplot(iris["species"])
plt.title("Iris Species - Count Plot")
_ = plt.show()

# (F) Iris Joint Plot
sns.jointplot(x="sepal_length", y="sepal_width", data=iris)
plt.suptitle("Sepal Length and Width -- Joint Plot", y=1.02)
_ = plt.show()

# (G) Iris Joint Plot & KDE Plot
sns.jointplot(x="sepal_length", y="sepal_width", kind="kde", data=iris)
plt.suptitle("Sepal Length and Width -- Joint Plot & KDE Plot", y=1.02)
_ = plt.show()

# (H) Iris Pair Plot
sns.pairplot(iris)
plt.title("Iris Pair Plot")
_ = plt.show()

# (I) Iris Pair Plot with Categorical values
sns.pairplot(iris, hue="species")
plt.title("Iris Pair Plot with Hue")
_ = plt.show()

# (J) Titanic Dataset
# (a) load Titanic Data

# load TItanic data
titanic = sns.load_dataset("titanic")
print(titanic.describe)
print(titanic.columns)

# (b) Count Plot
sns.countplot(x = "class", data = titanic)
plt.title("Titanic Class - Count Plot")
_ = plt.show()


# II. Practice Statistics for Dataset
# 1) Write the following code.
import numpy as np
data = [99,86,87,88,111,86,103,87,94,78,77,85,86]

# 2) Add the following code and run it.
x = np.mean(data)
print(x)
# What mean (average) value do you get? Copy the result below the code. Comment out what you did.
# 89.76923076923077
# I got the average value from data vector

# 3) Add the following code and run it.
y = np.median(data)
print(y)
# What median value do you get? Copy the result below the code. Comment out what you did.
# 87.0 this is median value which is the middle value from data vector in order. 

# 4) Add the following code and run it.
from scipy import stats
z= stats.mode(data)
print(z)
# What mode value do you get? Copy the result below the code. Comment out what you did.
# ModeResult(mode=array([86]), count=array([3]))
# This is the most frequently occurrd value counted as 3

# 5) Add the following code and run it.
std = np.std(data)
print(std)
# What standard deviation value do you get? Copy the result below the code. Comment out what you did.
# 9.258292301032677
# this is square root of variance

# 6) Add the following code and run it.
v = np.var(data)
# print(v, sqrt(v))
# What result do you get? Copy the result below the code in the comment. In order to fix this error, which library module should be imported? 
# Exception has occurred: NameError
# name 'sqrt' is not defined
#   File "C:\Users\kmk45\OneDrive\바탕 화면\Applied ML\lab\KMK-lab03.py", line 153, in <module>
#     print(v, sqrt(v))
# NameError: name 'sqrt' is not defined

# numpy library
# Fix the problem and run again. What is the variance value? Comment out what you did.
print(v, np.sqrt(v)) # fixed
# 9.258292301032677

# III. Create Random Dataset
# 1) Make sure that all of part II is commented out. Copy the following code and run it.
import numpy as np
import matplotlib.pyplot as plt

x = np.random.uniform(0.0, 5.0, 1000)
print(x)
# What data type do you get? Do not copy the result. How many items does x have?
# float data type & 1000 items 

# Add the following code and run it.
plt.hist(x, 5)
plt.show()

# Comment the code from line 4 to line 7 out
# Copy the following code and run it.
x = np.random.normal(5.0, 1.0, 10000)
print(x)

# What data type do you get? Do not copy the result. How many items does x have?
# float data type & 1000 items 

# # Add the following code and run it.
plt.hist(x, 100)
plt.show()

# 6) Briefly describe the difference of graphs from 2) and 5) in the comment.
# 2) gragh follows a uniform distribution where the minimum is 0.0 and the maximum is 5.0
# 5 ) gragh follows a normal distribution where the mean is 5.0 and the standard deviation is 1.0

# IV. Simple Linear Regression
# 1) Make sure that all of part III is commented out. Copy the following code and run it.
import matplotlib.pyplot as plt
from scipy import stats

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

# Calculate a linear least-squares regression
result = stats.linregress(x, y)
print(result)

# What result do you get? Copy the result below the code in the comment. Comment out what you did from line 8.
# LinregressResult(slope=-1.7512877115526118, intercept=103.10596026490066, rvalue=-0.758591524376155, pvalue=0.002646873922456106, 
# stderr=0.453536157607742, intercept_stderr=3.9034928101545123)
# I got the result from regression model where independent variant is x and dependent variant is y

# 2) Copy the following code and run it.
slope, intercept, r, p, std_err = stats.linregress(x, y)
print("relationship", r)
# What r value do you get? Copy the result below the code in the comment. r is for relationship, 0 means no relationship, and 1 (and -1) means 100% related. 
# Comment out what you did.
# relationship -0.758591524376155 which means x and y have a negative and strong relatinship 

# 3) Add the following code and run it.
def myfunc(x):
    return slope * x + intercept

pred = myfunc(10)
print("prediction of", 10, pred)
# What result do you get? Copy the result below the code in the comment.
# prediction of 10 85.59308314937454

# 4) Add the following code and run it.
mymodel = list(map(myfunc, x))
plt.plot(x, y, 'o', label='original data')
plt.plot(x, mymodel, 'r', label='fitted line')
plt.plot(10, pred, marker='*', markersize=20, markerfacecolor="green", label='prediction')
plt.legend()
plt.show()
