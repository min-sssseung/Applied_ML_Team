# """lab6.ipynb"""

# I. Iris Data with K-Means

# 1) import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# 2) load the iris dataset

X, y = load_iris(return_X_y=True)

# 3) find optimal number of clusters using the elbow method

wcss_list= [] #Initializing the list for the values of WCSS

# Using for loop for iterations from 1 to 10.
for i in range(1, 11):
  kmeans = KMeans(n_clusters=i, init='k-means++', random_state= 42)
  kmeans.fit(X)
  wcss_list.append(kmeans.inertia_)

# Plot the Elbow graph to find the optimum number of cluster
plt.plot(range(1, 11), wcss_list)
plt.title('The Elobw Method Graph')
plt.xlabel('Number of clusters(k)')
plt.ylabel('wcss_list')
plt.show()

# What do you observe? Describe what you understand from the graph w.r.t. the elbow-like
# situation. What is your K value? Comment out your answer.
# There is an elbow graph which shows each algorithm's inertia
# With inertia, we can evaluate and compare many algorithms
# Proper K value is 3 according to the graph

# 4) complete the following code with your K value.

k = 3 #(Your Value)

# 5) build the K-means model on a dataset

kmeans = KMeans(n_clusters = k, init='k-means++', random_state=0)
pred= kmeans.fit_predict(X)
# compare two arrays: prediction vs target
comparison = pred == y
print("pred == y?", comparison.all())
print(pred)
print(y)

# How many data points are different? Comment out your answer.
# 114
# * print(comparison.sum()) = 36

# 6) visualize the clusters

# visulaizing the clusters
plt.figure(figsize=(12,5))
plt.subplot(2,2,1)
plt.scatter(X[:,0],X[:,1],c = pred, cmap=cm.Accent)
plt.grid(True)
for center in kmeans.cluster_centers_:
  center = center[:2]
plt.scatter(center[0],center[1],marker = '^',c = 'red')
plt.xlabel("petal length (cm)")
plt.ylabel("petal width (cm)")
plt.title("Predicted")

plt.subplot(2,2,2)
plt.scatter(X[:,2],X[:,3],c = pred, cmap=cm.Accent)
plt.grid(True)
for center in kmeans.cluster_centers_:
  center = center[2:4]
plt.scatter(center[0],center[1],marker = '^',c = 'red')
plt.xlabel("sepal length (cm)")
plt.ylabel("sepal width (cm)")
plt.title("Predicted")

plt.subplot(2,2,3)
plt.scatter(X[:,0],X[:,1],c = y, cmap=cm.hsv)
plt.grid(True)
for center in kmeans.cluster_centers_:
  center = center[:2]
plt.scatter(center[0],center[1],marker = '^',c = 'blue')
plt.xlabel("petal length (cm)")
plt.ylabel("petal width (cm)")
plt.title("Original Target")

plt.subplot(2,2,4)
plt.scatter(X[:,2],X[:,3],c = y, cmap=cm.hsv)
plt.grid(True)
for center in kmeans.cluster_centers_:
  center = center[2:4]
plt.scatter(center[0],center[1],marker = '^',c = 'blue')
plt.xlabel("sepal length (cm)")
plt.ylabel("sepal width (cm)")
plt.title("Original Target")

# set the spacing between subplots
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.6)
plt.show()

# What do you get? Describe your observation in your own word. Comment out your answer.
# Left upper graph shows how petal length and petal width are scattered with predicted label
# Left lower graph shows how petal length and petal width are scattered with original label
# Right upper graph shows how sepal length and sepal width are scattered with predicted label
# Right lower graph shows how sepal length and sepal width are scattered with orginal label
# classification by sepal length and width is easier to clarify than the algorithm with other features

# II. Mall Customers Data with K-Means

# 1) import libraries
import numpy as nm
import matplotlib.pyplot as plt
import pandas as pd

# 2) import the dataset
dataset = pd.read_csv('./Mall_Customers.csv')
print(dataset.head())
print(dataset.shape)
# What do you get? Copy the result below the code. Comment out the result and all print statements.
#    CustomerID   Genre  Age  Annual Income (k$)  Spending Score (1-100)
# 0           1    Male   19                  15                      39
# 1           2    Male   21                  15                      81
# 2           3  Female   20                  16                       6
# 3           4  Female   23                  16                      77
# 4           5  Female   31                  17                      40
# (200, 5)
# first 5 rows in dataset
# dimensions of dataset

# 3) extract independent variables
X = dataset.iloc[:, [3, 4]].values
print(X[:5])
# What do you get? Copy the result below the code. Comment out the result.
# X is a sliced data from dataset
# [[15 39]
#  [15 81]
#  [16  6]
#  [16 77]
#  [17 40]]

# 4) find optimal number of clusters using the elbow method
from sklearn.cluster import KMeans

wcss_list= [] #Initializing the list for the values of WCSS

# Using for loop for iterations from 1 to 15.
for i in range(1, 16):
  kmeans = KMeans(n_clusters=i, init='k-means++', random_state= 42)
  kmeans.fit(X)
  wcss_list.append(kmeans.inertia_)
plt.plot(range(1, 16), wcss_list)
plt.title('The Elobw Method Graph')
plt.xlabel('Number of clusters(k)')
plt.ylabel('wcss_list')
plt.show()
# What do you observe? Describe what you understand from the graph. What is your choice
# of K? Comment out your answer.
# the elbow graph which shows the proper value of K
# K is 5

# 5) train the K-means model on a dataset
k = 5 # your K value
kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, random_state=340)
pred= kmeans.fit_predict(X)
dataset['k-means-cluster'] = pred
print(dataset.head())
# What do you get? Comment out your answer.
# I get a 'k-menas-cluster' feature that divides each instance's label

# 6) visualize the clusters
colors = ['blue', 'green', 'red', 'cyan', 'magenta']
# visulaizing the clusters
for i in range(k):
  plt.scatter(X[pred == i, 0], X[pred == i, 1], s = 100,
              c= colors[i],
              label = 'Cluster '+str(i+1)) # for first cluster
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300,
            c ='yellow', label = 'Centroid')
plt.title('Clusters of customers - K-means')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(loc='center left', bbox_to_anchor=(0.9, 0.5))
plt.show()
# What do you get? Comment out your answer.
# A scatter graph where x and y values are annual income and speiding score
# the data are divided into 5 clusters with their own centroid

# III. Mall Customers Data with AHC

# 1) Use the same code as 1 to 3 of part II.
# 2) find the optimal number of clusters using the dendrogram
import scipy.cluster.hierarchy as shc
dendro = shc.dendrogram(shc.linkage(X, method="ward"))
plt.title("Dendrogram Plot")
plt.ylabel("Euclidean Distances")
plt.xlabel("Customers")
plt.show()
# What do you observe? Describe what you understand from the graph. Explain what your
# choice of number clusters is. Comment out your answer.
# Dendrogram shows

# 3) train the hierarchical model on dataset
k = 5 # your choice of number of clusters
from sklearn.cluster import AgglomerativeClustering
hc= AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward')
pred= hc.fit_predict(X)
dataset['AHC-cluster'] = pred
print(dataset.head())
# What do you get? Comment out your answer
# I get a 'AHC-cluster' feature that divides each instance's label according to AgglomerativeClustering

## 4) visualize the clusters
colors = ['blue', 'green', 'red', 'cyan', 'magenta']
# visulaizing the clusters
for i in range(k):
  plt.scatter(X[pred == i, 0], X[pred == i, 1], s = 100, c = colors[i],
              label = 'Cluster'+str(i+1)) # for first cluster
plt.title('Clusters of customers - AHC')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(loc='center left', bbox_to_anchor=(0.9, 0.5))
plt.show()
# What do you get? Comment out your answer.
# A scatter graph where x and y values are annual income and speiding score
# the data are divided into 5 clusters according to AHC

# IV.Marketing Data with K-Means

# 1) import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 2) read dataset from csv and check the information (the original data is separated by ’tab’)
# Reading dataset from tab separated data
dataset = pd.read_csv('./marketing.csv', sep='\t')
print(dataset.head())
# Finding the information of the dataset
print(dataset.shape)
print(dataset.info())
print(dataset.describe().T)

# 3) check Nan or null in data
# Checking if any NaN is present in column or not
print(dataset.isna().any())

# To check the null values in the dataset
for col in dataset.columns:
  temp = dataset[col].isnull().sum()
if temp > 0:
  print(f'Column {col} contains {temp} null values.')

# fill the missing value in the income by mean
dataset['Income'] = dataset['Income'].fillna(dataset['Income'].mean())
print(dataset.isna().any())

# drop missing values
df = dataset.dropna()
print("Total values are:", len(df))

# 4) drop columns because they will not contribute anything in model building
# Finding the number of unique values present in each column
print(df.nunique())

# Dropping columns because they will not contribute anything in model building
df=df.drop(columns=["Z_CostContact", "Z_Revenue"],axis=1)
print(df.head())

# 5) add/drop columns
# Adding a column "Age" in the dataframe
df['Age'] = 2023 - df["Year_Birth"]

# # Number of days a customer was engaged with company
# Changing Dt_customer into timestamp format
df['Dt_Customer'] = pd.to_datetime(df.Dt_Customer, format="%d-%m-%Y")
df['first_day'] = '01-01-2023'
df['first_day'] = pd.to_datetime(df.first_day, format="%d-%m-%Y")
df['day_engaged'] = (df['first_day'] - df['Dt_Customer']).dt.days
df=df.drop(columns=["ID", "Dt_Customer", "first_day", "Year_Birth", "Dt_Customer", "Recency", "Complain"],axis=1)
print(df.shape)

# 6) combine different dataframe columns into a single column to reduce the number of dimension
df['Kids'] = df['Kidhome'] + df['Teenhome']
df['Expenses'] = df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] + df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds']
df['TotalAcceptedCmp'] = df['AcceptedCmp1'] + df['AcceptedCmp2'] + df['AcceptedCmp3'] + df['AcceptedCmp4'] + df['AcceptedCmp5'] + df['Response']
df['NumTotalPurchases'] = df['NumWebPurchases'] + df['NumCatalogPurchases'] + df['NumStorePurchases'] + df['NumDealsPurchases']

# Deleting some column to reduce dimension and complexity of model
col_del = ["AcceptedCmp1" , "AcceptedCmp2", "AcceptedCmp3" , "AcceptedCmp4","AcceptedCmp5",
"Response","NumWebVisitsMonth",
"NumWebPurchases","NumCatalogPurchases","NumStorePurchases","NumDealsPurchases" ,
"Kidhome", "Teenhome","MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts",
"MntSweetProducts", "MntGoldProds"]
df=df.drop(columns=col_del,axis=1)
print(df.head())

# 7) simplify the categories
# Checking number of unique categories present in the "Marital_Status"
print(df['Marital_Status'].value_counts())
df['Marital_Status'] = df['Marital_Status'].replace(['Married', 'Together'],'relationship')
df['Marital_Status'] = df['Marital_Status'].replace(['Divorced', 'Widow', 'Alone', 'YOLO', 'Absurd'],'Single')
print(df['Marital_Status'].value_counts())

# Checking number of unique categories present in the "Education"
print(df['Education'].value_counts())
df['Education'] = df['Education'].replace(['PhD','2n Cycle','Graduation', 'Master'],'PG')
df['Education'] = df['Education'].replace(['Basic'], 'UG')
print(df['Education'].value_counts())

# 8) label encoding
# Label Encoding
cate = []
for i in df.columns:
  if (df[i].dtypes == "object"):
    cate.append(i)
print(cate)

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
lbl_encode = LabelEncoder()
for i in cate:
  df[i]=df[[i]].apply(lbl_encode.fit_transform)
print(df.head())

# 9) feature scaling
# Label Encoding
# Standardization
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df.values)
scaled_df = pd.DataFrame(scaled_data, index=df.index, columns=df.columns)
print(scaled_df.head())

# 10) find the number of clusters (K)
# Elbow Method
from sklearn.cluster import KMeans
wcss=[]
for i in range (1,21):
  kmeans=KMeans(n_clusters=i,init='k-means++', max_iter=500, random_state=42)
  kmeans.fit(scaled_df)
  wcss.append(kmeans.inertia_)
plt.figure(figsize=(12,8))
plt.plot(range(1,21),wcss, 'bx-')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
number_of_clusters = 5

# Silhouette Score
from sklearn.metrics import silhouette_score
silhouette_scores = []
for i in range(2,20):
  m1=KMeans(n_clusters=i, max_iter=500,random_state=42)
  c = m1.fit_predict(scaled_df)
  silhouette_scores.append(silhouette_score(scaled_df, m1.fit_predict(scaled_df)))
plt.bar(range(2,20), silhouette_scores)
plt.xlabel('Number of clusters', fontsize = 20)
plt.ylabel('S(i)', fontsize = 20)
plt.show()
sc = max(silhouette_scores)
number_of_clusters = silhouette_scores.index(sc)+2
print("Number of Cluster Required is : ", number_of_clusters)

# 11) train the model
# Training a predicting using K-Means Algorithm.
kmeans = KMeans(n_clusters = number_of_clusters, max_iter=500,random_state=42).fit(scaled_df)
pred = kmeans.predict(scaled_df)

# Appending those cluster value into main dataframe (without standard-scalar)
df['cluster'] = pred + 1
print(df.head())
pl = sns.countplot(x=df["cluster"])
pl.set_title("Distribution Of The Clusters")
plt.show()