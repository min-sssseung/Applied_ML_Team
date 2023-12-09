import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer

# Set a font that supports Hangul characters
### EDA ### 

plt.rcParams['font.family'] = 'Malgun Gothic'

df = pd.read_csv('./Financial/[1206]ESG.csv', dtype={'corp_code': str, 'stock_code': str})

# scaler= StandardScaler()
scaler= MinMaxScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer(output_distribution='uniform') # 새로운 모습
# scaler = PowerTransformer()

# DF 만들기 
saved_df = df.iloc[:, :5]
scaled_df = pd.DataFrame(scaler.fit_transform(df.iloc[:, 5:]), columns=df.columns[5:])
df2 = pd.concat([saved_df, scaled_df], axis=1)

# ESG 등급 나누기
binary_ESG = df2['ESG'].map({'A+': 1,'A': 1,'B+': 1,'B': 0,'C': 0,'D': 0,})

df2['ESG'] = pd.Categorical(df2['ESG'], categories=['A+', 'A', 'B+', 'B', 'C', 'D'], ordered=True)
df2['ESG'] = df2['ESG'].map({'A+': 6,'A': 5,'B+': 4,'B': 3,'C': 2,'D': 1})

# 데이터 분포를 살피기
plt.figure()
sns.countplot(data=df2, x='ESG') # imbalanced dataset
plt.show()

# Multi-label task
for i, col in enumerate(df2.columns[5:]):
    plt.figure(figsize=(10, 8))
    sns.kdeplot(data=df2, x=col, hue='ESG',  fill=True)
    plt.title(col)
    plt.tight_layout()
    plt.show()
# QuantileTransformer, PowerTransformer로 진행해도 괜찮을듯 


# # Binary Task
# df2['binary_ESG'] = binary_ESG

# for i, col in enumerate(df2.columns[5:-1]):
#     plt.figure(figsize=(10, 8))
#     sns.kdeplot(data=df2, x=col, hue='binary_ESG', fill=True)
#     plt.title(col)
#     plt.tight_layout()
#     plt.show()
# # 성능이 낮으면 이진분류로 가도 좋다

corr_matrix = df2.iloc[:, 5:].corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
fig, ax = plt.subplots(figsize=(12, 10))
# sns.heatmap(corr_matrix, mask=mask, annot=True)
sns.heatmap(corr_matrix, mask=mask, cmap="coolwarm", vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()

# 상관관계는 없는 듯하다

