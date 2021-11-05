# 作者：zgx
# 时间：2021/10/20 17:09
import pandas as pd
import numpy as np
from boto import sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# 解决matplotlib中文问题
from pylab import mpl

i = 1
fig1 = plt.figure(figsize=(2 * 3, 1 * 4))

# 获取数据
data = pd.read_csv("data/tmdb_5000_movies.csv")

data.head()

# 数据基本处理

# 空值去除
data = data.replace(0, np.nan)
data = data.dropna(axis=0, how="any")


# 特征值提取
feature_arr = data[['budget', 'popularity', 'runtime', 'vote_average', 'vote_count']].values




# print(feature_arr)

# 标签值提取
tag_arr = data['revenue']
# print(tag_arr)

tag_arr = np.array(pd.qcut(data['revenue'], 2, labels=['低','高']))

# print(tag_arr)
# 数据分割
# 这里将数据分割为训练数据集与预测数据集，这里是默认8比2的方式。
x_train, x_test, y_train, y_test = train_test_split(feature_arr, tag_arr,test_size=0.2)
pd.plotting.scatter_matrix(data, figsize=(15,15), marker='0',c=np.array(data['revenue']), hist_kwds={'bins':50},s=60,alpha=0.8)
# df = pd.DataFrame(np.random.randn(1000, 4), columns=['A','B','C','D'])
# pd.plotting.scatter_matrix(df, alpha=0.2)
plt.show()
# 特征工程 特征工程就对特征进行一些预处理操作，我们的训练特征均是连续型的，所以这里无需过多进行处理。这里只是对数据进行正则化转换，即把各个维度的数值拉到一个正态化的区间。
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 模型训练
# 实例化一个KNN 预测器
revenue_predict = KNeighborsClassifier(n_neighbors=9)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                               metric_params=None, n_jobs=1, n_neighbors=24, p=3,
                               weights='distance')

# 训练
revenue_predict.fit(x_train, y_train)
y_pre = revenue_predict.predict(x_test)
print("预测结果:\n", y_pre)
print("预测值和真实值的对比是:\n", y_pre == y_test)
rs = y_pre == y_test
i = 0
for r in rs:
    if r == True:
        i += 1
print("准确率：", i/len(x_test))
score = revenue_predict.score(x_test, y_test)
print(score)
from sklearn.metrics import confusion_matrix
import seaborn as sn
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
i = 1
fig1 = plt.figure(figsize=(2 * 3, 1 * 4))
pred_y = revenue_predict.predict(x_test)
matrix = pd.DataFrame(confusion_matrix(y_test, pred_y))
ax1 = fig1.add_subplot(2, 2, 1)
sn.heatmap(matrix, annot=True, cmap='OrRd')
plt.ylabel('实际值0/1', fontsize=12)
plt.xlabel('预测值0/1', fontsize=12)
plt.title('Random Forest ')
plt.show()