# 作者：zgx
# 时间：2021/10/20 19:51
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D  # 不要去掉这个import
from sklearn.metrics import mean_squared_error, r2_score
# 解决matplotlib中文问题
from pylab import mpl


plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 获取数据
data = pd.read_csv("data/tmdb_5000_movies.csv")

data.head()

# 数据基本处理

# 空值去除
data = data.replace(0, np.nan)
data = data.dropna(axis=0, how="any")
# 特征值提取
feature_arr = data[['budget', 'popularity', 'runtime', 'vote_average', 'vote_count']].values

# 标签值提取
tag_arr = data['revenue']

# tag_arr = np.array(pd.qcut(data['revenue'], 3, labels=['低', '中', '高']))

# 数据分割
# 这里将数据分割为训练数据集与预测数据集，这里是默认8比2的方式。
x_train, x_test, y_train, y_test = train_test_split(feature_arr, tag_arr,test_size=0.2)
print(y_train)
# 特征工程 特征工程就对特征进行一些预处理操作，我们的训练特征均是连续型的，所以这里无需过多进行处理。这里只是对数据进行正则化转换，即把各个维度的数值拉到一个正态化的区间。
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 训练模型

model = LinearRegression()

model.fit(x_train, y_train)
# 利用模型进行预测
y_predict = model.predict(x_test)
print(y_predict)
y_predicts = np.array(pd.qcut(y_predict, 2, labels=['低', '高']))
y_tests = np.array(pd.qcut(y_test, 2, labels=['低', '高']))
print("预测值和真实值的对比是:\n", y_predicts == y_tests)
rs = y_predicts == y_tests
i = 0
for r in rs:
    if r == True:
        i += 1
print("准确率：", i/len(x_test))
score = model.score(x_test, y_test)
print(score)
from sklearn.metrics import confusion_matrix
import seaborn as sn
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
i = 1
fig1 = plt.figure(figsize=(2 * 3, 1 * 4))
# pred_y = model.predict(x_test)
matrix = pd.DataFrame(confusion_matrix(y_tests, y_predicts))
ax1 = fig1.add_subplot(2, 2, 1)
sn.heatmap(matrix, annot=True, cmap='OrRd')
plt.ylabel('实际值0/1', fontsize=12)
plt.xlabel('预测值0/1', fontsize=12)
plt.title('Random Forest ')
plt.show()
