from NaiveBayes import NaiveBayes
import numpy as np
from sklearn import datasets
from time import time
import pandas as pd


df = pd.read_csv("watermelon3_0_Ch.csv")
df.drop("编号", axis=1, inplace=True)
t0 = time()
nb = NaiveBayes(df)
print("统计消耗时间: {:.3f}S".format(time() - t0))
precision = 0.0
t0 = time()
for i in range(len(df)):
    x = df.iloc[i, :].tolist()
    y = nb.predict(x[:-1])
    print("{}: 真实分类: {}, 预测结果: {}".format(i + 1, x[-1], y))
    if x[-1] == y:
        precision += 1 / (len(df) + 1)

print("准确率: {:.3f}".format(precision))
print("预测消耗时间: {:.3f}S".format(time() - t0))

iris = datasets.load_iris()
df = pd.DataFrame(
    data=np.c_[iris["data"], iris["target"]], columns=iris["feature_names"] + ["target"]
)
print(df)
t0 = time()
nb = NaiveBayes(df)
print("统计消耗时间: {:.3f}S".format(time() - t0))
precision = 0.0
t0 = time()
for i in range(len(df)):
    x = df.iloc[i, :].tolist()
    y = nb.predict(x[:-1])
    print("{}: 真实分类: {}, 预测结果: {}".format(i + 1, x[-1], y))
    if x[-1] == y:
        precision += 1 / (len(df) + 1)

print("准确率: {:.3f}".format(precision))
print("预测消耗时间: {:.3f}S".format(time() - t0))
