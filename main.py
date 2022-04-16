from DecisionTree import DecisionTree
from NaiveBayes import NaiveBayes
import numpy as np
from sklearn import datasets
from time import time
import pandas as pd


def splite_dataset(D, train_ratio):
    permutation = np.random.permutation(len(D))
    train_size = int(len(D) * train_ratio)
    train_i = permutation[:train_size]
    test_i = permutation[train_size:]
    return D.loc[train_i], D.loc[test_i]


dataset = datasets.load_wine()
D = pd.DataFrame(
    data=np.c_[dataset["data"], dataset["target"]],
    columns=dataset["feature_names"] + ["target"],
)

train_set, test_set = splite_dataset(D, 0.7)

print(train_set)
print(test_set)


t0 = time()
# model = NaiveBayes(train_set)
model = DecisionTree(train_set, max_depth=1)
train_time = time() - t0

print(test_set.values)

precision = 0.0
t0 = time()
for i in range(len(test_set)):
    x = test_set.iloc[i, :].tolist()
    y = model.predict(x[:-1])
    print("{}: 真实分类: {}, 预测结果: {}".format(i + 1, x[-1], y))
    if x[-1] == y:
        precision += 1
precision /= len(test_set)
predict_time = time() - t0


print("决策树节点数: {}".format(len(model.V)))
print("准确率: {:.3f}%".format(precision * 100))
print("统计消耗时间: {:.3f}S".format(train_time))
print("预测消耗时间: {:.3f}S".format(predict_time))
