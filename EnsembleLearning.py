from numpy import exp, float64, log, unique, sort, log2, sign
import pandas as pd
from DecisionTree import DecisionStump
from random import random
from sklearn import datasets
from time import time
import numpy as np


class AdaBoost:

    base_learner = None
    D = None
    P = None
    H = None
    a = None
    m = None

    def malloc(self):
        self.P = {}
        self.H = []
        self.a = []

    def train(self):
        # 重采样再训练
        ls = []
        for i in range(self.m):
            x = self.D.loc[i]
            p = self.P[i] * self.m
            n = int(p)
            f = p - n
            if random() < f:
                n += 1
            for j in range(n):
                ls.append(x)
        D = pd.DataFrame(ls, columns=self.D.keys())
        h = self.base_learner(D)
        return h

    def __init__(self, D, base_learner, T) -> None:
        self.malloc()
        self.base_learner = base_learner
        self.T = 0
        self.D = D
        self.m = len(D)
        for i in range(self.m):
            self.P[i] = 1 / self.m
        for t in range(T):
            print("Generating No.{} hypothesis...".format(t + 1))
            # 训练新模型
            h = self.train()
            self.H.append(h)
            # 计算ε和α
            e = 0.0
            for i in range(self.m):
                x = self.D.iloc[i, :].tolist()
                y = h.predict(x[:-1])
                if y != x[-1]:
                    e += self.P[i]
            if e > 0.5:
                print("Stop training.")
                break
            else:
                self.T += 1
                a = 0.5 * log((1 - e) / e)
                self.a.append(a)
            # 更新分布
            z = 0.0
            for i in range(self.m):
                x = self.D.iloc[i, :].tolist()
                y = h.predict(x[:-1])
                if x[-1] == y:
                    self.P[i] *= exp(-a)
                else:
                    self.P[i] *= exp(a)
                z += self.P[i]
            for i in range(self.m):
                self.P[i] /= z

    def predict(self, x):
        res, mx = None, 0.0
        vote = {}
        for i in range(self.T):
            y = self.H[i].predict(x)
            a = self.a[i]
            vote[y] = vote.get(y, 0.0) + a
        for key, val in vote.items():
            if val > mx:
                res, mx = key, val
        return res


D = pd.read_csv("watermelon3_0_α_Ch.csv")
D.drop("编号", axis=1, inplace=True)

"""
dataset = datasets.load_wine()
D = pd.DataFrame(
    data=np.c_[dataset["data"], dataset["target"]],
    columns=dataset["feature_names"] + ["target"],
)
"""

t0 = time()
model = AdaBoost(D, DecisionStump, 1)
print("Train consume: {:.03f}s".format(time() - t0))

p = 0.0
m = len(D)
for i in range(m):
    x = D.iloc[i, :].tolist()
    y = model.predict(x[:-1])
    if y == x[-1]:
        p += 1 / m
print("Precision: {:.2f}%".format(p * 100))

print("done.")
