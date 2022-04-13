from numpy import float64
import numpy as np
import pandas as pd
from time import time
from scipy import stats as st
from sklearn import datasets


class NaiveBayes:
    class U:
        def __init__(self) -> None:
            self.is_discrate = True
            self.cnt = {}

        is_discrate, cnt, = None, None

    data = None
    Y = None
    tot = None
    ycnt = None
    Ni = None

    def malloc(self):
        self.data = []
        self.Y = []
        self.tot = 0
        self.ycnt = {}
        self.Ni = []

    def __init__(self, D) -> None:
        self.malloc()
        self.tot = len(D)
        self.Y = set(D.iloc[:, -1])
        Dy = {}
        for y in self.Y:
            Dy[y] = D.loc[D[D.keys()[-1]] == y]
            self.ycnt[y] = len(Dy[y])
        for attr in D.keys()[:-1]:
            self.Ni.append(len(set(D[attr])))
            self.data.append(self.U())
            if D[attr].dtype == float64:
                self.data[-1].is_discrate = False
                for y in self.Y:
                    μ = Dy[y][attr].mean()
                    s = Dy[y][attr].std()
                    self.data[-1].cnt[y] = [μ, s]
            else:
                self.data[-1].is_discrate = True
                for y in self.Y:
                    self.data[-1].cnt[y] = {}
                    for xi in Dy[y][attr]:
                        self.data[-1].cnt[y][xi] = self.data[-1].cnt[y].get(xi, 0) + 1

    def predict(self, x):
        laplace = False
        res, maxp = None, 0.0
        # 判断是否需要laplace修正
        for y in self.Y:
            for xi, i in zip(x, range(len(x))):
                if self.data[i].is_discrate:
                    if self.data[i].cnt[y].get(xi, -1) == -1:
                        laplace = True

        for y in self.Y:
            p = 1.0
            for xi, i in zip(x, range(len(x))):
                if self.data[i].is_discrate:
                    if laplace:
                        p *= (self.data[i].cnt[y].get(xi, 0) + 1) / (
                            self.ycnt[y] + self.Ni[i]
                        )
                    else:
                        p *= self.data[i].cnt[y][xi] / self.ycnt[y]
                else:
                    p *= st.norm.pdf(
                        x=xi, loc=self.data[i].cnt[y][0], scale=self.data[i].cnt[y][1]
                    )

            if laplace:
                p *= (self.ycnt[y] + 1) / (self.tot + len(self.Y))
            else:
                p *= self.ycnt[y] / self.tot

            if p > maxp:
                maxp, res = p, y
        return res