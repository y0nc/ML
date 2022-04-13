from numpy import float64
import pandas as pd
from time import time
from scipy import stats as st


class NaiveBayes:
    class U:
        def __init__(self) -> None:
            self.is_discrate = True
            self.cnt = {}

        is_discrate, cnt, = None, None

    data = []
    Y = []
    tot, ycnt = 0, {}
    Ni = []

    def __init__(self, D) -> None:
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
