from numpy import float64, unique, sort, log2
import pandas as pd
from time import time


class DecisionTree:
    class node:
        def __init__(self) -> None:
            self.E = {}

        a, thresh = None, None  # 分类标准
        E = None  # 边集
        label = None  # 默认分类

    V = None
    a_dic = None
    remain_time = None
    max_depth = None
    continuous_a_usetimes = None

    def malloc(self):
        self.V = []
        self.a_dic = {}
        self.remain_time = {}

    def __init__(self, D, con_a_usetimes=10, max_depth=10) -> None:
        self.malloc()
        self.max_depth = max_depth
        keys = D.keys()[:-1]
        for i in range(len(keys)):
            self.a_dic[keys[i]] = i
            if D[keys[i]].dtype == float64:
                self.remain_time[keys[i]] = con_a_usetimes
            else:
                self.remain_time[keys[i]] = 1
        self.BuildTree(D)

    def Ent(self, D):
        lable_key = D.keys()[-1]
        y = D[lable_key]
        tot, res, cnt = len(y), 0.0, {}
        for yi in y:
            cnt[yi] = cnt.get(yi, 0) + 1
        for key, val in cnt.items():
            res += -(val / tot) * log2(val / tot)
        return res

    def divide(self, D, a, thresh=None):
        if D[a].dtype == float64:
            return [D.loc[D[a] <= thresh], D.loc[D[a] > thresh]]
        else:
            return [D.loc[D[a] == val] for val in set(D[a])]

    def Gain(self, D, a, thresh=None):
        lable_key = D.keys()[-1]
        res = self.Ent(D)
        for Dv in self.divide(D, a, thresh):
            res -= (len(Dv) / len(D)) * self.Ent(Dv)
        return res

    def Gain_ratio(self, D, a, thresh=None):
        IVa = 0.0
        for Dv in self.divide(D, a, thresh):
            IVa += -(len(Dv) / len(D)) * log2(len(Dv) / len(D))
        Gain = self.Gain(D, a, thresh)
        return Gain / IVa

    def chooseAttribute(self, D):
        gain = []
        best_thresh = []
        gain_ratio = []
        for a in D.keys()[:-1]:
            gain.append(0.0)
            best_thresh.append(0.0)
            gain_ratio.append(0.0)
            if D[a].dtype == float64:
                vals = sort(unique(D[a]))
                threshs = [(vals[i] + vals[i + 1]) / 2 for i in range(len(vals) - 1)]
                for thresh in threshs:
                    _gain_ = self.Gain(D, a, thresh)
                    _gain_ratio_ = self.Gain_ratio(D, a, thresh)
                    if _gain_ratio_ > gain_ratio[-1]:
                        gain[-1] = _gain_
                        best_thresh[-1] = thresh
                        gain_ratio[-1] = _gain_ratio_
            else:
                gain[-1] = self.Gain(D, a)
                gain_ratio[-1] = self.Gain_ratio(D, a)
        mean_gain_ratio = sum(gain_ratio) / len(gain_ratio)
        p, max_gain = None, 0
        for i in range(len(gain)):
            if gain_ratio[i] < mean_gain_ratio:
                pass
            else:
                if gain[i] > max_gain:
                    max_gain = gain[i]
                    p = i
        return [D.keys()[p], best_thresh[p]]

    def BuildTree(self, D, depth=0):
        root = len(self.V)
        self.V.append(self.node())
        cnt = D.iloc[:, -1].value_counts()
        self.V[root].label = cnt.keys()[cnt.argmax()]
        if depth == self.max_depth:
            return
        if self.Ent(D) == 0 or len(D.keys()) == 1:  # 标签一致，无需再划分 或属性耗尽
            return
        a, thresh = self.chooseAttribute(D)
        if self.Gain(D, a) == 0:  # 所有样本一致，不可再划分
            return
        self.V[root].a = a
        if self.remain_time[a] == 0:
            return
        else:
            self.remain_time[a] -= 1
        if D[a].dtype == float64:
            self.V[root].thresh = thresh
            Dvs = self.divide(D, a, thresh)
            for Dv, i in zip(Dvs, range(len(Dvs))):
                self.V[root].E[i] = len(self.V)
                self.BuildTree(Dv, depth + 1)
        else:
            for Dv in self.divide(D, a):
                _Dv_ = Dv.drop(a, axis=1, inplace=False)
                val = Dv[a][Dv.index[0]]
                self.V[root].E[val] = len(self.V)
                self.BuildTree(_Dv_, depth + 1)

    def predict(self, x, root=0):
        if len(self.V[root].E) == 0:
            return self.V[root].label
        a = self.V[root].a
        i = self.a_dic[a]
        if isinstance(x[i], float):
            res = None
            if x[i] <= self.V[root].thresh:
                res = self.predict(x, self.V[root].E[0])
            else:
                res = self.predict(x, self.V[root].E[1])
            return res
        else:
            res = self.predict(x, self.V[root].E[x[i]])
            return res


class DecisionStump:
    def malloc(self):
        self.dic = {}

    a, ia = None, None
    thresh = None
    dic = None

    def divide(self, D, a, thresh=None):
        if thresh == None:
            return [D.loc[D[a] == av] for av in set(D[a])]
        else:
            return [D.loc[D[a] <= thresh], D.loc[D[a] > thresh]]

    def Ent(self, D):
        y = D.iloc[:, -1]
        res = 0.0
        for cnt in y.value_counts():
            p = cnt / len(y)
            res += -p * log2(p)
        return res

    def Gain(self, D, a, thresh=None):
        Dvs = self.divide(D, a, thresh)
        gain = self.Ent(D)
        IVa = 0.0
        for Dv in Dvs:
            p = len(Dv) / len(D)
            gain += -p * self.Ent(Dv)
            IVa += -p * log2(p)
        gain_ratio = gain / IVa
        return gain, gain_ratio

    def __init__(self, D) -> None:
        self.malloc()
        A = D.keys()[:-1]
        m = len(A)
        gain_info = []  # gain,gain_ratio,thresh
        mean = 0.0
        for a in A:
            if D[a].dtype == float64:
                vals = sort(unique(D[a]))
                avs = [(vals[i] + vals[i + 1]) / 2 for i in range(len(vals) - 1)]
                mx_gain, mx_gain_ratio, thresh = 0.0, 0.0, None
                for av in avs:
                    gain, gain_ratio = self.Gain(D, a, av)
                    if gain_ratio > mx_gain_ratio:
                        mx_gain, mx_gain_ratio, thresh = gain, gain_ratio, av
                gain_info.append([mx_gain, mx_gain_ratio, thresh])
            else:
                gain, gain_ratio = self.Gain(D, a)
                gain_info.append([gain, gain_ratio, None])
            mean += gain_info[-1][1] / m
        mx_gain, p = 0.0, None
        for i in range(m):
            if gain_info[i][1] >= mean:
                if gain_info[i][0] > mx_gain:
                    mx_gain = gain_info[i][0]
                    p = i
        self.a = A[p]
        self.ia = p
        self.thresh = gain_info[p][2]
        if self.thresh == None:
            Dvs = self.divide(D, self.a)
            for Dv in Dvs:
                cnt = Dv.iloc[:, -1].value_counts()
                av = Dv[self.a].tolist()[0]
                y = cnt.keys()[cnt.argmax()]
                self.dic[av] = y
        else:
            Dvs = self.divide(D, self.a, self.thresh)
            for Dv, i in zip(Dvs, range(len(Dvs))):
                cnt = Dv.iloc[:, -1].value_counts()
                y = cnt.keys()[cnt.argmax()]
                self.dic[i] = y

    def predict(self, x):
        if self.thresh == None:
            return self.dic[x[self.ia]]
        else:
            if x[self.ia] <= self.thresh:
                return self.dic[0]
            else:
                return self.dic[1]
