from numpy import float64, unique, sort, log2
import pandas as pd
from time import time


class DecisionTree:
    class node:
        def __init__(self) -> None:
            self.attr = object
            self.E = dict()
            self.label = object
            pass

    T = []
    attr_index = {}

    def __init__(self, D) -> None:
        for attr, i in zip(D.keys()[:-1], range(D.keys().__len__()-1)):
            self.attr_index[attr] = i
        self.CreateDecisionTree(D)

    def Ent(self, D):
        cnt = D.iloc[:, -1].value_counts()
        ent = [-pk*log2(pk) for pk in cnt/cnt.sum()]
        return sum(ent)

    def Seperate(self, D):
        attrs = D.keys()[:-1]
        mn_ent = 100000000.
        (Dvs, vals, attr) = (None, None, None)
        for _attr_ in attrs:
            if D[_attr_].dtype == float64:
                ps = sort(unique(D[_attr_]))
                ps = [(ps[i]+ps[i+1])/2 for i in range(ps.__len__()-1)]
                for p in ps:
                    _vals_ = [-p, p]
                    _Dvs_ = [D.loc[D[_attr_] <= p], D.loc[D[_attr_] > p]]
                    _ent_ = sum([len(_Dv_)/len(D[_attr_]) *
                                 self.Ent(_Dv_) for _Dv_ in _Dvs_])
                    if _ent_ < mn_ent:
                        mn_ent = _ent_
                        Dvs, vals, attr = _Dvs_, _vals_, _attr_
            else:
                _vals_ = unique(D[_attr_])
                _Dvs_ = [D.loc[D[_attr_] == val] for val in _vals_]
                _ent_ = sum([len(_Dv_)/len(D[_attr_]) *
                            self.Ent(_Dv_) for _Dv_ in _Dvs_])
                if _ent_ < mn_ent:
                    mn_ent = _ent_
                    vals, attr = _vals_, _attr_
                    Dvs = [_Dv_.drop(_attr_, axis=1) for _Dv_ in _Dvs_]
        return (Dvs, vals, attr)

    def CreateDecisionTree(self, D):
        self.T.append(self.node())

        cnt = D.iloc[:, -1].value_counts()
        self.T[-1].label = cnt.keys()[cnt.argmax()]

        if self.Ent(D) == 0 or D.keys().__len__() == 1:
            return

        Dvs, vals, attr = self.Seperate(D)
        self.T[-1].attr = attr
        x = self.T.__len__()-1
        for Dv, val in zip(Dvs, vals):
            self.T[x].E[val] = self.T.__len__()
            self.CreateDecisionTree(Dv)

    def predict(self, x, root=0):
        attr = self.T[root].attr
        for val, nxt in self.T[root].E.items():
            if isinstance(val, float):
                if val <= 0:
                    if x[self.attr_index[attr]] <= -val:
                        return self.predict(x, nxt)
                else:
                    if x[self.attr_index[attr]] > val:
                        return self.predict(x, nxt)
            else:
                if x[self.attr_index[attr]] == val:
                    return self.predict(x, nxt)
        else:
            return self.T[root].label


df = pd.read_csv("watermelon3_0_Ch.csv")
df.drop('编号', axis=1, inplace=True)
t0 = time()
decision_tree = DecisionTree(df)
print('Train consume: {}s'.format(time()-t0))
t0 = time()
for xi in range(df.shape[0]):
    x = df.iloc[xi, :].tolist()
    x = [x, '预测结果：'+decision_tree.predict(x)]
    print(x)
print('Predict consum: {}s'.format(time()-t0))
