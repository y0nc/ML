import numpy as np
from numpy import array, where
from torch import narrow


class DecisionTree:
    class node:
        E = []
        res = None
    D = []
    narryD = None
    V = []

    def __init__(self, D):
        dim = len(D[0, :])
        nd = len(D[:, 0])
        for d in range(dim):
            try:
                if isinstance(D[0, d], float) or D[0, d].find('.') != -1:
                    self.D.append([float(D[i, d]) for i in range(nd)])
                    continue
            except:
                pass
            # defualt
            self.D.append([D[i, d] for i in range(nd)])
        self.narryD = np.array(self.D)
        D_i = np.arange(nd)
        a_i = np.arange(dim-1)
        self.BuildDecisionTree(D_i, a_i)

    def Ent(self, y):
        ys, cnt = np.unique(y, return_counts=True)
        tmp = [-pk*np.log2(pk) for pk in cnt/len(y)]
        return sum(tmp)

    def BestSeparation(self, D_i, a_i):
        x = self.narryD[a_i, :]
        z = x[:, D_i]
        x = x[:, D_i]
        y = self.narryD[-1, D_i]
        ents = np.zeros(len(a_i))
        return_thresh = np.zeros(len(a_i))
        for dim in range(len(a_i)):
            if isinstance(self.D[a_i[dim]][0], float):
                # 连续值
                ents[dim] = 100000000.
                atts = np.array(self.D[a_i[dim]])[D_i]
                v = np.sort(np.unique(atts))
                for i in range(len(v)-1):
                    thresh = (v[i]+v[i+1])/2
                    y1 = y[np.where(atts <= thresh)]
                    y2 = y[np.where(atts > thresh)]
                    ent = len(y1)/len(y) * self.Ent(y1) + \
                        len(y2)/len(y)*self.Ent(y2)
                    if ent < ents[dim]:
                        ents[dim] = ent
                        return_thresh[dim] = thresh
            else:
                # 离散值
                for val in np.unique(x[dim, :]):
                    yv = y[np.where(x[dim, :] == val)]
                    ents[dim] += len(yv)/len(y)*self.Ent(yv)
        gains = self.Ent(y) - ents
        return (np.argmax(gains), return_thresh[np.argmax(gains)])

    def BuildDecisionTree(self, D_i, a_i):
        self.V.append(self.node())
        self_node = self.V[len(self.V)-1]
        x = self.narryD[a_i, :][:, D_i]
        y = self.narryD[-1, D_i]

        if self.Ent(y) == 0:
            # 所有向量类别均相等
            # 叶节点，标记为所有向量所共有的类别
            self_node.res = y[0]
            return

        if len(a_i) == 0 or np.array([x[0] == Xi for Xi in x]).all():
            # 属性空间维度为0，或所有向量在剩余属性空间上相等
            # 叶节点，标记为众数类别
            ys, cnt = np.unique(y, return_counts=True)
            self_node.res = ys[np.argmax(cnt)]
            return

        best_sp = self.BestSeparation(D_i, a_i)
        i, thresh = best_sp
        attrs = np.array(self.D[a_i[i]])
        _a_i = np.delete(a_i, i)
        if isinstance(self.D[a_i[i]][0], float):
            # 连续属性作为最佳划分
            self_node.E.append([i, -thresh, len(self.V)])
            self.BuildDecisionTree(np.where(attrs <= thresh)[0], a_i)
            self_node.E.append([i, thresh, len(self.V)])
            self.BuildDecisionTree(np.where(attrs > thresh)[0], a_i)
        else:
            # 离散属性作为最佳划分
            for att in np.unique(attrs):
                self_node.E.append([i, att, len(self.V)])
                _D_i = np.where(attrs == att)
                self.BuildDecisionTree(np.where(attrs == att)[0], _a_i)
            # 空节点
            self_node.E.append([i, -1, len(self.V)])
            self.V.append(self.node())
            ys, cnt = np.unique(attrs, return_counts=True)
            self.V[len(self.V)-1].res = ys[np.argmax(cnt)]


with open('watermelon3_0_Ch.csv', encoding='utf-8') as f:
    D = np.loadtxt(f, str, delimiter=",", skiprows=1)
    D = np.delete(D, 0, axis=1)
    # print(D)
    decision_tree = DecisionTree(D)
