from json import load
from os import remove
from cv2 import randShuffle
import numpy as np


class DecisionTree:
    class node:
        E = []
        res = None

    V = []
    mask = []

    def __init__(self, D):
        if isinstance(D, list):
            D = np.array(D)
        ls = []
        continuous = []
        for dim in range(D.shape[1]):
            # 构建映射
            self.mask.append({})
            try:
                float(D[0, dim])
                continuous.append(True)
                pass
            except:
                continuous.append(False)
                atts = np.unique(D[:, dim])
                for i in range(len(atts)):
                    self.mask[dim][i] = atts[i]
                    self.mask[dim][atts[i]] = i

            # 修改数据集
            for i in range(D.shape[0]):
                try:
                    ls.append(float(D[i, dim]))
                except:
                    ls.append(self.mask[dim][D[i, dim]])

        D = np.array(ls).reshape(D.shape, order='F')
        self.BuildDecisionTree(D, continuous)

    def Ent(self, D):
        Y, cnt = np.unique(D[:, -1], return_counts=True)
        return sum([-pk*np.log2(pk) for pk in cnt/len(D)])

    def BestSeparation(self, D, continuous):
        ndim = D.shape[1]-1
        gains = np.zeros(ndim)
        thresh = np.zeros(ndim)
        for dim in range(ndim):
            if self.continuous[dim]:
                # 连续值
                v = np.sort(np.unique(D[:, dim]))
                threshs = [(v[i]+v[i+1])/2 for i in range(len(v)-1)]
                tmpgains = np.zeros(len(threshs))
                for i in range(len(threshs)):
                    D1 = D[np.where(D[:, dim] <= threshs[i])]
                    D2 = D[np.where(D[:, dim] > threshs[i])]
                    tmpgains[i] += len(D1)/len(D)*self.Ent(D1)
                    tmpgains[i] += len(D2)/len(D)*self.Ent(D2)
                gains[dim] = min(tmpgains)
                thresh[dim] = threshs[np.argmin(tmpgains)]
            else:
                # 离散值
                for att in np.unique(D[:, dim]):
                    Dv = D[np.where(D[:, dim] == att)]
                    gains[dim] += len(Dv)/len(D)*self.Ent(Dv)
        gains = self.Ent(D) - gains
        return (np.argmax(gains), thresh[np.argmax(gains)])

    def BuildDecisionTree(self, D, continuous):
        self.V.append(self.node())
        if self.Ent(D) == 0:
            # 所有向量类别均相等
            # 叶节点，标记为所有向量所共有的类别
            self.V[len(self.V)-1].res = D[0, 0]
            return

        dim = len(D[0])-1
        if dim == 0 or np.array([np.array(D[0, :-1] == D[i, :-1]).all()
                                 for i in range(D.shape[0])]).all():
            # 属性空间维度为0，或所有向量在剩余属性空间上相等
            # 叶节点，标记为众数类别
            Y, cnt = np.unique(D[:, -1], return_counts=True)
            self.V[len(self.V)-1].res = Y[np.argmax(cnt)]
            return

        best_sp = self.BestSeparation(D)
        sp_dim, thresh = best_sp
        if continuous[sp_dim]:
            # 连续属性作为最佳划分
            self.V[len(self.V)-1].E.append([sp_dim, -thresh, len(self.V)])
            self.BuildDecisionTree(D[np.where(D[:, sp_dim] <= thresh)])
            self.V[len(self.V)-1].E.append([sp_dim, -thresh, len(self.V)])
            self.BuildDecisionTree(D[np.where(D[:, sp_dim] > thresh)],remove())
        else:
            # 离散属性作为最佳划分
            for att in np.unique(D[:, sp_dim]):
                Dv = D[np.where(D[:, sp_dim] == att)]
                self.V[len(self.V)-1].E.append([sp_dim, att, len(self.V)])
                self.BuildDecisionTree(np.delete(Dv, sp_dim, axis=1))
            # 空节点
            self.V[len(self.V)-1].E.append([sp_dim, -1, len(self.V)])
            self.V.append(self.node())
            Y, cnt = np.unique(D[:, -1], return_counts=True)
            self.V[len(self.V)-1].res = Y[np.argmax(cnt)]


with open('watermelon3_0_Ch.csv', encoding='utf-8') as f:
    D = np.loadtxt(f, str, delimiter=",", skiprows=1)
    D = np.delete(D, 0, axis=1)
    print(D)
    decision_tree = DecisionTree(D)
