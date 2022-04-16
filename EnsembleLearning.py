from numpy import exp, log

class AdaBoost:

    base_learner = None
    D = None
    P = None
    H = None
    a = None
    m = None

    def malloc(self):
        self.P = []
        self.H = []
        self.a = []

    def __init__(self, D, base_learner, T) -> None:
        self.malloc()
        self.base_learner = base_learner
        self.T = 0
        self.D = D
        self.m = len(D)
        for i in range(self.m):
            self.P.append(1 / self.m)
        for t in range(T):
            print("Generating No.{} hypothesis...".format(t + 1))
            # 训练新模型
            # h = self.train()
            h = self.base_learner(self.D.copy(), self.P)
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
