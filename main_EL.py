import matplotlib.pyplot as plt
from numpy import sort, unique, mean
import pandas as pd
from time import time
from DecisionTree import DecisionStump
from EnsembleLearning import AdaBoost


def visualize(model, title):
    plt.clf()
    plt.xlabel("Density")
    plt.ylabel("Sugar Content")
    plt.title(title)
    vline = [0.2, 0.8]
    hline = [0.0, 0.5]
    for i in range(model.T):
        a = model.H[i].a
        thresh = model.H[i].thresh
        if a == "密度":
            vline.append(thresh)
        else:
            hline.append(thresh)
    vline = sort(unique(vline))
    hline = sort(unique(hline))

    T = [[], []]
    F = [[], []]
    for i in range(len(D)):
        x, y = D.iloc[i, -3], D.iloc[i, -2]
        if D.iloc[i, -1] == "是":
            T[0].append(x)
            T[1].append(y)
        else:
            F[0].append(x)
            F[1].append(y)

    plt.plot(T[0], T[1], "r*", label="good")
    plt.plot(F[0], F[1], "b*", label="bad")

    for i in range(1, len(hline)):
        for j in range(1, len(vline)):
            x = [vline[j], vline[j - 1], vline[j - 1], vline[j]]
            y = [hline[i], hline[i], hline[i - 1], hline[i - 1]]
            mx, my = mean(x), mean(y)
            if model.predict([mx, my]) == "是":
                plt.fill(x, y, color="r", alpha=0.3)
            else:
                plt.fill(x, y, color="b", alpha=0.3)
    plt.legend()
    # plt.show()
    plt.savefig("charts\\{}.jpg".format(model.T))


D = pd.read_csv("watermelon3_0_α_Ch.csv")
D.drop("编号", axis=1, inplace=True)

"""
dataset = datasets.load_wine()
D = pd.DataFrame(
    data=np.c_[dataset["data"], dataset["target"]],
    columns=dataset["feature_names"] + ["target"],
)
"""

for t in range(1, 10):
    t0 = time()
    model = AdaBoost(D, DecisionStump, t)
    print("Train consume: {:.03f}s".format(time() - t0))
    p = 0.0
    m = len(D)
    for i in range(m):
        x = D.iloc[i, :].tolist()
        y = model.predict(x[:-1])
        if y == x[-1]:
            p += 1 / m
    print("Precision: {:.2f}%".format(p * 100))
    visualize(model, title="T={}   Precesion={:.03f}".format(t, p))
