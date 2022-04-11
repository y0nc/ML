


for i in range(len(df)):
    x = df.iloc[i+1, :].tolist()
    print(x)
    y = nb.predict(x[:-1])
    print("Predict res: {}".format(y))

print("done.")
