import pandas as pd
import numpy as np
import os

train_df = pd.read_csv("train (1).csv")
test_df = pd.read_csv("test.csv")

y_train = train_df["Churn"].values
x_train = train_df.drop(columns=["Churn"]).values

x_test = test_df.values

os.makedirs("data/raw", exist_ok=True)

np.save("data/raw/x_train_v1.npy", x_train)
np.save("data/raw/y_train_v1.npy", y_train)
np.save("data/raw/x_test.npy", x_test)

print("Train:", x_train.shape)
print("Test:", x_test.shape)