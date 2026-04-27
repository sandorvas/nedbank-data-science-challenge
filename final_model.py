import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestRegressor

# load data

train = pd.read_csv("Train.csv")

test = pd.read_csv("Test.csv")

demo = pd.read_parquet("demographics_clean.parquet")

# merge demographics only

df = train.merge(demo, on="UniqueID", how="left")

df_test = test.merge(demo, on="UniqueID", how="left")

# BirthDate → Age

for d in [df, df_test]:

    d["BirthDate"] = pd.to_datetime(d["BirthDate"], errors="coerce")

    d["Age"] = 2026 - d["BirthDate"].dt.year

    d.drop(columns=["BirthDate"], inplace=True)

# target/features

y = df["next_3m_txn_count"]

X = df.drop(columns=["UniqueID", "next_3m_txn_count"])

X_test = df_test.drop(columns=["UniqueID"])

# encode

X = pd.get_dummies(X)

X_test = pd.get_dummies(X_test)

# align

X, X_test = X.align(X_test, join="left", axis=1, fill_value=0)

# fill missing

X = X.fillna(0)

X_test = X_test.fillna(0)

# model

model = RandomForestRegressor(

    n_estimators=50,

    random_state=42

)

model.fit(X, y)

# predict

preds = model.predict(X_test)

# the strange but apparently working transformation

preds = np.log1p(preds)

# submission

submission = pd.DataFrame({

    "UniqueID": test["UniqueID"],

    "next_3m_txn_count": preds

})

submission.to_csv("submission.csv", index=False)

print("submission.csv created")
