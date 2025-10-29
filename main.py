import pathlib

import numpy as np
import pandas as pd
import torch


from torch.utils.data import DataLoader
from titanic_dataset import TitanicDataset


np.random.seed(42)

df_train = pd.read_csv('./train.csv')

columns = ["Sex", "Embarked"]
dict_encoding_maps = {}
for col in columns:
    unique_values = sorted(df_train[col].dropna().unique())
    dict_encoding_maps[col] = {val: i for i, val in enumerate(unique_values)}

    df_train[col] = df_train[col].map(dict_encoding_maps[col])
    # df[col] = df[col].map(lambda x: mapping.get(x, -1)) GPT 가 사용하라는 코드

n = len(df_train)
idx = np.random.permutation(n)
split = int(n * 0.8)
idx_train = idx[:split]
idx_val = idx[split:]

df = df_train.copy()
df_train = df.iloc[idx_train]
df_val = df.iloc[idx_val]

cat_features = ["Sex", "Embarked"]
num_features = ["Age", "Fare", "SibSp", "Parch"]
target = "Survived"

tensor_train = TitanicDataset(df_train, cat_features, num_features, target)
tensor_val = TitanicDataset(df_val, cat_features, num_features, target)

loader_train = DataLoader(tensor_train, batch_size=32, shuffle=True)
loader_val = DataLoader(tensor_val, batch_size=32, shuffle=False)