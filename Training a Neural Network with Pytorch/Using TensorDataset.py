import pandas as pd
animals = pd.read_csv('animal_dataset.csv')

import numpy as np

# Define input features
features = animals.iloc[:, 1:-1]

X = features.to_numpy()
print(X)

# Define target values (ground truth)
target = animals.iloc[:, -1]
y = target.to_numpy()
print(y)
