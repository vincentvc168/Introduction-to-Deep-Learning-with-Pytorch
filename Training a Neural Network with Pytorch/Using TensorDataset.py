import pandas as pd
animals = pd.read_csv('animal_dataset.csv')

---
import numpy as np

# Define input features
features = animals.iloc[:, 1:-1]

X = features.to_numpy()
print(X)

# Define target values (ground truth)
target = animals.iloc[:, -1]
y = target.to_numpy()
print(y)

---
import torch
from torch.utils.data import TensorDataset

X = animals.iloc[:, 1:-1].to_numpy()  
y = animals.iloc[:, -1].to_numpy()

# Create a dataset
dataset = TensorDataset(torch.tensor(X), torch.tensor(y))

# Print the first sample
input_sample, label_sample = dataset[0]
print('Input sample:', input_sample)
print('Label sample:', label_sample)

---
from torch.utils.data import DataLoader

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Iterate over the dataloader
for batch_inputs, batch_labels in dataloader:
    print('batch_inputs:', batch_inputs)
    print('batch_labels:', batch_labels)
