import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNetwork

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

#tokenizing
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w,tag))

#stemming
ignore_words = ['!','?','.',',']
all_words = sorted(set([stem(word) for word in all_words if word not in ignore_words]))
tags = sorted(set(tags))

#training set
X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)    

#torch dataset

class ChatDataset(Dataset):
    def __init__(self):
        self.num_samples = len(X_train)
        self.x = X_train
        self.y = y_train
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.num_samples

#hyperparameters
batch_size = 8
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
learning_rate = 0.001
num_epochs = 1000


dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size = batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNetwork(input_size, hidden_size, len(tags)).to(device=device)

#loss and optimiser
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not((epoch+1) % 100):
            print(f'epoch {epoch+1}: loss = {loss.item():.4f}')

print(f'final loss: {loss.item():.4f}')

data = {
    "model state": model.state_dict(),
    "input size": input_size,
    "output size": output_size,
    "hidden size": hidden_size,
    "tags": tags,
    "all words": all_words
}

FILE = 'data.pth'
torch.save(data, FILE)

print(f'Training complete, model saved to {FILE}')