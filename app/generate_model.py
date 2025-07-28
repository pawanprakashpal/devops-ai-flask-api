import pickle
from sklearn.dummy import DummyClassifier

model = DummyClassifier(strategy='most_frequent')
model.fit([[0, 0]], [1])  # Initialize with dummy data

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)